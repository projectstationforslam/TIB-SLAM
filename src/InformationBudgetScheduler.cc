#include "InformationBudgetScheduler.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

namespace TIB_SLAM
{
namespace
{
using Matrix6d = InformationBudgetScheduler::Matrix6d;
using Vector6d = InformationBudgetScheduler::Vector6d;

Eigen::Matrix3d Skew(const Eigen::Vector3d& value)
{
    Eigen::Matrix3d result;
    result << 0.0, -value.z(), value.y(),
              value.z(), 0.0, -value.x(),
              -value.y(), value.x(), 0.0;
    return result;
}

double StableLogDet(const Matrix6d& matrix)
{
    Eigen::LLT<Matrix6d> factor(0.5 * (matrix + matrix.transpose()));
    if(factor.info() != Eigen::Success)
        return -std::numeric_limits<double>::infinity();
    return 2.0 * factor.matrixL().toDenseMatrix().diagonal().array().log().sum();
}
} // namespace

InformationBudgetScheduler::Matrix6d
InformationBudgetScheduler::TransportInformation(
    const Matrix6d& information, const Eigen::Matrix4d& previousPose,
    const Eigen::Matrix4d& currentPose)
{
    if(!information.allFinite() || !previousPose.allFinite() ||
       !currentPose.allFinite())
        return Matrix6d::Zero();
    const Eigen::Matrix4d relative = currentPose * previousPose.inverse();
    const Eigen::Matrix3d rotation = relative.block<3,3>(0,0);
    const Eigen::Vector3d translation = relative.block<3,1>(0,3);
    Matrix6d adjoint = Matrix6d::Zero();
    // g2o tangent ordering is [rotation, translation].
    adjoint.block<3,3>(0,0) = rotation;
    adjoint.block<3,3>(3,0) = Skew(translation) * rotation;
    adjoint.block<3,3>(3,3) = rotation;
    const Matrix6d inverse = adjoint.inverse();
    Matrix6d transported = inverse.transpose() * information * inverse;
    return 0.5 * (transported + transported.transpose());
}

InformationBudgetScheduler::Vector6d
InformationBudgetScheduler::SoftWeakness(const Vector6d& eigenvalues,
                                          double weakEigenvalueRatio,
                                          double transition)
{
    Vector6d weights = Vector6d::Zero();
    const double maximum = std::max(1e-12, eigenvalues.maxCoeff());
    const double threshold = std::max(1e-12, weakEigenvalueRatio) * maximum;
    const double scale = std::max(1e-3, transition);
    for(int index = 0; index < 6; ++index)
    {
        const double ratio = std::max(1e-15, eigenvalues(index)) / threshold;
        const double argument = std::max(-60.0, std::min(60.0,
            std::log(ratio) / scale));
        weights(index) = 1.0 / (1.0 + std::exp(argument));
    }
    return weights;
}

InformationBudgetScheduler::Vector6d
InformationBudgetScheduler::HardWeakness(const Vector6d& eigenvalues,
                                          double weakEigenvalueRatio)
{
    Vector6d weights = Vector6d::Zero();
    const double maximum = std::max(1e-12, eigenvalues.maxCoeff());
    const double threshold = std::max(1e-12, weakEigenvalueRatio) * maximum;
    for(int index = 0; index < 6; ++index)
        weights(index) = eigenvalues(index) < threshold ? 1.0 : 0.0;
    return weights;
}

InformationBudgetScheduler::Result InformationBudgetScheduler::Select(
    const Matrix6d& pointInformation, const Eigen::Matrix4d& currentPose,
    std::uint64_t frameId,
    const std::vector<Candidate, Eigen::aligned_allocator<Candidate>>& candidates,
    const Config& config, State* state)
{
    Result result;
    if(!pointInformation.allFinite() || !currentPose.allFinite())
        return result;
    Matrix6d current = 0.5 * (pointInformation + pointInformation.transpose());
    Matrix6d history = Matrix6d::Zero();
    if(config.temporalTransport && state && state->valid && frameId > state->frameId)
        history = config.adjointTransport
            ? TransportInformation(state->information, state->pose, currentPose)
            : state->information;
    const double retention = std::max(0.0, std::min(0.999, config.temporalRetention));
    result.transportedInformation = (1.0 - retention) * current + retention * history;
    if(!state || !state->valid || !config.temporalTransport)
        result.transportedInformation = current;
    result.transportedInformation = 0.5 *
        (result.transportedInformation + result.transportedInformation.transpose());

    Eigen::SelfAdjointEigenSolver<Matrix6d> solver(result.transportedInformation);
    if(solver.info() != Eigen::Success)
        return result;
    result.eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    result.weakWeights = config.softObservability ?
        SoftWeakness(result.eigenvalues, config.weakEigenvalueRatio,
                     config.softTransition) :
        HardWeakness(result.eigenvalues, config.weakEigenvalueRatio);
    const Eigen::Matrix<double,6,6> vectors = solver.eigenvectors();
    const double maximum = std::max(1e-12, result.eigenvalues.maxCoeff());
    const Vector6d normalized = result.eigenvalues / maximum;
    const Matrix6d weakRoot = config.useWeakWeights
        ? Matrix6d(result.weakWeights.cwiseSqrt().asDiagonal())
        : Matrix6d::Identity();
    Matrix6d selectedInformation = normalized.asDiagonal().toDenseMatrix();
    selectedInformation.diagonal().array() +=
        std::max(1e-12, config.regularization);
    Vector6d observableMask = Vector6d::Zero();
    for(int direction = 0; direction < 6; ++direction)
        if(result.weakWeights(direction) < 0.5)
            observableMask(direction) = 1.0;
    result.informationProjector = vectors * observableMask.asDiagonal() *
        vectors.transpose();
    result.effectiveRank = static_cast<int>(observableMask.sum());

    const auto shapedAddition = [&](std::size_t index) -> Matrix6d {
        Matrix6d addition = vectors.transpose() *
            (0.5 * (candidates[index].information +
                    candidates[index].information.transpose())) * vectors;
        return weakRoot * (addition / maximum) * weakRoot;
    };

    std::vector<bool> used(candidates.size(), false);
    if(config.selectionPolicy == 3)
    {
        // Select-all: no budget, no cardinality, no score.
        for(std::size_t index = 0; index < candidates.size(); ++index)
        {
            if(!candidates[index].information.allFinite())
                continue;
            const Matrix6d addition = shapedAddition(index);
            const double gain = StableLogDet(selectedInformation + addition) -
                StableLogDet(selectedInformation);
            selectedInformation += addition;
            used[index] = true;
            result.selected.push_back(candidates[index].externalIndex);
            result.totalGain += gain;
            result.totalCost += std::max(1e-6, candidates[index].cost);
        }
    }
    else
    {
        std::vector<std::size_t> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        if(config.selectionPolicy == 2)
        {
            // Random order with a fixed seed for reproducibility.
            std::mt19937 rng(20240701u);
            std::shuffle(order.begin(), order.end(), rng);
        }
        else if(config.selectionPolicy == 1)
        {
            // One-shot: score every candidate once against the empty selection,
            // then fill in descending score order.
            const double baseLogDet = StableLogDet(selectedInformation);
            std::vector<std::pair<double, std::size_t>> scored;
            scored.reserve(candidates.size());
            for(std::size_t index = 0; index < candidates.size(); ++index)
            {
                if(!candidates[index].information.allFinite())
                    continue;
                const double cost = std::max(1e-6, candidates[index].cost);
                const Matrix6d addition = shapedAddition(index);
                const double gain =
                    StableLogDet(selectedInformation + addition) - baseLogDet;
                const double score = config.greedyLogDet ? gain / cost : gain;
                scored.emplace_back(score, index);
            }
            std::sort(scored.begin(), scored.end(),
                [&](const std::pair<double, std::size_t>& a,
                    const std::pair<double, std::size_t>& b) {
                    if(std::abs(a.first - b.first) > 1e-12)
                        return a.first > b.first;
                    return candidates[a.second].externalIndex <
                           candidates[b.second].externalIndex;
                });
            order.clear();
            for(const auto& item : scored)
                order.push_back(item.second);
        }

        while(static_cast<int>(result.selected.size()) <
              std::max(0, config.maximumFactors))
        {
            if(config.selectionPolicy == 0)
            {
                // Recomputed greedy: re-evaluate each marginal gain against the
                // current selection, then take the best gain-per-cost.
                int best = -1;
                double bestGain = -std::numeric_limits<double>::infinity();
                double bestScore = -std::numeric_limits<double>::infinity();
                const double baseLogDet = StableLogDet(selectedInformation);
                for(std::size_t index : order)
                {
                    if(used[index] || !candidates[index].information.allFinite())
                        continue;
                    const double cost = std::max(1e-6, candidates[index].cost);
                    if(result.totalCost + cost > config.factorBudget + 1e-12)
                        continue;
                    const Matrix6d addition = shapedAddition(index);
                    const double gain =
                        StableLogDet(selectedInformation + addition) - baseLogDet;
                    const double score = config.greedyLogDet ? gain / cost : gain;
                    if(score > bestScore + 1e-12 ||
                       (std::abs(score - bestScore) <= 1e-12 &&
                        candidates[index].externalIndex <
                            (best >= 0 ? candidates[best].externalIndex :
                             std::numeric_limits<int>::max())))
                    {
                        best = static_cast<int>(index);
                        bestGain = gain;
                        bestScore = score;
                    }
                }
                if(best < 0 || bestGain < config.minimumMarginalGain ||
                   result.weakWeights.maxCoeff() < config.minimumActivation)
                    break;
                const Matrix6d addition = shapedAddition(best);
                selectedInformation += addition;
                used[best] = true;
                result.selected.push_back(candidates[best].externalIndex);
                result.totalGain += bestGain;
                result.totalCost += std::max(1e-6, candidates[best].cost);
            }
            else
            {
                // One-shot or random: fill in the precomputed order.
                bool added = false;
                for(std::size_t index : order)
                {
                    if(used[index] || !candidates[index].information.allFinite())
                        continue;
                    const double cost = std::max(1e-6, candidates[index].cost);
                    if(result.totalCost + cost > config.factorBudget + 1e-12)
                        continue;
                    const Matrix6d addition = shapedAddition(index);
                    const double gain =
                        StableLogDet(selectedInformation + addition) -
                        StableLogDet(selectedInformation);
                    if(gain < config.minimumMarginalGain)
                    {
                        used[index] = true;
                        continue;
                    }
                    selectedInformation += addition;
                    used[index] = true;
                    result.selected.push_back(candidates[index].externalIndex);
                    result.totalGain += gain;
                    result.totalCost += cost;
                    added = true;
                    break;
                }
                if(!added)
                    break;
            }
        }
    }

    if(state)
    {
        state->information = result.transportedInformation;
        state->pose = currentPose;
        state->frameId = frameId;
        state->valid = true;
    }
    result.valid = true;
    return result;
}

} // namespace TIB_SLAM
