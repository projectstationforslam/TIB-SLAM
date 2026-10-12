#ifndef TIB_SLAM_INFORMATION_BUDGET_SCHEDULER_H
#define TIB_SLAM_INFORMATION_BUDGET_SCHEDULER_H

#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace TIB_SLAM
{

// Pure, deterministic information-subspace scheduler.  It is intentionally independent of g2o
// so the observability policy can be unit-tested without launching SLAM.
class InformationBudgetScheduler
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;

    struct Config
    {
        bool enabled = false;
        bool temporalTransport = true;
        bool softObservability = true;
        bool greedyLogDet = true;
        // Review-revision ablation switches.  Defaults reproduce the published
        // full scheduler, so existing settings files are unchanged.
        bool adjointTransport = true;   // false: add history without adjoint transport
        bool useWeakWeights = true;     // false: unweighted log-determinant (W = I)
        int selectionPolicy = 0;        // 0 = recomputed greedy, 1 = one-shot sort,
                                        // 2 = random order, 3 = select all
        double temporalRetention = 0.70;
        double weakEigenvalueRatio = 1e-4;
        double softTransition = 0.35;
        double minimumActivation = 0.05;
        double regularization = 1e-6;
        double minimumMarginalGain = 1e-4;
        double factorBudget = 24.0;
        int maximumFactors = 24;
    };

    struct Candidate
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Matrix6d information = Matrix6d::Zero();
        double cost = 1.0;
        int externalIndex = -1;
    };

    struct State
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Matrix6d information = Matrix6d::Zero();
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        std::uint64_t frameId = 0;
        bool valid = false;
    };

    struct Result
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Matrix6d transportedInformation = Matrix6d::Zero();
        Matrix6d informationProjector = Matrix6d::Zero();
        Vector6d eigenvalues = Vector6d::Zero();
        Vector6d weakWeights = Vector6d::Zero();
        std::vector<int> selected;
        double totalGain = 0.0;
        double totalCost = 0.0;
        int effectiveRank = 0;
        bool valid = false;
    };

    static Result Select(const Matrix6d& pointInformation,
                         const Eigen::Matrix4d& currentPose,
                         std::uint64_t frameId,
                         const std::vector<Candidate,
                             Eigen::aligned_allocator<Candidate>>& candidates,
                         const Config& config,
                         State* state);

    static Matrix6d TransportInformation(const Matrix6d& information,
                                         const Eigen::Matrix4d& previousPose,
                                         const Eigen::Matrix4d& currentPose);
    static Vector6d SoftWeakness(const Vector6d& eigenvalues,
                                double weakEigenvalueRatio,
                                double transition);
    static Vector6d HardWeakness(const Vector6d& eigenvalues,
                                double weakEigenvalueRatio);
};

} // namespace TIB_SLAM

#endif
