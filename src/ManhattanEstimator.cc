#include "ManhattanEstimator.h"

#include "KeyFrame.h"
#include "Map.h"
#include "MapPlane.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace TIB_SLAM
{
namespace
{

bool MakeBasis(const Eigen::Vector3f& firstInput, const Eigen::Vector3f& secondInput,
               Eigen::Matrix3f& basis)
{
    if(!firstInput.allFinite() || !secondInput.allFinite() ||
       firstInput.norm() < 1e-6f || secondInput.norm() < 1e-6f)
        return false;
    const Eigen::Vector3f first = firstInput.normalized();
    const Eigen::Vector3f secondUnit = secondInput.normalized();
    Eigen::Vector3f second = secondUnit - first * first.dot(secondUnit);
    if(second.norm() < 1e-5f)
        return false;
    second.normalize();
    Eigen::Vector3f third = first.cross(second);
    if(third.norm() < 1e-5f)
        return false;
    third.normalize();
    basis.col(0) = first;
    basis.col(1) = second;
    basis.col(2) = third;
    return basis.allFinite();
}

Eigen::Matrix3f ProjectToSO3(const Eigen::Matrix3f& input)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    const Eigen::Matrix3f V = svd.matrixV();
    if((U * V.transpose()).determinant() < 0.0f)
        U.col(2) *= -1.0f;
    return U * V.transpose();
}

} // namespace

ManhattanEstimator::ManhattanEstimator(float orthogonalCosineThreshold,
                                       float minimumConfidence,
                                       float maximumRotationDelta)
    : mOrthogonalCosineThreshold(std::max(0.01f, orthogonalCosineThreshold)),
      mMinimumConfidence(std::max(0.0f, std::min(1.0f, minimumConfidence))),
      mMaximumRotationDelta(std::max(0.05f, std::min(1.57f, maximumRotationDelta)))
{
}

bool ManhattanEstimator::Update(StructuralFrameData& data, const Eigen::Matrix3f& Rcw, Map* map)
{
    data.manhattan = ManhattanObservation();
    if(!map || !Rcw.allFinite() || data.planes.size() < 2)
        return false;

    Eigen::Matrix3f bestCameraBasis = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f bestWorldBasis = Eigen::Matrix3f::Identity();
    float bestScore = -1.0f;
    int mappedCount = 0;
    for(const StructuralPlaneObservation& plane : data.planes)
        if(plane.mapPlane && !plane.mapPlane->isBad())
            ++mappedCount;
    if(mappedCount < 2)
        return false;

    for(std::size_t i = 0; i < data.planes.size(); ++i)
    {
        const StructuralPlaneObservation& firstObservation = data.planes[i];
        if(!firstObservation.mapPlane || firstObservation.mapPlane->isBad())
            continue;
        Eigen::Vector3f firstWorld = firstObservation.mapPlane->GetWorldPos().head<3>();
        Eigen::Vector3f firstCamera = firstObservation.cameraPlane.head<3>();
        if(!firstWorld.allFinite() || !firstCamera.allFinite() ||
           firstWorld.norm() < 1e-6f || firstCamera.norm() < 1e-6f)
            continue;
        firstWorld.normalize();
        firstCamera.normalize();
        if((Rcw * firstWorld).dot(firstCamera) < 0.0f)
            firstCamera = -firstCamera;

        for(std::size_t j = i + 1; j < data.planes.size(); ++j)
        {
            const StructuralPlaneObservation& secondObservation = data.planes[j];
            if(!secondObservation.mapPlane || secondObservation.mapPlane->isBad() ||
               secondObservation.mapPlane == firstObservation.mapPlane)
                continue;
            Eigen::Vector3f secondWorld = secondObservation.mapPlane->GetWorldPos().head<3>();
            Eigen::Vector3f secondCamera = secondObservation.cameraPlane.head<3>();
            if(!secondWorld.allFinite() || !secondCamera.allFinite() ||
               secondWorld.norm() < 1e-6f || secondCamera.norm() < 1e-6f)
                continue;
            secondWorld.normalize();
            secondCamera.normalize();
            if((Rcw * secondWorld).dot(secondCamera) < 0.0f)
                secondCamera = -secondCamera;

            const float worldOrthogonality = std::abs(firstWorld.dot(secondWorld));
            const float cameraOrthogonality = std::abs(firstCamera.dot(secondCamera));
            if(worldOrthogonality > mOrthogonalCosineThreshold ||
               cameraOrthogonality > mOrthogonalCosineThreshold)
                continue;

            KeyFrame* anchor = map->GetPartialManhattanObservation(
                firstObservation.mapPlane, secondObservation.mapPlane);
            if(!anchor)
                continue;
            const int firstAnchorIndex =
                firstObservation.mapPlane->GetIndexInKeyFrame(anchor);
            const int secondAnchorIndex =
                secondObservation.mapPlane->GetIndexInKeyFrame(anchor);
            if(firstAnchorIndex < 0 || secondAnchorIndex < 0)
                continue;

            Eigen::Vector4f firstAnchorPlane, secondAnchorPlane;
            std::size_t firstAnchorSupport = 0;
            std::size_t secondAnchorSupport = 0;
            if(!anchor->GetPlaneObservation(
                   static_cast<std::size_t>(firstAnchorIndex),
                   firstAnchorPlane, firstAnchorSupport) ||
               !anchor->GetPlaneObservation(
                   static_cast<std::size_t>(secondAnchorIndex),
                   secondAnchorPlane, secondAnchorSupport))
                continue;

            const Eigen::Matrix3f anchorRcw = anchor->GetPose().rotationMatrix();
            Eigen::Vector3f firstAnchor = firstAnchorPlane.head<3>();
            Eigen::Vector3f secondAnchor = secondAnchorPlane.head<3>();
            if((anchorRcw * firstWorld).dot(firstAnchor) < 0.0f)
                firstAnchor = -firstAnchor;
            if((anchorRcw * secondWorld).dot(secondAnchor) < 0.0f)
                secondAnchor = -secondAnchor;

            Eigen::Matrix3f anchorBasis, cameraBasis;
            if(!MakeBasis(firstAnchor, secondAnchor, anchorBasis) ||
               !MakeBasis(firstCamera, secondCamera, cameraBasis))
                continue;
            const Eigen::Matrix3f worldBasis =
                ProjectToSO3(anchorRcw.transpose() * anchorBasis);
            const float support = std::min(1.0f, 2.0f *
                (firstObservation.supportRatio + secondObservation.supportRatio));
            const float refinement = 0.5f *
                (firstObservation.inlierRatio + secondObservation.inlierRatio);
            const float anchorSupport = std::min(1.0f,
                static_cast<float>(firstAnchorSupport + secondAnchorSupport) / 800.0f);
            const float observations = std::min(1.0f, 0.2f * static_cast<float>(
                firstObservation.mapPlane->Observations() +
                secondObservation.mapPlane->Observations()));
            const float score = support * std::max(0.5f, refinement) *
                                std::max(0.5f, observations) *
                                std::max(0.5f, anchorSupport) *
                                (1.0f - std::max(worldOrthogonality, cameraOrthogonality));
            if(score > bestScore)
            {
                bestScore = score;
                bestWorldBasis = worldBasis;
                bestCameraBasis = cameraBasis;
            }
        }
    }
    if(bestScore < mMinimumConfidence)
        return false;

    const Eigen::Matrix3f measuredRcw = ProjectToSO3(
        bestCameraBasis * bestWorldBasis.transpose());
    if(!measuredRcw.allFinite() || measuredRcw.determinant() < 0.99f)
        return false;
    const float delta = Eigen::AngleAxisf(measuredRcw * Rcw.transpose()).angle();
    if(!std::isfinite(delta) || delta > mMaximumRotationDelta)
        return false;

    State& state = mStates[map];
    if(state.initialized)
    {
        const float temporalDelta = Eigen::AngleAxisf(measuredRcw * state.lastRcw.transpose()).angle();
        if(!std::isfinite(temporalDelta) || temporalDelta > 1.2f)
            return false;
    }
    state.initialized = true;
    state.lastRcw = measuredRcw;

    data.manhattan.valid = true;
    data.manhattan.confidence = std::min(1.0f, bestScore);
    data.manhattan.cameraAxes = bestCameraBasis;
    data.manhattan.measuredRcw = measuredRcw;
    data.manhattan.matchedPlaneCount = mappedCount;
    return true;
}

void ManhattanEstimator::Reset(Map* map)
{
    if(map)
        mStates.erase(map);
    else
        mStates.clear();
}

} // namespace TIB_SLAM
