#include "MapPlane.h"

#include "KeyFrame.h"
#include "Map.h"

#include <algorithm>
#include <cmath>

namespace TIB_SLAM
{

std::atomic<unsigned long> MapPlane::nNextId(0);

bool MapPlane::NormalizePlane(Eigen::Vector4f& plane)
{
    const float norm = plane.head<3>().norm();
    if(!plane.allFinite() || norm < 1e-6f)
        return false;
    plane /= norm;
    return true;
}

MapPlane::MapPlane(const Eigen::Vector4f& worldPlane, const PointVector& supportWorld,
                   KeyFrame* referenceKeyFrame, Map* map)
    : mnId(nNextId.fetch_add(1)),
      mnFirstKFid(referenceKeyFrame ? static_cast<long>(referenceKeyFrame->mnId) : -1),
      mpRefKF(referenceKeyFrame), mpMap(map)
{
    mWorldPlane = worldPlane;
    if(!NormalizePlane(mWorldPlane))
        mWorldPlane << 0.0f, 0.0f, 1.0f, 0.0f;
    mSupportWorld = supportWorld;

    // Deterministic color avoids rand() data races in the original code.
    color << static_cast<int>((53 * mnId + 97) % 255),
             static_cast<int>((101 * mnId + 37) % 255),
             static_cast<int>((193 * mnId + 17) % 255);
}

Eigen::Vector4f MapPlane::GetWorldPos() const
{
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    return mWorldPlane;
}

void MapPlane::SetWorldPos(const Eigen::Vector4f& worldPlane)
{
    Eigen::Vector4f normalized = worldPlane;
    if(!NormalizePlane(normalized))
        return;
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    mWorldPlane = normalized;
}

MapPlane::PointVector MapPlane::GetSupportPoints() const
{
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    return mSupportWorld;
}

void MapPlane::FuseObservation(const Eigen::Vector4f& worldPlane,
                               const PointVector& supportWorld)
{
    Eigen::Vector4f normalized = worldPlane;
    if(!NormalizePlane(normalized))
        return;
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    if(normalized.head<3>().dot(mWorldPlane.head<3>()) < 0.0f)
        normalized = -normalized;
    Eigen::Vector4f fused = 0.8f * mWorldPlane + 0.2f * normalized;
    if(NormalizePlane(fused))
        mWorldPlane = fused;
    for(const Eigen::Vector3f& point : supportWorld)
        if(point.allFinite())
            mSupportWorld.push_back(point);
    // Bound memory independently of the number of keyframe observations.
    if(mSupportWorld.size() > 2000)
    {
        PointVector reduced;
        reduced.reserve(1000);
        for(std::size_t i = 0; i < mSupportWorld.size(); i += 2)
            reduced.push_back(mSupportWorld[i]);
        mSupportWorld.swap(reduced);
    }
}

void MapPlane::AddObservation(KeyFrame* keyFrame, std::size_t index)
{
    if(!keyFrame)
        return;
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mObservations.emplace(keyFrame, index);
    if(!mpRefKF)
        mpRefKF = keyFrame;
}

void MapPlane::EraseObservation(KeyFrame* keyFrame)
{
    if(!keyFrame)
        return;
    bool shouldBeBad = false;
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        const auto it = mObservations.find(keyFrame);
        if(it == mObservations.end())
            return;
        mObservations.erase(it);
        if(mpRefKF == keyFrame)
            mpRefKF = mObservations.empty() ? nullptr : mObservations.begin()->first;
        shouldBeBad = mObservations.empty();
    }
    if(shouldBeBad)
        SetBadFlag();
}

std::map<KeyFrame*, std::size_t> MapPlane::GetObservations() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPlane::Observations() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return static_cast<int>(mObservations.size());
}

int MapPlane::GetIndexInKeyFrame(KeyFrame* keyFrame) const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(keyFrame);
    return it == mObservations.end() ? -1 : static_cast<int>(it->second);
}

void MapPlane::SetBadFlag()
{
    std::map<KeyFrame*, std::size_t> observations;
    {
        std::lock_guard<std::mutex> lock(mMutexFeatures);
        if(mbBad)
            return;
        mbBad = true;
        observations.swap(mObservations);
        mpRefKF = nullptr;
    }
    for(const auto& observation : observations)
        if(observation.first)
            observation.first->EraseMapPlaneMatch(observation.second);
    if(mpMap)
        mpMap->EraseMapPlane(this);
}

bool MapPlane::isBad() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mbBad;
}

void MapPlane::IncreaseVisible(int n)
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mnVisible += std::max(0, n);
}

void MapPlane::IncreaseFound(int n)
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mnFound += std::max(0, n);
}

float MapPlane::GetFoundRatio() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mnVisible > 0 ? static_cast<float>(mnFound) / static_cast<float>(mnVisible) : 0.0f;
}

void MapPlane::ApplyScaledRotation(const Eigen::Matrix3f& rotation,
                                   const Eigen::Vector3f& translation, float scale)
{
    if(!rotation.allFinite() || !translation.allFinite() || !std::isfinite(scale) || scale <= 0.0f)
        return;
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    Eigen::Vector3f transformedNormal = rotation * mWorldPlane.head<3>();
    mWorldPlane(3) = scale * mWorldPlane(3) - transformedNormal.dot(translation);
    mWorldPlane.head<3>() = transformedNormal;
    NormalizePlane(mWorldPlane);
    for(Eigen::Vector3f& point : mSupportWorld)
        point = scale * rotation * point + translation;
}

} // namespace TIB_SLAM
