#include "MapLine.h"

#include "KeyFrame.h"
#include "Map.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace TIB_SLAM
{

std::atomic<unsigned long> MapLine::nNextId(0);

MapLine::MapLine(const Eigen::Vector3f& startWorld, const Eigen::Vector3f& endWorld,
                 KeyFrame* referenceKeyFrame, Map* map)
    : mnId(nNextId.fetch_add(1)),
      mnFirstKFid(referenceKeyFrame ? static_cast<long>(referenceKeyFrame->mnId) : -1),
      mpRefKF(referenceKeyFrame), mpMap(map)
{
    mWorldPos.head<3>() = startWorld;
    mWorldPos.tail<3>() = endWorld;
}

MapLine::Vector6f MapLine::GetWorldPos() const
{
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    return mWorldPos;
}

void MapLine::SetWorldPos(const Eigen::Vector3f& startWorld, const Eigen::Vector3f& endWorld)
{
    if(!startWorld.allFinite() || !endWorld.allFinite())
        return;
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    mWorldPos.head<3>() = startWorld;
    mWorldPos.tail<3>() = endWorld;
}

Eigen::Vector3f MapLine::GetMidPoint() const
{
    const Vector6f line = GetWorldPos();
    return 0.5f * (line.head<3>() + line.tail<3>());
}

Eigen::Vector3f MapLine::GetDirection() const
{
    const Vector6f line = GetWorldPos();
    Eigen::Vector3f direction = line.tail<3>() - line.head<3>();
    const float norm = direction.norm();
    if(norm <= 1e-6f)
        return Eigen::Vector3f::Zero();
    direction /= norm;
    return direction;
}

cv::Mat MapLine::GetDescriptor() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

void MapLine::AddDescriptor(const cv::Mat& descriptor)
{
    if(descriptor.empty() || descriptor.type() != CV_8U || descriptor.rows != 1)
        return;
    const cv::Mat candidate = descriptor.clone();
    if(candidate.empty() || !candidate.isContinuous())
        return;

    std::lock_guard<std::mutex> lock(mMutexFeatures);
    if(!mDescriptorHistory.empty() && candidate.cols != mDescriptorHistory.front().cols)
        return;
    // Keep a bounded history. LBD is binary, so the medoid under Hamming
    // distance is a stable representative and is safe to compare directly.
    if(mDescriptorHistory.size() == 20)
        mDescriptorHistory.erase(mDescriptorHistory.begin());
    mDescriptorHistory.push_back(candidate);

    std::size_t best = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for(std::size_t i = 0; i < mDescriptorHistory.size(); ++i)
    {
        double distance = 0.0;
        for(std::size_t j = 0; j < mDescriptorHistory.size(); ++j)
            distance += cv::norm(mDescriptorHistory[i], mDescriptorHistory[j], cv::NORM_HAMMING);
        if(distance < bestDistance)
        {
            bestDistance = distance;
            best = i;
        }
    }
    mDescriptor = mDescriptorHistory[best].clone();
}

void MapLine::AddObservation(KeyFrame* keyFrame, std::size_t index)
{
    if(!keyFrame)
        return;
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mObservations.emplace(keyFrame, index);
    if(!mpRefKF)
        mpRefKF = keyFrame;
}

void MapLine::EraseObservation(KeyFrame* keyFrame)
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

std::map<KeyFrame*, std::size_t> MapLine::GetObservations() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapLine::Observations() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return static_cast<int>(mObservations.size());
}

bool MapLine::IsInKeyFrame(KeyFrame* keyFrame) const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mObservations.count(keyFrame) != 0;
}

void MapLine::SetBadFlag()
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

    // Never call back into KeyFrame while holding the line mutex: KeyFrame
    // erasure can take its own feature lock.
    for(const auto& observation : observations)
        if(observation.first)
            observation.first->EraseMapLineMatch(observation.second);

    if(mpMap)
        mpMap->EraseMapLine(this);
}

bool MapLine::isBad() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mbBad;
}

void MapLine::IncreaseVisible(int n)
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mnVisible += std::max(0, n);
}

void MapLine::IncreaseFound(int n)
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    mnFound += std::max(0, n);
}

float MapLine::GetFoundRatio() const
{
    std::lock_guard<std::mutex> lock(mMutexFeatures);
    return mnVisible > 0 ? static_cast<float>(mnFound) / static_cast<float>(mnVisible) : 0.0f;
}

void MapLine::ApplyScaledRotation(const Eigen::Matrix3f& rotation,
                                  const Eigen::Vector3f& translation, float scale)
{
    if(!rotation.allFinite() || !translation.allFinite() || !std::isfinite(scale) || scale <= 0.0f)
        return;
    std::lock_guard<std::mutex> lock(mMutexGeometry);
    mWorldPos.head<3>() = scale * rotation * mWorldPos.head<3>() + translation;
    mWorldPos.tail<3>() = scale * rotation * mWorldPos.tail<3>() + translation;
}

} // namespace TIB_SLAM
