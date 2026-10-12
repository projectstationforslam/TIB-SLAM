#ifndef TIB_SLAM_MAP_LINE_H
#define TIB_SLAM_MAP_LINE_H

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

namespace TIB_SLAM
{

class KeyFrame;
class Map;

class MapLine
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector6f = Eigen::Matrix<float, 6, 1>;

    MapLine(const Eigen::Vector3f& startWorld, const Eigen::Vector3f& endWorld,
            KeyFrame* referenceKeyFrame, Map* map);

    Vector6f GetWorldPos() const;
    void SetWorldPos(const Eigen::Vector3f& startWorld, const Eigen::Vector3f& endWorld);
    Eigen::Vector3f GetMidPoint() const;
    Eigen::Vector3f GetDirection() const;
    cv::Mat GetDescriptor() const;
    void AddDescriptor(const cv::Mat& descriptor);

    void AddObservation(KeyFrame* keyFrame, std::size_t index);
    void EraseObservation(KeyFrame* keyFrame);
    std::map<KeyFrame*, std::size_t> GetObservations() const;
    int Observations() const;
    bool IsInKeyFrame(KeyFrame* keyFrame) const;

    void SetBadFlag();
    bool isBad() const;

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio() const;

    void ApplyScaledRotation(const Eigen::Matrix3f& rotation,
                             const Eigen::Vector3f& translation, float scale);

public:
    const unsigned long mnId;
    const long mnFirstKFid;
    unsigned long mnLastFrameSeen = 0;
    unsigned long mnTrackReferenceForFrame = 0;

private:
    static std::atomic<unsigned long> nNextId;

    mutable std::mutex mMutexGeometry;
    mutable std::mutex mMutexFeatures;
    Vector6f mWorldPos;
    std::map<KeyFrame*, std::size_t> mObservations;
    std::vector<cv::Mat> mDescriptorHistory;
    cv::Mat mDescriptor;
    KeyFrame* mpRefKF;
    Map* mpMap;
    int mnVisible = 1;
    int mnFound = 1;
    bool mbBad = false;
};

} // namespace TIB_SLAM

#endif
