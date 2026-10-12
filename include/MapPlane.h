#ifndef TIB_SLAM_MAP_PLANE_H
#define TIB_SLAM_MAP_PLANE_H

#include <Eigen/Core>

#include <atomic>
#include <map>
#include <mutex>
#include <vector>

namespace TIB_SLAM
{

class KeyFrame;
class Map;

class MapPlane
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using PointVector = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >;

    MapPlane(const Eigen::Vector4f& worldPlane, const PointVector& supportWorld,
             KeyFrame* referenceKeyFrame, Map* map);

    Eigen::Vector4f GetWorldPos() const;
    void SetWorldPos(const Eigen::Vector4f& worldPlane);
    PointVector GetSupportPoints() const;
    void FuseObservation(const Eigen::Vector4f& worldPlane,
                         const PointVector& supportWorld);

    void AddObservation(KeyFrame* keyFrame, std::size_t index);
    void EraseObservation(KeyFrame* keyFrame);
    std::map<KeyFrame*, std::size_t> GetObservations() const;
    int Observations() const;
    int GetIndexInKeyFrame(KeyFrame* keyFrame) const;

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
    Eigen::Vector3i color;

private:
    static std::atomic<unsigned long> nNextId;
    static bool NormalizePlane(Eigen::Vector4f& plane);

    mutable std::mutex mMutexGeometry;
    mutable std::mutex mMutexFeatures;
    Eigen::Vector4f mWorldPlane;
    PointVector mSupportWorld;
    std::map<KeyFrame*, std::size_t> mObservations;
    KeyFrame* mpRefKF;
    Map* mpMap;
    int mnVisible = 1;
    int mnFound = 1;
    bool mbBad = false;
};

} // namespace TIB_SLAM

#endif
