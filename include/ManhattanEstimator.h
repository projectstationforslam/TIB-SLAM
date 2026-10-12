#ifndef TIB_SLAM_MANHATTAN_ESTIMATOR_H
#define TIB_SLAM_MANHATTAN_ESTIMATOR_H

#include "StructuralTypes.h"

#include <Eigen/Core>
#include <map>

namespace TIB_SLAM
{

class Map;

class ManhattanEstimator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit ManhattanEstimator(float orthogonalCosineThreshold = 0.08716f,
                                float minimumConfidence = 0.25f,
                                float maximumRotationDelta = 0.61f);

    bool Update(StructuralFrameData& data, const Eigen::Matrix3f& Rcw, Map* map);
    void Reset(Map* map);

private:
    struct State
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        bool initialized = false;
        Eigen::Matrix3f lastRcw = Eigen::Matrix3f::Identity();
    };

    float mOrthogonalCosineThreshold;
    float mMinimumConfidence;
    float mMaximumRotationDelta;
    std::map<Map*, State> mStates;
};

} // namespace TIB_SLAM

#endif
