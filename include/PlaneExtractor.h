#ifndef TIB_SLAM_PLANE_EXTRACTOR_H
#define TIB_SLAM_PLANE_EXTRACTOR_H

#include "StructuralTypes.h"

#include <opencv2/core.hpp>

namespace TIB_SLAM
{

class GeometricCamera;

class PlaneExtractor
{
public:
    PlaneExtractor(int stride = 2, int ransacIterations = 160,
                   int minInliers = 500, int maxPlanes = 8,
                   float distanceThreshold = 0.035f,
                   float minDepth = 0.15f, float maxDepth = 8.0f,
                   float voxelLeaf = 0.02f, float minimumInlierRatio = 0.72f);

    std::vector<StructuralPlaneObservation> Extract(const cv::Mat& depth,
                                                    GeometricCamera* camera,
                                                    unsigned long frameId) const;

private:
    int mStride;
    int mRansacIterations;
    int mMinInliers;
    int mMaxPlanes;
    float mDistanceThreshold;
    float mMinDepth;
    float mMaxDepth;
    float mVoxelLeaf;
    float mMinimumInlierRatio;
};

} // namespace TIB_SLAM

#endif
