#ifndef TIB_SLAM_LINE_EXTRACTOR_H
#define TIB_SLAM_LINE_EXTRACTOR_H

#include "StructuralTypes.h"

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

namespace TIB_SLAM
{

class GeometricCamera;

class LineExtractor
{
public:
    LineExtractor(float minPixelLength = 35.0f, int maxLines = 80,
                  int depthWindowRadius = 2, float minDepth = 0.15f,
                  float maxDepth = 8.0f, int scale = 2,
                  int octaves = 2);

    std::vector<StructuralLineObservation> Extract(const cv::Mat& gray,
                                                    const cv::Mat& depth,
                                                    GeometricCamera* camera) const;

private:
    bool ReadDepthPoint(const cv::Mat& depth, const cv::Point2f& pixel,
                        GeometricCamera* camera, Eigen::Vector3f& point) const;

    float mMinPixelLength;
    int mMaxLines;
    int mDepthWindowRadius;
    float mMinDepth;
    float mMaxDepth;
    int mScale;
    int mOctaves;
    cv::Ptr<cv::line_descriptor::LSDDetector> mpDetector;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> mpDescriptor;
};

} // namespace TIB_SLAM

#endif
