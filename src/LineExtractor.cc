#include "LineExtractor.h"

#include "CameraModels/GeometricCamera.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <algorithm>
#include <cmath>

namespace TIB_SLAM
{

LineExtractor::LineExtractor(float minPixelLength, int maxLines,
                             int depthWindowRadius, float minDepth, float maxDepth,
                             int scale, int octaves)
    : mMinPixelLength(std::max(5.0f, minPixelLength)),
      mMaxLines(std::max(1, maxLines)),
      mDepthWindowRadius(std::max(0, depthWindowRadius)),
      mMinDepth(std::max(0.01f, minDepth)),
      mMaxDepth(std::max(mMinDepth, maxDepth)),
      mScale(std::max(2, scale)),
      mOctaves(std::max(1, octaves)),
      mpDetector(cv::line_descriptor::LSDDetector::createLSDDetector()),
      mpDescriptor(cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor())
{
}

bool LineExtractor::ReadDepthPoint(const cv::Mat& depth, const cv::Point2f& pixel,
                                   GeometricCamera* camera, Eigen::Vector3f& point) const
{
    if(!camera || depth.empty() || depth.type() != CV_32F)
        return false;

    const int centerX = cvRound(pixel.x);
    const int centerY = cvRound(pixel.y);
    std::vector<float> samples;
    samples.reserve((2 * mDepthWindowRadius + 1) * (2 * mDepthWindowRadius + 1));
    for(int y = centerY - mDepthWindowRadius; y <= centerY + mDepthWindowRadius; ++y)
    {
        if(y < 0 || y >= depth.rows)
            continue;
        for(int x = centerX - mDepthWindowRadius; x <= centerX + mDepthWindowRadius; ++x)
        {
            if(x < 0 || x >= depth.cols)
                continue;
            const float value = depth.at<float>(y, x);
            if(std::isfinite(value) && value >= mMinDepth && value <= mMaxDepth)
                samples.push_back(value);
        }
    }
    if(samples.empty())
        return false;

    const std::size_t middle = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
    const float z = samples[middle];
    Eigen::Vector3f ray = camera->unprojectEig(pixel);
    if(!ray.allFinite() || std::abs(ray.z()) < 1e-6f)
        return false;
    point = ray * (z / ray.z());
    return point.allFinite() && point.z() > 0.0f;
}

std::vector<StructuralLineObservation> LineExtractor::Extract(const cv::Mat& gray,
                                                               const cv::Mat& depth,
                                                               GeometricCamera* camera) const
{
    std::vector<StructuralLineObservation> result;
    if(gray.empty() || !camera || !mpDetector || !mpDescriptor)
        return result;

    cv::Mat gray8;
    if(gray.type() == CV_8UC1)
        gray8 = gray;
    else
        gray.convertTo(gray8, CV_8U);

    // OpenCV's multi-octave LSDDetector enumerates every segment before the
    // caller can apply mMaxLines.  On high-texture Replica views that step
    // grows from ~50 ms to 30 s.  Detect on a bounded image with the standard
    // linear-time LSD implementation, rank there, and compute LBD only for
    // the retained segments at original resolution.
    const float detectionScale = std::min(
        1.0f, 640.0f / static_cast<float>(std::max(gray8.cols, gray8.rows)));
    cv::Mat detectionImage;
    if(detectionScale < 0.999f)
        cv::resize(gray8, detectionImage, cv::Size(), detectionScale,
                   detectionScale, cv::INTER_AREA);
    else
        detectionImage = gray8;
    std::vector<cv::Vec4f> segments;
    cv::Ptr<cv::LineSegmentDetector> detector =
        cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
    detector->detect(detectionImage, segments);

    std::vector<cv::line_descriptor::KeyLine> detected;
    detected.reserve(segments.size());
    const float inverseDetectionScale = 1.0f / detectionScale;
    for(const cv::Vec4f& segment : segments)
    {
        const float x0 = segment[0] * inverseDetectionScale;
        const float y0 = segment[1] * inverseDetectionScale;
        const float x1 = segment[2] * inverseDetectionScale;
        const float y1 = segment[3] * inverseDetectionScale;
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        const float length = std::sqrt(dx * dx + dy * dy);
        if(!std::isfinite(length) || length < mMinPixelLength)
            continue;
        cv::line_descriptor::KeyLine line;
        line.startPointX = x0;
        line.startPointY = y0;
        line.endPointX = x1;
        line.endPointY = y1;
        line.sPointInOctaveX = x0;
        line.sPointInOctaveY = y0;
        line.ePointInOctaveX = x1;
        line.ePointInOctaveY = y1;
        line.lineLength = length;
        line.numOfPixels = std::max(1, cvRound(length));
        line.angle = std::atan2(dy, dx);
        line.response = length /
            static_cast<float>(std::max(gray8.cols, gray8.rows));
        line.octave = 0;
        line.size = length;
        line.pt = cv::Point2f(0.5f * (x0 + x1), 0.5f * (y0 + y1));
        detected.push_back(line);
    }
    std::sort(detected.begin(), detected.end(), [](const cv::line_descriptor::KeyLine& lhs,
                                                    const cv::line_descriptor::KeyLine& rhs) {
        return lhs.response > rhs.response;
    });

    detected.erase(std::remove_if(detected.begin(), detected.end(), [this](const cv::line_descriptor::KeyLine& line) {
        return !std::isfinite(line.lineLength) || line.lineLength < mMinPixelLength ||
               !std::isfinite(line.startPointX) || !std::isfinite(line.startPointY) ||
               !std::isfinite(line.endPointX) || !std::isfinite(line.endPointY);
    }), detected.end());
    if(detected.size() > static_cast<std::size_t>(mMaxLines))
        detected.resize(static_cast<std::size_t>(mMaxLines));
    for(std::size_t i = 0; i < detected.size(); ++i)
        detected[i].class_id = static_cast<int>(i);

    cv::Mat descriptors;
    if(!detected.empty())
        mpDescriptor->compute(gray8, detected, descriptors);
    if(descriptors.empty() || descriptors.type() != CV_8U ||
       descriptors.rows != static_cast<int>(detected.size()))
        return result;

    result.reserve(std::min<std::size_t>(detected.size(), static_cast<std::size_t>(mMaxLines)));
    for(std::size_t index = 0; index < detected.size(); ++index)
    {
        if(static_cast<int>(result.size()) >= mMaxLines)
            break;

        StructuralLineObservation observation;
        observation.keyLine = detected[index];
        observation.startPixel = cv::Point2f(detected[index].startPointX, detected[index].startPointY);
        observation.endPixel = cv::Point2f(detected[index].endPointX, detected[index].endPointY);
        observation.pixelLength = cv::norm(observation.endPixel - observation.startPixel);
        if(!std::isfinite(observation.pixelLength) || observation.pixelLength < mMinPixelLength)
            continue;

        Eigen::Vector3f startH(observation.startPixel.x, observation.startPixel.y, 1.0f);
        Eigen::Vector3f endH(observation.endPixel.x, observation.endPixel.y, 1.0f);
        observation.imageLine = startH.cross(endH);
        const float normalNorm = observation.imageLine.head<2>().norm();
        if(normalNorm < 1e-6f || !observation.imageLine.allFinite())
            continue;
        observation.imageLine /= normalNorm;

        observation.hasDepth = ReadDepthPoint(depth, observation.startPixel, camera,
                                              observation.startCamera) &&
                               ReadDepthPoint(depth, observation.endPixel, camera,
                                              observation.endCamera);
        if(observation.hasDepth &&
           (observation.endCamera - observation.startCamera).norm() < 0.03f)
            observation.hasDepth = false;

        observation.descriptor = descriptors.row(static_cast<int>(index)).clone();
        if(observation.descriptor.empty() || !observation.descriptor.isContinuous())
            continue;
        result.push_back(std::move(observation));
    }
    return result;
}

} // namespace TIB_SLAM
