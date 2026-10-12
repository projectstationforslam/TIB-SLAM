#ifndef TIB_SLAM_PATCHTST_PREDICTOR_H
#define TIB_SLAM_PATCHTST_PREDICTOR_H

#include "StructuralTypes.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace TIB_SLAM
{

class Frame;

// Optional OpenCV-DNN bridge for a fixed-shape PatchTST ONNX model. The class
// is owned and called only by Tracking; it has no Python or LibTorch runtime
// dependency and is a no-op unless explicitly enabled in the camera YAML.
class PatchTSTPredictor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit PatchTSTPredictor(const std::string& settingsPath);
    ~PatchTSTPredictor();

    bool Enabled() const;
    bool Ready() const;
    void Reset();

    void ComputeImageStatistics(Frame& frame, const cv::Mat& gray,
                                const cv::Mat& depth) const;
    void AttachPrediction(Frame& current, const Frame& last);
    void Observe(Frame& frame, int trackingState, bool mapUpdated);

    static const char* FeatureName(std::size_t index);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace TIB_SLAM

#endif
