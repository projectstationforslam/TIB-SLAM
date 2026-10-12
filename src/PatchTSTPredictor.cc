#include "PatchTSTPredictor.h"

#include "Frame.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Eigenvalues>
#include <sophus/se3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include <system_error>

namespace TIB_SLAM
{
namespace
{

template<typename T>
T ReadSetting(const cv::FileStorage& settings, const char* key, const T& fallback)
{
    const cv::FileNode node = settings[key];
    if(node.empty())
        return fallback;
    T value = fallback;
    node >> value;
    return value;
}

float ClampFinite(float value, float low, float high, float fallback = 0.0f)
{
    return std::isfinite(value) ? std::max(low, std::min(high, value)) : fallback;
}

float Logistic(float value)
{
    value = ClampFinite(value, -20.0f, 20.0f);
    return 1.0f / (1.0f + std::exp(-value));
}

std::string ResolvePath(const std::string& settingsPath, const std::string& value)
{
    if(value.empty())
        return value;
    std::filesystem::path path(value);
    if(path.is_relative())
        path = std::filesystem::path(settingsPath).parent_path() / path;
    return path.lexically_normal().string();
}

float ResidualConsistency(float first, float second)
{
    if(first <= 0.0f || second <= 0.0f || !std::isfinite(first) || !std::isfinite(second))
        return 0.0f;
    return std::exp(-std::abs(std::log((first + 1e-4f) / (second + 1e-4f))));
}

} // namespace

struct PatchTSTPredictor::Impl
{
    bool enabled = false;
    bool collectTelemetry = false;
    bool usePoseInitialization = false;
    bool useTemporalPrior = false;
    bool useFactorWeights = false;
    bool ready = false;
    int contextLength = 60;
    float maximumTranslation = 0.35f;
    float maximumRotation = 0.6f;
    float maximumVariance = 1.0f;
    float minimumFactorWeight = 0.20f;
    float maximumFactorWeight = 5.0f;
    float minimumConfidence = 0.05f;
    float minimumDegenerationProbability = 0.5f;

    cv::dnn::Net net;
    std::deque<std::array<float, kSlamFeatureCount> > history;
    LearnedPosePrior cachedPrediction;
    bool hasPrediction = false;
    bool hasLastPose = false;
    Sophus::SE3f lastTwc;
    double lastTimestamp = 0.0;
    float lastDt = 1.0f / 30.0f;
    std::ofstream telemetry;

    void RunModel()
    {
        hasPrediction = false;
        cachedPrediction = LearnedPosePrior();
        if(!ready || history.size() < static_cast<std::size_t>(contextLength))
            return;

        int dimensions[3] = {1, contextLength, static_cast<int>(kSlamFeatureCount)};
        cv::Mat input(3, dimensions, CV_32F);
        for(int t = 0; t < contextLength; ++t)
        {
            const auto& row = history[history.size() - contextLength + t];
            for(std::size_t feature = 0; feature < kSlamFeatureCount; ++feature)
                input.at<float>(0, t, static_cast<int>(feature)) = row[feature];
        }

        try
        {
            net.setInput(input);
            cv::Mat output = net.forward();
            if(output.empty() || output.total() < 18 || output.type() != CV_32F)
                return;
            cv::Mat flat = output.reshape(1, 1);
            LearnedPosePrior prediction;
            for(int i = 0; i < 6; ++i)
            {
                prediction.relativeIncrement(i) = flat.at<float>(0, i);
                const float logVariance = ClampFinite(flat.at<float>(0, 6 + i), -10.0f, 5.0f);
                prediction.variance(i) = std::exp(logVariance);
            }
            for(std::size_t factor = 0; factor < kLearnedFactorCount; ++factor)
            {
                const float logWeight = ClampFinite(
                    flat.at<float>(0, 12 + static_cast<int>(factor)), -4.0f, 4.0f);
                prediction.factorWeights[factor] = ClampFinite(
                    std::exp(logWeight), minimumFactorWeight, maximumFactorWeight, 1.0f);
            }
            prediction.degenerationProbability = Logistic(flat.at<float>(0, 17));

            const float translation = prediction.relativeIncrement.head<3>().norm();
            const float rotation = prediction.relativeIncrement.tail<3>().norm();
            const float meanVariance = prediction.variance.mean();
            prediction.confidence = 1.0f / (1.0f + std::max(0.0f, meanVariance));
            prediction.usePoseInitialization = usePoseInitialization;
            prediction.useTemporalPrior = useTemporalPrior;
            prediction.useFactorWeights = useFactorWeights;
            prediction.valid = prediction.relativeIncrement.allFinite() &&
                prediction.variance.allFinite() && translation <= maximumTranslation &&
                rotation <= maximumRotation &&
                prediction.variance.maxCoeff() <= maximumVariance &&
                prediction.confidence >= minimumConfidence;
            if(prediction.valid)
            {
                cachedPrediction = prediction;
                hasPrediction = true;
            }
        }
        catch(const cv::Exception& error)
        {
            std::cerr << "[PatchTST] inference disabled after OpenCV-DNN error: "
                      << error.what() << std::endl;
            ready = false;
        }
    }
};

static const std::array<const char*, kSlamFeatureCount> kFeatureNames{{
    "dt",
    "relative_tx", "relative_ty", "relative_tz",
    "relative_rx", "relative_ry", "relative_rz",
    "velocity_tx", "velocity_ty", "velocity_tz",
    "velocity_rx", "velocity_ry", "velocity_rz",
    "orb_keypoint_count", "point_inliers", "line_inliers", "plane_inliers",
    "point_reprojection_rms", "line_reprojection_rms",
    "plane_normal_rms", "plane_distance_rms",
    "image_gradient_mean", "image_gradient_std", "image_entropy",
    "texture_ratio", "depth_valid_ratio",
    "manhattan_confidence", "manhattan_rotation_error", "manhattan_valid",
    "hessian_min_eigenvalue", "hessian_log_condition",
    "point_grid_occupancy", "point_grid_entropy",
    "line_orientation_coverage", "plane_normal_diversity",
    "point_line_residual_consistency", "point_plane_residual_consistency",
    "line_plane_residual_consistency", "tracking_state", "map_updated"
}};

PatchTSTPredictor::PatchTSTPredictor(const std::string& settingsPath)
    : mImpl(new Impl())
{
    cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
    if(!settings.isOpened())
        return;

    mImpl->enabled = ReadSetting<int>(settings, "Structure.PatchTST.Enable", 0) != 0;
    mImpl->collectTelemetry =
        ReadSetting<int>(settings, "Structure.PatchTST.CollectTelemetry", 0) != 0;
    mImpl->usePoseInitialization = ReadSetting<int>(
        settings, "Structure.PatchTST.UsePoseInitialization", 0) != 0;
    mImpl->useTemporalPrior = ReadSetting<int>(
        settings, "Structure.PatchTST.UseTemporalPrior", 0) != 0;
    mImpl->useFactorWeights = ReadSetting<int>(
        settings, "Structure.PatchTST.UseFactorWeights", 0) != 0;
    mImpl->contextLength = std::max(
        4, ReadSetting<int>(settings, "Structure.PatchTST.ContextLength", 60));
    mImpl->maximumTranslation = std::max(
        0.01f, ReadSetting<float>(settings, "Structure.PatchTST.MaxTranslation", 0.35f));
    mImpl->maximumRotation = std::max(
        0.01f, ReadSetting<float>(settings, "Structure.PatchTST.MaxRotation", 0.6f));
    mImpl->maximumVariance = std::max(
        1e-4f, ReadSetting<float>(settings, "Structure.PatchTST.MaxVariance", 1.0f));
    mImpl->minimumFactorWeight = std::max(
        0.01f, ReadSetting<float>(settings, "Structure.PatchTST.MinFactorWeight", 0.20f));
    mImpl->maximumFactorWeight = std::max(
        mImpl->minimumFactorWeight,
        ReadSetting<float>(settings, "Structure.PatchTST.MaxFactorWeight", 5.0f));
    mImpl->minimumConfidence = ClampFinite(
        ReadSetting<float>(settings, "Structure.PatchTST.MinConfidence", 0.05f),
        0.0f, 1.0f, 0.05f);
    mImpl->minimumDegenerationProbability = ClampFinite(
        ReadSetting<float>(settings,
            "Structure.PatchTST.MinDegenerationProbability", 0.5f),
        0.0f, 1.0f, 0.5f);

    const std::string telemetryPath = ResolvePath(
        settingsPath,
        ReadSetting<std::string>(settings, "Structure.PatchTST.TelemetryPath", ""));
    if(mImpl->collectTelemetry && !telemetryPath.empty())
    {
        const std::filesystem::path parent =
            std::filesystem::path(telemetryPath).parent_path();
        std::error_code directoryError;
        if(!parent.empty())
            std::filesystem::create_directories(parent, directoryError);
        mImpl->telemetry.open(telemetryPath);
        if(mImpl->telemetry)
        {
            mImpl->telemetry << std::setprecision(17);
            mImpl->telemetry << "timestamp";
            for(const char* name : kFeatureNames)
                mImpl->telemetry << ',' << name;
            mImpl->telemetry
                << ",twc_tx,twc_ty,twc_tz,twc_qx,twc_qy,twc_qz,twc_qw,tracking_ok\n";
        }
        else
            std::cerr << "[PatchTST] cannot open telemetry file: "
                      << telemetryPath << std::endl;
    }

    if(!mImpl->enabled)
        return;

    const std::string modelPath = ResolvePath(
        settingsPath,
        ReadSetting<std::string>(settings, "Structure.PatchTST.ModelPath", ""));
    if(modelPath.empty() || !std::filesystem::exists(modelPath))
    {
        std::cerr << "[PatchTST] model not found; learned prior is disabled: "
                  << modelPath << std::endl;
        return;
    }

    try
    {
        mImpl->net = cv::dnn::readNetFromONNX(modelPath);
        mImpl->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        mImpl->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        mImpl->ready = !mImpl->net.empty();
        if(mImpl->ready)
            std::cout << "[PatchTST] loaded ONNX model: " << modelPath << std::endl;
    }
    catch(const cv::Exception& error)
    {
        std::cerr << "[PatchTST] cannot load ONNX model: " << error.what() << std::endl;
    }
}

PatchTSTPredictor::~PatchTSTPredictor() = default;

bool PatchTSTPredictor::Enabled() const
{
    return mImpl && (mImpl->enabled || mImpl->collectTelemetry);
}

bool PatchTSTPredictor::Ready() const
{
    return mImpl && mImpl->ready;
}

void PatchTSTPredictor::Reset()
{
    if(!mImpl)
        return;
    mImpl->history.clear();
    mImpl->cachedPrediction = LearnedPosePrior();
    mImpl->hasPrediction = false;
    mImpl->hasLastPose = false;
    mImpl->lastTimestamp = 0.0;
}

const char* PatchTSTPredictor::FeatureName(std::size_t index)
{
    return index < kFeatureNames.size() ? kFeatureNames[index] : "";
}

void PatchTSTPredictor::ComputeImageStatistics(
    Frame& frame, const cv::Mat& gray, const cv::Mat& depth) const
{
    if(!Enabled() || !frame.mpStructuralData || gray.empty() || depth.empty())
        return;

    SlamTelemetry& telemetry = frame.mpStructuralData->telemetry;
    cv::Mat gray8;
    if(gray.type() == CV_8U)
        gray8 = gray;
    else
        gray.convertTo(gray8, CV_8U);

    cv::Mat gradientX, gradientY, magnitude;
    cv::Sobel(gray8, gradientX, CV_32F, 1, 0, 3);
    cv::Sobel(gray8, gradientY, CV_32F, 0, 1, 3);
    cv::magnitude(gradientX, gradientY, magnitude);
    cv::Scalar mean, standardDeviation;
    cv::meanStdDev(magnitude, mean, standardDeviation);
    telemetry.imageGradientMean = static_cast<float>(mean[0]);
    telemetry.imageGradientStd = static_cast<float>(standardDeviation[0]);
    telemetry.textureRatio = static_cast<float>(
        cv::countNonZero(magnitude > 20.0f)) /
        static_cast<float>(std::max<std::size_t>(1, magnitude.total()));

    int histogramSize = 256;
    float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};
    cv::Mat histogram;
    cv::calcHist(&gray8, 1, nullptr, cv::Mat(), histogram, 1,
                 &histogramSize, ranges, true, false);
    const float totalPixels = static_cast<float>(std::max<std::size_t>(1, gray8.total()));
    float entropy = 0.0f;
    for(int value = 0; value < histogramSize; ++value)
    {
        const float probability = histogram.at<float>(value) / totalPixels;
        if(probability > 0.0f)
            entropy -= probability * std::log2(probability);
    }
    telemetry.imageEntropy = entropy;

    std::size_t validDepth = 0;
    if(depth.type() == CV_32F)
    {
        for(int row = 0; row < depth.rows; ++row)
        {
            const float* values = depth.ptr<float>(row);
            for(int column = 0; column < depth.cols; ++column)
                if(std::isfinite(values[column]) && values[column] > 0.0f)
                    ++validDepth;
        }
    }
    telemetry.depthValidRatio = static_cast<float>(validDepth) /
        static_cast<float>(std::max<std::size_t>(1, depth.total()));
}

void PatchTSTPredictor::AttachPrediction(Frame& current, const Frame& last)
{
    if(!mImpl || !mImpl->ready || !mImpl->hasPrediction ||
       !current.mpStructuralData || !last.isSet())
        return;

    LearnedPosePrior prediction = mImpl->cachedPrediction;
    // Current image/depth and raw structural counts are already available at
    // GrabImageRGBD time. Apply a deterministic last-mile gate so an abrupt
    // low-texture frame is handled immediately rather than one frame later.
    const SlamTelemetry& currentTelemetry =
        current.mpStructuralData->telemetry;
    const bool currentDegenerate =
        currentTelemetry.textureRatio < 0.06f || current.N < 150 ||
        currentTelemetry.depthValidRatio < 0.20f;
    if(currentDegenerate)
    {
        prediction.degenerationProbability =
            std::max(0.75f, prediction.degenerationProbability);
        prediction.factorWeights[LearnedPointFactor] = ClampFinite(
            prediction.factorWeights[LearnedPointFactor] * 0.5f,
            mImpl->minimumFactorWeight, mImpl->maximumFactorWeight, 1.0f);
        const float structuralBoost =
            current.mpStructuralData->lines.empty() &&
            current.mpStructuralData->planes.empty() ? 1.0f : 1.35f;
        for(std::size_t factor : {LearnedLineFactor, LearnedPlaneFactor,
                                  LearnedManhattanFactor})
            prediction.factorWeights[factor] = ClampFinite(
                prediction.factorWeights[factor] * structuralBoost,
                mImpl->minimumFactorWeight, mImpl->maximumFactorWeight, 1.0f);
        prediction.factorWeights[LearnedTemporalPriorFactor] = ClampFinite(
            prediction.factorWeights[LearnedTemporalPriorFactor] * 1.5f,
            mImpl->minimumFactorWeight, mImpl->maximumFactorWeight, 1.0f);
    }
    if(prediction.degenerationProbability <
       mImpl->minimumDegenerationProbability)
        return;

    const double dt = current.mTimeStamp - last.mTimeStamp;
    if(!std::isfinite(dt) || dt <= 0.0 || dt > 1.0)
        return;
    const float timeScale = ClampFinite(
        static_cast<float>(dt) / std::max(1e-3f, mImpl->lastDt), 0.25f, 4.0f, 1.0f);
    prediction.relativeIncrement *= timeScale;
    prediction.variance *= timeScale * timeScale;

    if(!prediction.relativeIncrement.allFinite() ||
       prediction.relativeIncrement.head<3>().norm() >
           4.0f * mImpl->maximumTranslation ||
       prediction.relativeIncrement.tail<3>().norm() >
           4.0f * mImpl->maximumRotation)
        return;
    const Sophus::SE3f predictedTwc =
        last.GetPose().inverse() * Sophus::SE3f::exp(prediction.relativeIncrement);
    const Sophus::SE3f predictedTcw = predictedTwc.inverse();
    if(!predictedTcw.matrix().allFinite())
        return;
    prediction.predictedTcw = predictedTcw.matrix();
    prediction.valid = prediction.relativeIncrement.head<3>().norm() <=
                           mImpl->maximumTranslation * std::max(1.0f, timeScale) &&
                       prediction.relativeIncrement.tail<3>().norm() <=
                           mImpl->maximumRotation * std::max(1.0f, timeScale) &&
                       prediction.variance.maxCoeff() <=
                           mImpl->maximumVariance * std::max(1.0f, timeScale * timeScale);
    if(prediction.valid)
        current.mpStructuralData->learnedPrior = prediction;
}

void PatchTSTPredictor::Observe(Frame& frame, int trackingState, bool mapUpdated)
{
    if(!Enabled() || !frame.mpStructuralData || !frame.isSet())
        return;

    StructuralFrameData& structural = *frame.mpStructuralData;
    auto& features = structural.telemetry.features;
    features.fill(0.0f);

    const Sophus::SE3f currentTwc = frame.GetPose().inverse();
    float dt = 0.0f;
    Eigen::Matrix<float, 6, 1> relative =
        Eigen::Matrix<float, 6, 1>::Zero();
    if(mImpl->hasLastPose)
    {
        dt = static_cast<float>(frame.mTimeStamp - mImpl->lastTimestamp);
        if(std::isfinite(dt) && dt > 1e-4f && dt <= 1.0f)
            relative = (mImpl->lastTwc.inverse() * currentTwc).log();
        else
            dt = 0.0f;
    }
    if(!relative.allFinite())
        relative.setZero();

    features[0] = dt;
    for(int i = 0; i < 6; ++i)
    {
        features[1 + i] = relative(i);
        features[7 + i] = dt > 1e-4f ? relative(i) / dt : 0.0f;
    }
    features[13] = static_cast<float>(std::max(0, frame.N));

    int pointInliers = 0;
    std::array<int, 48> pointCells{};
    const std::size_t pointCount = std::min(
        frame.mvpMapPoints.size(), frame.mvbOutlier.size());
    for(std::size_t index = 0; index < pointCount; ++index)
    {
        if(!frame.mvpMapPoints[index] || frame.mvbOutlier[index])
            continue;
        ++pointInliers;
        if(index < frame.mvKeysUn.size())
        {
            const cv::Point2f point = frame.mvKeysUn[index].pt;
            const float width = std::max(1.0f, Frame::mnMaxX - Frame::mnMinX);
            const float height = std::max(1.0f, Frame::mnMaxY - Frame::mnMinY);
            const int x = std::max(0, std::min(7,
                static_cast<int>(8.0f * (point.x - Frame::mnMinX) / width)));
            const int y = std::max(0, std::min(5,
                static_cast<int>(6.0f * (point.y - Frame::mnMinY) / height)));
            ++pointCells[static_cast<std::size_t>(y * 8 + x)];
        }
    }
    features[14] = static_cast<float>(pointInliers);

    int lineInliers = 0;
    std::array<int, 12> lineBins{};
    for(const StructuralLineObservation& line : structural.lines)
    {
        if(!line.mapLine || line.outlier)
            continue;
        ++lineInliers;
        const cv::Point2f direction = line.endPixel - line.startPixel;
        float angle = std::atan2(direction.y, direction.x);
        if(angle < 0.0f)
            angle += static_cast<float>(CV_PI);
        if(angle >= static_cast<float>(CV_PI))
            angle -= static_cast<float>(CV_PI);
        const int bin = std::max(0, std::min(11,
            static_cast<int>(12.0f * angle / static_cast<float>(CV_PI))));
        ++lineBins[static_cast<std::size_t>(bin)];
    }
    features[15] = static_cast<float>(lineInliers);

    int planeInliers = 0;
    std::vector<Eigen::Vector3f> planeNormals;
    for(const StructuralPlaneObservation& plane : structural.planes)
    {
        if(!plane.mapPlane || plane.outlier)
            continue;
        ++planeInliers;
        if(plane.cameraPlane.head<3>().allFinite())
            planeNormals.push_back(plane.cameraPlane.head<3>().normalized());
    }
    features[16] = static_cast<float>(planeInliers);
    features[17] = structural.telemetry.pointResidualRms;
    features[18] = structural.telemetry.lineResidualRms;
    features[19] = structural.telemetry.planeNormalResidualRms;
    features[20] = structural.telemetry.planeDistanceResidualRms;
    features[21] = structural.telemetry.imageGradientMean;
    features[22] = structural.telemetry.imageGradientStd;
    features[23] = structural.telemetry.imageEntropy;
    features[24] = structural.telemetry.textureRatio;
    features[25] = structural.telemetry.depthValidRatio;
    features[26] = structural.manhattan.confidence;
    features[28] = structural.manhattan.valid ? 1.0f : 0.0f;
    if(structural.manhattan.valid)
    {
        const Eigen::Matrix3f errorRotation =
            structural.manhattan.measuredRcw * frame.GetPose().rotationMatrix().transpose();
        features[27] = Eigen::AngleAxisf(errorRotation).angle();
    }
    features[29] = structural.telemetry.hessianMinEigenvalue;
    features[30] = structural.telemetry.hessianLogCondition;

    int occupiedCells = 0;
    float pointEntropy = 0.0f;
    for(int count : pointCells)
    {
        if(count > 0)
            ++occupiedCells;
        if(pointInliers > 0 && count > 0)
        {
            const float probability = static_cast<float>(count) /
                                      static_cast<float>(pointInliers);
            pointEntropy -= probability * std::log(probability);
        }
    }
    features[31] = static_cast<float>(occupiedCells) /
                   static_cast<float>(pointCells.size());
    features[32] = pointCells.size() > 1 ?
        pointEntropy / std::log(static_cast<float>(pointCells.size())) : 0.0f;

    const int occupiedLineBins = static_cast<int>(std::count_if(
        lineBins.begin(), lineBins.end(), [](int count) { return count > 0; }));
    features[33] = static_cast<float>(occupiedLineBins) /
                   static_cast<float>(lineBins.size());

    float planeDiversity = 0.0f;
    int planePairs = 0;
    for(std::size_t first = 0; first < planeNormals.size(); ++first)
        for(std::size_t second = first + 1; second < planeNormals.size(); ++second)
        {
            planeDiversity += 1.0f -
                std::abs(planeNormals[first].dot(planeNormals[second]));
            ++planePairs;
        }
    features[34] = planePairs > 0 ? planeDiversity / planePairs : 0.0f;
    const float planeResidual = std::hypot(
        structural.telemetry.planeNormalResidualRms,
        structural.telemetry.planeDistanceResidualRms);
    features[35] = ResidualConsistency(
        structural.telemetry.pointResidualRms, structural.telemetry.lineResidualRms);
    features[36] = ResidualConsistency(
        structural.telemetry.pointResidualRms, planeResidual);
    features[37] = ResidualConsistency(
        structural.telemetry.lineResidualRms, planeResidual);
    features[38] = static_cast<float>(trackingState);
    features[39] = mapUpdated ? 1.0f : 0.0f;
    structural.telemetry.featuresValid = true;

    if(mImpl->telemetry)
    {
        const Eigen::Vector3f translation = currentTwc.translation();
        const Eigen::Quaternionf quaternion = currentTwc.unit_quaternion();
        mImpl->telemetry << frame.mTimeStamp;
        for(float value : features)
            mImpl->telemetry << ',' << value;
        mImpl->telemetry << ',' << translation.x() << ',' << translation.y()
                         << ',' << translation.z() << ',' << quaternion.x()
                         << ',' << quaternion.y() << ',' << quaternion.z()
                         << ',' << quaternion.w() << ','
                         << (trackingState == 2 ? 1 : 0) << std::endl;
    }

    mImpl->history.push_back(features);
    while(mImpl->history.size() > static_cast<std::size_t>(mImpl->contextLength))
        mImpl->history.pop_front();
    if(dt > 1e-4f)
        mImpl->lastDt = dt;
    mImpl->lastTwc = currentTwc;
    mImpl->lastTimestamp = frame.mTimeStamp;
    mImpl->hasLastPose = true;
    mImpl->RunModel();
}

} // namespace TIB_SLAM
