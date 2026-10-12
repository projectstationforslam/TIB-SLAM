#include "DirectRgbdPoseRefiner.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace TIB_SLAM
{
namespace
{

template<typename T>
void ReadScalar(const cv::FileStorage& file, const char* key, T& value)
{
    const cv::FileNode node = file[key];
    if(!node.empty()) node >> value;
}

} // namespace

DirectRgbdPoseRefiner::DirectRgbdPoseRefiner(const std::string& settingsPath)
{
    cv::FileStorage file(settingsPath, cv::FileStorage::READ);
    int enabled = mConfig.enabled ? 1 : 0;
    int recoveryEnabled = mConfig.recoveryEnabled ? 1 : 0;
    int uncertaintyGate = mConfig.uncertaintyGate ? 1 : 0;
    int commitOnlyWhenDegenerate = mConfig.commitOnlyWhenDegenerate ? 1 : 0;
    int applyPoseCorrection = mConfig.applyPoseCorrection ? 1 : 0;
    int degradationAwareSubspace = mConfig.degradationAwareSubspace ? 1 : 0;
    ReadScalar(file, "DirectPose.Enable", enabled);
    ReadScalar(file, "DirectPose.RecoveryEnable", recoveryEnabled);
    ReadScalar(file, "DirectPose.UncertaintyGate", uncertaintyGate);
    ReadScalar(file, "DirectPose.CommitOnlyWhenDegenerate", commitOnlyWhenDegenerate);
    ReadScalar(file, "DirectPose.ApplyPoseCorrection", applyPoseCorrection);
    ReadScalar(file, "DirectPose.DegradationAwareSubspace", degradationAwareSubspace);
    mConfig.enabled = enabled != 0;
    mConfig.recoveryEnabled = recoveryEnabled != 0;
    mConfig.uncertaintyGate = uncertaintyGate != 0;
    mConfig.commitOnlyWhenDegenerate = commitOnlyWhenDegenerate != 0;
    mConfig.applyPoseCorrection = applyPoseCorrection != 0;
    mConfig.degradationAwareSubspace = degradationAwareSubspace != 0;
    ReadScalar(file, "DirectPose.FrameInterval", mConfig.frameInterval);
    ReadScalar(file, "DirectPose.PyramidLevels", mConfig.pyramidLevels);
    ReadScalar(file, "DirectPose.IterationsPerLevel", mConfig.iterationsPerLevel);
    ReadScalar(file, "DirectPose.PixelStride", mConfig.pixelStride);
    ReadScalar(file, "DirectPose.MinCorrespondences", mConfig.minimumCorrespondences);
    ReadScalar(file, "DirectPose.MinDepth", mConfig.minimumDepth);
    ReadScalar(file, "DirectPose.MaxDepth", mConfig.maximumDepth);
    ReadScalar(file, "DirectPose.MaxDepthDifference", mConfig.maximumDepthDifference);
    ReadScalar(file, "DirectPose.MaxNormalDepthDifference", mConfig.maximumNormalDepthDifference);
    ReadScalar(file, "DirectPose.MinImageGradient", mConfig.minimumImageGradient);
    ReadScalar(file, "DirectPose.PhotometricSigma", mConfig.photometricSigma);
    ReadScalar(file, "DirectPose.GeometricSigma", mConfig.geometricSigma);
    ReadScalar(file, "DirectPose.HuberDelta", mConfig.huberDelta);
    ReadScalar(file, "DirectPose.MinRelativeReduction", mConfig.minimumRelativeReduction);
    ReadScalar(file, "DirectPose.MaxTranslationCorrection", mConfig.maximumTranslationCorrection);
    ReadScalar(file, "DirectPose.MaxRotationCorrection", mConfig.maximumRotationCorrection);
    ReadScalar(file, "DirectPose.MaxMahalanobis", mConfig.maximumMahalanobis);
    ReadScalar(file, "DirectPose.MinDirectionalGain", mConfig.minimumDirectionalGain);
    ReadScalar(file, "DirectPose.DegradationGainPenalty", mConfig.degradationGainPenalty);
    ReadScalar(file, "DirectPose.MaxDirectionalNis", mConfig.maximumDirectionalNis);
    ReadScalar(file, "DirectPose.MinSubspaceRank", mConfig.minimumSubspaceRank);
    ReadScalar(file, "DirectPose.RotationVarianceFloor", mConfig.rotationVarianceFloor);
    ReadScalar(file, "DirectPose.TranslationVarianceFloor", mConfig.translationVarianceFloor);
    ReadScalar(file, "DirectPose.MaxIterationTranslation", mConfig.maximumIterationTranslation);
    ReadScalar(file, "DirectPose.MaxIterationRotation", mConfig.maximumIterationRotation);
    ReadScalar(file, "DirectPose.TelemetryPath", mConfig.telemetryPath);
    ReadScalar(file, "DirectPose.SummaryPath", mConfig.summaryPath);

    mConfig.pyramidLevels = std::max(1, std::min(4, mConfig.pyramidLevels));
    mConfig.frameInterval = std::max(1, mConfig.frameInterval);
    mConfig.iterationsPerLevel = std::max(1, std::min(12, mConfig.iterationsPerLevel));
    mConfig.pixelStride = std::max(2, mConfig.pixelStride);
    mConfig.minimumCorrespondences = std::max(30, mConfig.minimumCorrespondences);
    mConfig.minimumDepth = std::max(0.01f, mConfig.minimumDepth);
    mConfig.maximumDepth = std::max(mConfig.minimumDepth + 0.1f, mConfig.maximumDepth);
    mConfig.photometricSigma = std::max(1e-4f, mConfig.photometricSigma);
    mConfig.geometricSigma = std::max(1e-4f, mConfig.geometricSigma);
    mConfig.huberDelta = std::max(0.1f, mConfig.huberDelta);
    mConfig.rotationVarianceFloor = std::max(1e-10f, mConfig.rotationVarianceFloor);
    mConfig.translationVarianceFloor = std::max(1e-10f, mConfig.translationVarianceFloor);
    mConfig.minimumDirectionalGain = std::max(0.0f,
        std::min(1.0f, mConfig.minimumDirectionalGain));
    mConfig.degradationGainPenalty = std::max(0.0f,
        std::min(1.0f - mConfig.minimumDirectionalGain,
                 mConfig.degradationGainPenalty));
    mConfig.maximumDirectionalNis = std::max(0.1f,
        mConfig.maximumDirectionalNis);
    mConfig.minimumSubspaceRank = std::max(1,
        std::min(6, mConfig.minimumSubspaceRank));

    if((mConfig.enabled || mConfig.recoveryEnabled) && !mConfig.telemetryPath.empty())
    {
        mTelemetry.open(mConfig.telemetryPath.c_str(), std::ios::out | std::ios::trunc);
        if(mTelemetry)
        {
            mTelemetry << "frame_id,timestamp,accepted,correspondences,iterations,"
                       << "objective_before,objective_after,relative_reduction,"
                       << "translation_correction,rotation_correction,"
                       << "applied_translation_correction,applied_rotation_correction,mahalanobis,"
                       << "covariance_valid,direct_covariance_valid,fused_covariance_valid,"
                       << "innovation_covariance_trace,"
                       << "degradation_health,information_subspace_rank,"
                       << "minimum_selected_directional_gain,maximum_selected_directional_gain,"
                       << "discarded_whitened_innovation,"
                       << "geometrically_degenerate,"
                       << "elapsed_ms,rejection_reason\n";
        }
    }
}

DirectRgbdPoseRefiner::~DirectRgbdPoseRefiner()
{
    ExportSummary();
}

cv::Mat DirectRgbdPoseRefiner::IntensityFloat(const cv::Mat& image)
{
    cv::Mat gray;
    if(image.channels() == 1) gray = image;
    else if(image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if(image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else return cv::Mat();

    cv::Mat intensity;
    double scale = 1.0;
    if(gray.depth() == CV_8U) scale = 1.0 / 255.0;
    else if(gray.depth() == CV_16U) scale = 1.0 / 65535.0;
    gray.convertTo(intensity, CV_32F, scale);
    return intensity;
}

float DirectRgbdPoseRefiner::Bilinear(const cv::Mat& image, float x, float y)
{
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    if(ix < 0 || iy < 0 || ix + 1 >= image.cols || iy + 1 >= image.rows)
        return std::numeric_limits<float>::quiet_NaN();
    const float dx = x - static_cast<float>(ix);
    const float dy = y - static_cast<float>(iy);
    const float a = image.at<float>(iy, ix);
    const float b = image.at<float>(iy, ix + 1);
    const float c = image.at<float>(iy + 1, ix);
    const float d = image.at<float>(iy + 1, ix + 1);
    return (1.0f - dy) * ((1.0f - dx) * a + dx * b) +
           dy * ((1.0f - dx) * c + dx * d);
}

Eigen::Matrix3d DirectRgbdPoseRefiner::Skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d matrix;
    matrix << 0.0, -v.z(), v.y(),
              v.z(), 0.0, -v.x(),
             -v.y(), v.x(), 0.0;
    return matrix;
}

float DirectRgbdPoseRefiner::HuberWeight(float residual, float delta)
{
    const float magnitude = std::abs(residual);
    if(!std::isfinite(magnitude)) return 0.0f;
    return magnitude <= delta ? 1.0f : delta / std::max(magnitude, 1e-12f);
}

double DirectRgbdPoseRefiner::HuberLoss(double residual, double delta)
{
    const double magnitude = std::abs(residual);
    return magnitude <= delta ? 0.5 * residual * residual :
        delta * (magnitude - 0.5 * delta);
}

DirectRgbdPoseRefiner::Linearization DirectRgbdPoseRefiner::Linearize(
    const cv::Mat& referenceIntensity,
    const cv::Mat& referenceDepth,
    const cv::Mat& currentIntensity,
    const cv::Mat& currentDepth,
    const cv::Mat& gradientX,
    const cv::Mat& gradientY,
    const Eigen::Matrix3d& K,
    const Sophus::SE3d& referenceTcw,
    const Sophus::SE3d& currentTcw,
    int stride,
    bool buildSystem) const
{
    Linearization output;
    if(referenceIntensity.empty() || referenceDepth.empty() ||
       currentIntensity.empty() || currentDepth.empty()) return output;

    const double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    // ICL-NUIM's Freiburg-format calibration uses a negative vertical focal
    // length to encode its image-axis convention.  The projection Jacobian is
    // valid for either sign; only a near-zero magnitude is invalid.
    if(!(std::abs(fx) > 1e-9 && std::abs(fy) > 1e-9)) return output;
    const Sophus::SE3d currentFromReference = currentTcw * referenceTcw.inverse();

    for(int v = stride; v < referenceDepth.rows - stride; v += stride)
    {
        for(int u = stride; u < referenceDepth.cols - stride; u += stride)
        {
            const float depth = referenceDepth.at<float>(v,u);
            if(!std::isfinite(depth) || depth < mConfig.minimumDepth ||
               depth > mConfig.maximumDepth) continue;

            const Eigen::Vector3d referencePoint(
                (static_cast<double>(u) - cx) * depth / fx,
                (static_cast<double>(v) - cy) * depth / fy,
                depth);
            const Eigen::Vector3d point = currentFromReference * referencePoint;
            if(!point.allFinite() || point.z() <= mConfig.minimumDepth ||
               point.z() > mConfig.maximumDepth) continue;

            const float x = static_cast<float>(fx * point.x() / point.z() + cx);
            const float y = static_cast<float>(fy * point.y() / point.z() + cy);
            if(x < 2.0f || y < 2.0f || x >= currentDepth.cols - 3.0f ||
               y >= currentDepth.rows - 3.0f) continue;

            const int xi = static_cast<int>(std::lround(x));
            const int yi = static_cast<int>(std::lround(y));
            const float observedDepth = currentDepth.at<float>(yi,xi);
            if(!std::isfinite(observedDepth) || observedDepth < mConfig.minimumDepth ||
               observedDepth > mConfig.maximumDepth ||
               std::abs(observedDepth - point.z()) > mConfig.maximumDepthDifference)
                continue;

            Eigen::Matrix<double,3,6> pointJacobian;
            pointJacobian.block<3,3>(0,0) = -Skew(point);
            pointJacobian.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
            Eigen::Matrix<double,2,3> projectionJacobian;
            const double inverseZ = 1.0 / point.z();
            const double inverseZ2 = inverseZ * inverseZ;
            projectionJacobian << fx * inverseZ, 0.0, -fx * point.x() * inverseZ2,
                                  0.0, fy * inverseZ, -fy * point.y() * inverseZ2;

            bool used = false;
            const float gx = Bilinear(gradientX, x, y);
            const float gy = Bilinear(gradientY, x, y);
            const float currentValue = Bilinear(currentIntensity, x, y);
            const float referenceValue = referenceIntensity.at<float>(v,u);
            if(std::isfinite(gx) && std::isfinite(gy) &&
               std::isfinite(currentValue) && std::isfinite(referenceValue) &&
               std::sqrt(gx * gx + gy * gy) >= mConfig.minimumImageGradient)
            {
                const double residual =
                    (static_cast<double>(currentValue) - referenceValue) /
                    mConfig.photometricSigma;
                Eigen::Matrix<double,1,2> imageGradient;
                imageGradient << gx, gy;
                const Eigen::Matrix<double,1,6> jacobian =
                    imageGradient * projectionJacobian * pointJacobian /
                    mConfig.photometricSigma;
                const double weight = HuberWeight(
                    static_cast<float>(residual), mConfig.huberDelta);
                output.objective += HuberLoss(residual, mConfig.huberDelta);
                if(buildSystem)
                {
                    output.hessian.noalias() += weight * jacobian.transpose() * jacobian;
                    output.gradient.noalias() += weight * jacobian.transpose() * residual;
                }
                used = true;
            }

            const float depthRight = currentDepth.at<float>(yi,xi + 1);
            const float depthDown = currentDepth.at<float>(yi + 1,xi);
            if(std::isfinite(depthRight) && std::isfinite(depthDown) &&
               depthRight > mConfig.minimumDepth && depthDown > mConfig.minimumDepth &&
               std::abs(depthRight - observedDepth) < mConfig.maximumNormalDepthDifference &&
               std::abs(depthDown - observedDepth) < mConfig.maximumNormalDepthDifference)
            {
                const Eigen::Vector3d observedPoint(
                    (x - cx) * observedDepth / fx,
                    (y - cy) * observedDepth / fy,
                    observedDepth);
                const Eigen::Vector3d rightPoint(
                    (static_cast<double>(xi + 1) - cx) * depthRight / fx,
                    (static_cast<double>(yi) - cy) * depthRight / fy,
                    depthRight);
                const Eigen::Vector3d downPoint(
                    (static_cast<double>(xi) - cx) * depthDown / fx,
                    (static_cast<double>(yi + 1) - cy) * depthDown / fy,
                    depthDown);
                Eigen::Vector3d normal = (rightPoint - observedPoint).cross(
                    downPoint - observedPoint);
                const double normalNorm = normal.norm();
                if(normalNorm > 1e-9 && std::isfinite(normalNorm))
                {
                    normal /= normalNorm;
                    const double residual = normal.dot(point - observedPoint) /
                        mConfig.geometricSigma;
                    const Eigen::Matrix<double,1,6> jacobian =
                        normal.transpose() * pointJacobian / mConfig.geometricSigma;
                    const double weight = HuberWeight(
                        static_cast<float>(residual), mConfig.huberDelta);
                    output.objective += HuberLoss(residual, mConfig.huberDelta);
                    if(buildSystem)
                    {
                        output.hessian.noalias() += weight * jacobian.transpose() * jacobian;
                        output.gradient.noalias() += weight * jacobian.transpose() * residual;
                    }
                    used = true;
                }
            }
            if(used) ++output.correspondences;
        }
    }
    return output;
}

bool DirectRgbdPoseRefiner::CorrectionSafe(
    const DirectPoseConfig& config,
    const Eigen::Matrix<float, 6, 1>& correction,
    const SlamTelemetry* telemetry,
    const Eigen::Matrix<float, 6, 6>* directCovariance,
    float& mahalanobis,
    float* innovationTrace,
    std::string* rejectionReason,
    Eigen::Matrix<float, 6, 1>* fusedCorrection,
    Eigen::Matrix<float, 6, 6>* fusedCovariance,
    Eigen::Matrix<float, 6, 6>* informationProjector,
    int* informationSubspaceRank,
    float* minimumSelectedDirectionalGain,
    float* maximumSelectedDirectionalGain,
    float* discardedWhitenedInnovation,
    float* degradationHealth)
{
    mahalanobis = 0.0f;
    if(innovationTrace) *innovationTrace = 0.0f;
    if(rejectionReason) rejectionReason->clear();
    if(fusedCorrection) fusedCorrection->setZero();
    if(fusedCovariance) fusedCovariance->setZero();
    if(informationProjector) informationProjector->setZero();
    if(informationSubspaceRank) *informationSubspaceRank = 0;
    if(minimumSelectedDirectionalGain) *minimumSelectedDirectionalGain = 0.0f;
    if(maximumSelectedDirectionalGain) *maximumSelectedDirectionalGain = 0.0f;
    if(discardedWhitenedInnovation) *discardedWhitenedInnovation = 0.0f;
    if(degradationHealth) *degradationHealth = 0.5f;
    if(!correction.allFinite() ||
       correction.head<3>().norm() > config.maximumRotationCorrection ||
       correction.tail<3>().norm() > config.maximumTranslationCorrection)
    {
        if(rejectionReason) *rejectionReason = "correction_limit";
        return false;
    }
    // A direct inter-frame objective is locally valid but can accumulate
    // drift when sparse map tracking is already well constrained.  In the
    // safety-first mode it is therefore a shadow proposal on ordinary frames
    // and may be committed only when the frontend reports degeneracy.
    if(config.commitOnlyWhenDegenerate &&
       (!telemetry || !telemetry->geometricallyDegenerate))
    {
        if(rejectionReason) *rejectionReason = "nondegenerate_shadow";
        return false;
    }
    if(!config.uncertaintyGate)
    {
        if(fusedCorrection) *fusedCorrection = correction;
        if(informationProjector)
            informationProjector->setIdentity();
        if(informationSubspaceRank) *informationSubspaceRank = 6;
        if(minimumSelectedDirectionalGain) *minimumSelectedDirectionalGain = 1.0f;
        if(maximumSelectedDirectionalGain) *maximumSelectedDirectionalGain = 1.0f;
        return true;
    }
    // Never interprets a missing covariance as permission to write a pose.
    // This differs deliberately from the mapper's neutral fallback: a map
    // sample may be skipped, while an unjustified pose correction would be
    // propagated by the motion model to every following frame.
    if(!telemetry || !telemetry->poseCovarianceValid)
    {
        if(rejectionReason) *rejectionReason = "missing_frontend_covariance";
        return false;
    }

    Eigen::Matrix<float,6,6> frontendCovariance = 0.5f *
        (telemetry->poseCovariance + telemetry->poseCovariance.transpose());
    if(!directCovariance || !directCovariance->allFinite())
    {
        if(rejectionReason) *rejectionReason = "invalid_direct_covariance";
        return false;
    }
    Eigen::Matrix<float,6,6> directNoise = 0.5f *
        (*directCovariance + directCovariance->transpose());
    for(int index = 0; index < 3; ++index)
        directNoise(index,index) += config.rotationVarianceFloor;
    for(int index = 3; index < 6; ++index)
        directNoise(index,index) += config.translationVarianceFloor;
    const Eigen::Matrix<float,6,6> covariance = frontendCovariance + directNoise;
    if(!covariance.allFinite())
    {
        if(rejectionReason) *rejectionReason = "invalid_innovation_covariance";
        return false;
    }
    if(innovationTrace) *innovationTrace = covariance.trace();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6> > solver(covariance);
    if(solver.info() != Eigen::Success)
    {
        if(rejectionReason) *rejectionReason = "innovation_eigendecomposition";
        return false;
    }
    const float largest = std::max(1e-9f, solver.eigenvalues().maxCoeff());
    Eigen::Matrix<float,6,1> inverseEigenvalues;
    Eigen::Matrix<float,6,1> inverseSquareRootEigenvalues;
    Eigen::Matrix<float,6,1> squareRootEigenvalues;
    for(int i = 0; i < 6; ++i)
    {
        const float value = std::max(solver.eigenvalues()[i], largest * 1e-6f);
        inverseEigenvalues[i] = 1.0f / value;
        inverseSquareRootEigenvalues[i] = 1.0f / std::sqrt(value);
        squareRootEigenvalues[i] = std::sqrt(value);
    }
    const Eigen::Matrix<float,6,6> innovationVectors = solver.eigenvectors();
    const Eigen::Matrix<float,6,6> information = innovationVectors *
        inverseEigenvalues.asDiagonal() * innovationVectors.transpose();
    const Eigen::Matrix<float,6,6> inverseSquareRoot = innovationVectors *
        inverseSquareRootEigenvalues.asDiagonal() * innovationVectors.transpose();
    const Eigen::Matrix<float,6,6> squareRoot = innovationVectors *
        squareRootEigenvalues.asDiagonal() * innovationVectors.transpose();
    mahalanobis = correction.dot(information * correction);
    const bool safe = std::isfinite(mahalanobis) &&
        mahalanobis <= config.maximumMahalanobis;
    if(!safe && rejectionReason) *rejectionReason = "mahalanobis";
    if(safe && fusedCorrection)
    {
        Eigen::Matrix<float,6,6> gain = frontendCovariance * information;
        Eigen::Matrix<float,6,6> projector =
            Eigen::Matrix<float,6,6>::Identity();
        int selectedRank = 6;
        float minimumGain = 1.0f;
        float maximumGain = 1.0f;
        float discardedEnergy = 0.0f;
        float health = 0.5f;

        if(config.degradationAwareSubspace)
        {
            if(telemetry && telemetry->featuresValid)
            {
                const float gradientHealth = std::min(1.0f,
                    std::max(0.0f, telemetry->imageGradientMean / 0.025f));
                const float textureHealth = std::min(1.0f,
                    std::max(0.0f, telemetry->textureRatio / 0.15f));
                const float depthHealth = std::min(1.0f,
                    std::max(0.0f, telemetry->depthValidRatio / 0.30f));
                const float conditionHealth = std::exp(-std::max(0.0f,
                    telemetry->hessianLogCondition - 3.0f) / 3.0f);
                health = 0.20f * gradientHealth + 0.20f * textureHealth +
                         0.35f * depthHealth + 0.25f * conditionHealth;
                health = std::min(1.0f, std::max(0.0f, health));
            }

            // G = S^{-1/2} Sigma_s S^{-1/2} is symmetric PSD and bounded by
            // the identity because S = Sigma_s + R with R PSD.  Its
            // eigenvalues are therefore coordinate-free directional gains.
            Eigen::Matrix<float,6,6> normalizedPrior = inverseSquareRoot *
                frontendCovariance * inverseSquareRoot;
            normalizedPrior = 0.5f *
                (normalizedPrior + normalizedPrior.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6> >
                gainSolver(normalizedPrior);
            if(gainSolver.info() != Eigen::Success)
            {
                if(rejectionReason) *rejectionReason = "subspace_eigendecomposition";
                return false;
            }
            const Eigen::Matrix<float,6,1> whitened =
                gainSolver.eigenvectors().transpose() *
                inverseSquareRoot * correction;
            const float gainThreshold = std::min(0.999f,
                config.minimumDirectionalGain +
                (1.0f - health) * config.degradationGainPenalty);
            const float nisThreshold = config.maximumDirectionalNis *
                (0.65f + 0.35f * health);
            Eigen::Matrix<float,6,1> selected =
                Eigen::Matrix<float,6,1>::Zero();
            Eigen::Matrix<float,6,1> selectedGain =
                Eigen::Matrix<float,6,1>::Zero();
            selectedRank = 0;
            minimumGain = 1.0f;
            maximumGain = 0.0f;
            for(int i = 0; i < 6; ++i)
            {
                const float directionalGain = std::min(1.0f,
                    std::max(0.0f, gainSolver.eigenvalues()[i]));
                const float directionalNis = whitened[i] * whitened[i];
                if(directionalGain >= gainThreshold &&
                   directionalNis <= nisThreshold)
                {
                    selected[i] = 1.0f;
                    selectedGain[i] = directionalGain;
                    ++selectedRank;
                    minimumGain = std::min(minimumGain, directionalGain);
                    maximumGain = std::max(maximumGain, directionalGain);
                }
                else
                    discardedEnergy += directionalNis;
            }
            if(selectedRank < config.minimumSubspaceRank)
            {
                if(rejectionReason) *rejectionReason = "subspace_rank";
                if(informationSubspaceRank) *informationSubspaceRank = selectedRank;
                if(discardedWhitenedInnovation)
                    *discardedWhitenedInnovation = discardedEnergy;
                if(degradationHealth) *degradationHealth = health;
                return false;
            }
            const Eigen::Matrix<float,6,6> modes = gainSolver.eigenvectors();
            projector = squareRoot * modes * selected.asDiagonal() *
                modes.transpose() * inverseSquareRoot;
            gain = squareRoot * modes * selectedGain.asDiagonal() *
                modes.transpose() * inverseSquareRoot;
        }

        // This is a strict information-subspace update.  In whitened
        // innovation coordinates, rejected modes have zero gain exactly; no
        // component of their direct proposal reaches the motion model.
        *fusedCorrection = gain * correction;
        if(!fusedCorrection->allFinite())
        {
            fusedCorrection->setZero();
            if(rejectionReason) *rejectionReason = "invalid_fused_correction";
            return false;
        }
        const float proposedRotation = correction.head<3>().norm();
        const float proposedTranslation = correction.tail<3>().norm();
        const float fusedRotation = fusedCorrection->head<3>().norm();
        const float fusedTranslation = fusedCorrection->tail<3>().norm();
        if(fusedRotation > proposedRotation && fusedRotation > 1e-12f)
            fusedCorrection->head<3>() *= proposedRotation / fusedRotation;
        if(fusedTranslation > proposedTranslation && fusedTranslation > 1e-12f)
            fusedCorrection->tail<3>() *= proposedTranslation / fusedTranslation;
        if(fusedCovariance)
        {
            const Eigen::Matrix<float,6,6> identity =
                Eigen::Matrix<float,6,6>::Identity();
            const Eigen::Matrix<float,6,6> joseph =
                (identity - gain) * frontendCovariance * (identity - gain).transpose() +
                gain * directNoise * gain.transpose();
            *fusedCovariance = 0.5f * (joseph + joseph.transpose());
            if(!fusedCovariance->allFinite())
            {
                fusedCovariance->setZero();
                fusedCorrection->setZero();
                if(rejectionReason) *rejectionReason = "invalid_fused_covariance";
                return false;
            }
        }
        if(informationProjector) *informationProjector = projector;
        if(informationSubspaceRank) *informationSubspaceRank = selectedRank;
        if(minimumSelectedDirectionalGain)
            *minimumSelectedDirectionalGain = minimumGain;
        if(maximumSelectedDirectionalGain)
            *maximumSelectedDirectionalGain = maximumGain;
        if(discardedWhitenedInnovation)
            *discardedWhitenedInnovation = discardedEnergy;
        if(degradationHealth) *degradationHealth = health;
    }
    return safe;
}

bool DirectRgbdPoseRefiner::Refine(
    const cv::Mat& referenceGray,
    const cv::Mat& referenceDepth,
    const cv::Mat& currentGray,
    const cv::Mat& currentDepth,
    const Eigen::Matrix3f& cameraMatrix,
    const Sophus::SE3f& referenceTcw,
    const Sophus::SE3f& initialTcw,
    const SlamTelemetry* telemetry,
    std::uint64_t frameId,
    double timestamp,
    Sophus::SE3f& refinedTcw,
    DirectPoseResult& result)
{
    result = DirectPoseResult();
    if(!mConfig.enabled && !mConfig.recoveryEnabled) return false;
    result.attempted = true;
    result.covarianceValid = telemetry && telemetry->poseCovarianceValid;
    result.geometricallyDegenerate = telemetry && telemetry->geometricallyDegenerate;
    const auto start = std::chrono::steady_clock::now();

    const cv::Mat referenceIntensity = IntensityFloat(referenceGray);
    const cv::Mat currentIntensity = IntensityFloat(currentGray);
    if(referenceIntensity.empty() || currentIntensity.empty() ||
       referenceDepth.empty() || currentDepth.empty() ||
       referenceDepth.type() != CV_32F || currentDepth.type() != CV_32F ||
       referenceIntensity.size() != referenceDepth.size() ||
       currentIntensity.size() != currentDepth.size())
    {
        result.rejectionReason = "invalid_input";
    }
    else
    {
        std::vector<cv::Mat> referenceImages(mConfig.pyramidLevels);
        std::vector<cv::Mat> referenceDepths(mConfig.pyramidLevels);
        std::vector<cv::Mat> currentImages(mConfig.pyramidLevels);
        std::vector<cv::Mat> currentDepths(mConfig.pyramidLevels);
        referenceImages[0] = referenceIntensity;
        referenceDepths[0] = referenceDepth;
        currentImages[0] = currentIntensity;
        currentDepths[0] = currentDepth;
        for(int level = 1; level < mConfig.pyramidLevels; ++level)
        {
            cv::pyrDown(referenceImages[level - 1], referenceImages[level]);
            cv::pyrDown(currentImages[level - 1], currentImages[level]);
            cv::resize(referenceDepths[level - 1], referenceDepths[level],
                       referenceImages[level].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize(currentDepths[level - 1], currentDepths[level],
                       currentImages[level].size(), 0.0, 0.0, cv::INTER_NEAREST);
        }

        Eigen::Quaterniond referenceQuaternion =
            referenceTcw.unit_quaternion().cast<double>();
        Eigen::Quaterniond initialQuaternion =
            initialTcw.unit_quaternion().cast<double>();
        referenceQuaternion.normalize();
        initialQuaternion.normalize();
        const Sophus::SE3d reference(
            referenceQuaternion, referenceTcw.translation().cast<double>());
        const Sophus::SE3d initial(
            initialQuaternion, initialTcw.translation().cast<double>());
        Sophus::SE3d estimate = initial;

        cv::Mat fullGradientX, fullGradientY;
        cv::Sobel(currentImages[0], fullGradientX, CV_32F, 1, 0, 3, 0.125);
        cv::Sobel(currentImages[0], fullGradientY, CV_32F, 0, 1, 3, 0.125);
        // Build degradation telemetry directly from the lightweight images.
        // The real-time configuration deliberately disables the expensive
        // structural extractor, so treating missing features as a neutral
        // health value would make the subspace gate blind exactly when it is
        // needed most.
        SlamTelemetry localTelemetry = telemetry ? *telemetry : SlamTelemetry();
        double gradientSum = 0.0;
        std::uint64_t textured = 0;
        std::uint64_t validDepth = 0;
        std::uint64_t sampled = 0;
        const int healthStride = std::max(2, mConfig.pixelStride);
        for(int y = 1; y < currentImages[0].rows - 1; y += healthStride)
        {
            const float* gx = fullGradientX.ptr<float>(y);
            const float* gy = fullGradientY.ptr<float>(y);
            const float* depth = currentDepths[0].ptr<float>(y);
            for(int x = 1; x < currentImages[0].cols - 1; x += healthStride)
            {
                const float magnitude = std::sqrt(gx[x] * gx[x] + gy[x] * gy[x]);
                gradientSum += magnitude;
                textured += magnitude >= mConfig.minimumImageGradient ? 1 : 0;
                validDepth += std::isfinite(depth[x]) &&
                    depth[x] >= mConfig.minimumDepth &&
                    depth[x] <= mConfig.maximumDepth ? 1 : 0;
                ++sampled;
            }
        }
        if(sampled > 0)
        {
            localTelemetry.featuresValid = true;
            localTelemetry.imageGradientMean = static_cast<float>(
                gradientSum / static_cast<double>(sampled));
            localTelemetry.textureRatio = static_cast<float>(textured) /
                static_cast<float>(sampled);
            localTelemetry.depthValidRatio = static_cast<float>(validDepth) /
                static_cast<float>(sampled);
        }
        const Linearization before = Linearize(
            referenceImages[0], referenceDepths[0], currentImages[0],
            currentDepths[0], fullGradientX, fullGradientY,
            cameraMatrix.cast<double>(), reference, initial,
            mConfig.pixelStride, false);
        result.objectiveBefore = before.correspondences > 0 ?
            static_cast<float>(before.objective / before.correspondences) : 0.0f;

        bool numericalFailure = false;
        for(int level = mConfig.pyramidLevels - 1; level >= 0 && !numericalFailure; --level)
        {
            const double scale = 1.0 / static_cast<double>(1 << level);
            Eigen::Matrix3d levelK = cameraMatrix.cast<double>();
            levelK(0,0) *= scale; levelK(1,1) *= scale;
            levelK(0,2) *= scale; levelK(1,2) *= scale;
            cv::Mat gradientX, gradientY;
            cv::Sobel(currentImages[level], gradientX, CV_32F, 1, 0, 3, 0.125);
            cv::Sobel(currentImages[level], gradientY, CV_32F, 0, 1, 3, 0.125);
            const int stride = std::max(2,
                static_cast<int>(std::lround(mConfig.pixelStride * scale)));

            for(int iteration = 0; iteration < mConfig.iterationsPerLevel; ++iteration)
            {
                Linearization system = Linearize(
                    referenceImages[level], referenceDepths[level],
                    currentImages[level], currentDepths[level], gradientX, gradientY,
                    levelK, reference, estimate, stride, true);
                ++result.iterations;
                if(system.correspondences < std::max(30,
                    static_cast<int>(mConfig.minimumCorrespondences * scale * scale)))
                    break;
                const double diagonalScale = std::max(1.0,
                    system.hessian.diagonal().mean());
                system.hessian.diagonal().array() += 1e-6 * diagonalScale;
                Eigen::Matrix<double,6,1> increment =
                    -system.hessian.ldlt().solve(system.gradient);
                if(!increment.allFinite())
                {
                    numericalFailure = true;
                    break;
                }
                const double rotationNorm = increment.head<3>().norm();
                const double translationNorm = increment.tail<3>().norm();
                if(rotationNorm > mConfig.maximumIterationRotation)
                    increment.head<3>() *= mConfig.maximumIterationRotation / rotationNorm;
                if(translationNorm > mConfig.maximumIterationTranslation)
                    increment.tail<3>() *= mConfig.maximumIterationTranslation / translationNorm;
                const Sophus::SE3d update(
                    Sophus::SO3d::exp(increment.head<3>()), increment.tail<3>());
                estimate = update * estimate;
                if(!estimate.matrix().allFinite())
                {
                    numericalFailure = true;
                    break;
                }
                if(increment.norm() < 1e-6) break;
            }
        }

        const Linearization after = Linearize(
            referenceImages[0], referenceDepths[0], currentImages[0],
            currentDepths[0], fullGradientX, fullGradientY,
            cameraMatrix.cast<double>(), reference, estimate,
            mConfig.pixelStride, false);
        result.correspondences = after.correspondences;
        result.objectiveAfter = after.correspondences > 0 ?
            static_cast<float>(after.objective / after.correspondences) : 0.0f;
        result.relativeReduction = result.objectiveBefore > 1e-9f ?
            (result.objectiveBefore - result.objectiveAfter) / result.objectiveBefore : 0.0f;

        const Sophus::SE3d correctionPose = estimate * initial.inverse();
        Eigen::Matrix<float,6,1> correction;
        correction.head<3>() = correctionPose.so3().log().cast<float>();
        correction.tail<3>() = correctionPose.translation().cast<float>();
        result.rotationCorrection = correction.head<3>().norm();
        result.translationCorrection = correction.tail<3>().norm();

        Eigen::Matrix<float,6,6> directCovariance =
            Eigen::Matrix<float,6,6>::Zero();
        const Linearization finalSystem = Linearize(
            referenceImages[0], referenceDepths[0], currentImages[0],
            currentDepths[0], fullGradientX, fullGradientY,
            cameraMatrix.cast<double>(), reference, estimate,
            mConfig.pixelStride, true);
        if(finalSystem.hessian.allFinite())
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6> >
                healthSolver(0.5 * (finalSystem.hessian +
                                    finalSystem.hessian.transpose()));
            if(healthSolver.info() == Eigen::Success)
            {
                const double maximum = std::max(1e-12,
                    healthSolver.eigenvalues().maxCoeff());
                const double minimum = std::max(maximum * 1e-12,
                    healthSolver.eigenvalues().minCoeff());
                localTelemetry.hessianLogCondition = static_cast<float>(
                    std::log10(maximum / minimum));
            }
        }
        if(finalSystem.correspondences >= mConfig.minimumCorrespondences &&
           finalSystem.hessian.allFinite())
        {
            Eigen::Matrix<double,6,6> hessian = 0.5 *
                (finalSystem.hessian + finalSystem.hessian.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6> >
                covarianceSolver(hessian);
            if(covarianceSolver.info() == Eigen::Success &&
               covarianceSolver.eigenvalues().maxCoeff() > 0.0)
            {
                const double largest = covarianceSolver.eigenvalues().maxCoeff();
                Eigen::Matrix<double,6,1> inverseEigenvalues;
                for(int index = 0; index < 6; ++index)
                    inverseEigenvalues[index] = 1.0 / std::max(
                        largest * 1e-9, covarianceSolver.eigenvalues()[index]);
                const Eigen::Matrix<double,6,6> covariance =
                    covarianceSolver.eigenvectors() *
                    inverseEigenvalues.asDiagonal() *
                    covarianceSolver.eigenvectors().transpose();
                if(covariance.allFinite())
                {
                    directCovariance = (0.5 *
                        (covariance + covariance.transpose())).cast<float>();
                    result.directCovarianceValid = directCovariance.allFinite();
                }
            }
        }

        if(numericalFailure || !std::isfinite(result.objectiveAfter))
            result.rejectionReason = "numerical";
        else if(before.correspondences < mConfig.minimumCorrespondences ||
                after.correspondences < mConfig.minimumCorrespondences)
            result.rejectionReason = "correspondences";
        else if(result.relativeReduction < mConfig.minimumRelativeReduction)
            result.rejectionReason = "objective";
        else
        {
            std::string safetyReason;
            Eigen::Matrix<float,6,1> fusedCorrection = correction;
            Eigen::Matrix<float,6,6> fusedCovariance =
                Eigen::Matrix<float,6,6>::Zero();
            const Eigen::Matrix<float,6,6>* covariancePointer =
                result.directCovarianceValid ? &directCovariance : nullptr;
            if(!CorrectionSafe(mConfig, correction, &localTelemetry,
                               covariancePointer, result.mahalanobis,
                               &result.innovationCovarianceTrace,
                               &safetyReason, &fusedCorrection,
                               &fusedCovariance,
                               &result.informationProjector,
                               &result.informationSubspaceRank,
                               &result.minimumSelectedDirectionalGain,
                               &result.maximumSelectedDirectionalGain,
                               &result.discardedWhitenedInnovation,
                               &result.degradationHealth))
                result.rejectionReason = safetyReason.empty() ?
                    "safety_gate" : safetyReason;
            else
            {
                result.appliedRotationCorrection = fusedCorrection.head<3>().norm();
                result.appliedTranslationCorrection = fusedCorrection.tail<3>().norm();
                result.fusedCovariance = fusedCovariance;
                result.fusedCovarianceValid = fusedCovariance.allFinite() &&
                    fusedCovariance.diagonal().minCoeff() >= 0.0f &&
                    fusedCovariance.trace() > 0.0f;
                const Sophus::SE3f fusedUpdate(
                    Sophus::SO3f::exp(fusedCorrection.head<3>()),
                    fusedCorrection.tail<3>());
                refinedTcw = fusedUpdate * initialTcw;
                result.accepted = refinedTcw.matrix().allFinite();
                if(!result.accepted) result.rejectionReason = "numerical";
            }
        }
    }

    result.elapsedMilliseconds = std::chrono::duration_cast<
        std::chrono::duration<double,std::milli> >(
            std::chrono::steady_clock::now() - start).count();
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ++mSummary.attemptedFrames;
        mSummary.totalMilliseconds += result.elapsedMilliseconds;
        if(result.accepted)
        {
            ++mSummary.acceptedFrames;
            mSummary.totalRelativeReduction += result.relativeReduction;
        }
        else if(result.rejectionReason == "correspondences" ||
                result.rejectionReason == "invalid_input")
            ++mSummary.rejectedCorrespondences;
        else if(result.rejectionReason == "objective")
            ++mSummary.rejectedObjective;
        else if(result.rejectionReason == "mahalanobis" ||
                result.rejectionReason == "missing_frontend_covariance" ||
                result.rejectionReason == "invalid_direct_covariance" ||
                result.rejectionReason == "invalid_innovation_covariance" ||
                result.rejectionReason == "innovation_eigendecomposition" ||
                result.rejectionReason == "subspace_eigendecomposition" ||
                result.rejectionReason == "subspace_rank")
            ++mSummary.rejectedUncertainty;
        else if(result.rejectionReason == "correction_limit" ||
                result.rejectionReason == "nondegenerate_shadow" ||
                result.rejectionReason == "safety_gate")
            ++mSummary.rejectedCorrection;
        else
            ++mSummary.rejectedNumerical;
    }
    WriteTelemetry(frameId, timestamp, result);
    return result.accepted;
}

void DirectRgbdPoseRefiner::WriteTelemetry(
    std::uint64_t frameId, double timestamp, const DirectPoseResult& result)
{
    if(!mTelemetry) return;
    mTelemetry << frameId << ',' << std::setprecision(15) << timestamp << ','
               << (result.accepted ? 1 : 0) << ',' << result.correspondences << ','
               << result.iterations << ',' << result.objectiveBefore << ','
               << result.objectiveAfter << ',' << result.relativeReduction << ','
               << result.translationCorrection << ',' << result.rotationCorrection << ','
               << result.appliedTranslationCorrection << ','
               << result.appliedRotationCorrection << ','
               << result.mahalanobis << ',' << (result.covarianceValid ? 1 : 0) << ','
               << (result.directCovarianceValid ? 1 : 0) << ','
               << (result.fusedCovarianceValid ? 1 : 0) << ','
               << result.innovationCovarianceTrace << ','
               << result.degradationHealth << ','
               << result.informationSubspaceRank << ','
               << result.minimumSelectedDirectionalGain << ','
               << result.maximumSelectedDirectionalGain << ','
               << result.discardedWhitenedInnovation << ','
               << (result.geometricallyDegenerate ? 1 : 0) << ','
               << result.elapsedMilliseconds << ','
               << result.rejectionReason << '\n';
    mTelemetry.flush();
}

DirectPoseSummary DirectRgbdPoseRefiner::Summary() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mSummary;
}

void DirectRgbdPoseRefiner::Reset()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mSummary = DirectPoseSummary();
}

void DirectRgbdPoseRefiner::ExportSummary() const
{
    if((!mConfig.enabled && !mConfig.recoveryEnabled) || mConfig.summaryPath.empty()) return;
    const DirectPoseSummary summary = Summary();
    std::ofstream output(mConfig.summaryPath.c_str(), std::ios::out | std::ios::trunc);
    if(!output) return;
    const double accepted = static_cast<double>(summary.acceptedFrames);
    output << std::setprecision(10)
           << "{\n"
           << "  \"attempted_frames\": " << summary.attemptedFrames << ",\n"
           << "  \"accepted_frames\": " << summary.acceptedFrames << ",\n"
           << "  \"acceptance_rate\": "
           << (summary.attemptedFrames ? accepted / summary.attemptedFrames : 0.0) << ",\n"
           << "  \"mean_time_ms\": "
           << (summary.attemptedFrames ? summary.totalMilliseconds / summary.attemptedFrames : 0.0) << ",\n"
           << "  \"mean_accepted_reduction\": "
           << (summary.acceptedFrames ? summary.totalRelativeReduction / accepted : 0.0) << ",\n"
           << "  \"rejected_correspondences\": " << summary.rejectedCorrespondences << ",\n"
           << "  \"rejected_objective\": " << summary.rejectedObjective << ",\n"
           << "  \"rejected_correction\": " << summary.rejectedCorrection << ",\n"
           << "  \"rejected_uncertainty\": " << summary.rejectedUncertainty << ",\n"
           << "  \"rejected_numerical\": " << summary.rejectedNumerical << "\n"
           << "}\n";
}

} // namespace TIB_SLAM
