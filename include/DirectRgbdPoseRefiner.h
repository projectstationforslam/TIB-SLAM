#ifndef TIB_SLAM_DIRECT_RGBD_POSE_REFINER_H
#define TIB_SLAM_DIRECT_RGBD_POSE_REFINER_H

#include "StructuralTypes.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <string>

namespace TIB_SLAM
{

struct DirectPoseConfig
{
    bool enabled = false;
    bool recoveryEnabled = false;
    bool uncertaintyGate = true;
    bool commitOnlyWhenDegenerate = false;
    // Shadow mode still estimates the information subspace for map fusion,
    // but does not write an inter-frame direct correction into the pose.
    bool applyPoseCorrection = true;
    int frameInterval = 3;
    int pyramidLevels = 2;
    int iterationsPerLevel = 4;
    int pixelStride = 6;
    int minimumCorrespondences = 160;
    float minimumDepth = 0.20f;
    float maximumDepth = 6.0f;
    float maximumDepthDifference = 0.12f;
    float maximumNormalDepthDifference = 0.05f;
    float minimumImageGradient = 0.008f;
    float photometricSigma = 0.05f;
    float geometricSigma = 0.02f;
    float huberDelta = 1.5f;
    float minimumRelativeReduction = 0.05f;
    float maximumTranslationCorrection = 0.008f;
    float maximumRotationCorrection = 0.012f;
    // 99% chi-square gate for a six-dimensional innovation.
    float maximumMahalanobis = 16.812f;
    // Select update directions in the innovation-covariance metric.  The
    // normalized prior covariance has eigenvalues in [0,1]; each eigenvalue is
    // the fraction of that innovation mode that a direct measurement may
    // correct.  Appearance/depth degradation raises the selection threshold.
    bool degradationAwareSubspace = true;
    float minimumDirectionalGain = 0.08f;
    float degradationGainPenalty = 0.12f;
    float maximumDirectionalNis = 6.635f;
    int minimumSubspaceRank = 1;
    // The innovation is the difference between sparse and direct pose
    // estimates.  These floors model residual calibration/linearization error
    // that is absent from either optimizer's formal inverse Hessian.
    float rotationVarianceFloor = 6.25e-6f;
    float translationVarianceFloor = 2.25e-6f;
    float maximumIterationTranslation = 0.015f;
    float maximumIterationRotation = 0.020f;
    std::string telemetryPath = "DirectPoseTelemetry.csv";
    std::string summaryPath = "DirectPoseSummary.json";
};

struct DirectPoseResult
{
    bool attempted = false;
    bool accepted = false;
    int correspondences = 0;
    int iterations = 0;
    float objectiveBefore = 0.0f;
    float objectiveAfter = 0.0f;
    float relativeReduction = 0.0f;
    float translationCorrection = 0.0f;
    float rotationCorrection = 0.0f;
    float appliedTranslationCorrection = 0.0f;
    float appliedRotationCorrection = 0.0f;
    float mahalanobis = 0.0f;
    bool covarianceValid = false;
    bool directCovarianceValid = false;
    bool fusedCovarianceValid = false;
    Eigen::Matrix<float, 6, 6> fusedCovariance =
        Eigen::Matrix<float, 6, 6>::Zero();
    float innovationCovarianceTrace = 0.0f;
    float degradationHealth = 0.0f;
    int informationSubspaceRank = 0;
    float minimumSelectedDirectionalGain = 0.0f;
    float maximumSelectedDirectionalGain = 0.0f;
    float discardedWhitenedInnovation = 0.0f;
    Eigen::Matrix<float, 6, 6> informationProjector =
        Eigen::Matrix<float, 6, 6>::Zero();
    bool geometricallyDegenerate = false;
    double elapsedMilliseconds = 0.0;
    std::string rejectionReason;
};

struct DirectPoseSummary
{
    std::uint64_t attemptedFrames = 0;
    std::uint64_t acceptedFrames = 0;
    std::uint64_t rejectedCorrespondences = 0;
    std::uint64_t rejectedObjective = 0;
    std::uint64_t rejectedCorrection = 0;
    std::uint64_t rejectedUncertainty = 0;
    std::uint64_t rejectedNumerical = 0;
    double totalMilliseconds = 0.0;
    double totalRelativeReduction = 0.0;
};

class DirectRgbdPoseRefiner
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit DirectRgbdPoseRefiner(const std::string& settingsPath);
    ~DirectRgbdPoseRefiner();

    bool Enabled() const { return mConfig.enabled; }
    bool RecoveryEnabled() const { return mConfig.recoveryEnabled; }
    bool ApplyPoseCorrection() const { return mConfig.applyPoseCorrection; }
    bool ShouldRefine(std::uint64_t frameId) const
    {
        return mConfig.enabled && frameId %
            static_cast<std::uint64_t>(std::max(1, mConfig.frameInterval)) == 0;
    }
    void Reset();

    bool Refine(const cv::Mat& referenceGray,
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
                DirectPoseResult& result);

    DirectPoseSummary Summary() const;
    void FlushSummary() const { ExportSummary(); }

    static float HuberWeight(float normalizedResidual, float delta);
    static bool CorrectionSafe(const DirectPoseConfig& config,
                               const Eigen::Matrix<float, 6, 1>& correction,
                               const SlamTelemetry* telemetry,
                               const Eigen::Matrix<float, 6, 6>* directCovariance,
                               float& mahalanobis,
                               float* innovationTrace = nullptr,
                               std::string* rejectionReason = nullptr,
                               Eigen::Matrix<float, 6, 1>* fusedCorrection = nullptr,
                               Eigen::Matrix<float, 6, 6>* fusedCovariance = nullptr,
                               Eigen::Matrix<float, 6, 6>* informationProjector = nullptr,
                               int* informationSubspaceRank = nullptr,
                               float* minimumSelectedDirectionalGain = nullptr,
                               float* maximumSelectedDirectionalGain = nullptr,
                               float* discardedWhitenedInnovation = nullptr,
                               float* degradationHealth = nullptr);

private:
    struct Linearization
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Matrix<double, 6, 6> hessian =
            Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> gradient =
            Eigen::Matrix<double, 6, 1>::Zero();
        double objective = 0.0;
        int correspondences = 0;
    };

    Linearization Linearize(const cv::Mat& referenceIntensity,
                            const cv::Mat& referenceDepth,
                            const cv::Mat& currentIntensity,
                            const cv::Mat& currentDepth,
                            const cv::Mat& gradientX,
                            const cv::Mat& gradientY,
                            const Eigen::Matrix3d& cameraMatrix,
                            const Sophus::SE3d& referenceTcw,
                            const Sophus::SE3d& currentTcw,
                            int stride,
                            bool buildSystem) const;

    static cv::Mat IntensityFloat(const cv::Mat& image);
    static float Bilinear(const cv::Mat& image, float x, float y);
    static Eigen::Matrix3d Skew(const Eigen::Vector3d& vector);
    static double HuberLoss(double residual, double delta);
    void WriteTelemetry(std::uint64_t frameId, double timestamp,
                        const DirectPoseResult& result);
    void ExportSummary() const;

    DirectPoseConfig mConfig;
    mutable std::mutex mMutex;
    DirectPoseSummary mSummary;
    std::ofstream mTelemetry;
};

} // namespace TIB_SLAM

#endif
