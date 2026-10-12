#ifndef TIB_SLAM_OBSERVABILITY_GAUSSIAN_MAPPER_H
#define TIB_SLAM_OBSERVABILITY_GAUSSIAN_MAPPER_H

#include "StructuralTypes.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace TIB_SLAM
{

struct GaussianMappingConfig
{
    bool enabled = false;
    bool rejectUnrecoveredDegeneracy = true;
    bool exportOnShutdown = true;
    int minimumFrameInterval = 5;
    int depthStride = 6;
    int maximumQueueSize = 3;
    int maximumGaussians = 500000;
    // Deterministic view hold-out for leakage-free rendering evaluation.
    // Zero disables it; otherwise frames with id % modulo == residue are
    // excluded from map construction and reserved for rendering metrics.
    int heldOutModulo = 0;
    int heldOutResidue = 0;
    int minimumSelectedFactors = 1;
    float minimumSelectedGain = 1e-3f;
    float minimumTranslation = 0.04f;
    float minimumRotation = 0.035f;
    float minimumDepth = 0.15f;
    float maximumDepth = 8.0f;
    float voxelSize = 0.015f;
    float minimumScale = 0.002f;
    float maximumScale = 0.05f;
    float scaleMultiplier = 1.25f;
    float normalScaleRatio = 0.25f;
    float baseOpacity = 0.70f;
    // Turn pose observability into a continuous mapping policy instead of
    // using it only as a binary keyframe gate.
    bool confidenceGuidance = true;
    float eigenvalueReference = 1e-3f;
    float goodLogCondition = 3.0f;
    float conditionDecay = 2.0f;
    float informationGainSaturation = 0.1f;
    float degeneracyBaseConfidence = 0.10f;
    float minimumPoseConfidence = 0.15f;
    float minimumInsertionConfidence = 0.45f;
    float maximumStrideMultiplier = 2.5f;
    float weakDirectionInflation = 1.5f;
    float minimumFusionWeight = 0.05f;
    float maximumFusionDistance = 0.025f;
    float minimumNormalCosine = 0.50f;
    float maximumColorDistance = 0.90f;
    // Covariance propagation and robust local photometric refinement.
    bool fullCovariancePropagation = true;
    float poseCovarianceMultiplier = 1.0f;
    float depthNoiseBase = 0.001f;
    float depthNoiseQuadratic = 0.0015f;
    bool localPhotometricOptimization = true;
    // Single-factor ablation switch. Disabling this keeps all confidence,
    // covariance, and photometric policies but replaces the geometry projector
    // by the identity during Gaussian fusion.
    bool directionalGeometryUpdate = true;
    float photometricHuberDelta = 0.10f;
    float photometricMinimumWeight = 0.10f;
    std::string plyPath = "GaussianMap.ply";
    std::string telemetryPath = "GaussianTelemetry.csv";
    std::string summaryPath = "GaussianSummary.json";
};

struct GaussianMappingSummary
{
    std::uint64_t submittedFrames = 0;
    std::uint64_t acceptedFrames = 0;
    std::uint64_t rejectedInterval = 0;
    std::uint64_t rejectedHeldOut = 0;
    std::uint64_t rejectedMotion = 0;
    std::uint64_t rejectedDegeneracy = 0;
    std::uint64_t rejectedLowConfidence = 0;
    std::uint64_t droppedQueueFrames = 0;
    std::uint64_t validDepthSamples = 0;
    std::uint64_t insertedGaussians = 0;
    std::uint64_t fusedGaussians = 0;
    std::uint64_t skippedLowConfidenceInsertions = 0;
    std::uint64_t rejectedFusionConflicts = 0;
    std::size_t gaussianCount = 0;
    double totalProcessingMilliseconds = 0.0;
    double summedPoseConfidence = 0.0;
    std::uint64_t covariancePropagatedSamples = 0;
    std::uint64_t photometricCorrespondences = 0;
    double photometricResidualBefore = 0.0;
    double photometricResidualAfter = 0.0;
    double geometricResidualBefore = 0.0;
    double geometricResidualAfter = 0.0;
};

struct GaussianMappingPolicy
{
    float poseConfidence = 1.0f;
    float fusionWeight = 1.0f;
    float opacityMultiplier = 1.0f;
    float strideMultiplier = 1.0f;
    bool allowInsertion = true;
    bool rejectFrame = false;
    bool hasWeakTranslation = false;
    Eigen::Vector3f weakTranslationCamera = Eigen::Vector3f::Zero();
};

class ObservabilityGaussianMapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit ObservabilityGaussianMapper(const std::string& settingsPath);
    ~ObservabilityGaussianMapper();

    bool Enabled() const { return mConfig.enabled; }

    void SubmitKeyFrame(const cv::Mat& color,
                        const cv::Mat& depthMeters,
                        const Eigen::Matrix4f& Tcw,
                        const Eigen::Matrix3f& calibration,
                        bool inputIsRgb,
                        std::uint64_t frameId,
                        double timestamp,
                        const SlamTelemetry* observability);

    void Stop();
    void Reset();
    GaussianMappingSummary Summary() const;

    // Pure, deterministic mapping policy exposed for regression tests and ablation.
    static GaussianMappingPolicy EvaluatePolicy(
        const GaussianMappingConfig& config,
        const SlamTelemetry* observability);
    static bool PropagatePointCovariance(
        const Eigen::Matrix<float,6,6>& poseCovariance,
        const Eigen::Matrix3f& Rwc,
        const Eigen::Vector3f& cameraPoint,
        const Eigen::Matrix3f& worldSurfaceBasis,
        const Eigen::Vector3f& surfaceScale,
        float depthSigma,
        float poseCovarianceMultiplier,
        float minimumScale,
        float maximumScale,
        Eigen::Vector3f& propagatedScale,
        Eigen::Quaternionf& propagatedRotation);

private:
    struct FramePacket
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        cv::Mat color;
        cv::Mat depth;
        Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
        SlamTelemetry observability;
        bool hasObservability = false;
        bool inputIsRgb = false;
        std::uint64_t frameId = 0;
        double timestamp = 0.0;
    };

    struct Gaussian
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
        Eigen::Vector3f color = Eigen::Vector3f::Zero();
        Eigen::Vector3f scale = Eigen::Vector3f::Constant(0.01f);
        Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
        float opacity = 0.7f;
        float confidence = 1.0f;
        float accumulatedWeight = 1.0f;
        std::uint32_t observations = 1;
    };

    struct VoxelKey
    {
        int x = 0;
        int y = 0;
        int z = 0;
        bool operator==(const VoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash
    {
        std::size_t operator()(const VoxelKey& key) const;
    };

    void WorkerLoop();
    void Process(const FramePacket& packet);
    bool MotionAccepted(const Eigen::Matrix4f& Tcw) const;
    bool ObservabilityAccepted(const FramePacket& packet) const;
    Eigen::Vector3f BackProject(int x, int y, float depth,
                                const Eigen::Matrix3f& K) const;
    bool SurfaceFrame(const cv::Mat& depth, int x, int y,
                      const Eigen::Matrix3f& K,
                      int sampleStep,
                      Eigen::Matrix3f& cameraBasis,
                      float& tangentialScale) const;
    enum class FusionResult
    {
        Inserted,
        Fused,
        SkippedLowConfidence,
        RejectedConflict,
        CapacityLimited
    };
    FusionResult Fuse(const Gaussian& gaussian, bool allowInsertion,
                      float fusionWeight,
                      const Eigen::Matrix3f& geometryProjector,
                      float& photometricResidualBefore,
                      float& photometricResidualAfter,
                      float& geometricResidualBefore,
                      float& geometricResidualAfter);
    bool PropagateCovariance(const FramePacket& packet,
                             const Eigen::Vector3f& cameraPoint,
                             const Eigen::Matrix3f& worldSurfaceBasis,
                             const Eigen::Vector3f& surfaceScale,
                             Gaussian& gaussian) const;
    void ExportPly() const;
    void ExportSummary() const;
    void WriteTelemetryHeader();
    void WriteTelemetry(const FramePacket& packet, const char* decision,
                        std::uint64_t samples, std::uint64_t inserted,
                        std::uint64_t fused,
                        std::uint64_t skippedLowConfidence,
                        std::uint64_t rejectedConflicts,
                        const GaussianMappingPolicy& policy,
                        int effectiveStride, double milliseconds);

    GaussianMappingConfig mConfig;
    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    std::deque<FramePacket> mQueue;
    std::thread mWorker;
    bool mStopping = false;
    bool mStopped = false;
    bool mResetRequested = false;
    bool mHaveAcceptedPose = false;
    bool mHaveSubmittedFrame = false;
    std::uint64_t mLastSubmittedFrame = 0;
    Eigen::Matrix4f mLastAcceptedTcw = Eigen::Matrix4f::Identity();
    GaussianMappingSummary mSummary;
    std::vector<Gaussian, Eigen::aligned_allocator<Gaussian> > mGaussians;
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> mVoxelIndex;
    std::ofstream mTelemetry;
};

} // namespace TIB_SLAM

#endif
