/**
 * Optional structural RGB-D observations for TIB-SLAM.
 *
 * The structures deliberately live behind a shared_ptr in Frame/KeyFrame so
 * the standard point-only and inertial code paths remain unchanged when the
 * module is disabled.
 */
#ifndef TIB_SLAM_STRUCTURAL_TYPES_H
#define TIB_SLAM_STRUCTURAL_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <array>
#include <memory>
#include <vector>

namespace TIB_SLAM
{

class MapLine;
class MapPlane;

struct StructuralLineObservation
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cv::line_descriptor::KeyLine keyLine;
    // One-row CV_8U LBD descriptor. clone() is used when an observation is
    // promoted so this never aliases a temporary extractor output matrix.
    cv::Mat descriptor;
    cv::Point2f startPixel;
    cv::Point2f endPixel;
    Eigen::Vector3f startCamera = Eigen::Vector3f::Zero();
    Eigen::Vector3f endCamera = Eigen::Vector3f::Zero();
    Eigen::Vector3f imageLine = Eigen::Vector3f::Zero();
    float pixelLength = 0.0f;
    bool hasDepth = false;
    bool outlier = false;
    bool observabilitySelected = true;
    float observabilityGain = 0.0f;
    MapLine* mapLine = nullptr;
};

struct StructuralPlaneObservation
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Unit normal followed by signed distance: n.dot(x) + d = 0.
    Eigen::Vector4f cameraPlane = Eigen::Vector4f::Zero();
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > supportCamera;
    // Indices in the downsampled organized PEAC image. Retaining membership
    // prevents disconnected, merely coplanar surfaces from being merged.
    std::vector<int> pixelIndices;
    Eigen::Vector3f centroidCamera = Eigen::Vector3f::Zero();
    Eigen::Matrix3f covarianceCamera = Eigen::Matrix3f::Zero();
    float supportRatio = 0.0f;
    float inlierRatio = 0.0f;
    int imageWidth = 0;
    int imageHeight = 0;
    bool outlier = false;
    bool observabilitySelected = true;
    float observabilityGain = 0.0f;
    MapPlane* mapPlane = nullptr;
    MapPlane* parallelPlane = nullptr;
    MapPlane* perpendicularPlane = nullptr;
};

struct ManhattanObservation
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool valid = false;
    float confidence = 0.0f;
    // Detected Manhattan axes expressed in the current camera frame.
    Eigen::Matrix3f cameraAxes = Eigen::Matrix3f::Identity();
    // Rotation measurement used by the g2o rotation edge.
    Eigen::Matrix3f measuredRcw = Eigen::Matrix3f::Identity();
    int matchedPlaneCount = 0;
};

// Runtime contract shared by the telemetry logger, the ONNX model and the
// pose optimizer. The network predicts the next body-frame motion
// Twc(k)^-1 * Twc(k+1), represented as a Sophus se(3) tangent vector.
constexpr std::size_t kSlamFeatureCount = 40;
constexpr std::size_t kLearnedFactorCount = 5;

enum LearnedFactor : std::size_t
{
    LearnedPointFactor = 0,
    LearnedLineFactor = 1,
    LearnedPlaneFactor = 2,
    LearnedManhattanFactor = 3,
    LearnedTemporalPriorFactor = 4
};

struct LearnedPosePrior
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool valid = false;
    bool usePoseInitialization = false;
    bool useTemporalPrior = false;
    bool useFactorWeights = false;
    Eigen::Matrix<float, 6, 1> relativeIncrement =
        Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> variance =
        Eigen::Matrix<float, 6, 1>::Ones();
    Eigen::Matrix4f predictedTcw = Eigen::Matrix4f::Identity();
    std::array<float, kLearnedFactorCount> factorWeights{{1.f, 1.f, 1.f, 1.f, 1.f}};
    float degenerationProbability = 0.0f;
    float confidence = 0.0f;
};

struct SlamTelemetry
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::array<float, kSlamFeatureCount> features{};
    bool featuresValid = false;
    float imageGradientMean = 0.0f;
    float imageGradientStd = 0.0f;
    float imageEntropy = 0.0f;
    float textureRatio = 0.0f;
    float depthValidRatio = 0.0f;
    float pointResidualRms = 0.0f;
    float lineResidualRms = 0.0f;
    float planeNormalResidualRms = 0.0f;
    float planeDistanceResidualRms = 0.0f;
    float hessianMinEigenvalue = 0.0f;
    float hessianLogCondition = 0.0f;
    Eigen::Matrix<float, 6, 1> weakestDirection =
        Eigen::Matrix<float, 6, 1>::Zero();
    // Inverse point-feature Hessian in g2o tangent order
    // [rotation, translation]. Eigenvalues are regularized before inversion.
    Eigen::Matrix<float, 6, 6> poseCovariance =
        Eigen::Matrix<float, 6, 6>::Zero();
    bool poseCovarianceValid = false;
    // Innovation-metric projector.  Rejected modes receive exactly zero
    // direct pose correction; the posterior covariance retains the sparse
    // prior in their complementary subspace.
    Eigen::Matrix<float, 6, 6> informationProjector =
        Eigen::Matrix<float, 6, 6>::Zero();
    bool informationSubspaceValid = false;
    int informationSubspaceRank = 0;
    float degradationHealth = 0.0f;
    bool geometricallyDegenerate = false;
    int selectedStructuralFactors = 0;
    float selectedInformationGain = 0.0f;
    float selectedFactorCost = 0.0f;
    float selectedLineCost = 0.0f;
    float selectedPlaneCost = 0.0f;
    float selectedManhattanCost = 0.0f;
    float observabilityActivation = 0.0f;
    Eigen::Matrix<float, 6, 1> softWeakness =
        Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 6> temporalInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    bool temporalInformationValid = false;
};

struct ObservabilityConfig
{
    bool enabled = false;
    bool structuralLocalBA = false;
    bool degenerateKeyframes = false;
    float minimumEigenvalue = 1e-3f;
    float maximumLogCondition = 6.0f;
    float minimumDirectionalGain = 1e-3f;
    int weakDirectionCount = 2;
    int maximumSelectedFactors = 24;
    int minimumKeyframeInterval = 5;
    // Information-subspace budgeted scheduler.
    bool budgetedSubspace = false;
    bool temporalTransport = true;
    bool softObservability = true;
    bool greedyLogDet = true;
    bool adjointTransport = true;
    bool useWeakWeights = true;
    int selectionPolicy = 0;
    float temporalRetention = 0.70f;
    float weakEigenvalueRatio = 1e-4f;
    float softTransition = 0.35f;
    float minimumActivation = 0.05f;
    float logDetRegularization = 1e-6f;
    float factorBudget = 24.0f;
};

struct StructuralFrameData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<StructuralLineObservation> lines;
    std::vector<StructuralPlaneObservation> planes;
    ManhattanObservation manhattan;
    LearnedPosePrior learnedPrior;
    SlamTelemetry telemetry;
    ObservabilityConfig observability;
    bool extracted = false;

    // Optimizer weights are carried with the observation so Frame/KeyFrame
    // copies use the exact settings that produced them. Defaults match the
    // published ManhattanSLAM RGB-D configuration.
    float planeAngleInformation = 13131.2f;
    float planeDistanceInformation = 2500.0f;
    float planeParallelInformation = 13131.2f;
    float planePerpendicularInformation = 13131.2f;
    float planeChi2 = 100.0f;
    float planeRelationChi2 = 50.0f;
};

using StructuralFrameDataPtr = std::shared_ptr<StructuralFrameData>;

} // namespace TIB_SLAM

#endif
