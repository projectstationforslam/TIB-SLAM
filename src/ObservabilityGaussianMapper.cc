#include "ObservabilityGaussianMapper.h"

#include <opencv2/imgproc.hpp>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

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

float Clamp(float value, float minimum, float maximum)
{
    return std::max(minimum, std::min(maximum, value));
}

float Logit(float probability)
{
    probability = Clamp(probability, 1e-4f, 1.0f - 1e-4f);
    return std::log(probability / (1.0f - probability));
}

} // namespace

std::size_t ObservabilityGaussianMapper::VoxelKeyHash::operator()(const VoxelKey& key) const
{
    std::size_t seed = 1469598103934665603ull;
    const auto mix = [&seed](int value)
    {
        seed ^= static_cast<std::uint32_t>(value);
        seed *= 1099511628211ull;
    };
    mix(key.x);
    mix(key.y);
    mix(key.z);
    return seed;
}

GaussianMappingPolicy ObservabilityGaussianMapper::EvaluatePolicy(
    const GaussianMappingConfig& config,
    const SlamTelemetry* observability)
{
    GaussianMappingPolicy policy;
    if(!config.confidenceGuidance || !observability)
        return policy;

    const bool hasSignal = observability->geometricallyDegenerate ||
        observability->hessianMinEigenvalue > 0.0f ||
        observability->hessianLogCondition > 0.0f ||
        observability->selectedStructuralFactors > 0 ||
        observability->selectedInformationGain > 0.0f;
    if(!hasSignal)
        return policy;

    const float minimumEigenvalue = std::isfinite(observability->hessianMinEigenvalue) ?
        std::max(0.0f, observability->hessianMinEigenvalue) : 0.0f;
    const float eigenReference = std::max(1e-9f, config.eigenvalueReference);
    const float eigenScore = minimumEigenvalue / (minimumEigenvalue + eigenReference);
    float conditionScore = 0.0f;
    if(std::isfinite(observability->hessianLogCondition))
    {
        const float excess = std::max(0.0f,
            observability->hessianLogCondition - config.goodLogCondition);
        conditionScore = std::exp(-excess / std::max(1e-3f, config.conditionDecay));
    }
    const float geometricScore = std::sqrt(Clamp(eigenScore * conditionScore,
                                                  0.0f, 1.0f));

    if(observability->geometricallyDegenerate)
    {
        const float gain = std::max(0.0f, observability->selectedInformationGain);
        const float gainScore = gain /
            (gain + std::max(1e-9f, config.informationGainSaturation));
        const float factorScore = 1.0f - std::exp(-std::max(0,
            observability->selectedStructuralFactors) / 3.0f);
        const float recoveryScore = Clamp(0.7f * gainScore + 0.3f * factorScore,
                                          0.0f, 1.0f);
        const float base = Clamp(config.degeneracyBaseConfidence, 0.0f, 1.0f);
        policy.poseConfidence = Clamp(std::max(geometricScore,
            base + (1.0f - base) * recoveryScore), 0.0f, 1.0f);
    }
    else
    {
        // A non-degenerate Hessian receives a conservative floor while still
        // preserving a continuous penalty near the observability boundary.
        policy.poseConfidence = Clamp(0.25f + 0.75f * geometricScore,
                                      0.0f, 1.0f);
    }

    policy.fusionWeight = Clamp(policy.poseConfidence,
        config.minimumFusionWeight, 1.0f);
    policy.opacityMultiplier = Clamp(0.25f + 0.75f * policy.poseConfidence,
                                     0.05f, 1.0f);
    policy.strideMultiplier = 1.0f + (1.0f - policy.poseConfidence) *
        std::max(0.0f, config.maximumStrideMultiplier - 1.0f);
    policy.allowInsertion = policy.poseConfidence >=
        config.minimumInsertionConfidence;
    policy.rejectFrame = policy.poseConfidence < config.minimumPoseConfidence;

    const Eigen::Vector3f weakTranslation =
        observability->weakestDirection.tail<3>();
    if(weakTranslation.allFinite() && weakTranslation.norm() > 1e-6f)
    {
        policy.hasWeakTranslation = true;
        policy.weakTranslationCamera = weakTranslation.normalized();
    }
    return policy;
}

ObservabilityGaussianMapper::ObservabilityGaussianMapper(
    const std::string& settingsPath)
{
    cv::FileStorage file(settingsPath, cv::FileStorage::READ);
    if(!file.isOpened()) return;

    int enabled = 0;
    int rejectDegeneracy = 1;
    int exportOnShutdown = 1;
    int confidenceGuidance = 1;
    int fullCovariancePropagation = 1;
    int localPhotometricOptimization = 1;
    int directionalGeometryUpdate = 1;
    ReadScalar(file, "GaussianMapping.Enable", enabled);
    ReadScalar(file, "GaussianMapping.RejectUnrecoveredDegeneracy", rejectDegeneracy);
    ReadScalar(file, "GaussianMapping.ExportOnShutdown", exportOnShutdown);
    mConfig.enabled = enabled != 0;
    mConfig.rejectUnrecoveredDegeneracy = rejectDegeneracy != 0;
    mConfig.exportOnShutdown = exportOnShutdown != 0;
    ReadScalar(file, "GaussianMapping.ConfidenceGuidance", confidenceGuidance);
    mConfig.confidenceGuidance = confidenceGuidance != 0;
    ReadScalar(file, "GaussianMapping.FullCovariancePropagation",
               fullCovariancePropagation);
    ReadScalar(file, "GaussianMapping.LocalPhotometricOptimization",
               localPhotometricOptimization);
    ReadScalar(file, "GaussianMapping.DirectionalGeometryUpdate",
               directionalGeometryUpdate);
    mConfig.fullCovariancePropagation = fullCovariancePropagation != 0;
    mConfig.localPhotometricOptimization = localPhotometricOptimization != 0;
    mConfig.directionalGeometryUpdate = directionalGeometryUpdate != 0;
    ReadScalar(file, "GaussianMapping.MinFrameInterval", mConfig.minimumFrameInterval);
    ReadScalar(file, "GaussianMapping.DepthStride", mConfig.depthStride);
    ReadScalar(file, "GaussianMapping.MaxQueueSize", mConfig.maximumQueueSize);
    ReadScalar(file, "GaussianMapping.MaxGaussians", mConfig.maximumGaussians);
    ReadScalar(file, "GaussianMapping.HeldOutModulo", mConfig.heldOutModulo);
    ReadScalar(file, "GaussianMapping.HeldOutResidue", mConfig.heldOutResidue);
    ReadScalar(file, "GaussianMapping.MinSelectedFactors", mConfig.minimumSelectedFactors);
    ReadScalar(file, "GaussianMapping.MinSelectedGain", mConfig.minimumSelectedGain);
    ReadScalar(file, "GaussianMapping.MinTranslation", mConfig.minimumTranslation);
    ReadScalar(file, "GaussianMapping.MinRotation", mConfig.minimumRotation);
    ReadScalar(file, "GaussianMapping.MinDepth", mConfig.minimumDepth);
    ReadScalar(file, "GaussianMapping.MaxDepth", mConfig.maximumDepth);
    ReadScalar(file, "GaussianMapping.VoxelSize", mConfig.voxelSize);
    ReadScalar(file, "GaussianMapping.MinScale", mConfig.minimumScale);
    ReadScalar(file, "GaussianMapping.MaxScale", mConfig.maximumScale);
    ReadScalar(file, "GaussianMapping.ScaleMultiplier", mConfig.scaleMultiplier);
    ReadScalar(file, "GaussianMapping.NormalScaleRatio", mConfig.normalScaleRatio);
    ReadScalar(file, "GaussianMapping.BaseOpacity", mConfig.baseOpacity);
    ReadScalar(file, "GaussianMapping.EigenvalueReference", mConfig.eigenvalueReference);
    ReadScalar(file, "GaussianMapping.GoodLogCondition", mConfig.goodLogCondition);
    ReadScalar(file, "GaussianMapping.ConditionDecay", mConfig.conditionDecay);
    ReadScalar(file, "GaussianMapping.InformationGainSaturation", mConfig.informationGainSaturation);
    ReadScalar(file, "GaussianMapping.DegeneracyBaseConfidence", mConfig.degeneracyBaseConfidence);
    ReadScalar(file, "GaussianMapping.MinPoseConfidence", mConfig.minimumPoseConfidence);
    ReadScalar(file, "GaussianMapping.MinInsertionConfidence", mConfig.minimumInsertionConfidence);
    ReadScalar(file, "GaussianMapping.MaxStrideMultiplier", mConfig.maximumStrideMultiplier);
    ReadScalar(file, "GaussianMapping.WeakDirectionInflation", mConfig.weakDirectionInflation);
    ReadScalar(file, "GaussianMapping.MinFusionWeight", mConfig.minimumFusionWeight);
    ReadScalar(file, "GaussianMapping.MaxFusionDistance", mConfig.maximumFusionDistance);
    ReadScalar(file, "GaussianMapping.MinNormalCosine", mConfig.minimumNormalCosine);
    ReadScalar(file, "GaussianMapping.MaxColorDistance", mConfig.maximumColorDistance);
    ReadScalar(file, "GaussianMapping.PoseCovarianceMultiplier",
               mConfig.poseCovarianceMultiplier);
    ReadScalar(file, "GaussianMapping.DepthNoiseBase", mConfig.depthNoiseBase);
    ReadScalar(file, "GaussianMapping.DepthNoiseQuadratic",
               mConfig.depthNoiseQuadratic);
    ReadScalar(file, "GaussianMapping.PhotometricHuberDelta",
               mConfig.photometricHuberDelta);
    ReadScalar(file, "GaussianMapping.PhotometricMinimumWeight",
               mConfig.photometricMinimumWeight);
    ReadScalar(file, "GaussianMapping.PlyPath", mConfig.plyPath);
    ReadScalar(file, "GaussianMapping.TelemetryPath", mConfig.telemetryPath);
    ReadScalar(file, "GaussianMapping.SummaryPath", mConfig.summaryPath);

    mConfig.minimumFrameInterval = std::max(1, mConfig.minimumFrameInterval);
    mConfig.depthStride = std::max(1, mConfig.depthStride);
    mConfig.maximumQueueSize = std::max(1, mConfig.maximumQueueSize);
    mConfig.maximumGaussians = std::max(1000, mConfig.maximumGaussians);
    mConfig.heldOutModulo = std::max(0, mConfig.heldOutModulo);
    if(mConfig.heldOutModulo > 0)
        mConfig.heldOutResidue = ((mConfig.heldOutResidue % mConfig.heldOutModulo) +
                                  mConfig.heldOutModulo) % mConfig.heldOutModulo;
    mConfig.voxelSize = std::max(1e-4f, mConfig.voxelSize);
    mConfig.minimumScale = std::max(1e-5f, mConfig.minimumScale);
    mConfig.maximumScale = std::max(mConfig.minimumScale, mConfig.maximumScale);
    mConfig.eigenvalueReference = std::max(1e-9f, mConfig.eigenvalueReference);
    mConfig.conditionDecay = std::max(1e-3f, mConfig.conditionDecay);
    mConfig.informationGainSaturation = std::max(1e-9f, mConfig.informationGainSaturation);
    mConfig.minimumPoseConfidence = Clamp(mConfig.minimumPoseConfidence, 0.0f, 1.0f);
    mConfig.minimumInsertionConfidence = Clamp(mConfig.minimumInsertionConfidence,
                                               mConfig.minimumPoseConfidence, 1.0f);
    mConfig.maximumStrideMultiplier = std::max(1.0f, mConfig.maximumStrideMultiplier);
    mConfig.weakDirectionInflation = std::max(0.0f, mConfig.weakDirectionInflation);
    mConfig.minimumFusionWeight = Clamp(mConfig.minimumFusionWeight, 1e-3f, 1.0f);
    mConfig.maximumFusionDistance = std::max(1e-4f, mConfig.maximumFusionDistance);
    mConfig.minimumNormalCosine = Clamp(mConfig.minimumNormalCosine, -1.0f, 1.0f);
    mConfig.maximumColorDistance = std::max(0.0f, mConfig.maximumColorDistance);
    mConfig.poseCovarianceMultiplier = std::max(0.0f,
        mConfig.poseCovarianceMultiplier);
    mConfig.depthNoiseBase = std::max(1e-6f, mConfig.depthNoiseBase);
    mConfig.depthNoiseQuadratic = std::max(0.0f,
        mConfig.depthNoiseQuadratic);
    mConfig.photometricHuberDelta = std::max(1e-4f,
        mConfig.photometricHuberDelta);
    mConfig.photometricMinimumWeight = Clamp(
        mConfig.photometricMinimumWeight, 1e-3f, 1.0f);

    if(!mConfig.enabled) return;
    mGaussians.reserve(static_cast<std::size_t>(mConfig.maximumGaussians));
    mVoxelIndex.reserve(static_cast<std::size_t>(mConfig.maximumGaussians));
    mTelemetry.open(mConfig.telemetryPath, std::ios::out | std::ios::trunc);
    WriteTelemetryHeader();
    mWorker = std::thread(&ObservabilityGaussianMapper::WorkerLoop, this);
    std::cout << "Covariance-guided Gaussian mapper enabled (max "
              << mConfig.maximumGaussians << " gaussians, base stride "
              << mConfig.depthStride << ")" << std::endl;
}

ObservabilityGaussianMapper::~ObservabilityGaussianMapper()
{
    Stop();
}

void ObservabilityGaussianMapper::SubmitKeyFrame(
    const cv::Mat& color, const cv::Mat& depthMeters,
    const Eigen::Matrix4f& Tcw, const Eigen::Matrix3f& calibration,
    bool inputIsRgb, std::uint64_t frameId, double timestamp,
    const SlamTelemetry* observability)
{
    if(!mConfig.enabled || color.empty() || depthMeters.empty() ||
       depthMeters.type() != CV_32F || !Tcw.allFinite() || !calibration.allFinite())
        return;

    // Reject over-dense keyframes before cloning two full-resolution images.
    // This is the only mapper-side work that runs on the tracking thread.
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if(mStopping || mStopped) return;
        ++mSummary.submittedFrames;
        if(mConfig.heldOutModulo > 0 &&
           static_cast<int>(frameId % static_cast<std::uint64_t>(mConfig.heldOutModulo)) ==
               mConfig.heldOutResidue)
        {
            ++mSummary.rejectedHeldOut;
            return;
        }
        if(mHaveSubmittedFrame && frameId < mLastSubmittedFrame +
           static_cast<std::uint64_t>(mConfig.minimumFrameInterval))
        {
            ++mSummary.rejectedInterval;
            return;
        }
        mHaveSubmittedFrame = true;
        mLastSubmittedFrame = frameId;
    }

    FramePacket packet;
    packet.color = color.clone();
    packet.depth = depthMeters.clone();
    packet.Tcw = Tcw;
    packet.K = calibration;
    packet.inputIsRgb = inputIsRgb;
    packet.frameId = frameId;
    packet.timestamp = timestamp;
    if(observability)
    {
        packet.observability = *observability;
        packet.hasObservability = true;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    if(mStopping || mStopped) return;
    if(static_cast<int>(mQueue.size()) >= mConfig.maximumQueueSize)
    {
        mQueue.pop_front();
        ++mSummary.droppedQueueFrames;
    }
    mQueue.emplace_back(std::move(packet));
    mCondition.notify_one();
}

bool ObservabilityGaussianMapper::MotionAccepted(const Eigen::Matrix4f& Tcw) const
{
    if(!mHaveAcceptedPose) return true;
    const Eigen::Matrix4f relative = Tcw * mLastAcceptedTcw.inverse();
    const float translation = relative.block<3,1>(0,3).norm();
    const float cosine = Clamp((relative.block<3,3>(0,0).trace() - 1.0f) * 0.5f,
                               -1.0f, 1.0f);
    const float rotation = std::acos(cosine);
    return translation >= mConfig.minimumTranslation ||
           rotation >= mConfig.minimumRotation;
}

bool ObservabilityGaussianMapper::ObservabilityAccepted(
    const FramePacket& packet) const
{
    if(!mConfig.rejectUnrecoveredDegeneracy || !packet.hasObservability ||
       !packet.observability.geometricallyDegenerate)
        return true;
    return packet.observability.selectedStructuralFactors >=
               mConfig.minimumSelectedFactors &&
           packet.observability.selectedInformationGain >=
               mConfig.minimumSelectedGain;
}

Eigen::Vector3f ObservabilityGaussianMapper::BackProject(
    int x, int y, float depth, const Eigen::Matrix3f& K) const
{
    return Eigen::Vector3f((static_cast<float>(x) - K(0,2)) * depth / K(0,0),
                           (static_cast<float>(y) - K(1,2)) * depth / K(1,1),
                           depth);
}

bool ObservabilityGaussianMapper::SurfaceFrame(
    const cv::Mat& depth, int x, int y, const Eigen::Matrix3f& K,
    int sampleStep,
    Eigen::Matrix3f& cameraBasis, float& tangentialScale) const
{
    const int step = std::max(1, sampleStep);
    if(x + step >= depth.cols || y + step >= depth.rows) return false;
    const float z = depth.at<float>(y,x);
    const float zx = depth.at<float>(y,x + step);
    const float zy = depth.at<float>(y + step,x);
    if(!std::isfinite(zx) || !std::isfinite(zy) ||
       zx < mConfig.minimumDepth || zx > mConfig.maximumDepth ||
       zy < mConfig.minimumDepth || zy > mConfig.maximumDepth)
        return false;
    Eigen::Vector3f tangentX = BackProject(x + step, y, zx, K) -
                               BackProject(x, y, z, K);
    Eigen::Vector3f tangentY = BackProject(x, y + step, zy, K) -
                               BackProject(x, y, z, K);
    Eigen::Vector3f normal = tangentX.cross(tangentY);
    if(!normal.allFinite() || normal.norm() < 1e-6f) return false;
    tangentX.normalize();
    normal.normalize();
    tangentY = normal.cross(tangentX).normalized();
    cameraBasis.col(0) = tangentX;
    cameraBasis.col(1) = tangentY;
    cameraBasis.col(2) = normal;
    tangentialScale = Clamp(z * static_cast<float>(step) /
                            std::max(1.0f, 0.5f * (K(0,0) + K(1,1))) *
                            mConfig.scaleMultiplier,
                            mConfig.minimumScale, mConfig.maximumScale);
    return cameraBasis.allFinite();
}

bool ObservabilityGaussianMapper::PropagatePointCovariance(
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
    Eigen::Quaternionf& propagatedRotation)
{
    if(!poseCovariance.allFinite() || !Rwc.allFinite() ||
       !cameraPoint.allFinite() || !worldSurfaceBasis.allFinite() ||
       !surfaceScale.allFinite() || !std::isfinite(depthSigma))
        return false;
    Eigen::Matrix3f skew;
    skew << 0.0f, -cameraPoint.z(), cameraPoint.y(),
            cameraPoint.z(), 0.0f, -cameraPoint.x(),
            -cameraPoint.y(), cameraPoint.x(), 0.0f;
    Eigen::Matrix<float,3,6> pointJacobian;
    // Tcw is left-perturbed in g2o. Twc exp(-delta) gives
    // dp_world/d_delta = [Rwc [p_camera]_x, -Rwc].
    pointJacobian.block<3,3>(0,0) = Rwc * skew;
    pointJacobian.block<3,3>(0,3) = -Rwc;

    Eigen::Matrix3f covariance = worldSurfaceBasis *
        surfaceScale.array().square().matrix().asDiagonal() *
        worldSurfaceBasis.transpose();
    const Eigen::Vector3f viewingDirection = Rwc.col(2).normalized();
    covariance += depthSigma * depthSigma *
        (viewingDirection * viewingDirection.transpose());
    covariance += std::max(0.0f, poseCovarianceMultiplier) * pointJacobian *
        poseCovariance * pointJacobian.transpose();
    covariance = 0.5f * (covariance + covariance.transpose());
    if(!covariance.allFinite()) return false;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if(solver.info() != Eigen::Success) return false;
    Eigen::Vector3f variances = solver.eigenvalues();
    Eigen::Matrix3f axes = solver.eigenvectors();
    if(!variances.allFinite() || !axes.allFinite()) return false;
    if(axes.determinant() < 0.0f) axes.col(0) = -axes.col(0);
    for(int axis = 0; axis < 3; ++axis)
        variances(axis) = Clamp(variances(axis),
            minimumScale * minimumScale,
            maximumScale * maximumScale);
    propagatedScale = variances.array().sqrt().matrix();
    propagatedRotation = Eigen::Quaternionf(axes).normalized();
    return propagatedScale.allFinite() &&
           propagatedRotation.coeffs().allFinite();
}

bool ObservabilityGaussianMapper::PropagateCovariance(
    const FramePacket& packet, const Eigen::Vector3f& cameraPoint,
    const Eigen::Matrix3f& worldSurfaceBasis,
    const Eigen::Vector3f& surfaceScale, Gaussian& gaussian) const
{
    if(!mConfig.fullCovariancePropagation ||
       !packet.hasObservability ||
       !packet.observability.poseCovarianceValid)
        return false;
    const Eigen::Matrix3f Rwc = packet.Tcw.inverse().block<3,3>(0,0);
    const float depthSigma = mConfig.depthNoiseBase +
        mConfig.depthNoiseQuadratic * cameraPoint.z() * cameraPoint.z();
    return PropagatePointCovariance(packet.observability.poseCovariance,
        Rwc, cameraPoint, worldSurfaceBasis, surfaceScale, depthSigma,
        mConfig.poseCovarianceMultiplier, mConfig.minimumScale,
        mConfig.maximumScale, gaussian.scale, gaussian.rotation);
}

ObservabilityGaussianMapper::FusionResult ObservabilityGaussianMapper::Fuse(
    const Gaussian& candidate, bool allowInsertion, float fusionWeight,
    const Eigen::Matrix3f& geometryProjector,
    float& photometricResidualBefore, float& photometricResidualAfter,
    float& geometricResidualBefore, float& geometricResidualAfter)
{
    photometricResidualBefore = 0.0f;
    photometricResidualAfter = 0.0f;
    geometricResidualBefore = 0.0f;
    geometricResidualAfter = 0.0f;
    const float inverseVoxel = 1.0f / mConfig.voxelSize;
    const VoxelKey key{static_cast<int>(std::floor(candidate.mean.x() * inverseVoxel)),
                       static_cast<int>(std::floor(candidate.mean.y() * inverseVoxel)),
                       static_cast<int>(std::floor(candidate.mean.z() * inverseVoxel))};
    const auto found = mVoxelIndex.find(key);
    if(found == mVoxelIndex.end())
    {
        if(!allowInsertion)
            return FusionResult::SkippedLowConfidence;
        if(static_cast<int>(mGaussians.size()) >= mConfig.maximumGaussians)
            return FusionResult::CapacityLimited;
        mVoxelIndex.emplace(key, mGaussians.size());
        mGaussians.push_back(candidate);
        return FusionResult::Inserted;
    }

    Gaussian& gaussian = mGaussians[found->second];
    const float normalCosine = std::abs(gaussian.normal.dot(candidate.normal));
    if((gaussian.mean - candidate.mean).norm() > mConfig.maximumFusionDistance ||
       normalCosine < mConfig.minimumNormalCosine ||
       (gaussian.color - candidate.color).norm() > mConfig.maximumColorDistance)
        return FusionResult::RejectedConflict;

    photometricResidualBefore =
        (gaussian.color - candidate.color).norm() / std::sqrt(3.0f);
    geometricResidualBefore = (gaussian.mean - candidate.mean).norm();
    float robustPhotometricWeight = 1.0f;
    if(mConfig.localPhotometricOptimization &&
       photometricResidualBefore > mConfig.photometricHuberDelta)
        robustPhotometricWeight = std::max(mConfig.photometricMinimumWeight,
            mConfig.photometricHuberDelta / photometricResidualBefore);

    const float oldWeight = std::min(32.0f,
        std::max(mConfig.minimumFusionWeight, gaussian.accumulatedWeight));
    const float newWeight = Clamp(fusionWeight * robustPhotometricWeight,
        mConfig.minimumFusionWeight * mConfig.photometricMinimumWeight, 1.0f);
    const float alpha = newWeight / (oldWeight + newWeight);
    const Eigen::Vector3f alignedNormal = gaussian.normal.dot(candidate.normal) < 0.0f ?
        -candidate.normal : candidate.normal;
    // Information-subspace geometry update.  The translational block is
    // expressed in the world tangent frame by Process().  Consequently an
    // eigenvector rejected by the degradation gate receives exactly zero
    // Gaussian-mean correction, instead of merely a smaller scalar weight.
    gaussian.mean += alpha * geometryProjector *
        (candidate.mean - gaussian.mean);
    gaussian.color = (1.0f - alpha) * gaussian.color + alpha * candidate.color;
    gaussian.normal = ((1.0f - alpha) * gaussian.normal +
                       alpha * alignedNormal).normalized();
    gaussian.scale = (1.0f - alpha) * gaussian.scale + alpha * candidate.scale;
    gaussian.rotation = gaussian.rotation.slerp(alpha, candidate.rotation).normalized();
    gaussian.confidence = Clamp((1.0f - alpha) * gaussian.confidence +
                                alpha * candidate.confidence, 0.0f, 1.0f);
    gaussian.accumulatedWeight = std::min(32.0f,
        gaussian.accumulatedWeight + newWeight);
    gaussian.opacity = Clamp(gaussian.opacity +
                             0.03f * newWeight * (1.0f - gaussian.opacity),
                             0.05f, 0.98f);
    ++gaussian.observations;
    photometricResidualAfter =
        (gaussian.color - candidate.color).norm() / std::sqrt(3.0f);
    geometricResidualAfter = (gaussian.mean - candidate.mean).norm();
    return FusionResult::Fused;
}

void ObservabilityGaussianMapper::Process(const FramePacket& packet)
{
    const auto started = std::chrono::steady_clock::now();
    const GaussianMappingPolicy policy = EvaluatePolicy(mConfig,
        packet.hasObservability ? &packet.observability : nullptr);
    const int effectiveStride = std::max(1, static_cast<int>(std::lround(
        static_cast<float>(mConfig.depthStride) * policy.strideMultiplier)));
    if(!MotionAccepted(packet.Tcw))
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ++mSummary.rejectedMotion;
        WriteTelemetry(packet, "motion", 0, 0, 0, 0, 0,
                       policy, effectiveStride, 0.0);
        return;
    }
    if(!ObservabilityAccepted(packet))
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ++mSummary.rejectedDegeneracy;
        WriteTelemetry(packet, "degenerate", 0, 0, 0, 0, 0,
                       policy, effectiveStride, 0.0);
        return;
    }
    if(policy.rejectFrame)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ++mSummary.rejectedLowConfidence;
        WriteTelemetry(packet, "low_confidence", 0, 0, 0, 0, 0,
                       policy, effectiveStride, 0.0);
        return;
    }

    const Eigen::Matrix4f Twc = packet.Tcw.inverse();
    const Eigen::Matrix3f Rwc = Twc.block<3,3>(0,0);
    const Eigen::Vector3f twc = Twc.block<3,1>(0,3);
    std::uint64_t samples = 0;
    std::uint64_t inserted = 0;
    std::uint64_t fused = 0;
    std::uint64_t skippedLowConfidence = 0;
    std::uint64_t rejectedConflicts = 0;
    std::uint64_t covariancePropagated = 0;
    std::uint64_t photometricCorrespondences = 0;
    double photometricResidualBefore = 0.0;
    double photometricResidualAfter = 0.0;
    double geometricResidualBefore = 0.0;
    double geometricResidualAfter = 0.0;
    const int step = effectiveStride;
    Eigen::Vector3f weakTranslationWorld = Eigen::Vector3f::Zero();
    if(policy.hasWeakTranslation)
        weakTranslationWorld = (Rwc * policy.weakTranslationCamera).normalized();
    Eigen::Matrix3f geometryProjector = Eigen::Matrix3f::Identity();
    if(mConfig.directionalGeometryUpdate && packet.hasObservability &&
       packet.observability.informationSubspaceValid &&
       packet.observability.informationProjector.allFinite())
    {
        const Eigen::Matrix3f cameraProjector =
            packet.observability.informationProjector.block<3,3>(3,3);
        Eigen::Matrix3f worldProjector = Rwc * cameraProjector * Rwc.transpose();
        worldProjector = 0.5f * (worldProjector + worldProjector.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> projectorSolver(worldProjector);
        if(projectorSolver.info() == Eigen::Success)
        {
            const Eigen::Vector3f bounded = projectorSolver.eigenvalues()
                .cwiseMax(0.0f).cwiseMin(1.0f);
            geometryProjector = projectorSolver.eigenvectors() *
                bounded.asDiagonal() * projectorSolver.eigenvectors().transpose();
        }
    }

    for(int y = 0; y < packet.depth.rows; y += step)
    {
        for(int x = 0; x < packet.depth.cols; x += step)
        {
            const float z = packet.depth.at<float>(y,x);
            if(!std::isfinite(z) || z < mConfig.minimumDepth || z > mConfig.maximumDepth)
                continue;
            ++samples;
            Eigen::Matrix3f cameraBasis;
            float tangentScale = mConfig.minimumScale;
            if(!SurfaceFrame(packet.depth, x, y, packet.K,
                             step, cameraBasis, tangentScale))
                cameraBasis = Eigen::Matrix3f::Identity();

            Gaussian gaussian;
            const Eigen::Vector3f cameraPoint = BackProject(x, y, z, packet.K);
            gaussian.mean = Rwc * cameraPoint + twc;
            const Eigen::Matrix3f worldBasis = Rwc * cameraBasis;
            gaussian.normal = worldBasis.col(2).normalized();
            gaussian.rotation = Eigen::Quaternionf(worldBasis).normalized();
            gaussian.scale = Eigen::Vector3f(tangentScale, tangentScale,
                std::max(mConfig.minimumScale,
                         tangentScale * mConfig.normalScaleRatio));
            const bool haveFullCovariance = mConfig.fullCovariancePropagation &&
                packet.hasObservability &&
                packet.observability.poseCovarianceValid;
            if(policy.hasWeakTranslation && !haveFullCovariance)
            {
                // Encode pose uncertainty geometrically: a Gaussian is wider
                // only along axes aligned with the weakest translational
                // direction. High-confidence frames converge to the baseline scales.
                const float uncertainty = (1.0f - policy.poseConfidence) *
                    mConfig.weakDirectionInflation;
                for(int axis = 0; axis < 3; ++axis)
                {
                    const float alignment = std::abs(
                        worldBasis.col(axis).dot(weakTranslationWorld));
                    gaussian.scale(axis) = Clamp(gaussian.scale(axis) *
                        (1.0f + uncertainty * alignment),
                        mConfig.minimumScale, mConfig.maximumScale);
                }
            }
            if(PropagateCovariance(packet, cameraPoint, worldBasis,
                                   gaussian.scale, gaussian))
                ++covariancePropagated;
            if(packet.color.channels() == 1)
            {
                const float value = packet.color.at<unsigned char>(y,x) / 255.0f;
                gaussian.color = Eigen::Vector3f::Constant(value);
            }
            else
            {
                const cv::Vec3b value = packet.color.at<cv::Vec3b>(y,x);
                if(packet.inputIsRgb)
                    gaussian.color = Eigen::Vector3f(value[0], value[1], value[2]) / 255.0f;
                else
                    gaussian.color = Eigen::Vector3f(value[2], value[1], value[0]) / 255.0f;
            }
            gaussian.confidence = policy.poseConfidence;
            gaussian.accumulatedWeight = policy.fusionWeight;
            gaussian.opacity = Clamp(mConfig.baseOpacity *
                policy.opacityMultiplier, 0.05f, 0.98f);
            float photoBefore = 0.0f;
            float photoAfter = 0.0f;
            float geometryBefore = 0.0f;
            float geometryAfter = 0.0f;
            switch(Fuse(gaussian, policy.allowInsertion, policy.fusionWeight,
                        geometryProjector,
                        photoBefore, photoAfter, geometryBefore, geometryAfter))
            {
                case FusionResult::Inserted: ++inserted; break;
                case FusionResult::Fused:
                    ++fused;
                    ++photometricCorrespondences;
                    photometricResidualBefore += photoBefore;
                    photometricResidualAfter += photoAfter;
                    geometricResidualBefore += geometryBefore;
                    geometricResidualAfter += geometryAfter;
                    break;
                case FusionResult::SkippedLowConfidence:
                    ++skippedLowConfidence; break;
                case FusionResult::RejectedConflict:
                    ++rejectedConflicts; break;
                case FusionResult::CapacityLimited: break;
            }
        }
    }

    const double milliseconds = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::lock_guard<std::mutex> lock(mMutex);
    mHaveAcceptedPose = true;
    mLastAcceptedTcw = packet.Tcw;
    ++mSummary.acceptedFrames;
    mSummary.validDepthSamples += samples;
    mSummary.insertedGaussians += inserted;
    mSummary.fusedGaussians += fused;
    mSummary.skippedLowConfidenceInsertions += skippedLowConfidence;
    mSummary.rejectedFusionConflicts += rejectedConflicts;
    mSummary.covariancePropagatedSamples += covariancePropagated;
    mSummary.photometricCorrespondences += photometricCorrespondences;
    mSummary.photometricResidualBefore += photometricResidualBefore;
    mSummary.photometricResidualAfter += photometricResidualAfter;
    mSummary.geometricResidualBefore += geometricResidualBefore;
    mSummary.geometricResidualAfter += geometricResidualAfter;
    mSummary.gaussianCount = mGaussians.size();
    mSummary.totalProcessingMilliseconds += milliseconds;
    mSummary.summedPoseConfidence += policy.poseConfidence;
    WriteTelemetry(packet, policy.allowInsertion ? "accepted" : "fusion_only",
                   samples, inserted, fused, skippedLowConfidence,
                   rejectedConflicts, policy, effectiveStride, milliseconds);
}

void ObservabilityGaussianMapper::WorkerLoop()
{
    while(true)
    {
        FramePacket packet;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCondition.wait(lock, [this]{
                return mStopping || mResetRequested || !mQueue.empty();
            });
            if(mResetRequested)
            {
                mQueue.clear();
                mGaussians.clear();
                mVoxelIndex.clear();
                mSummary = GaussianMappingSummary();
                mHaveAcceptedPose = false;
                mHaveSubmittedFrame = false;
                mLastSubmittedFrame = 0;
                mResetRequested = false;
                continue;
            }
            if(mQueue.empty() && mStopping) break;
            packet = std::move(mQueue.front());
            mQueue.pop_front();
        }
        Process(packet);
    }
}

void ObservabilityGaussianMapper::Stop()
{
    if(!mConfig.enabled) return;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if(mStopped) return;
        mStopping = true;
    }
    mCondition.notify_all();
    if(mWorker.joinable()) mWorker.join();
    if(mTelemetry.is_open()) mTelemetry.flush();
    if(mConfig.exportOnShutdown) ExportPly();
    ExportSummary();
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mStopped = true;
    }
    const GaussianMappingSummary summary = Summary();
    const double mean = summary.acceptedFrames > 0 ?
        summary.totalProcessingMilliseconds / summary.acceptedFrames : 0.0;
    std::cout << "Gaussian mapper: " << summary.gaussianCount
              << " gaussians from " << summary.acceptedFrames << "/"
              << summary.submittedFrames << " keyframes, mean "
              << mean << " ms/accepted keyframe" << std::endl;
}

void ObservabilityGaussianMapper::Reset()
{
    if(!mConfig.enabled) return;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if(mStopping || mStopped) return;
        mResetRequested = true;
    }
    mCondition.notify_one();
}

GaussianMappingSummary ObservabilityGaussianMapper::Summary() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    GaussianMappingSummary result = mSummary;
    result.gaussianCount = mGaussians.size();
    return result;
}

void ObservabilityGaussianMapper::WriteTelemetryHeader()
{
    if(mTelemetry.is_open())
        mTelemetry << "frame_id,timestamp,decision,degenerate,min_eigenvalue,"
                   << "log_condition,selected_factors,selected_gain,samples,"
                   << "inserted,fused,skipped_low_confidence,rejected_conflicts,"
                   << "pose_confidence,fusion_weight,allow_insertion,effective_stride,"
                   << "line_cost,plane_cost,manhattan_cost,total_gaussians,processing_ms\n";
}

void ObservabilityGaussianMapper::WriteTelemetry(
    const FramePacket& packet, const char* decision, std::uint64_t samples,
    std::uint64_t inserted, std::uint64_t fused,
    std::uint64_t skippedLowConfidence, std::uint64_t rejectedConflicts,
    const GaussianMappingPolicy& policy, int effectiveStride,
    double milliseconds)
{
    if(!mTelemetry.is_open()) return;
    mTelemetry << packet.frameId << ',' << std::setprecision(15) << packet.timestamp
               << ',' << decision << ','
               << (packet.hasObservability && packet.observability.geometricallyDegenerate)
               << ',' << packet.observability.hessianMinEigenvalue
               << ',' << packet.observability.hessianLogCondition
               << ',' << packet.observability.selectedStructuralFactors
               << ',' << packet.observability.selectedInformationGain
               << ',' << samples << ',' << inserted << ',' << fused
               << ',' << skippedLowConfidence << ',' << rejectedConflicts
               << ',' << policy.poseConfidence << ',' << policy.fusionWeight
               << ',' << policy.allowInsertion << ',' << effectiveStride
               << ',' << packet.observability.selectedLineCost
               << ',' << packet.observability.selectedPlaneCost
               << ',' << packet.observability.selectedManhattanCost
               << ',' << mGaussians.size() << ',' << milliseconds << '\n';
}

void ObservabilityGaussianMapper::ExportPly() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::ofstream output(mConfig.plyPath, std::ios::out | std::ios::trunc);
    if(!output.is_open())
    {
        std::cerr << "Cannot write Gaussian map: " << mConfig.plyPath << std::endl;
        return;
    }
    output << "ply\nformat ascii 1.0\nelement vertex " << mGaussians.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "property float nx\nproperty float ny\nproperty float nz\n"
           << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
           << "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
           << "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
           << "property float rot_2\nproperty float rot_3\n"
           << "property float confidence\nproperty uint observations\nend_header\n";
    constexpr float kShDc = 0.28209479177387814f;
    output << std::setprecision(9);
    for(const Gaussian& gaussian : mGaussians)
    {
        const Eigen::Vector3f sh = (gaussian.color - Eigen::Vector3f::Constant(0.5f)) / kShDc;
        output << gaussian.mean.x() << ' ' << gaussian.mean.y() << ' ' << gaussian.mean.z() << ' '
               << gaussian.normal.x() << ' ' << gaussian.normal.y() << ' ' << gaussian.normal.z() << ' '
               << sh.x() << ' ' << sh.y() << ' ' << sh.z() << ' '
               << Logit(gaussian.opacity) << ' '
               << std::log(std::max(1e-6f, gaussian.scale.x())) << ' '
               << std::log(std::max(1e-6f, gaussian.scale.y())) << ' '
               << std::log(std::max(1e-6f, gaussian.scale.z())) << ' '
               << gaussian.rotation.w() << ' ' << gaussian.rotation.x() << ' '
               << gaussian.rotation.y() << ' ' << gaussian.rotation.z() << ' '
               << gaussian.confidence << ' ' << gaussian.observations << '\n';
    }
}

void ObservabilityGaussianMapper::ExportSummary() const
{
    const GaussianMappingSummary summary = Summary();
    std::ofstream output(mConfig.summaryPath, std::ios::out | std::ios::trunc);
    if(!output.is_open()) return;
    const double mean = summary.acceptedFrames > 0 ?
        summary.totalProcessingMilliseconds / summary.acceptedFrames : 0.0;
    const double meanConfidence = summary.acceptedFrames > 0 ?
        summary.summedPoseConfidence / summary.acceptedFrames : 0.0;
    const double meanPhotoBefore = summary.photometricCorrespondences > 0 ?
        summary.photometricResidualBefore / summary.photometricCorrespondences : 0.0;
    const double meanPhotoAfter = summary.photometricCorrespondences > 0 ?
        summary.photometricResidualAfter / summary.photometricCorrespondences : 0.0;
    const double meanGeometryBefore = summary.photometricCorrespondences > 0 ?
        summary.geometricResidualBefore / summary.photometricCorrespondences : 0.0;
    const double meanGeometryAfter = summary.photometricCorrespondences > 0 ?
        summary.geometricResidualAfter / summary.photometricCorrespondences : 0.0;
    output << "{\n"
           << "  \"submitted_frames\": " << summary.submittedFrames << ",\n"
           << "  \"accepted_frames\": " << summary.acceptedFrames << ",\n"
           << "  \"rejected_interval\": " << summary.rejectedInterval << ",\n"
           << "  \"rejected_held_out\": " << summary.rejectedHeldOut << ",\n"
           << "  \"rejected_motion\": " << summary.rejectedMotion << ",\n"
           << "  \"rejected_degeneracy\": " << summary.rejectedDegeneracy << ",\n"
           << "  \"rejected_low_confidence\": " << summary.rejectedLowConfidence << ",\n"
           << "  \"dropped_queue_frames\": " << summary.droppedQueueFrames << ",\n"
           << "  \"valid_depth_samples\": " << summary.validDepthSamples << ",\n"
           << "  \"inserted_gaussians\": " << summary.insertedGaussians << ",\n"
           << "  \"fused_gaussians\": " << summary.fusedGaussians << ",\n"
           << "  \"skipped_low_confidence_insertions\": "
           << summary.skippedLowConfidenceInsertions << ",\n"
           << "  \"rejected_fusion_conflicts\": "
           << summary.rejectedFusionConflicts << ",\n"
           << "  \"covariance_propagated_samples\": "
           << summary.covariancePropagatedSamples << ",\n"
           << "  \"photometric_correspondences\": "
           << summary.photometricCorrespondences << ",\n"
           << "  \"photometric_residual_before\": "
           << meanPhotoBefore << ",\n"
           << "  \"photometric_residual_after\": "
           << meanPhotoAfter << ",\n"
           << "  \"geometric_residual_before_m\": "
           << meanGeometryBefore << ",\n"
           << "  \"geometric_residual_after_m\": "
           << meanGeometryAfter << ",\n"
           << "  \"gaussian_count\": " << summary.gaussianCount << ",\n"
           << "  \"total_processing_ms\": " << summary.totalProcessingMilliseconds << ",\n"
           << "  \"mean_processing_ms\": " << mean << ",\n"
           << "  \"mean_pose_confidence\": " << meanConfidence << "\n}\n";
}

} // namespace TIB_SLAM
