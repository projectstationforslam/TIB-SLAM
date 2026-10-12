#include "ObservabilityGaussianMapper.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace
{

void Require(bool condition, const char* message)
{
    if(condition) return;
    std::cerr << "FAILED: " << message << std::endl;
    std::exit(EXIT_FAILURE);
}

} // namespace

int main()
{
    using namespace TIB_SLAM;
    GaussianMappingConfig config;
    config.confidenceGuidance = true;
    config.minimumPoseConfidence = 0.15f;
    config.minimumInsertionConfidence = 0.45f;

    const GaussianMappingPolicy neutral =
        ObservabilityGaussianMapper::EvaluatePolicy(config, nullptr);
    Require(std::abs(neutral.poseConfidence - 1.0f) < 1e-6f,
            "missing telemetry must be neutral");
    Require(neutral.allowInsertion && !neutral.rejectFrame,
            "neutral policy must preserve baseline behavior");

    SlamTelemetry healthy;
    healthy.hessianMinEigenvalue = 1.0f;
    healthy.hessianLogCondition = 2.0f;
    GaussianMappingPolicy policy =
        ObservabilityGaussianMapper::EvaluatePolicy(config, &healthy);
    Require(policy.poseConfidence > 0.95f && policy.allowInsertion,
            "well-observed pose must permit insertion");
    Require(policy.strideMultiplier < 1.1f,
            "well-observed pose must retain dense sampling");

    SlamTelemetry unrecovered;
    unrecovered.geometricallyDegenerate = true;
    unrecovered.hessianMinEigenvalue = 0.0f;
    unrecovered.hessianLogCondition =
        std::numeric_limits<float>::infinity();
    policy = ObservabilityGaussianMapper::EvaluatePolicy(config, &unrecovered);
    Require(policy.rejectFrame && !policy.allowInsertion,
            "unrecovered degeneracy must not contaminate the map");

    SlamTelemetry recovered = unrecovered;
    recovered.selectedStructuralFactors = 6;
    recovered.selectedInformationGain = 0.5f;
    recovered.weakestDirection.tail<3>() = Eigen::Vector3f(2.0f, 0.0f, 0.0f);
    policy = ObservabilityGaussianMapper::EvaluatePolicy(config, &recovered);
    Require(!policy.rejectFrame && policy.allowInsertion,
            "structurally recovered degeneracy must remain usable");
    Require(policy.hasWeakTranslation &&
            std::abs(policy.weakTranslationCamera.norm() - 1.0f) < 1e-6f,
            "weak translation direction must be normalized");

    SlamTelemetry illConditioned;
    illConditioned.hessianMinEigenvalue = 1.0f;
    illConditioned.hessianLogCondition = 10.0f;
    policy = ObservabilityGaussianMapper::EvaluatePolicy(config, &illConditioned);
    Require(!policy.rejectFrame && !policy.allowInsertion,
            "uncertain pose should fuse existing geometry without inserting");
    Require(policy.strideMultiplier > 1.0f && policy.fusionWeight < 0.5f,
            "uncertain pose must consume fewer resources and carry less weight");

    const Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    const Eigen::Vector3f cameraPoint(0.2f, -0.1f, 2.0f);
    const Eigen::Vector3f surfaceScale(0.01f, 0.02f, 0.03f);
    Eigen::Vector3f scaleWithoutPose;
    Eigen::Quaternionf rotationWithoutPose;
    Require(ObservabilityGaussianMapper::PropagatePointCovariance(
                Eigen::Matrix<float, 6, 6>::Zero(), identity, cameraPoint,
                identity, surfaceScale, 0.0f, 1.0f, 1e-4f, 1.0f,
                scaleWithoutPose, rotationWithoutPose),
            "zero-pose covariance propagation must succeed");
    Require(scaleWithoutPose.allFinite() &&
                std::abs(rotationWithoutPose.norm() - 1.0f) < 1e-5f,
            "propagated Gaussian axes and rotation must be finite");
    Require((scaleWithoutPose.array().square().matrix().sum() -
             surfaceScale.array().square().matrix().sum()) < 1e-6f,
            "zero pose/depth uncertainty must preserve total surface variance");

    Eigen::Matrix<float, 6, 6> translationCovariance =
        Eigen::Matrix<float, 6, 6>::Zero();
    // g2o::SE3Quat tangent order is [rotation, translation].
    translationCovariance(3, 3) = 0.04f;
    Eigen::Vector3f scaleWithPose;
    Eigen::Quaternionf rotationWithPose;
    Require(ObservabilityGaussianMapper::PropagatePointCovariance(
                translationCovariance, identity, cameraPoint, identity,
                surfaceScale, 0.001f, 1.0f, 1e-4f, 1.0f,
                scaleWithPose, rotationWithPose),
            "translation-covariance propagation must succeed");
    Require(scaleWithPose.allFinite() &&
                std::abs(rotationWithPose.norm() - 1.0f) < 1e-5f,
            "uncertain propagated Gaussian must stay finite and normalized");
    Require(scaleWithPose.squaredNorm() >
                scaleWithoutPose.squaredNorm() + 0.01f,
            "translation uncertainty must increase total spatial variance");

    std::cout << "All Gaussian policies and covariance tests passed."
              << std::endl;
    return EXIT_SUCCESS;
}
