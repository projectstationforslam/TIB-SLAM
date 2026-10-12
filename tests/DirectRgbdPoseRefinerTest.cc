#include "DirectRgbdPoseRefiner.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{
void Require(bool condition, const char* message)
{
    if(condition) return;
    std::cerr << "FAILED: " << message << std::endl;
    std::exit(EXIT_FAILURE);
}
}

int main()
{
    using namespace TIB_SLAM;
    Require(std::abs(DirectRgbdPoseRefiner::HuberWeight(0.5f, 1.5f) - 1.0f) < 1e-6f,
            "Huber inlier must retain unit weight");
    Require(std::abs(DirectRgbdPoseRefiner::HuberWeight(3.0f, 1.5f) - 0.5f) < 1e-6f,
            "Huber outlier must be downweighted");

    DirectPoseConfig config;
    Eigen::Matrix<float,6,1> correction = Eigen::Matrix<float,6,1>::Zero();
    correction[3] = 0.005f;
    float mahalanobis = 0.0f;
    Require(!DirectRgbdPoseRefiner::CorrectionSafe(config, correction, nullptr, nullptr, mahalanobis),
            "missing covariance must fail when uncertainty gate is enabled");
    config.uncertaintyGate = false;
    Require(DirectRgbdPoseRefiner::CorrectionSafe(config, correction, nullptr, nullptr, mahalanobis),
            "small finite correction must pass when uncertainty gate is disabled");
    correction[3] = 0.10f;
    Require(!DirectRgbdPoseRefiner::CorrectionSafe(config, correction, nullptr, nullptr, mahalanobis),
            "large translation correction must be rejected");

    SlamTelemetry telemetry;
    config.uncertaintyGate = true;
    config.maximumTranslationCorrection = 0.05f;
    telemetry.poseCovarianceValid = true;
    telemetry.poseCovariance = Eigen::Matrix<float,6,6>::Identity() * 1e-4f;
    Eigen::Matrix<float,6,6> directCovariance =
        Eigen::Matrix<float,6,6>::Identity() * 1e-4f;
    correction.setZero();
    correction[3] = 0.01f;
    Eigen::Matrix<float,6,1> fusedCorrection;
    Eigen::Matrix<float,6,6> fusedCovariance;
    Require(DirectRgbdPoseRefiner::CorrectionSafe(config, correction, &telemetry,
            &directCovariance, mahalanobis, nullptr, nullptr, &fusedCorrection,
            &fusedCovariance),
            "one-sigma covariance-consistent correction must pass");
    Require(fusedCorrection[3] > 0.0f && fusedCorrection[3] < correction[3],
            "covariance fusion must damp a finite direct innovation");
    Require(fusedCovariance.allFinite() && fusedCovariance.trace() > 0.0f &&
            fusedCovariance.trace() < telemetry.poseCovariance.trace(),
            "Joseph covariance update must be finite and reduce uncertainty");
    correction[3] = 0.03f;
    config.maximumMahalanobis = 4.0f;
    Require(!DirectRgbdPoseRefiner::CorrectionSafe(config, correction, &telemetry,
            &directCovariance, mahalanobis),
            "three-sigma correction must fail a two-sigma gate");
    Require(std::isfinite(mahalanobis), "Mahalanobis result must remain finite");

    correction[3] = 0.01f;
    config.maximumMahalanobis = 9.0f;
    config.commitOnlyWhenDegenerate = true;
    Require(!DirectRgbdPoseRefiner::CorrectionSafe(config, correction, &telemetry,
            &directCovariance, mahalanobis),
            "well-constrained frames must keep direct proposals in shadow mode");
    telemetry.geometricallyDegenerate = true;
    Require(DirectRgbdPoseRefiner::CorrectionSafe(config, correction, &telemetry,
            &directCovariance, mahalanobis),
            "covariance-consistent correction may pass on a degenerate frame");

    config.commitOnlyWhenDegenerate = false;
    config.maximumMahalanobis = 16.812f;
    config.degradationAwareSubspace = true;
    config.minimumDirectionalGain = 0.08f;
    telemetry.featuresValid = true;
    telemetry.imageGradientMean = 0.03f;
    telemetry.textureRatio = 0.2f;
    telemetry.depthValidRatio = 0.5f;
    telemetry.hessianLogCondition = 2.0f;
    telemetry.poseCovariance = Eigen::Matrix<float,6,6>::Identity() * 1e-4f;
    telemetry.poseCovariance(3,3) = 1e-8f;
    correction.setZero();
    correction[3] = 0.005f;
    Eigen::Matrix<float,6,6> projector;
    int rank = 0;
    float minGain = 0.0f, maxGain = 0.0f, discarded = 0.0f, health = 0.0f;
    Require(DirectRgbdPoseRefiner::CorrectionSafe(config, correction, &telemetry,
            &directCovariance, mahalanobis, nullptr, nullptr, &fusedCorrection,
            &fusedCovariance, &projector, &rank, &minGain, &maxGain,
            &discarded, &health),
            "a valid mixed-rank innovation must retain observable modes");
    Require(rank == 5, "one low-information direction must be removed");
    Require(std::abs(fusedCorrection[3]) < 1e-9f,
            "rejected information direction must receive exactly zero correction");
    Require(projector.allFinite() && discarded > 0.0f && health > 0.9f,
            "subspace telemetry must be finite and expose discarded innovation");

    std::cout << "All direct-pose and information-subspace tests passed." << std::endl;
    return EXIT_SUCCESS;
}
