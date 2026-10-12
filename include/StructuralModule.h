#ifndef TIB_SLAM_STRUCTURAL_MODULE_H
#define TIB_SLAM_STRUCTURAL_MODULE_H

#include "LineExtractor.h"
#include "ManhattanEstimator.h"
#include "PlaneExtractor.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

namespace TIB_SLAM
{

class Frame;
class KeyFrame;
class Map;

class StructuralModule
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Config
    {
        bool enabled = false;
        int extractionFrameInterval = 1;
        float lineMinLength = 35.0f;
        int lineMaxCount = 80;
        float lineAngleCosine = 0.965f;
        float linePixelDistance = 18.0f;
        float line3dDistance = 0.60f;
        int lineDescriptorDistance = 48;
        int lineScale = 2;
        int lineOctaves = 2;
        int maximumAssociationLines = 512;

        int planeStride = 2;
        int planeRansacIterations = 160;
        int planeMinInliers = 500;
        int planeMaxCount = 8;
        float planeDistanceThreshold = 0.04f;
        float planeVoxelLeaf = 0.02f;
        float planeMinimumInlierRatio = 0.72f;
        float planeCentroidDistance = 1.20f;
        float planeMatchDistance = 0.05f;
        float planeMatchCosine = 0.985f;
        float parallelCosine = 0.9962f;
        float perpendicularCosine = 0.08716f;
        float planeAngleInformation = 0.5f;
        float planeDistanceInformation = 50.0f;
        float planeParallelInformation = 0.5f;
        float planePerpendicularInformation = 0.5f;
        float planeChi2 = 100.0f;
        float planeRelationChi2 = 50.0f;
        int maximumAssociationPlanes = 128;

        float minDepth = 0.15f;
        float maxDepth = 8.0f;
        float manhattanOrthogonalCosine = 0.08716f;
        float manhattanMinimumConfidence = 0.25f;
        float manhattanMaxRotationDelta = 0.61f;

        bool observabilityEnabled = false;
        bool structuralLocalBA = false;
        bool degenerateKeyframes = false;
        float observabilityMinimumEigenvalue = 1e-3f;
        float observabilityMaximumLogCondition = 6.0f;
        float observabilityMinimumDirectionalGain = 1e-3f;
        int observabilityWeakDirectionCount = 2;
        int observabilityMaximumSelectedFactors = 24;
        int degenerateKeyframeMinimumInterval = 5;
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

    explicit StructuralModule(const std::string& settingsPath);

    bool Enabled() const { return mConfig.enabled; }
    bool ShouldExtract(std::uint64_t frameId) const
    {
        return mConfig.enabled && frameId % static_cast<std::uint64_t>(
            std::max(1, mConfig.extractionFrameInterval)) == 0;
    }
    const Config& GetConfig() const { return mConfig; }

    void Extract(Frame& frame, const cv::Mat& gray, const cv::Mat& depth) const;
    int Associate(Frame& frame, Map* map);
    void PromoteToKeyFrame(KeyFrame* keyFrame, Frame& frame, Map* map);
    void Reset(Map* map = nullptr);

private:
    static Config LoadConfig(const std::string& settingsPath);
    static bool ProjectLine(const Eigen::Matrix<float, 6, 1>& worldLine,
                            Frame& frame, cv::Point2f& start, cv::Point2f& end,
                            Eigen::Vector3f& startCamera, Eigen::Vector3f& endCamera);
    static Eigen::Vector4f WorldPlaneToCamera(const Eigen::Vector4f& worldPlane,
                                              const Frame& frame);

    Config mConfig;
    LineExtractor mLineExtractor;
    PlaneExtractor mPlaneExtractor;
    ManhattanEstimator mManhattanEstimator;
};

} // namespace TIB_SLAM

#endif
