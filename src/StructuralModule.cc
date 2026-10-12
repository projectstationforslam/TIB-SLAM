#include "StructuralModule.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapLine.h"
#include "MapPlane.h"

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <tuple>

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

template<typename T>
T ReadSettingCompat(const cv::FileStorage& settings, const char* currentKey,
                    const char* legacyKey, const T& fallback)
{
    const cv::FileNode current = settings[currentKey];
    if(!current.empty())
    {
        T value = fallback;
        current >> value;
        return value;
    }
    return ReadSetting<T>(settings, legacyKey, fallback);
}

Eigen::Vector3f NormalizedImageLine(const cv::Point2f& start, const cv::Point2f& end)
{
    Eigen::Vector3f startH(start.x, start.y, 1.0f);
    Eigen::Vector3f endH(end.x, end.y, 1.0f);
    Eigen::Vector3f line = startH.cross(endH);
    const float norm = line.head<2>().norm();
    if(norm <= 1e-6f)
        return Eigen::Vector3f::Zero();
    line /= norm;
    return line;
}

float SegmentDirectionCosine(const cv::Point2f& a0, const cv::Point2f& a1,
                             const cv::Point2f& b0, const cv::Point2f& b1)
{
    const cv::Point2f da = a1 - a0;
    const cv::Point2f db = b1 - b0;
    const float denominator = cv::norm(da) * cv::norm(db);
    return denominator > 1e-6f ? std::abs(da.dot(db) / denominator) : 0.0f;
}

} // namespace

StructuralModule::Config StructuralModule::LoadConfig(const std::string& settingsPath)
{
    Config config;
    cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
    if(!settings.isOpened())
        return config;

    config.enabled = ReadSetting<int>(settings, "Structure.Enable", 0) != 0;
    config.extractionFrameInterval = std::max(1, ReadSetting<int>(
        settings, "Structure.FrameInterval", config.extractionFrameInterval));
    config.lineMinLength = ReadSetting<float>(settings, "Structure.Line.MinLength", config.lineMinLength);
    config.lineMaxCount = ReadSetting<int>(settings, "Structure.Line.MaxCount", config.lineMaxCount);
    config.lineAngleCosine = ReadSetting<float>(settings, "Structure.Line.MatchCosine", config.lineAngleCosine);
    config.linePixelDistance = ReadSetting<float>(settings, "Structure.Line.MatchPixelDistance", config.linePixelDistance);
    config.line3dDistance = ReadSetting<float>(settings, "Structure.Line.Match3DDistance", config.line3dDistance);
    config.lineDescriptorDistance = ReadSetting<int>(settings, "Structure.Line.DescriptorDistance", config.lineDescriptorDistance);
    config.lineScale = ReadSetting<int>(settings, "Structure.Line.Scale", config.lineScale);
    config.lineOctaves = ReadSetting<int>(settings, "Structure.Line.Octaves", config.lineOctaves);
    config.maximumAssociationLines = std::max(32, ReadSetting<int>(
        settings, "Structure.Association.MaxLines", config.maximumAssociationLines));

    config.planeStride = ReadSetting<int>(settings, "Structure.Plane.Stride", config.planeStride);
    config.planeRansacIterations = ReadSetting<int>(settings, "Structure.Plane.RansacIterations", config.planeRansacIterations);
    config.planeMinInliers = ReadSetting<int>(settings, "Structure.Plane.MinInliers", config.planeMinInliers);
    config.planeMaxCount = ReadSetting<int>(settings, "Structure.Plane.MaxCount", config.planeMaxCount);
    config.planeDistanceThreshold = ReadSettingCompat<float>(settings, "Structure.Plane.DistanceThreshold", "Plane.DistanceThreshold", config.planeDistanceThreshold);
    config.planeVoxelLeaf = ReadSetting<float>(settings, "Structure.Plane.VoxelLeaf", config.planeVoxelLeaf);
    config.planeMinimumInlierRatio = ReadSetting<float>(settings, "Structure.Plane.MinInlierRatio", config.planeMinimumInlierRatio);
    config.planeCentroidDistance = ReadSetting<float>(settings, "Structure.Plane.MatchCentroidDistance", config.planeCentroidDistance);
    config.planeMatchDistance = ReadSettingCompat<float>(settings, "Structure.Plane.MatchDistance", "Plane.AssociationDisRef", config.planeMatchDistance);
    config.planeMatchCosine = ReadSettingCompat<float>(settings, "Structure.Plane.MatchCosine", "Plane.AssociationAngRef", config.planeMatchCosine);
    config.parallelCosine = ReadSettingCompat<float>(settings, "Structure.Plane.ParallelCosine", "Plane.ParallelThreshold", config.parallelCosine);
    config.perpendicularCosine = ReadSettingCompat<float>(settings, "Structure.Plane.PerpendicularCosine", "Plane.VerticalThreshold", config.perpendicularCosine);
    config.planeAngleInformation = ReadSettingCompat<float>(settings, "Structure.Plane.AngleInfo", "Plane.AngleInfo", config.planeAngleInformation);
    config.planeDistanceInformation = ReadSettingCompat<float>(settings, "Structure.Plane.DistanceInfo", "Plane.DistanceInfo", config.planeDistanceInformation);
    config.planeParallelInformation = ReadSettingCompat<float>(settings, "Structure.Plane.RelationInfo", "Plane.ParallelInfo", config.planeParallelInformation);
    config.planeParallelInformation = ReadSetting<float>(settings, "Structure.Plane.ParallelInfo", config.planeParallelInformation);
    config.planePerpendicularInformation = ReadSettingCompat<float>(settings, "Structure.Plane.PerpendicularInfo", "Plane.VerticalInfo", config.planeParallelInformation);
    config.planeChi2 = ReadSettingCompat<float>(settings, "Structure.Plane.Chi", "Plane.Chi", config.planeChi2);
    config.planeRelationChi2 = ReadSettingCompat<float>(settings, "Structure.Plane.RelationChi", "Plane.VPChi", config.planeRelationChi2);
    config.maximumAssociationPlanes = std::max(16, ReadSetting<int>(
        settings, "Structure.Association.MaxPlanes", config.maximumAssociationPlanes));
    config.minDepth = ReadSetting<float>(settings, "Structure.Depth.Min", config.minDepth);
    config.maxDepth = ReadSetting<float>(settings, "Structure.Depth.Max", config.maxDepth);
    config.manhattanOrthogonalCosine = ReadSetting<float>(settings, "Structure.Manhattan.OrthogonalCosine", config.manhattanOrthogonalCosine);
    config.manhattanMinimumConfidence = ReadSetting<float>(settings, "Structure.Manhattan.MinConfidence", config.manhattanMinimumConfidence);
    config.manhattanMaxRotationDelta = ReadSetting<float>(settings, "Structure.Manhattan.MaxRotationDelta", config.manhattanMaxRotationDelta);
    config.observabilityEnabled = ReadSetting<int>(settings, "Structure.Observability.Enable", 0) != 0;
    config.structuralLocalBA = ReadSetting<int>(settings, "Structure.Observability.StructuralLocalBA", 0) != 0;
    config.degenerateKeyframes = ReadSetting<int>(settings, "Structure.Observability.DegenerateKeyframes", 0) != 0;
    config.observabilityMinimumEigenvalue = ReadSetting<float>(settings, "Structure.Observability.MinEigenvalue", config.observabilityMinimumEigenvalue);
    config.observabilityMaximumLogCondition = ReadSetting<float>(settings, "Structure.Observability.MaxLogCondition", config.observabilityMaximumLogCondition);
    config.observabilityMinimumDirectionalGain = ReadSetting<float>(settings, "Structure.Observability.MinDirectionalGain", config.observabilityMinimumDirectionalGain);
    config.observabilityWeakDirectionCount = ReadSetting<int>(settings, "Structure.Observability.WeakDirectionCount", config.observabilityWeakDirectionCount);
    config.observabilityMaximumSelectedFactors = ReadSetting<int>(settings, "Structure.Observability.MaxSelectedFactors", config.observabilityMaximumSelectedFactors);
    config.degenerateKeyframeMinimumInterval = ReadSetting<int>(settings, "Structure.Observability.MinKeyframeInterval", config.degenerateKeyframeMinimumInterval);
    config.budgetedSubspace = ReadSetting<int>(settings, "Structure.Scheduler.Enable", 0) != 0;
    config.temporalTransport = ReadSetting<int>(settings, "Structure.Scheduler.TemporalTransport", 1) != 0;
    config.softObservability = ReadSetting<int>(settings, "Structure.Scheduler.SoftObservability", 1) != 0;
    config.greedyLogDet = ReadSetting<int>(settings, "Structure.Scheduler.GreedyLogDet", 1) != 0;
    config.adjointTransport = ReadSetting<int>(settings, "Structure.Scheduler.AdjointTransport", 1) != 0;
    config.useWeakWeights = ReadSetting<int>(settings, "Structure.Scheduler.UseWeakWeights", 1) != 0;
    config.selectionPolicy = ReadSetting<int>(settings, "Structure.Scheduler.SelectionPolicy", 0);
    config.temporalRetention = ReadSetting<float>(settings, "Structure.Scheduler.TemporalRetention", config.temporalRetention);
    config.weakEigenvalueRatio = ReadSetting<float>(settings, "Structure.Scheduler.WeakEigenvalueRatio", config.weakEigenvalueRatio);
    config.softTransition = ReadSetting<float>(settings, "Structure.Scheduler.SoftTransition", config.softTransition);
    config.minimumActivation = ReadSetting<float>(settings, "Structure.Scheduler.MinimumActivation", config.minimumActivation);
    config.logDetRegularization = ReadSetting<float>(settings, "Structure.Scheduler.LogDetRegularization", config.logDetRegularization);
    config.factorBudget = ReadSetting<float>(settings, "Structure.Scheduler.FactorBudget", config.factorBudget);
    return config;
}

StructuralModule::StructuralModule(const std::string& settingsPath)
    : mConfig(LoadConfig(settingsPath)),
      mLineExtractor(mConfig.lineMinLength, mConfig.lineMaxCount, 2,
                     mConfig.minDepth, mConfig.maxDepth,
                     mConfig.lineScale, mConfig.lineOctaves),
      mPlaneExtractor(mConfig.planeStride, mConfig.planeRansacIterations,
                      mConfig.planeMinInliers, mConfig.planeMaxCount,
                      mConfig.planeDistanceThreshold, mConfig.minDepth,
                      mConfig.maxDepth, mConfig.planeVoxelLeaf,
                      mConfig.planeMinimumInlierRatio),
      mManhattanEstimator(mConfig.manhattanOrthogonalCosine,
                          mConfig.manhattanMinimumConfidence,
                          mConfig.manhattanMaxRotationDelta)
{
}

void StructuralModule::Extract(Frame& frame, const cv::Mat& gray, const cv::Mat& depth) const
{
    if(!mConfig.enabled || !frame.mpCamera || gray.empty() || depth.empty())
        return;

    StructuralFrameDataPtr data(new StructuralFrameData());
    data->lines = mLineExtractor.Extract(gray, depth, frame.mpCamera);
    data->planes = mPlaneExtractor.Extract(depth, frame.mpCamera, frame.mnId);
    // ManhattanSLAM's *Info YAML entries are noise scales, despite their
    // names. Reproduce its conversion to inverse variances exactly:
    // angular: 1/sigma_rad^2, distance: configured_scale^2.
    constexpr float degreesSquaredPerRadianSquared = 3282.80635f;
    const float angleScale = std::max(1e-4f, mConfig.planeAngleInformation);
    const float distanceScale = std::max(1e-4f, mConfig.planeDistanceInformation);
    const float parallelScale = std::max(1e-4f, mConfig.planeParallelInformation);
    const float perpendicularScale =
        std::max(1e-4f, mConfig.planePerpendicularInformation);
    data->planeAngleInformation =
        degreesSquaredPerRadianSquared / (angleScale * angleScale);
    data->planeDistanceInformation = distanceScale * distanceScale;
    data->planeParallelInformation =
        degreesSquaredPerRadianSquared / (parallelScale * parallelScale);
    data->planePerpendicularInformation =
        degreesSquaredPerRadianSquared /
        (perpendicularScale * perpendicularScale);
    data->planeChi2 = std::max(1e-3f, mConfig.planeChi2);
    data->planeRelationChi2 = std::max(1e-3f, mConfig.planeRelationChi2);
    data->observability.enabled = mConfig.observabilityEnabled;
    data->observability.structuralLocalBA = mConfig.structuralLocalBA;
    data->observability.degenerateKeyframes = mConfig.degenerateKeyframes;
    data->observability.minimumEigenvalue = std::max(0.0f, mConfig.observabilityMinimumEigenvalue);
    data->observability.maximumLogCondition = std::max(0.0f, mConfig.observabilityMaximumLogCondition);
    data->observability.minimumDirectionalGain = std::max(0.0f, mConfig.observabilityMinimumDirectionalGain);
    data->observability.weakDirectionCount = std::max(1, std::min(6, mConfig.observabilityWeakDirectionCount));
    data->observability.maximumSelectedFactors = std::max(1, mConfig.observabilityMaximumSelectedFactors);
    data->observability.minimumKeyframeInterval = std::max(1, mConfig.degenerateKeyframeMinimumInterval);
    data->observability.budgetedSubspace = mConfig.budgetedSubspace;
    data->observability.temporalTransport = mConfig.temporalTransport;
    data->observability.softObservability = mConfig.softObservability;
    data->observability.greedyLogDet = mConfig.greedyLogDet;
    data->observability.adjointTransport = mConfig.adjointTransport;
    data->observability.useWeakWeights = mConfig.useWeakWeights;
    data->observability.selectionPolicy = mConfig.selectionPolicy;
    data->observability.temporalRetention = std::max(0.0f, std::min(0.999f, mConfig.temporalRetention));
    data->observability.weakEigenvalueRatio = std::max(1e-9f, mConfig.weakEigenvalueRatio);
    data->observability.softTransition = std::max(1e-3f, mConfig.softTransition);
    data->observability.minimumActivation = std::max(0.0f, std::min(1.0f, mConfig.minimumActivation));
    data->observability.logDetRegularization = std::max(1e-12f, mConfig.logDetRegularization);
    data->observability.factorBudget = std::max(0.0f, mConfig.factorBudget);
    data->extracted = true;
    frame.mpStructuralData = data;
}

bool StructuralModule::ProjectLine(const Eigen::Matrix<float, 6, 1>& worldLine,
                                   Frame& frame, cv::Point2f& start, cv::Point2f& end,
                                   Eigen::Vector3f& startCamera, Eigen::Vector3f& endCamera)
{
    if(!frame.mpCamera || !frame.isSet() || !worldLine.allFinite())
        return false;
    const Sophus::SE3f Tcw = frame.GetPose();
    startCamera = Tcw * worldLine.head<3>();
    endCamera = Tcw * worldLine.tail<3>();
    if(!startCamera.allFinite() || !endCamera.allFinite() ||
       startCamera.z() <= 1e-4f || endCamera.z() <= 1e-4f)
        return false;
    const Eigen::Vector2f startProjection = frame.mpCamera->project(startCamera);
    const Eigen::Vector2f endProjection = frame.mpCamera->project(endCamera);
    if(!startProjection.allFinite() || !endProjection.allFinite())
        return false;
    start = cv::Point2f(startProjection.x(), startProjection.y());
    end = cv::Point2f(endProjection.x(), endProjection.y());
    return true;
}

Eigen::Vector4f StructuralModule::WorldPlaneToCamera(const Eigen::Vector4f& worldPlane,
                                                     const Frame& frame)
{
    Eigen::Vector4f cameraPlane = Eigen::Vector4f::Zero();
    if(!frame.isSet() || !worldPlane.allFinite())
        return cameraPlane;
    const Sophus::SE3f Tcw = frame.GetPose();
    cameraPlane.head<3>() = Tcw.rotationMatrix() * worldPlane.head<3>();
    cameraPlane(3) = worldPlane(3) - cameraPlane.head<3>().dot(Tcw.translation());
    const float norm = cameraPlane.head<3>().norm();
    if(norm > 1e-6f)
        cameraPlane /= norm;
    return cameraPlane;
}

int StructuralModule::Associate(Frame& frame, Map* map)
{
    if(!mConfig.enabled || !map || !frame.mpStructuralData || !frame.isSet())
        return 0;

    StructuralFrameData& data = *frame.mpStructuralData;
    for(StructuralLineObservation& line : data.lines)
    {
        line.mapLine = nullptr;
        line.outlier = false;
    }
    for(StructuralPlaneObservation& plane : data.planes)
    {
        plane.mapPlane = nullptr;
        plane.parallelPlane = nullptr;
        plane.perpendicularPlane = nullptr;
        plane.outlier = false;
    }

    struct Candidate { float score; std::size_t observation; std::size_t landmark; };
    std::vector<MapLine*> mapLines = map->GetAllMapLines();
    // Global structural landmark scans made the synchronous frontend grow
    // from milliseconds to tens of seconds as a map accumulated thousands of
    // line/plane fragments.  Keep a deterministic, recency-first working set;
    // loop recognition remains responsible for global place association.
    const auto linePriority = [](MapLine* lhs, MapLine* rhs) {
        if(!lhs) return false;
        if(!rhs) return true;
        if(lhs->mnLastFrameSeen != rhs->mnLastFrameSeen)
            return lhs->mnLastFrameSeen > rhs->mnLastFrameSeen;
        return lhs->Observations() > rhs->Observations();
    };
    if(mapLines.size() > static_cast<std::size_t>(mConfig.maximumAssociationLines))
    {
        std::partial_sort(mapLines.begin(),
                          mapLines.begin() + mConfig.maximumAssociationLines,
                          mapLines.end(), linePriority);
        mapLines.resize(static_cast<std::size_t>(mConfig.maximumAssociationLines));
    }
    std::vector<Candidate> lineCandidates;
    for(std::size_t landmarkIndex = 0; landmarkIndex < mapLines.size(); ++landmarkIndex)
    {
        MapLine* mapLine = mapLines[landmarkIndex];
        if(!mapLine || mapLine->isBad())
            continue;
        cv::Point2f projectedStart, projectedEnd;
        Eigen::Vector3f startCamera, endCamera;
        if(!ProjectLine(mapLine->GetWorldPos(), frame, projectedStart, projectedEnd,
                        startCamera, endCamera))
            continue;
        const cv::Mat mapDescriptor = mapLine->GetDescriptor();
        if(mapDescriptor.empty() || mapDescriptor.type() != CV_8U || mapDescriptor.rows != 1)
            continue;
        const Eigen::Vector3f projectedLine = NormalizedImageLine(projectedStart, projectedEnd);
        if(projectedLine.head<2>().norm() < 0.5f)
            continue;

        for(std::size_t observationIndex = 0; observationIndex < data.lines.size(); ++observationIndex)
        {
            const StructuralLineObservation& observation = data.lines[observationIndex];
            if(observation.descriptor.empty() || observation.descriptor.type() != CV_8U ||
               observation.descriptor.rows != 1 || observation.descriptor.cols != mapDescriptor.cols)
                continue;
            const double descriptorDistance = cv::norm(
                observation.descriptor, mapDescriptor, cv::NORM_HAMMING);
            if(!std::isfinite(descriptorDistance) ||
               descriptorDistance > static_cast<double>(mConfig.lineDescriptorDistance))
                continue;
            const float cosine = SegmentDirectionCosine(projectedStart, projectedEnd,
                                                         observation.startPixel, observation.endPixel);
            if(cosine < mConfig.lineAngleCosine)
                continue;
            const float pixelDistance = 0.5f *
                (std::abs(observation.imageLine.dot(Eigen::Vector3f(projectedStart.x, projectedStart.y, 1.0f))) +
                 std::abs(observation.imageLine.dot(Eigen::Vector3f(projectedEnd.x, projectedEnd.y, 1.0f))));
            if(pixelDistance > mConfig.linePixelDistance)
                continue;

            float distance3d = 0.0f;
            if(observation.hasDepth)
            {
                const float sameOrder = (startCamera - observation.startCamera).norm() +
                                        (endCamera - observation.endCamera).norm();
                const float swapped = (startCamera - observation.endCamera).norm() +
                                      (endCamera - observation.startCamera).norm();
                distance3d = 0.5f * std::min(sameOrder, swapped);
                if(distance3d > mConfig.line3dDistance)
                    continue;
            }
            lineCandidates.push_back({static_cast<float>(0.08 * descriptorDistance) + pixelDistance +
                                      20.0f * (1.0f - cosine) + 5.0f * distance3d,
                                      observationIndex, landmarkIndex});
        }
    }
    std::sort(lineCandidates.begin(), lineCandidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.score < rhs.score; });
    std::set<std::size_t> usedLineObservations, usedLineLandmarks;
    int matches = 0;
    for(const Candidate& candidate : lineCandidates)
    {
        if(usedLineObservations.count(candidate.observation) ||
           usedLineLandmarks.count(candidate.landmark))
            continue;
        data.lines[candidate.observation].mapLine = mapLines[candidate.landmark];
        if(mapLines[candidate.landmark]->mnLastFrameSeen != frame.mnId)
        {
            mapLines[candidate.landmark]->IncreaseVisible();
            mapLines[candidate.landmark]->mnLastFrameSeen = frame.mnId;
        }
        usedLineObservations.insert(candidate.observation);
        usedLineLandmarks.insert(candidate.landmark);
        ++matches;
    }

    std::vector<MapPlane*> mapPlanes = map->GetAllMapPlanes();
    const auto planePriority = [](MapPlane* lhs, MapPlane* rhs) {
        if(!lhs) return false;
        if(!rhs) return true;
        if(lhs->mnLastFrameSeen != rhs->mnLastFrameSeen)
            return lhs->mnLastFrameSeen > rhs->mnLastFrameSeen;
        return lhs->Observations() > rhs->Observations();
    };
    if(mapPlanes.size() > static_cast<std::size_t>(mConfig.maximumAssociationPlanes))
    {
        std::partial_sort(mapPlanes.begin(),
                          mapPlanes.begin() + mConfig.maximumAssociationPlanes,
                          mapPlanes.end(), planePriority);
        mapPlanes.resize(static_cast<std::size_t>(mConfig.maximumAssociationPlanes));
    }
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > cameraMapPlanes(mapPlanes.size());
    std::vector<Candidate> planeCandidates;
    for(std::size_t landmarkIndex = 0; landmarkIndex < mapPlanes.size(); ++landmarkIndex)
    {
        MapPlane* mapPlane = mapPlanes[landmarkIndex];
        if(!mapPlane || mapPlane->isBad())
            continue;
        cameraMapPlanes[landmarkIndex] = WorldPlaneToCamera(mapPlane->GetWorldPos(), frame);
        if(cameraMapPlanes[landmarkIndex].head<3>().norm() < 0.5f)
            continue;
        for(std::size_t observationIndex = 0; observationIndex < data.planes.size(); ++observationIndex)
        {
            Eigen::Vector4f predicted = cameraMapPlanes[landmarkIndex];
            const Eigen::Vector4f& observed = data.planes[observationIndex].cameraPlane;
            float cosine = predicted.head<3>().dot(observed.head<3>());
            if(cosine < 0.0f)
            {
                predicted = -predicted;
                cosine = -cosine;
            }
            const float distance = std::abs(predicted(3) - observed(3));
            float centroidDistance = 0.0f;
            const MapPlane::PointVector support = mapPlane->GetSupportPoints();
            if(!support.empty() && data.planes[observationIndex].centroidCamera.allFinite())
            {
                Eigen::Vector3f mapCentroid = Eigen::Vector3f::Zero();
                std::size_t validSupport = 0;
                for(const Eigen::Vector3f& worldPoint : support)
                    if(worldPoint.allFinite())
                    {
                        mapCentroid += frame.GetPose() * worldPoint;
                        ++validSupport;
                    }
                if(validSupport > 0)
                    centroidDistance = (mapCentroid / static_cast<float>(validSupport) -
                                        data.planes[observationIndex].centroidCamera).norm();
            }
            if(cosine >= mConfig.planeMatchCosine && distance <= mConfig.planeMatchDistance &&
               std::isfinite(centroidDistance) && centroidDistance <= mConfig.planeCentroidDistance)
                planeCandidates.push_back({20.0f * (1.0f - cosine) + distance + 0.05f * centroidDistance,
                                           observationIndex, landmarkIndex});
        }
    }
    std::sort(planeCandidates.begin(), planeCandidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.score < rhs.score; });
    std::set<std::size_t> usedPlaneObservations, usedPlaneLandmarks;
    for(const Candidate& candidate : planeCandidates)
    {
        if(usedPlaneObservations.count(candidate.observation) ||
           usedPlaneLandmarks.count(candidate.landmark))
            continue;
        data.planes[candidate.observation].mapPlane = mapPlanes[candidate.landmark];
        if(mapPlanes[candidate.landmark]->mnLastFrameSeen != frame.mnId)
        {
            mapPlanes[candidate.landmark]->IncreaseVisible();
            mapPlanes[candidate.landmark]->mnLastFrameSeen = frame.mnId;
        }
        usedPlaneObservations.insert(candidate.observation);
        usedPlaneLandmarks.insert(candidate.landmark);
        ++matches;
    }

    // Structural-only relations are allowed to reuse landmarks. They are not
    // treated as direct plane observations and receive separate robust edges.
    for(StructuralPlaneObservation& observation : data.planes)
    {
        if(observation.mapPlane)
            continue;
        float bestParallel = mConfig.parallelCosine;
        float bestPerpendicular = mConfig.perpendicularCosine;
        for(std::size_t landmarkIndex = 0; landmarkIndex < mapPlanes.size(); ++landmarkIndex)
        {
            if(!mapPlanes[landmarkIndex] || mapPlanes[landmarkIndex]->isBad())
                continue;
            const float cosine = std::abs(cameraMapPlanes[landmarkIndex].head<3>().dot(
                                          observation.cameraPlane.head<3>()));
            if(cosine > bestParallel)
            {
                bestParallel = cosine;
                observation.parallelPlane = mapPlanes[landmarkIndex];
            }
            if(cosine < bestPerpendicular)
            {
                bestPerpendicular = cosine;
                observation.perpendicularPlane = mapPlanes[landmarkIndex];
            }
        }
    }

    mManhattanEstimator.Update(data, frame.GetPose().rotationMatrix(), map);
    return matches;
}

void StructuralModule::PromoteToKeyFrame(KeyFrame* keyFrame, Frame& frame, Map* map)
{
    if(!mConfig.enabled || !keyFrame || !map || !frame.mpStructuralData || !frame.isSet())
        return;

    StructuralFrameData& data = *frame.mpStructuralData;
    const Sophus::SE3f Twc = frame.GetPose().inverse();
    for(std::size_t i = 0; i < data.lines.size(); ++i)
    {
        StructuralLineObservation& observation = data.lines[i];
        if(observation.outlier || !observation.hasDepth)
            continue;
        MapLine* mapLine = observation.mapLine;
        if(!mapLine || mapLine->isBad())
        {
            std::unique_ptr<MapLine> owned(new MapLine(Twc * observation.startCamera,
                                                       Twc * observation.endCamera,
                                                       keyFrame, map));
            mapLine = map->AddMapLine(std::move(owned));
            observation.mapLine = mapLine;
        }
        if(mapLine)
        {
            mapLine->AddDescriptor(observation.descriptor);
            mapLine->AddObservation(keyFrame, i);
            keyFrame->AddMapLine(mapLine, i);
        }
    }

    for(std::size_t i = 0; i < data.planes.size(); ++i)
    {
        StructuralPlaneObservation& observation = data.planes[i];
        if(observation.outlier)
            continue;
        MapPlane* mapPlane = observation.mapPlane;
        bool created = false;
        if(!mapPlane || mapPlane->isBad())
        {
            Eigen::Vector4f worldPlane;
            worldPlane.head<3>() = Twc.rotationMatrix() * observation.cameraPlane.head<3>();
            worldPlane(3) = observation.cameraPlane(3) +
                            observation.cameraPlane.head<3>().dot(frame.GetPose().translation());
            MapPlane::PointVector supportWorld;
            supportWorld.reserve(observation.supportCamera.size());
            for(const Eigen::Vector3f& point : observation.supportCamera)
                supportWorld.push_back(Twc * point);
            std::unique_ptr<MapPlane> owned(new MapPlane(worldPlane, supportWorld, keyFrame, map));
            mapPlane = map->AddMapPlane(std::move(owned));
            observation.mapPlane = mapPlane;
            created = mapPlane != nullptr;
        }
        if(mapPlane)
        {
            Eigen::Vector4f worldPlane;
            worldPlane.head<3>() = Twc.rotationMatrix() * observation.cameraPlane.head<3>();
            worldPlane(3) = observation.cameraPlane(3) +
                            observation.cameraPlane.head<3>().dot(frame.GetPose().translation());
            MapPlane::PointVector supportWorld;
            supportWorld.reserve(observation.supportCamera.size());
            for(const Eigen::Vector3f& point : observation.supportCamera)
                if(point.allFinite())
                    supportWorld.push_back(Twc * point);
            if(!created)
                mapPlane->FuseObservation(worldPlane, supportWorld);
            mapPlane->AddObservation(keyFrame, i);
            keyFrame->AddMapPlane(mapPlane, i);
        }
    }
}

void StructuralModule::Reset(Map* map)
{
    mManhattanEstimator.Reset(map);
}

} // namespace TIB_SLAM
