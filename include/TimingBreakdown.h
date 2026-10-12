// TimingBreakdown.h
//
// Accumulates per-frame wall-clock time for the structural/scheduler pipeline
// and per-factor-type candidate-information compute, so the paper can
//   (a) report a phased timing breakdown (feature extraction, association,
//       candidate information, scheduling, pose optimization, direct recovery),
//   (b) calibrate the scheduler costs in Table I against measured compute.
//
// Only the tracking thread touches this accumulator (PoseOptimization and the
// structural stages it wraps run on the tracking thread), so no locking is
// needed. The inline function-local static is a single object shared across
// translation units.
#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace TIB_SLAM
{

struct TimingBreakdown
{
    // Phased per-frame wall-clock totals (seconds), summed over frames.
    double featureExtract = 0.0;    // ORB extraction + StructuralModule::Extract
    double association = 0.0;       // StructuralModule::Associate
    double graphBuild = 0.0;        // g2o point+structural edge construction
    double candidateInfo = 0.0;     // PoseInformation() + directional gain
    double scheduling = 0.0;        // InformationBudgetScheduler::Select
    double poseOptimization = 0.0;  // g2o 4x optimize + outlier reclassification
    double directRecovery = 0.0;    // DirectRgbdPoseRefiner::Refine
    long frames = 0;                // frames reaching the scheduler path
    long associationFrames = 0;     // frames reaching structural association

    // Per-factor-type candidate-information compute (for cost calibration).
    double candidateLine = 0.0, candidatePlane = 0.0, candidateParallel = 0.0,
           candidatePerpendicular = 0.0, candidateManhattan = 0.0;
    long countLine = 0, countPlane = 0, countParallel = 0,
         countPerpendicular = 0, countManhattan = 0;

    // Factors that actually entered the pose update (per type).
    long selectedLine = 0, selectedPlane = 0, selectedParallel = 0,
         selectedPerpendicular = 0, selectedManhattan = 0;
};

inline TimingBreakdown& Timing()
{
    static TimingBreakdown instance;
    return instance;
}

inline double NowSeconds()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// RAII stopwatch that adds its elapsed seconds to the target on destruction.
struct ScopedTimer
{
    double* target;
    double start;
    explicit ScopedTimer(double* t) : target(t), start(NowSeconds()) {}
    ~ScopedTimer() { *target += NowSeconds() - start; }
};

// Writes the accumulated breakdown as JSON to `path`. Idempotent; safe to call
// after SLAM.Shutdown() from the executable.
inline void DumpTimingBreakdown(const std::string& path)
{
    TimingBreakdown& t = Timing();
    FILE* out = std::fopen(path.c_str(), "w");
    if(!out)
        return;
    const auto write = [&](const char* key, double value, bool last)
    {
        std::fprintf(out, "    \"%s\": %.9f%s\n", key, value, last ? "" : ",");
    };
    std::fprintf(out, "{\n");
    std::fprintf(out, "  \"frames\": %ld,\n", t.frames);
    std::fprintf(out, "  \"association_frames\": %ld,\n", t.associationFrames);
    write("feature_extract_s", t.featureExtract, false);
    write("association_s", t.association, false);
    write("graph_build_s", t.graphBuild, false);
    write("candidate_info_s", t.candidateInfo, false);
    write("scheduling_s", t.scheduling, false);
    write("pose_optimization_s", t.poseOptimization, false);
    write("direct_recovery_s", t.directRecovery, true);
    std::fprintf(out, "}\n");
    std::fclose(out);

    // Per-factor-type candidate-information compute + counts, so the paper can
    // calibrate the Table I scheduler costs (0.5/1.0/1.5) against measured
    // per-factor compute. Written to a sibling file to keep the stage file
    // clean.
    std::string factorPath = path;
    const std::string marker = ".json";
    if(factorPath.size() >= marker.size() &&
       factorPath.compare(factorPath.size() - marker.size(), marker.size(),
                          marker) == 0)
        factorPath.replace(factorPath.size() - marker.size(), marker.size(),
                           "_factors.csv");
    else
        factorPath += "_factors.csv";
    FILE* fc = std::fopen(factorPath.c_str(), "w");
    if(!fc)
        return;
    std::fprintf(fc, "factor,candidate_time_s,candidate_count,selected_count\n");
    std::fprintf(fc, "line,%.9f,%ld,%ld\n", t.candidateLine, t.countLine,
                 t.selectedLine);
    std::fprintf(fc, "plane,%.9f,%ld,%ld\n", t.candidatePlane, t.countPlane,
                 t.selectedPlane);
    std::fprintf(fc, "parallel,%.9f,%ld,%ld\n", t.candidateParallel,
                 t.countParallel, t.selectedParallel);
    std::fprintf(fc, "perpendicular,%.9f,%ld,%ld\n", t.candidatePerpendicular,
                 t.countPerpendicular, t.selectedPerpendicular);
    std::fprintf(fc, "manhattan,%.9f,%ld,%ld\n", t.candidateManhattan,
                 t.countManhattan, t.selectedManhattan);
    std::fclose(fc);
}

} // namespace TIB_SLAM
