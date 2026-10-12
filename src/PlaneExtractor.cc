#include "PlaneExtractor.h"

#include "CameraModels/GeometricCamera.h"
#include "peac/AHCPlaneFitter.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>

namespace TIB_SLAM
{
namespace
{

struct OrganizedCloud
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PointVector = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >;
    PointVector points;
    int columns = 0;
    int rows = 0;

    int width() const { return columns; }
    int height() const { return rows; }
    bool get(int row, int column, double& x, double& y, double& z) const
    {
        if(row < 0 || column < 0 || row >= rows || column >= columns)
            return false;
        const Eigen::Vector3d& point = points[static_cast<std::size_t>(row * columns + column)];
        if(!point.allFinite() || point.z() <= 0.0)
            return false;
        x = point.x(); y = point.y(); z = point.z();
        return true;
    }
};

bool NormalizePlane(Eigen::Vector4f& plane)
{
    if(!plane.allFinite())
        return false;
    const float norm = plane.head<3>().norm();
    if(norm < 1e-6f)
        return false;
    plane /= norm;
    if(plane(3) < 0.0f)
        plane = -plane;
    return plane.allFinite();
}

} // namespace

PlaneExtractor::PlaneExtractor(int stride, int ransacIterations, int minInliers,
                               int maxPlanes, float distanceThreshold,
                               float minDepth, float maxDepth, float voxelLeaf,
                               float minimumInlierRatio)
    : mStride(std::max(1, stride)),
      mRansacIterations(std::max(20, ransacIterations)),
      mMinInliers(std::max(100, minInliers)),
      mMaxPlanes(std::max(1, maxPlanes)),
      mDistanceThreshold(std::max(0.005f, distanceThreshold)),
      mMinDepth(std::max(0.01f, minDepth)),
      mMaxDepth(std::max(mMinDepth, maxDepth)),
      mVoxelLeaf(std::max(0.005f, voxelLeaf)),
      mMinimumInlierRatio(std::max(0.25f, std::min(1.0f, minimumInlierRatio)))
{
}

std::vector<StructuralPlaneObservation> PlaneExtractor::Extract(const cv::Mat& depth,
                                                                 GeometricCamera* camera,
                                                                 unsigned long) const
{
    std::vector<StructuralPlaneObservation> result;
    if(!camera || depth.empty() || depth.type() != CV_32F || depth.rows < 20 || depth.cols < 20)
        return result;

    // Keep the organized segmentation cost approximately resolution
    // invariant.  A fixed stride of two is suitable for VGA RGB-D, but it
    // feeds more than 200k points to AHC on Replica (1200x680) and creates
    // deterministic multi-second spikes.  The configured stride remains the
    // lower bound; higher-resolution inputs are sampled to a bounded cloud.
    constexpr double kMaximumOrganizedSamples = 80000.0;
    const double requiredStride = std::sqrt(
        static_cast<double>(depth.rows) * static_cast<double>(depth.cols) /
        kMaximumOrganizedSamples);
    const int effectiveStride = std::max(
        mStride, static_cast<int>(std::ceil(requiredStride)));

    OrganizedCloud organized;
    organized.columns = (depth.cols + effectiveStride - 1) / effectiveStride;
    organized.rows = (depth.rows + effectiveStride - 1) / effectiveStride;
    // PEAC initializes fixed windows. Crop the sampled dimensions to complete
    // 10x10 windows instead of allowing an out-of-range partial block.
    organized.columns -= organized.columns % 10;
    organized.rows -= organized.rows % 10;
    if(organized.columns < 20 || organized.rows < 20)
        return result;
    const Eigen::Vector3d invalid = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    organized.points.assign(static_cast<std::size_t>(organized.columns * organized.rows), invalid);

    std::size_t validCount = 0;
    for(int row = 0; row < organized.rows; ++row)
    {
        const int y = row * effectiveStride;
        for(int column = 0; column < organized.columns; ++column)
        {
            const int x = column * effectiveStride;
            const float z = depth.at<float>(y, x);
            if(!std::isfinite(z) || z < mMinDepth || z > mMaxDepth)
                continue;
            const Eigen::Vector3f ray = camera->unprojectEig(
                cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            if(!ray.allFinite() || std::abs(ray.z()) < 1e-6f)
                continue;
            const Eigen::Vector3f point = ray * (z / ray.z());
            if(!point.allFinite() || point.z() <= 0.0f)
                continue;
            organized.points[static_cast<std::size_t>(row * organized.columns + column)] =
                point.cast<double>();
            ++validCount;
        }
    }
    if(validCount < static_cast<std::size_t>(mMinInliers))
        return result;

    ahc::PlaneFitter<OrganizedCloud> fitter;
    fitter.minSupport = mMinInliers;
    fitter.windowWidth = 10;
    fitter.windowHeight = 10;
    // Convert the Kinect noise model from millimetres to metres.
    fitter.params.depthSigma = 1.6e-3;
    fitter.params.stdTol_init = 0.005;
    fitter.params.stdTol_merge = 0.008;
    fitter.params.z_near = mMinDepth;
    fitter.params.z_far = mMaxDepth;
    fitter.params.depthChangeTol = 0.02;

    std::vector<std::vector<int> > memberships;
    if(fitter.run(&organized, &memberships, nullptr, nullptr, false) <= 0.0)
        return result;
    const std::size_t planeCount = std::min(memberships.size(), fitter.extractedPlanes.size());

    struct RankedPlane
    {
        StructuralPlaneObservation observation;
        std::size_t support = 0;
    };
    std::vector<RankedPlane> ranked;
    ranked.reserve(planeCount);
    for(std::size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex)
    {
        const std::vector<int>& membership = memberships[planeIndex];
        if(membership.size() < static_cast<std::size_t>(mMinInliers))
            continue;

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > points;
        points.reserve(membership.size());
        for(int index : membership)
        {
            if(index < 0 || static_cast<std::size_t>(index) >= organized.points.size())
                continue;
            const Eigen::Vector3d& point = organized.points[static_cast<std::size_t>(index)];
            if(point.allFinite() && point.z() > 0.0)
                points.push_back(point.cast<float>());
        }
        if(points.size() < static_cast<std::size_t>(mMinInliers))
            continue;

        // AHC has already segmented a coherent plane membership.  The old
        // implementation voxelized every membership and launched another
        // PCL RANSAC, producing multi-second (occasionally 30 s) synchronous
        // tracking spikes.  Fit the segmented members once with deterministic
        // total least squares and retain the same residual-ratio safety gate.
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for(const Eigen::Vector3f& point : points)
            centroid += point.cast<double>();
        centroid /= static_cast<double>(points.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for(const Eigen::Vector3f& point : points)
        {
            const Eigen::Vector3d centered = point.cast<double>() - centroid;
            covariance.noalias() += centered * centered.transpose();
        }
        covariance /= static_cast<double>(points.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if(solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
            continue;
        Eigen::Vector3d normal = solver.eigenvectors().col(0);
        if(!normal.allFinite() || normal.norm() < 1e-9)
            continue;
        normal.normalize();
        double offset = -normal.dot(centroid);
        if(offset < 0.0)
        {
            normal = -normal;
            offset = -offset;
        }
        std::size_t inliers = 0;
        for(const Eigen::Vector3f& point : points)
            if(std::abs(normal.dot(point.cast<double>()) + offset) <=
               static_cast<double>(mDistanceThreshold))
                ++inliers;
        const float inlierRatio = static_cast<float>(inliers) /
                                  static_cast<float>(points.size());
        if(!std::isfinite(inlierRatio) || inlierRatio < mMinimumInlierRatio)
            continue;

        StructuralPlaneObservation observation;
        observation.cameraPlane << static_cast<float>(normal.x()),
                                   static_cast<float>(normal.y()),
                                   static_cast<float>(normal.z()),
                                   static_cast<float>(offset);
        if(!NormalizePlane(observation.cameraPlane))
            continue;
        observation.imageWidth = organized.columns;
        observation.imageHeight = organized.rows;
        observation.pixelIndices = membership;
        observation.inlierRatio = inlierRatio;
        observation.supportRatio = static_cast<float>(points.size()) /
                                   static_cast<float>(validCount);

        const std::size_t sampleStep = std::max<std::size_t>(1, points.size() / 500);
        observation.supportCamera.reserve(std::min<std::size_t>(500, points.size()));
        for(std::size_t i = 0; i < points.size(); i += sampleStep)
            observation.supportCamera.push_back(points[i]);
        if(observation.supportCamera.size() < 3)
            continue;
        for(const Eigen::Vector3f& point : observation.supportCamera)
            observation.centroidCamera += point;
        observation.centroidCamera /= static_cast<float>(observation.supportCamera.size());
        for(const Eigen::Vector3f& point : observation.supportCamera)
        {
            const Eigen::Vector3f centered = point - observation.centroidCamera;
            observation.covarianceCamera.noalias() += centered * centered.transpose();
        }
        observation.covarianceCamera /= static_cast<float>(observation.supportCamera.size());
        if(!observation.centroidCamera.allFinite() || !observation.covarianceCamera.allFinite())
            continue;
        ranked.push_back({std::move(observation), points.size()});
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedPlane& lhs, const RankedPlane& rhs) {
        return lhs.support > rhs.support;
    });
    if(ranked.size() > static_cast<std::size_t>(mMaxPlanes))
        ranked.resize(static_cast<std::size_t>(mMaxPlanes));
    result.reserve(ranked.size());
    for(RankedPlane& plane : ranked)
        result.push_back(std::move(plane.observation));
    return result;
}

} // namespace TIB_SLAM
