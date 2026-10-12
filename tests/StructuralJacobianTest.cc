#include "CameraModels/GeometricCamera.h"
#include "StructuralG2oTypes.h"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace
{

class TestPinhole : public TIB_SLAM::GeometricCamera
{
public:
    TestPinhole(double fx, double fy, double cx, double cy)
        : GeometricCamera(std::vector<float>{static_cast<float>(fx), static_cast<float>(fy),
                                             static_cast<float>(cx), static_cast<float>(cy)}),
          mFx(fx), mFy(fy), mCx(cx), mCy(cy) {}

    cv::Point2f project(const cv::Point3f& p) override
    { return cv::Point2f(static_cast<float>(mFx * p.x / p.z + mCx),
                         static_cast<float>(mFy * p.y / p.z + mCy)); }
    Eigen::Vector2d project(const Eigen::Vector3d& p) override
    { return Eigen::Vector2d(mFx * p.x() / p.z() + mCx, mFy * p.y() / p.z() + mCy); }
    Eigen::Vector2f project(const Eigen::Vector3f& p) override
    {
        const Eigen::Vector3d point = p.cast<double>();
        const Eigen::Vector2d projected = project(point);
        return projected.cast<float>();
    }
    Eigen::Vector2f projectMat(const cv::Point3f& p) override
    {
        const cv::Point2f projected = project(p);
        return Eigen::Vector2f(projected.x, projected.y);
    }
    float uncertainty2(const Eigen::Matrix<double,2,1>&) override { return 1.0f; }
    Eigen::Vector3f unprojectEig(const cv::Point2f& p) override
    { return Eigen::Vector3f((p.x - mCx) / mFx, (p.y - mCy) / mFy, 1.0f); }
    cv::Point3f unproject(const cv::Point2f& p) override
    { const Eigen::Vector3f ray = unprojectEig(p); return cv::Point3f(ray.x(), ray.y(), ray.z()); }
    Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& p) override
    {
        Eigen::Matrix<double,2,3> jacobian;
        const double inverseZ = 1.0 / p.z();
        jacobian << mFx * inverseZ, 0.0, -mFx * p.x() * inverseZ * inverseZ,
                    0.0, mFy * inverseZ, -mFy * p.y() * inverseZ * inverseZ;
        return jacobian;
    }
    bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>&, const std::vector<cv::KeyPoint>&,
                                 const std::vector<int>&, Sophus::SE3f&,
                                 std::vector<cv::Point3f>&, std::vector<bool>&) override { return false; }
    cv::Mat toK() override
    { return (cv::Mat_<float>(3,3) << mFx, 0.0, mCx, 0.0, mFy, mCy, 0.0, 0.0, 1.0); }
    Eigen::Matrix3f toK_() override
    {
        Eigen::Matrix3f K;
        K << mFx, 0.0, mCx, 0.0, mFy, mCy, 0.0, 0.0, 1.0;
        return K;
    }
    bool epipolarConstrain(TIB_SLAM::GeometricCamera*, const cv::KeyPoint&, const cv::KeyPoint&,
                           const Eigen::Matrix3f&, const Eigen::Vector3f&, float, float) override
    { return false; }
    bool matchAndtriangulate(const cv::KeyPoint&, const cv::KeyPoint&, TIB_SLAM::GeometricCamera*,
                             Sophus::SE3f&, Sophus::SE3f&, float, float, Eigen::Vector3f&) override
    { return false; }

private:
    double mFx, mFy, mCx, mCy;
};

template<typename EdgeType, int ErrorDimension>
class TestableEdge : public EdgeType
{
public:
    using EdgeType::EdgeType;

    void LinearizeForTest(Eigen::Matrix<double, ErrorDimension, 6>& storage)
    {
        using JacobianMap = typename EdgeType::JacobianXiOplusType;
        new (&this->_jacobianOplusXi)
            JacobianMap(storage.data(), ErrorDimension, 6);
        this->linearizeOplus();
    }
};

template<int ErrorDimension, typename EdgeType>
double CheckJacobian(const std::string& name, EdgeType& edge,
                     g2o::VertexSE3Expmap& vertex, double epsilon = 1e-7)
{
    edge.setVertex(0, &vertex);
    edge.computeError();
    Eigen::Matrix<double, ErrorDimension, 6> jacobianStorage;
    edge.LinearizeForTest(jacobianStorage);
    const Eigen::Matrix<double, ErrorDimension, 6> analytic = edge.jacobianOplusXi();
    Eigen::Matrix<double, ErrorDimension, 6> numeric;
    const g2o::SE3Quat original = vertex.estimate();

    for(int column = 0; column < 6; ++column)
    {
        g2o::Vector6d update = g2o::Vector6d::Zero();
        update(column) = epsilon;
        vertex.setEstimate(g2o::SE3Quat::exp(update) * original);
        edge.computeError();
        const Eigen::Matrix<double, ErrorDimension, 1> plus = edge.error();

        update(column) = -epsilon;
        vertex.setEstimate(g2o::SE3Quat::exp(update) * original);
        edge.computeError();
        const Eigen::Matrix<double, ErrorDimension, 1> minus = edge.error();
        numeric.col(column) = (plus - minus) / (2.0 * epsilon);
    }

    vertex.setEstimate(original);
    edge.computeError();
    const double maximumError = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << name << " max|J_analytic-J_numeric| = " << maximumError << std::endl;
    return maximumError;
}

template<int ErrorDimension, typename EdgeType>
bool CheckInvalidInputSafety(const std::string& name, EdgeType& edge,
                             g2o::VertexSE3Expmap& vertex)
{
    edge.setVertex(0, &vertex);
    edge.computeError();
    Eigen::Matrix<double, ErrorDimension, 6> jacobianStorage;
    edge.LinearizeForTest(jacobianStorage);
    const bool safe = edge.error().allFinite() && edge.jacobianOplusXi().allFinite() &&
                      edge.error().isZero(0.0) && edge.jacobianOplusXi().isZero(0.0);
    std::cout << name << " invalid-input guard = " << (safe ? "passed" : "failed") << std::endl;
    return safe;
}

template<typename EdgeType, int ErrorDimension>
class TestableTranslationEdge : public EdgeType
{
public:
    using EdgeType::EdgeType;
    void LinearizeForTest(Eigen::Matrix<double, ErrorDimension, 3>& storage)
    {
        using JacobianMap = typename EdgeType::JacobianXiOplusType;
        new (&this->_jacobianOplusXi) JacobianMap(storage.data(), ErrorDimension, 3);
        this->linearizeOplus();
    }
};

template<int ErrorDimension, typename EdgeType>
double CheckTranslationJacobian(const std::string& name, EdgeType& edge,
                                TIB_SLAM::VertexTranslation& vertex,
                                double epsilon = 1e-7)
{
    edge.setVertex(0, &vertex);
    edge.computeError();
    Eigen::Matrix<double, ErrorDimension, 3> storage;
    edge.LinearizeForTest(storage);
    const Eigen::Matrix<double, ErrorDimension, 3> analytic = edge.jacobianOplusXi();
    Eigen::Matrix<double, ErrorDimension, 3> numeric;
    const Eigen::Vector3d original = vertex.estimate();
    for(int column = 0; column < 3; ++column)
    {
        Eigen::Vector3d update = Eigen::Vector3d::Zero();
        update(column) = epsilon;
        vertex.setEstimate(original + update);
        edge.computeError();
        const Eigen::Matrix<double, ErrorDimension, 1> plus = edge.error();
        vertex.setEstimate(original - update);
        edge.computeError();
        const Eigen::Matrix<double, ErrorDimension, 1> minus = edge.error();
        numeric.col(column) = (plus - minus) / (2.0 * epsilon);
    }
    vertex.setEstimate(original);
    edge.computeError();
    const double maximumError = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << name << " max|J_analytic-J_numeric| = " << maximumError << std::endl;
    return maximumError;
}

Eigen::Vector4d TransformPlane(const Eigen::Vector4d& worldPlane,
                               const g2o::SE3Quat& pose)
{
    Eigen::Vector4d cameraPlane;
    cameraPlane.head<3>() = pose.rotation().toRotationMatrix() * worldPlane.head<3>();
    cameraPlane(3) = worldPlane(3) - cameraPlane.head<3>().dot(pose.translation());
    cameraPlane /= cameraPlane.head<3>().norm();
    return cameraPlane;
}

} // namespace

int main()
{
    using namespace TIB_SLAM;

    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(0.17, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-0.11, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const g2o::SE3Quat pose(Eigen::Quaterniond(rotation), Eigen::Vector3d(0.2, -0.1, 0.3));
    g2o::VertexSE3Expmap vertex;
    vertex.setEstimate(pose);

    TestPinhole camera(520.0, 515.0, 320.0, 240.0);
    double worst = 0.0;

    const Eigen::Vector3d worldPoint(0.4, -0.25, 3.2);
    const Eigen::Vector3d cameraPoint = pose.map(worldPoint);
    const Eigen::Vector2d pixel = camera.project(cameraPoint);
    Eigen::Vector3d imageLine(0.6, -0.8, 0.0);
    imageLine(2) = -imageLine.head<2>().dot(pixel + Eigen::Vector2d(1.5, -0.7));
    TestableEdge<EdgeLineProjectXYZOnlyPose, 1> lineEdge(worldPoint, imageLine, &camera);
    lineEdge.setMeasurement(0.0);
    worst = std::max(worst, CheckJacobian<1>("line", lineEdge, vertex));

    Eigen::Vector4d worldPlane;
    worldPlane << 0.3, -0.4, 0.866025403784, -1.2;
    worldPlane /= worldPlane.head<3>().norm();
    TestableEdge<EdgePlaneOnlyPose, 3> planeEdge(worldPlane);
    planeEdge.setMeasurement(TransformPlane(worldPlane, pose));
    worst = std::max(worst, CheckJacobian<3>("plane", planeEdge, vertex));

    const Eigen::Vector3d parallelWorld = Eigen::Vector3d(0.2, 0.7, 0.68).normalized();
    const Eigen::Vector3d parallelObserved =
        (rotation * parallelWorld + Eigen::Vector3d(0.01, -0.02, 0.015)).normalized();
    TestableEdge<EdgeParallelPlaneOnlyPose, 2> parallelEdge(parallelWorld);
    parallelEdge.setMeasurement(parallelObserved);
    worst = std::max(worst, CheckJacobian<2>("parallel", parallelEdge, vertex));

    const Eigen::Vector3d perpendicularWorld = Eigen::Vector3d(-0.5, 0.2, 0.84).normalized();
    const Eigen::Vector3d perpendicularObserved = Eigen::Vector3d(0.7, 0.65, -0.2).normalized();
    TestableEdge<EdgePerpendicularPlaneOnlyPose, 1> perpendicularEdge(
        perpendicularWorld, perpendicularObserved);
    perpendicularEdge.setMeasurement(0.0);
    worst = std::max(worst, CheckJacobian<1>("perpendicular", perpendicularEdge, vertex));

    TestableEdge<EdgeManhattanRotationOnlyPose, 9> manhattanEdge;
    manhattanEdge.setMeasurement(FlattenRotation(Eigen::Matrix3d::Identity()));
    worst = std::max(worst, CheckJacobian<9>("manhattan", manhattanEdge, vertex));

    const g2o::SE3Quat priorMeasurement(
        Eigen::Quaterniond(Eigen::AngleAxisd(
            0.08, Eigen::Vector3d(0.2, 0.9, -0.1).normalized())),
        Eigen::Vector3d(0.16, -0.08, 0.27));
    TestableEdge<EdgeSE3MotionPrior, 6> motionPrior;
    motionPrior.setMeasurement(priorMeasurement);
    worst = std::max(worst, CheckJacobian<6>("motion-prior", motionPrior, vertex));

    VertexTranslation translationVertex;
    translationVertex.setEstimate(pose.translation());
    TestableTranslationEdge<EdgePointTranslation, 2> pointTranslation(
        worldPoint, rotation, &camera);
    pointTranslation.setMeasurement(pixel + Eigen::Vector2d(0.3, -0.2));
    worst = std::max(worst, CheckTranslationJacobian<2>(
        "translation-point", pointTranslation, translationVertex));

    TestableTranslationEdge<EdgeStereoPointTranslation, 3> stereoTranslation(
        worldPoint, rotation, &camera, 40.0);
    stereoTranslation.setMeasurement(Eigen::Vector3d(pixel.x(), pixel.y(),
        pixel.x() - 40.0 / cameraPoint.z()));
    worst = std::max(worst, CheckTranslationJacobian<3>(
        "translation-stereo", stereoTranslation, translationVertex));

    TestableTranslationEdge<EdgeLineTranslation, 1> lineTranslation(
        worldPoint, imageLine, rotation, &camera);
    lineTranslation.setMeasurement(0.0);
    worst = std::max(worst, CheckTranslationJacobian<1>(
        "translation-line", lineTranslation, translationVertex));

    const Eigen::Vector4d observedPlane = TransformPlane(worldPlane, pose);
    TestableTranslationEdge<EdgePlaneDistanceTranslation, 1> planeTranslation(
        worldPlane, observedPlane.head<3>(), rotation);
    planeTranslation.setMeasurement(observedPlane(3));
    worst = std::max(worst, CheckTranslationJacobian<1>(
        "translation-plane", planeTranslation, translationVertex));

    TestableEdge<EdgeParallelPlaneOnlyPose, 2> invalidParallel(Eigen::Vector3d::Zero());
    invalidParallel.setMeasurement(Eigen::Vector3d::UnitX());
    TestableEdge<EdgePerpendicularPlaneOnlyPose, 1> invalidPerpendicular(
        Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY());
    invalidPerpendicular.setMeasurement(0.0);
    TestableEdge<EdgeManhattanRotationOnlyPose, 9> invalidManhattan;
    Eigen::Matrix<double, 9, 1> invalidRotation = Eigen::Matrix<double, 9, 1>::Zero();
    invalidRotation(0) = std::numeric_limits<double>::quiet_NaN();
    invalidManhattan.setMeasurement(invalidRotation);
    const bool invalidInputsSafe =
        CheckInvalidInputSafety<2>("parallel", invalidParallel, vertex) &&
        CheckInvalidInputSafety<1>("perpendicular", invalidPerpendicular, vertex) &&
        CheckInvalidInputSafety<9>("manhattan", invalidManhattan, vertex);

    if(!std::isfinite(worst) || worst > 2e-5 || !invalidInputsSafe)
    {
        std::cerr << "Structural Jacobian test failed; worst error = " << worst << std::endl;
        return 1;
    }
    std::cout << "All structural Jacobians passed." << std::endl;
    return 0;
}
