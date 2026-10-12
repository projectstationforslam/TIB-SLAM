#include "StructuralG2oTypes.h"

#include <cmath>

namespace TIB_SLAM
{
namespace
{

Eigen::Matrix3d Skew(const Eigen::Vector3d& vector)
{
    Eigen::Matrix3d result;
    result << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
             -vector.y(), vector.x(), 0.0;
    return result;
}

const g2o::VertexSE3Expmap* PoseVertex(const g2o::HyperGraph::Vertex* vertex)
{
    return vertex ? dynamic_cast<const g2o::VertexSE3Expmap*>(vertex) : nullptr;
}

g2o::VertexSE3Expmap* MutablePoseVertex(g2o::HyperGraph::Vertex* vertex)
{
    return vertex ? dynamic_cast<g2o::VertexSE3Expmap*>(vertex) : nullptr;
}

const VertexTranslation* TranslationVertex(const g2o::HyperGraph::Vertex* vertex)
{
    return vertex ? dynamic_cast<const VertexTranslation*>(vertex) : nullptr;
}

Eigen::Vector3d SafeUnitVector(const Eigen::Vector3d& vector)
{
    if(!vector.allFinite() || vector.norm() <= 1e-12)
        return Eigen::Vector3d::Zero();
    return vector.normalized();
}

bool TangentBasis(const Eigen::Vector3d& normalInput, Eigen::Matrix<double,3,2>& tangent)
{
    const Eigen::Vector3d normal = SafeUnitVector(normalInput);
    if(normal.squaredNorm() < 0.5)
        return false;
    Eigen::Vector3d seed = std::abs(normal.z()) < 0.9 ? Eigen::Vector3d::UnitZ()
                                                      : Eigen::Vector3d::UnitX();
    tangent.col(0) = (seed - normal * normal.dot(seed)).normalized();
    tangent.col(1) = normal.cross(tangent.col(0)).normalized();
    return tangent.allFinite();
}

} // namespace

void EdgeSE3MotionPrior::computeError()
{
    _error.setZero();
    const g2o::VertexSE3Expmap* vertex =
        PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex)
        return;
    _error = (vertex->estimate() * _measurement.inverse()).log();
    if(!_error.allFinite())
        _error.setZero();
}

void EdgeSE3MotionPrior::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    g2o::VertexSE3Expmap* vertex =
        MutablePoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex)
        return;

    const g2o::SE3Quat original = vertex->estimate();
    constexpr double epsilon = 1e-7;
    for(int column = 0; column < 6; ++column)
    {
        g2o::Vector6d increment = g2o::Vector6d::Zero();
        increment(column) = epsilon;
        vertex->setEstimate(g2o::SE3Quat::exp(increment) * original);
        computeError();
        const g2o::Vector6d positive = _error;
        vertex->setEstimate(g2o::SE3Quat::exp(-increment) * original);
        computeError();
        const g2o::Vector6d negative = _error;
        _jacobianOplusXi.col(column) =
            (positive - negative) / (2.0 * epsilon);
    }
    vertex->setEstimate(original);
    computeError();
}

Eigen::Matrix<double, 9, 1> FlattenRotation(const Eigen::Matrix3d& rotation)
{
    Eigen::Matrix<double, 9, 1> flattened;
    int index = 0;
    for(int row = 0; row < 3; ++row)
        for(int column = 0; column < 3; ++column)
            flattened(index++) = rotation(row, column);
    return flattened;
}

void VertexTranslation::oplusImpl(const double* update)
{
    if(!update)
        return;
    const Eigen::Map<const Eigen::Vector3d> increment(update);
    if(increment.allFinite())
        _estimate += increment;
}

EdgePointTranslation::EdgePointTranslation(const Eigen::Vector3d& worldPoint,
                                           const Eigen::Matrix3d& fixedRcw,
                                           GeometricCamera* camera)
    : mWorldPoint(worldPoint), mRcw(fixedRcw), mpCamera(camera)
{
}

bool EdgePointTranslation::CameraPoint(Eigen::Vector3d& point) const
{
    const VertexTranslation* vertex = TranslationVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera || !mWorldPoint.allFinite() || !mRcw.allFinite())
        return false;
    point = mRcw * mWorldPoint + vertex->estimate();
    return point.allFinite() && point.z() > 1e-6;
}

void EdgePointTranslation::computeError()
{
    mValid = false;
    _error.setZero();
    Eigen::Vector3d point;
    if(!CameraPoint(point) || !_measurement.allFinite())
        return;
    const Eigen::Vector2d projection = mpCamera->project(point);
    if(!projection.allFinite())
        return;
    _error = projection - _measurement;
    mValid = _error.allFinite();
    if(!mValid) _error.setZero();
}

void EdgePointTranslation::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector3d point;
    if(CameraPoint(point))
    {
        const Eigen::Matrix<double,2,3> jacobian = mpCamera->projectJac(point);
        if(jacobian.allFinite()) _jacobianOplusXi = jacobian;
    }
}

EdgeStereoPointTranslation::EdgeStereoPointTranslation(const Eigen::Vector3d& worldPoint,
                                                       const Eigen::Matrix3d& fixedRcw,
                                                       GeometricCamera* camera, double bf)
    : mWorldPoint(worldPoint), mRcw(fixedRcw), mpCamera(camera), mBf(bf)
{
}

bool EdgeStereoPointTranslation::CameraPoint(Eigen::Vector3d& point) const
{
    const VertexTranslation* vertex = TranslationVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera || !mWorldPoint.allFinite() || !mRcw.allFinite() || !std::isfinite(mBf))
        return false;
    point = mRcw * mWorldPoint + vertex->estimate();
    return point.allFinite() && point.z() > 1e-6;
}

void EdgeStereoPointTranslation::computeError()
{
    mValid = false;
    _error.setZero();
    Eigen::Vector3d point;
    if(!CameraPoint(point) || !_measurement.allFinite())
        return;
    const Eigen::Vector2d projection = mpCamera->project(point);
    if(!projection.allFinite())
        return;
    Eigen::Vector3d prediction;
    prediction << projection.x(), projection.y(), projection.x() - mBf / point.z();
    _error = prediction - _measurement;
    mValid = _error.allFinite();
    if(!mValid) _error.setZero();
}

void EdgeStereoPointTranslation::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector3d point;
    if(!CameraPoint(point))
        return;
    const Eigen::Matrix<double,2,3> jacobian = mpCamera->projectJac(point);
    if(!jacobian.allFinite())
        return;
    _jacobianOplusXi.block<2,3>(0,0) = jacobian;
    _jacobianOplusXi.row(2) = jacobian.row(0);
    _jacobianOplusXi(2,2) += mBf / (point.z() * point.z());
}

EdgeLineTranslation::EdgeLineTranslation(const Eigen::Vector3d& worldPoint,
                                         const Eigen::Vector3d& imageLine,
                                         const Eigen::Matrix3d& fixedRcw,
                                         GeometricCamera* camera)
    : mWorldPoint(worldPoint), mImageLine(imageLine), mRcw(fixedRcw), mpCamera(camera)
{
}

bool EdgeLineTranslation::CameraPoint(Eigen::Vector3d& point) const
{
    const VertexTranslation* vertex = TranslationVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera || !mWorldPoint.allFinite() || !mImageLine.allFinite() || !mRcw.allFinite())
        return false;
    point = mRcw * mWorldPoint + vertex->estimate();
    return point.allFinite() && point.z() > 1e-6;
}

void EdgeLineTranslation::computeError()
{
    mValid = false;
    _error.setZero();
    Eigen::Vector3d point;
    if(!CameraPoint(point))
        return;
    const Eigen::Vector2d projection = mpCamera->project(point);
    if(!projection.allFinite())
        return;
    _error(0) = mImageLine.x() * projection.x() + mImageLine.y() * projection.y() + mImageLine.z();
    mValid = std::isfinite(_error(0));
    if(!mValid) _error.setZero();
}

void EdgeLineTranslation::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector3d point;
    if(!CameraPoint(point))
        return;
    const Eigen::Matrix<double,2,3> jacobian = mpCamera->projectJac(point);
    if(jacobian.allFinite())
        _jacobianOplusXi = mImageLine.head<2>().transpose() * jacobian;
}

EdgePlaneDistanceTranslation::EdgePlaneDistanceTranslation(
    const Eigen::Vector4d& worldPlane, const Eigen::Vector3d& observedNormal,
    const Eigen::Matrix3d& fixedRcw)
    : mWorldPlane(worldPlane), mObservedNormal(SafeUnitVector(observedNormal)), mRcw(fixedRcw)
{
}

bool EdgePlaneDistanceTranslation::Parameters(Eigen::Vector3d& normal, double& sign,
                                               const VertexTranslation*& vertex) const
{
    vertex = TranslationVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mWorldPlane.allFinite() || !mRcw.allFinite() ||
       mObservedNormal.squaredNorm() < 0.5)
        return false;
    const double norm = mWorldPlane.head<3>().norm();
    if(norm < 1e-9)
        return false;
    normal = mRcw * (mWorldPlane.head<3>() / norm);
    sign = normal.dot(mObservedNormal) < 0.0 ? -1.0 : 1.0;
    return normal.allFinite();
}

void EdgePlaneDistanceTranslation::computeError()
{
    _error.setZero();
    Eigen::Vector3d normal;
    double sign = 1.0;
    const VertexTranslation* vertex = nullptr;
    if(!Parameters(normal, sign, vertex) || !std::isfinite(_measurement))
        return;
    const double norm = mWorldPlane.head<3>().norm();
    const double predictedDistance = mWorldPlane(3) / norm - normal.dot(vertex->estimate());
    _error(0) = sign * predictedDistance - _measurement;
    if(!std::isfinite(_error(0))) _error.setZero();
}

void EdgePlaneDistanceTranslation::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector3d normal;
    double sign = 1.0;
    const VertexTranslation* vertex = nullptr;
    if(Parameters(normal, sign, vertex))
        _jacobianOplusXi = -sign * normal.transpose();
}

EdgeLineProjectXYZOnlyPose::EdgeLineProjectXYZOnlyPose(const Eigen::Vector3d& worldPoint,
                                                       const Eigen::Vector3d& imageLine,
                                                       GeometricCamera* camera)
    : mWorldPoint(worldPoint), mImageLine(imageLine), mpCamera(camera)
{
}

void EdgeLineProjectXYZOnlyPose::computeError()
{
    mValid = false;
    _error.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera || !mWorldPoint.allFinite() || !mImageLine.allFinite())
        return;
    const Eigen::Vector3d cameraPoint = vertex->estimate().map(mWorldPoint);
    if(!cameraPoint.allFinite() || cameraPoint.z() <= 1e-6)
        return;
    const Eigen::Vector2d projected = mpCamera->project(cameraPoint);
    if(!projected.allFinite())
        return;
    _error(0) = mImageLine.x() * projected.x() + mImageLine.y() * projected.y() + mImageLine.z();
    mValid = std::isfinite(_error(0));
    if(!mValid) _error.setZero();
}

void EdgeLineProjectXYZOnlyPose::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera)
        return;
    const Eigen::Vector3d cameraPoint = vertex->estimate().map(mWorldPoint);
    if(!cameraPoint.allFinite() || cameraPoint.z() <= 1e-6)
        return;
    const Eigen::Matrix<double,2,3> projectionJacobian = mpCamera->projectJac(cameraPoint);
    if(!projectionJacobian.allFinite())
        return;
    Eigen::Matrix<double,3,6> se3Jacobian;
    se3Jacobian.block<3,3>(0,0) = -Skew(cameraPoint);
    se3Jacobian.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    _jacobianOplusXi = mImageLine.head<2>().transpose() * projectionJacobian * se3Jacobian;
}

Eigen::Matrix<double,6,6> EdgeLineProjectXYZOnlyPose::PoseInformation()
{
    Eigen::Matrix<double,1,6> jacobian = Eigen::Matrix<double,1,6>::Zero();
    const g2o::VertexSE3Expmap* vertex =
        PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !mpCamera)
        return Eigen::Matrix<double,6,6>::Zero();
    const Eigen::Vector3d cameraPoint = vertex->estimate().map(mWorldPoint);
    if(!cameraPoint.allFinite() || cameraPoint.z() <= 1e-6)
        return Eigen::Matrix<double,6,6>::Zero();
    const Eigen::Matrix<double,2,3> projectionJacobian =
        mpCamera->projectJac(cameraPoint);
    if(!projectionJacobian.allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    Eigen::Matrix<double,3,6> se3Jacobian;
    se3Jacobian.block<3,3>(0,0) = -Skew(cameraPoint);
    se3Jacobian.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    jacobian = mImageLine.head<2>().transpose() * projectionJacobian * se3Jacobian;
    if(!jacobian.allFinite() || !information().allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    return jacobian.transpose() * information() * jacobian;
}

EdgePlaneOnlyPose::EdgePlaneOnlyPose(const Eigen::Vector4d& worldPlane)
    : mWorldPlane(worldPlane)
{
}

bool EdgePlaneOnlyPose::PredictedPlane(Eigen::Vector4d& plane, double& sign,
                                       Eigen::Matrix<double,3,2>& tangent) const
{
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    const Eigen::Vector3d observedNormal = SafeUnitVector(_measurement.head<3>());
    if(!vertex || !mWorldPlane.allFinite() || observedNormal.squaredNorm() < 0.5 ||
       !TangentBasis(observedNormal, tangent))
        return false;
    const double worldNorm = mWorldPlane.head<3>().norm();
    if(worldNorm < 1e-9)
        return false;
    const g2o::SE3Quat pose = vertex->estimate();
    plane.head<3>() = pose.rotation().toRotationMatrix() * (mWorldPlane.head<3>() / worldNorm);
    plane(3) = mWorldPlane(3) / worldNorm - plane.head<3>().dot(pose.translation());
    sign = plane.head<3>().dot(observedNormal) < 0.0 ? -1.0 : 1.0;
    plane *= sign;
    return plane.allFinite();
}

void EdgePlaneOnlyPose::computeError()
{
    _error.setZero();
    Eigen::Vector4d predicted;
    Eigen::Matrix<double,3,2> tangent;
    double sign = 1.0;
    if(!PredictedPlane(predicted, sign, tangent))
        return;
    const Eigen::Vector3d observedNormal = SafeUnitVector(_measurement.head<3>());
    _error.head<2>() = tangent.transpose() * (predicted.head<3>() - observedNormal);
    _error(2) = predicted(3) - _measurement(3) / _measurement.head<3>().norm();
    if(!_error.allFinite()) _error.setZero();
}

void EdgePlaneOnlyPose::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector4d predicted;
    Eigen::Matrix<double,3,2> tangent;
    double sign = 1.0;
    if(!PredictedPlane(predicted, sign, tangent))
        return;
    _jacobianOplusXi.block<2,3>(0,0) = tangent.transpose() * (-Skew(predicted.head<3>()));
    _jacobianOplusXi.block<1,3>(2,3) = -predicted.head<3>().transpose();
}

Eigen::Matrix<double,6,6> EdgePlaneOnlyPose::PoseInformation()
{
    Eigen::Matrix<double,3,6> jacobian = Eigen::Matrix<double,3,6>::Zero();
    Eigen::Vector4d predicted;
    Eigen::Matrix<double,3,2> tangent;
    double sign = 1.0;
    if(!PredictedPlane(predicted, sign, tangent))
        return Eigen::Matrix<double,6,6>::Zero();
    jacobian.block<2,3>(0,0) = tangent.transpose() * (-Skew(predicted.head<3>()));
    jacobian.block<1,3>(2,3) = -predicted.head<3>().transpose();
    if(!jacobian.allFinite() || !information().allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    return jacobian.transpose() * information() * jacobian;
}

EdgeParallelPlaneOnlyPose::EdgeParallelPlaneOnlyPose(const Eigen::Vector3d& worldNormal)
    : mWorldNormal(SafeUnitVector(worldNormal))
{
}

bool EdgeParallelPlaneOnlyPose::Parameters(Eigen::Vector3d& predicted,
                                           Eigen::Matrix<double,3,2>& tangent) const
{
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    const Eigen::Vector3d observed = SafeUnitVector(_measurement);
    if(!vertex || mWorldNormal.squaredNorm() < 0.5 || observed.squaredNorm() < 0.5 ||
       !TangentBasis(observed, tangent))
        return false;
    predicted = vertex->estimate().rotation().toRotationMatrix() * mWorldNormal;
    if(predicted.dot(observed) < 0.0)
        predicted = -predicted;
    return predicted.allFinite();
}

void EdgeParallelPlaneOnlyPose::computeError()
{
    _error.setZero();
    Eigen::Vector3d predicted;
    Eigen::Matrix<double,3,2> tangent;
    if(Parameters(predicted, tangent))
        _error = tangent.transpose() * predicted;
}

void EdgeParallelPlaneOnlyPose::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    Eigen::Vector3d predicted;
    Eigen::Matrix<double,3,2> tangent;
    if(Parameters(predicted, tangent))
        _jacobianOplusXi.block<2,3>(0,0) = tangent.transpose() * (-Skew(predicted));
}

Eigen::Matrix<double,6,6> EdgeParallelPlaneOnlyPose::PoseInformation()
{
    Eigen::Matrix<double,2,6> jacobian = Eigen::Matrix<double,2,6>::Zero();
    Eigen::Vector3d predicted;
    Eigen::Matrix<double,3,2> tangent;
    if(!Parameters(predicted, tangent))
        return Eigen::Matrix<double,6,6>::Zero();
    jacobian.block<2,3>(0,0) = tangent.transpose() * (-Skew(predicted));
    if(!jacobian.allFinite() || !information().allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    return jacobian.transpose() * information() * jacobian;
}

EdgePerpendicularPlaneOnlyPose::EdgePerpendicularPlaneOnlyPose(
    const Eigen::Vector3d& worldNormal, const Eigen::Vector3d& observedNormal)
    : mWorldNormal(SafeUnitVector(worldNormal)), mObservedNormal(SafeUnitVector(observedNormal))
{
}

void EdgePerpendicularPlaneOnlyPose::computeError()
{
    _error.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || mWorldNormal.squaredNorm() < 0.5 || mObservedNormal.squaredNorm() < 0.5)
        return;
    const Eigen::Vector3d predicted = vertex->estimate().rotation().toRotationMatrix() * mWorldNormal;
    _error(0) = predicted.dot(mObservedNormal);
}

void EdgePerpendicularPlaneOnlyPose::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || mWorldNormal.squaredNorm() < 0.5 || mObservedNormal.squaredNorm() < 0.5)
        return;
    const Eigen::Vector3d predicted = vertex->estimate().rotation().toRotationMatrix() * mWorldNormal;
    _jacobianOplusXi.block<1,3>(0,0) = -mObservedNormal.transpose() * Skew(predicted);
}

Eigen::Matrix<double,6,6> EdgePerpendicularPlaneOnlyPose::PoseInformation()
{
    Eigen::Matrix<double,1,6> jacobian = Eigen::Matrix<double,1,6>::Zero();
    const g2o::VertexSE3Expmap* vertex =
        PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || mWorldNormal.squaredNorm() < 0.5 ||
       mObservedNormal.squaredNorm() < 0.5)
        return Eigen::Matrix<double,6,6>::Zero();
    const Eigen::Vector3d predicted =
        vertex->estimate().rotation().toRotationMatrix() * mWorldNormal;
    jacobian.block<1,3>(0,0) = -mObservedNormal.transpose() * Skew(predicted);
    if(!jacobian.allFinite() || !information().allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    return jacobian.transpose() * information() * jacobian;
}

void EdgeManhattanRotationOnlyPose::computeError()
{
    _error.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(vertex && _measurement.allFinite())
        _error = FlattenRotation(vertex->estimate().rotation().toRotationMatrix()) - _measurement;
}

void EdgeManhattanRotationOnlyPose::linearizeOplus()
{
    _jacobianOplusXi.setZero();
    const g2o::VertexSE3Expmap* vertex = PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex || !_measurement.allFinite())
        return;
    const Eigen::Matrix3d rotation = vertex->estimate().rotation().toRotationMatrix();
    for(int axis = 0; axis < 3; ++axis)
    {
        Eigen::Vector3d basis = Eigen::Vector3d::Zero();
        basis(axis) = 1.0;
        _jacobianOplusXi.col(axis) = FlattenRotation(Skew(basis) * rotation);
    }
}

Eigen::Matrix<double,6,6> EdgeManhattanRotationOnlyPose::PoseInformation()
{
    Eigen::Matrix<double,9,6> jacobian = Eigen::Matrix<double,9,6>::Zero();
    const g2o::VertexSE3Expmap* vertex =
        PoseVertex(_vertices.empty() ? nullptr : _vertices[0]);
    if(!vertex)
        return Eigen::Matrix<double,6,6>::Zero();
    const Eigen::Matrix3d rotation = vertex->estimate().rotation().toRotationMatrix();
    for(int axis = 0; axis < 3; ++axis)
    {
        Eigen::Vector3d basis = Eigen::Vector3d::Zero();
        basis(axis) = 1.0;
        jacobian.col(axis) = FlattenRotation(Skew(basis) * rotation);
    }
    if(!jacobian.allFinite() || !information().allFinite())
        return Eigen::Matrix<double,6,6>::Zero();
    return jacobian.transpose() * information() * jacobian;
}

} // namespace TIB_SLAM
