#ifndef TIB_SLAM_STRUCTURAL_G2O_TYPES_H
#define TIB_SLAM_STRUCTURAL_G2O_TYPES_H

#include "CameraModels/GeometricCamera.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <Eigen/Core>

namespace TIB_SLAM
{

class VertexTranslation : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexTranslation() { setToOriginImpl(); }
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* update) override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class EdgePointTranslation
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTranslation>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTranslation>::linearizeOplus;
    EdgePointTranslation(const Eigen::Vector3d& worldPoint,
                         const Eigen::Matrix3d& fixedRcw,
                         GeometricCamera* camera);
    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    bool IsValid() const { return mValid; }
private:
    bool CameraPoint(Eigen::Vector3d& point) const;
    Eigen::Vector3d mWorldPoint;
    Eigen::Matrix3d mRcw;
    GeometricCamera* mpCamera = nullptr;
    bool mValid = false;
};

class EdgeStereoPointTranslation
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexTranslation>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexTranslation>::linearizeOplus;
    EdgeStereoPointTranslation(const Eigen::Vector3d& worldPoint,
                               const Eigen::Matrix3d& fixedRcw,
                               GeometricCamera* camera, double bf);
    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    bool IsValid() const { return mValid; }
private:
    bool CameraPoint(Eigen::Vector3d& point) const;
    Eigen::Vector3d mWorldPoint;
    Eigen::Matrix3d mRcw;
    GeometricCamera* mpCamera = nullptr;
    double mBf = 0.0;
    bool mValid = false;
};

class EdgeLineTranslation
    : public g2o::BaseUnaryEdge<1, double, VertexTranslation>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<1, double, VertexTranslation>::linearizeOplus;
    EdgeLineTranslation(const Eigen::Vector3d& worldPoint,
                        const Eigen::Vector3d& imageLine,
                        const Eigen::Matrix3d& fixedRcw,
                        GeometricCamera* camera);
    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    bool IsValid() const { return mValid; }
private:
    bool CameraPoint(Eigen::Vector3d& point) const;
    Eigen::Vector3d mWorldPoint;
    Eigen::Vector3d mImageLine;
    Eigen::Matrix3d mRcw;
    GeometricCamera* mpCamera = nullptr;
    bool mValid = false;
};

class EdgePlaneDistanceTranslation
    : public g2o::BaseUnaryEdge<1, double, VertexTranslation>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<1, double, VertexTranslation>::linearizeOplus;
    EdgePlaneDistanceTranslation(const Eigen::Vector4d& worldPlane,
                                 const Eigen::Vector3d& observedNormal,
                                 const Eigen::Matrix3d& fixedRcw);
    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
private:
    bool Parameters(Eigen::Vector3d& normal, double& sign,
                    const VertexTranslation*& vertex) const;
    Eigen::Vector4d mWorldPlane;
    Eigen::Vector3d mObservedNormal;
    Eigen::Matrix3d mRcw;
};

class EdgeLineProjectXYZOnlyPose
    : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>::linearizeOplus;
    EdgeLineProjectXYZOnlyPose(const Eigen::Vector3d& worldPoint,
                               const Eigen::Vector3d& imageLine,
                               GeometricCamera* camera);
    void computeError() override;
    void linearizeOplus() override;
    Eigen::Matrix<double,6,6> PoseInformation();
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    bool IsValid() const { return mValid; }
private:
    Eigen::Vector3d mWorldPoint;
    Eigen::Vector3d mImageLine;
    GeometricCamera* mpCamera;
    bool mValid = false;
};

// Minimal plane residual: two components in the observed normal's tangent
// space plus signed plane distance. This avoids optimizing a redundant fourth
// coefficient and follows ManhattanSLAM's 3-DoF plane error convention.
class EdgePlaneOnlyPose
    : public g2o::BaseUnaryEdge<3, Eigen::Vector4d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<3, Eigen::Vector4d, g2o::VertexSE3Expmap>::linearizeOplus;
    explicit EdgePlaneOnlyPose(const Eigen::Vector4d& worldPlane);
    void computeError() override;
    void linearizeOplus() override;
    Eigen::Matrix<double,6,6> PoseInformation();
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
private:
    bool PredictedPlane(Eigen::Vector4d& plane, double& sign,
                        Eigen::Matrix<double,3,2>& tangent) const;
    Eigen::Vector4d mWorldPlane;
};

class EdgeParallelPlaneOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<2, Eigen::Vector3d, g2o::VertexSE3Expmap>::linearizeOplus;
    explicit EdgeParallelPlaneOnlyPose(const Eigen::Vector3d& worldNormal);
    void computeError() override;
    void linearizeOplus() override;
    Eigen::Matrix<double,6,6> PoseInformation();
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
private:
    bool Parameters(Eigen::Vector3d& predicted,
                    Eigen::Matrix<double,3,2>& tangent) const;
    Eigen::Vector3d mWorldNormal;
};

class EdgePerpendicularPlaneOnlyPose
    : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>::linearizeOplus;
    EdgePerpendicularPlaneOnlyPose(const Eigen::Vector3d& worldNormal,
                                   const Eigen::Vector3d& observedNormal);
    void computeError() override;
    void linearizeOplus() override;
    Eigen::Matrix<double,6,6> PoseInformation();
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
private:
    Eigen::Vector3d mWorldNormal;
    Eigen::Vector3d mObservedNormal;
};

class EdgeManhattanRotationOnlyPose
    : public g2o::BaseUnaryEdge<9, Eigen::Matrix<double, 9, 1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using g2o::BaseUnaryEdge<9, Eigen::Matrix<double, 9, 1>, g2o::VertexSE3Expmap>::linearizeOplus;
    void computeError() override;
    void linearizeOplus() override;
    Eigen::Matrix<double,6,6> PoseInformation();
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Short-lived learned pose prior. Its information matrix is uncertainty-
// calibrated and tightly capped by Optimizer; the finite-difference Jacobian
// keeps the SE(3) convention consistent with VertexSE3Expmap's left update.
class EdgeSE3MotionPrior
    : public g2o::BaseUnaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override;
    void linearizeOplus() override;
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

Eigen::Matrix<double, 9, 1> FlattenRotation(const Eigen::Matrix3d& rotation);

} // namespace TIB_SLAM

#endif
