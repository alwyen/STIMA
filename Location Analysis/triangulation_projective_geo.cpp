#include "triangulation_projective_geo.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;
using Eigen::JacobiSVD;

double sinc(double x) {
    if (x == 0.0) {
        return 1;
    }
    else{
        double v = sin(x) / x;
        return v;
    }
}
Matrix3d skew(Vector3d w) {
    Matrix3d skewM = Matrix3d::Zero();

    skewM(0, 1) = -w(2, 0);
    skewM(0, 2) = w(1, 0);
    skewM(1, 0) = w(2, 0);
    skewM(1, 2) = -w(0, 0);
    skewM(2, 0) = -w(1, 0);
    skewM(2, 1) = w(0, 0);
    
    return skewM;
}
Matrix3d deparameterize_Omega(Matrix3d w) {
    double mag_w = w.norm();
    Matrix3d R = Matrix3d::Identity();

    Matrix3d R1 = skew(w) * sinc(mag_w);

    Matrix3d R2 = ((1 - cos(mag_w)) / (pow(mag_w, 2.0)) * skew(w) * skew(w);

    R = R + R1 + R2;

    return R;
}

void parameterize_Rotation(Matrix3d R, Vector3d &w, double &theta) {
    Matrix3d newR = R - Matrix3d::Identity();
    //Matrix3d b = Matrix3d::Zero();
    //Vector3d V = newR.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    JacobiSVD<MatrixXd> svd(newR, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd V = svd.matrixV().col(2);

    RowVector3d V_hat{ { [R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1) } };

    double sin_theta = double((V * V_hat) / 2);


    double cos_theta = double((R.trace() - 1) / 2);

    theta = atan2(sin_theta, cos_theta);

    w = (theta / V.norm()) * V;

    if (theta > M_PI) {
        w = ((1 - 2 * M_PI / theta) * ceil((theta - M_PI) / 2 / M_PI)) * w;
    }
}

MatrixXd calc_camera_proj_matrix(Matrix3d K, Matrix3d R, VectorXd t) {
    MatrixXd Rnew(R.rows(), R.cols() + t.cols()); // Horizontal Cancatenation
    Rnew << R, t;

    return K * Rnew;
}
MatrixXd homogenize(MatrixXd x) {
    MatrixXd ones = MatrixXd::Ones(1, x.cols());
    MatrixXd xHomog(x.rows() + ones.rows(), x.cols());

    xHomog << x, ones;

    return xHomog;

}

MatrixXd dehomogenize(MatrixXd x) {
    MatrixXd x1Inhomog = x.block(0, 0, 2, x.cols());
    RowVectorXd lastRow = x.block(2, 0, 1, x.cols());

    MatrixXd xinHomog = x1Inhomog.array().rowwise() / lastRow.array();

    return xinHomog;
}
Matrix3d angle_axis_to_rotation(double angle, MatrixXd axis) {
    double c = cos(angle);
    double s = sin(angle);
    double t = 1.0 - c;
    MatrixXd norm_axis = axis / axis.norm();

    double x = norm_axis(0, 0);
    double y = norm_axis(1, 0);
    double z = norm_axis(2, 0);
    Matrix3d R = Matrix3d::Zero();
    R(0, 0) = t * pow(x, 2) + c;
    R(0, 1) = t * x * y - z * s;
    R(0, 2) = t * x * z + y * s;
    R(1, 0) = t * x * y + z * s;
    R(1, 1) = t * pow(y, 2) + c;
    R(1, 2) = t * y * z - x * s;
    R(2, 0) = t * x * z - y * s;
    R(2, 1) = t * y * z + x * s;
    R(2, 2) = t * pow(z, 2) + c;

    return R;
}
Matrix3d interpolate_rotation(double alpha, Matrix3d r1, Matrix3d r2) {
    Vector3d w;
    double theta;

    parameterize_Rotation(r2 * r1.transposeInPlace(), &w, &theta);

    return deparameterize_Omega(alpha * w) * r1;
}
Vector3d camera_translation(Matrix3d R1, VectorXd t1, Matrix3d R2) {
    return R2 * (R1.transposeInPlace() * t1);
}
Matrix3d calc_projective_transformation(Matrix3d K1, Matrix3d R1, Matrix3d K2, Matrix3d R2) {
    return K2 * R2 * R1.transposeInPlace() * K1.inverse();
}

void epipolar_rectification(Matrix3d K1, Matrix3d R1, Vector3d t1,
    Matrix3d K2, Matrix3d R2, Vector3d t2,
    Matrix3d& Krectified, Matrix3d& Rrectified, Vector3d& t1rectified,
    Vector3d& t2rectified, Matrix3d& H1, Matrix3d& H2) {

    Matrix3d Rinterp = interpolate_rotation(0.5, R1, R2);
    Vector3d u = camera_translation(R2, t2, Rinterp) - camera_translation(R1, t1, Rinterp);
    Vector3d vhat{ {1.0, 0.0, 0.0} };


    double neg = u.transpose() * vhat;

    if (0 > neg) {
        vhat(0, 0) = -1.0;
    }

    double angle = acos(u.transpose() * vhat / u.norm());


    Vector3d axis = u.cross(vhat);

    Matrix3d R_X = angle_axis_to_rotation(angle, axis);
    Rrectified = R_X * Rinterp;

    t1rectified = camera_translation(R1, t1, Rrectified);
    t2rectified = camera_translation(R2, t2, Rrectified);

    double alpha = (K1(0,0) + K2(0,0) + K1(1,1) + K2(1,1)) / 4;


    double x0 = (K1(0, 2) + K2(0, 2)) / 2;


    double y0 = (K1(1, 2) + K2(1, 2)) / 2;

    Krectified = Matrix3d::Zero();


    Krectified(0, 0) = alpha;
    Krectified(0, 2) = x0;
    Krectified(1, 1) = alpha;
    Krectified(1, 2) = y0;
    Krectified(2, 2) = 1.0;

    H1 = calc_projective_transformation(K1, R1, Krectified, Rrectified);

    H2 = calc_projective_transformation(K2, R2, Krectified, Rrectified);
}
MatrixXd calc_epipole(MatrixXd E) {
    JacobiSVD<MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd v = svd.matrixV().col(2);

    return v;
}
void backprojection(VectorXd x, MatrixXd P, RowVector4d& row1, RowVector4d& row2) {
    double x1_coeff = P(2, 0)* x(0, 0) - P(0, 0);
    double y1_coeff = P(2, 1) * x(0, 0) - P(0, 1);
    double z1_coeff = P(2, 2) * x(0, 0) - P(0, 2);
    double w1_coeff = P(2, 3) * x(0, 0) - P(0, 3);

    double x2_coeff = P(2, 0) * x(1, 0) - P(1, 0);
    double y2_coeff = P(2, 1) * x(1, 0) - P(1, 1);
    double z2_coeff = P(2, 2) * x(1, 0) - P(1, 2);
    double w2_coeff = P(2, 3) * x(1, 0) - P(1, 3);

    row1 << x1_coeff, y1_coeff, z1_coeff, w1_coeff;
    row2 << x2_coeff, y2_coeff, z2_coeff, w2_coeff;
}
MatrixXd triangulation2View(VectorXd x1, VectorXd x2, MatrixXd P1rectified, MatrixXd P2rectified) {
    RowVector4d row1;
    RowVector4d row2;
    RowVector4d row3;
    RowVector4d row4;

    backprojection(x1, P1rectified, &row1, &row2);
    backprojection(x2, P2rectified, &row3, &row4);

    MatrixXd C(row1.rows() + row2.rows() + row3.rows() + row4.rows(), row1.cols());

    C << row1, row2, row3, row4;

    JacobiSVD<MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd v = svd.matrixV().col(3);

    return v; //MAKE SURE TO ADD dehomogenize!!!
}