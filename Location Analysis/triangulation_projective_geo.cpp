#define _USE_MATH_DEFINES


#include "triangulation_projective_geo.h"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include "rotation.h"


using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixXf;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;
using Eigen::JacobiSVD;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::VectorXd;

double TPGeo::sinc(double x) {
    if (x == 0.0) {
        return 1;
    }
    else{
        double v = sin(x) / x;
        return v;
    }
}
Matrix3d TPGeo::skew(Vector3d w) {
    Matrix3d skewM = Matrix3d::Zero();

    skewM(0, 1) = -w(2, 0);
    skewM(0, 2) = w(1, 0);
    skewM(1, 0) = w(2, 0);
    skewM(1, 2) = -w(0, 0);
    skewM(2, 0) = -w(1, 0);
    skewM(2, 1) = w(0, 0);
    
    return skewM;
}
Matrix3d TPGeo::deparameterize_Omega(Vector3d w) {
    double mag_w = w.norm();
    Matrix3d R = Matrix3d::Identity();

    Matrix3d R1 = skew(w) * sinc(mag_w);

    Matrix3d R2 = ((1 - cos(mag_w)) / (pow(mag_w, 2.0)) * skew(w) * skew(w));

    R = R + R1 + R2;

    return R;
}

//w Must be a 3x1 Vector
void TPGeo::parameterize_Rotation(Matrix3d R, Vector3d *w, double *theta) {
    Matrix3d newR = R - Matrix3d::Identity();
    //Matrix3d b = Matrix3d::Zero();
    //Vector3d V = newR.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    JacobiSVD<MatrixXd> svd(newR, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd V = svd.matrixV().col(2);
    V.transposeInPlace();
    Vector3d V_hat{ { R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1) } };

    double sin_theta = double((V * V_hat)(0, 0) / 2);

    double cos_theta = double((R.trace() - 1) / 2);

    *theta = atan2(sin_theta, cos_theta);

    //std::cout << "BeforeW" << std::endl;
    Vector3d V1 = svd.matrixV().col(2);
    std::cout << "BeforeW" << std::endl;
    *w = (*theta / V1.norm()) * V1;
    std::cout << "After W" << std::endl;

    if (*theta > M_PI) {
        *w = ((1 - 2 * M_PI / *theta) * ceil((*theta - M_PI) / 2 / M_PI)) * (*w);
    }
}

MatrixXd TPGeo::calc_camera_proj_matrix(Matrix3d K, Matrix3d R, VectorXd t) {
    MatrixXd Rnew(R.rows(), R.cols() + t.cols()); // Horizontal Cancatenation
    Rnew << R, t;

    return K * Rnew;
}

MatrixXd TPGeo::homogenize(MatrixXd x) {
    MatrixXd ones = MatrixXd::Ones(1, x.cols());
    MatrixXd xHomog(x.rows() + ones.rows(), x.cols());

    xHomog << x, ones;

    return xHomog;

}

MatrixXd TPGeo::dehomogenize(MatrixXd x) {
    MatrixXd x1Inhomog = x.block(0, 0, x.rows() - 1, x.cols());

    RowVectorXd lastRow = x.block(x.rows() - 1, 0, 1, x.cols());
    std::cout << "X" << std::endl;
    std::cout << x << std::endl;
    std::cout << "inhomogoenous" << std::endl;
    std::cout << x1Inhomog << std::endl;
    std::cout << "last Row" << std::endl;
    std::cout << lastRow << std::endl;

    MatrixXd xinHomog = x1Inhomog.array().rowwise() / lastRow.array();

    return xinHomog;
}
Matrix3d TPGeo::angle_axis_to_rotation(double angle, MatrixXd axis) {
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
Matrix3d TPGeo::interpolate_rotation(double alpha, Matrix3d r1, Matrix3d r2) {
    Vector3d w = Vector3d::Zero();
    double theta = 0;
    r1.transposeInPlace();

    Matrix3d newR = r2 * r1;

    parameterize_Rotation(newR, &w, &theta);

    return deparameterize_Omega(alpha * w) * r1;
}
Vector3d TPGeo::camera_translation(Matrix3d R1, VectorXd t1, Matrix3d R2) {
    R1.transposeInPlace();
    return R2 * (R1 * t1);
}
Matrix3d TPGeo::calc_projective_transformation(Matrix3d K1, Matrix3d R1, Matrix3d K2, Matrix3d R2) {
    R1.transposeInPlace();

    return K2 * R2 * R1 * K1.inverse();
}

void TPGeo::epipolar_rectification(Matrix3d K1, Matrix3d R1, Vector3d t1,
    Matrix3d K2, Matrix3d R2, Vector3d t2,
    Matrix3d *Krectified, Matrix3d *Rrectified, Vector3d *t1rectified,
    Vector3d *t2rectified, Matrix3d *H1, Matrix3d *H2) {

    std::cout << "Inside epipolar_rectification" << std::endl;
    Matrix3d Rinterp = interpolate_rotation(0.5, R1, R2);

    std::cout << Rinterp << std::endl;
    Vector3d u = camera_translation(R2, t2, Rinterp) - camera_translation(R1, t1, Rinterp);
    Vector3d vhat{ {1.0, 0.0, 0.0} };
    
    std::cout << "U" << std::endl;
    std::cout << u << std::endl;
    MatrixXd uT = u.transpose();

    std::cout << "U.T" << std::endl;
    std::cout << uT << std::endl;


    double neg = (uT * vhat)(0,0);

    if (0 > neg) {
        vhat(0, 0) = -1.0;
    }

    double angle = acos((uT * vhat)(0,0) / u.norm());


    Vector3d axis = u.cross(vhat);

    Matrix3d R_X = angle_axis_to_rotation(angle, axis);
    *Rrectified = R_X * Rinterp;

    *t1rectified = camera_translation(R1, t1, *Rrectified);
    *t2rectified = camera_translation(R2, t2, *Rrectified);

    double alpha = (K1(0,0) + K2(0,0) + K1(1,1) + K2(1,1)) / 4;


    double x0 = (K1(0, 2) + K2(0, 2)) / 2;


    double y0 = (K1(1, 2) + K2(1, 2)) / 2;

    *Krectified = Matrix3d::Zero();


    Krectified->coeffRef(0, 0) = alpha;
    Krectified->coeffRef(0, 2) = x0;
    Krectified->coeffRef(1, 1) = alpha;
    Krectified->coeffRef(1, 2) = y0;
    Krectified->coeffRef(2, 2) = 1.0;

    *H1 = calc_projective_transformation(K1, R1, *Krectified, *Rrectified);

    *H2 = calc_projective_transformation(K2, R2, *Krectified, *Rrectified);
}
MatrixXd TPGeo::calc_epipole(MatrixXd E) {
    JacobiSVD<MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd v = svd.matrixV().col(2);

    return v;
}
void TPGeo::backprojection(VectorXd x, MatrixXd P, RowVector4d *row1, RowVector4d *row2) {
    double x1_coeff = P(2, 0)* x(0, 0) - P(0, 0);
    double y1_coeff = P(2, 1) * x(0, 0) - P(0, 1);
    double z1_coeff = P(2, 2) * x(0, 0) - P(0, 2);
    double w1_coeff = P(2, 3) * x(0, 0) - P(0, 3);

    double x2_coeff = P(2, 0) * x(1, 0) - P(1, 0);
    double y2_coeff = P(2, 1) * x(1, 0) - P(1, 1);
    double z2_coeff = P(2, 2) * x(1, 0) - P(1, 2);
    double w2_coeff = P(2, 3) * x(1, 0) - P(1, 3);

    *row1 << x1_coeff, y1_coeff, z1_coeff, w1_coeff;
    *row2 << x2_coeff, y2_coeff, z2_coeff, w2_coeff;
}
MatrixXd TPGeo::triangulation2View(VectorXd x1, VectorXd x2, MatrixXd P1rectified, MatrixXd P2rectified) {
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

    return dehomogenize(v);//MAKE SURE TO ADD dehomogenize!!!
}
//Testing functions

void rectified_calibration_matrices(Matrix3d Krectified, Matrix3d H1, Matrix3d H2, 
    const MatrixXf &I1, const MatrixXf &I2, Matrix3d* K1rectified, Matrix3d* K2rectified) {
    //points are defined by[x, y]!!!
    TPGeo tp = TPGeo();

    Vector2d I1_top_left{ {-0.5, -0.5} };
    Vector2d I1_top_right{ {I1.cols() - 0.5, -0.5}};

    Vector2d I1_bottom_left{ {-0.5, I1.rows() - 0.5}};
    Vector2d I1_bottom_right{ {I1.cols() - 0.5, I1.rows() - 0.5}};


    Vector2d I2_top_left{ {-0.5, -0.5} };
    Vector2d I2_top_right{ {I2.cols() - 0.5, -0.5} };

    Vector2d I2_bottom_left{ {-0.5, I2.rows() - 0.5} };
    Vector2d I2_bottom_right{ {I2.cols() - 0.5, I2.rows() - 0.5} };


    MatrixXd pts_I1(I1_top_left.rows(), I1_top_left.cols() + I1_top_right.cols() + I1_bottom_left.cols() + I1_bottom_right.cols());
    pts_I1 << I1_top_left, I1_top_right, I1_bottom_left, I1_bottom_right;

    MatrixXd pts_I2(I2_top_left.rows(), I2_top_left.cols() + I2_top_right.cols() + I2_bottom_left.cols() + I2_bottom_right.cols());
    pts_I2 << I2_top_left, I2_top_right, I2_bottom_left, I2_bottom_right;

    MatrixXd pts_I1_rec = tp.dehomogenize(H1 * tp.homogenize(pts_I1));

    MatrixXd pts_I2_rec = tp.dehomogenize(H2 * tp.homogenize(pts_I2));

    double min_row_I1 = ceil(pts_I1_rec.rowwise().minCoeff()(1, 0));
    double max_row_I1 = ceil(pts_I1_rec.rowwise().maxCoeff()(1, 0));

    double min_row_I2 = ceil(pts_I2_rec.rowwise().minCoeff()(1, 0));
    double max_row_I2 = ceil(pts_I2_rec.rowwise().maxCoeff()(1, 0));

    double min_col_I1 = ceil(pts_I1_rec.rowwise().minCoeff()(0, 0));
    double max_col_I1 = ceil(pts_I1_rec.rowwise().maxCoeff()(0, 0));

    double min_col_I2 = ceil(pts_I2_rec.rowwise().minCoeff()(0, 0));
    double max_col_I2 = ceil(pts_I2_rec.rowwise().maxCoeff()(0, 0));

    Matrix3d T1 = Matrix3d::Identity();
    Matrix3d T2 = Matrix3d::Identity();

    //if add values onto T1[1][2], answer gets closer to Ben's results
    T1(1, 2) = -1*std::min(min_row_I1, min_row_I2) - 0.5;
    T1(0, 2) = -min_col_I1 - 0.5;

    T2(1, 2) = -std::min(min_row_I1, min_row_I2) - 0.5;
    T2(0, 2) = -min_col_I2 - 0.5;

    *K1rectified = T1 * Krectified;

    *K2rectified = T2 * Krectified;
}


MatrixXd geocentric_triangulation2View(const MatrixXf &left_img, const MatrixXf &right_img, MatrixXd x1, MatrixXd x2,
    MatrixXd C, double latitude_origin, double longitude_origin, double plat_yaw, double plat_pitch, 
    double plat_roll, Matrix3d K1, Matrix3d K2, Matrix3d R12, Vector3d t12) {

    double gimbal_yaw = 0;
    double gimbal_pitch = -M_PI / 2;
    double gimbal_roll = 0;

    double latitude_radians = latitude_origin * 180.0 / M_PI;
    double longitude_radians = longitude_origin * 180.0 / M_PI;

    double plat_yaw_radians = plat_yaw * 180.0 / M_PI;
    double plat_pitch_radians = plat_pitch * 180.0 / M_PI;
    double plat_roll_radians = plat_roll * 180.0 / M_PI;

    Rotation R = Rotation();

    Matrix3d R1_geocentric = R.rotationFromWGS84GeocentricToCameraFame(latitude_radians, longitude_radians, gimbal_pitch, gimbal_roll, gimbal_yaw, plat_pitch_radians, plat_roll_radians, plat_yaw_radians);

    std::cout << "Before t1" << std::endl;
    Vector3d t1_geocentric = -R1_geocentric * C;
    
    Matrix3d R2_geocentric = R12 * R1_geocentric;
    Vector3d t2_geocentric = R12 * t1_geocentric + t12;
    std::cout << "After t2" << std::endl;

    // corresponding rectified :

    Matrix3d Krectified;
    Matrix3d Rrectified;
    Vector3d t1rectified;
    Vector3d t2rectified;
    Matrix3d H1;
    Matrix3d H2;
    
    TPGeo tp = TPGeo();

    std::cout << "Before epipolar_rectification" << std::endl;
    tp.epipolar_rectification(K1, R1_geocentric, t1_geocentric, K2, R2_geocentric, t2_geocentric,
        &Krectified, &Rrectified, &t1rectified, &t2rectified, &H1, &H2);

    std::cout << "After epipolar_rectification" << std::endl;
    Matrix3d K1rectified;
    Matrix3d K2rectified;

    rectified_calibration_matrices(Krectified, H1, H2, left_img, right_img, &K1rectified, &K1rectified);

    MatrixXd P1_geocentric = tp.calc_camera_proj_matrix(K1rectified, Rrectified, t1rectified);

    MatrixXd P2_geocentric = tp.calc_camera_proj_matrix(K2rectified, Rrectified, t2rectified);


    //frobius norm projection matrices
    MatrixXd norm_P1_geocentric = P1_geocentric / P1_geocentric.norm();
    MatrixXd norm_P2_geocentric = P2_geocentric / P2_geocentric.norm();


    MatrixXd triangulated_geocentric_point = tp.triangulation2View(x1, x2, norm_P1_geocentric, norm_P2_geocentric);

    return triangulated_geocentric_point;
}

//int main() {
//    std::string image_path1 = "C:/Users/aquir/Downloads/right0.jpg";
//    cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_GRAYSCALE);
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img12;
//    cv::cv2eigen(img1, img12);
//
//    std::cout << "IMAGE1" << std::endl;
//    std::cout << "Cols: " << img12.cols() << std::endl;
//    std::cout << "Rows: " << img12.rows() << std::endl;
//
//    std::string image_path2 = "C:/Users/aquir/Downloads/left0.jpg";
//    cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_GRAYSCALE);
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img22;
//    cv::cv2eigen(img2, img22);
//
//    std::cout << "IMAGE1" << std::endl;
//    std::cout << "Cols: " << img22.cols() << std::endl;
//    std::cout << "Rows: " << img22.rows() << std::endl;
//
//    MatrixXd x1{ {1862}, {1176} };
//
//    MatrixXd x2{ {1840}, {1176} };
//
//    MatrixXd C{ {-2452517.8384},
//        {-4768302.495884},
//        {3442352.572949} };
//
//        
//    double latitude_origin = 32.875045;
//    double longitude_origin = -117.218468;
//    double plat_yaw = 364.936; 
//    double plat_pitch = -110.848;
//
//    double plat_roll = 1.383;
//    Matrix3d K1{ {3.28387648e+03, 0.00000000e+00, 2.03036278e+03 }, 
//        {0.00000000e+00, 3.29903024e+03, 1.55033756e+03}, 
//        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
//    Matrix3d K2{ {3.26907563e+03, 0.00000000e+00, 2.08651620e+03}, 
//        {0.00000000e+00, 3.27623450e+03, 1.58700571e+03}, 
//        {0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
//    Matrix3d R12{ {0.99962162,  0.02428445, -0.01291845}, 
//        {-0.02436906, 0.99968233, -0.00643293}, 
//        { 0.01275813, 0.00674531, 0.99989586} };
//    Vector3d t12{ {- 0.33418658, 0.00541115, -0.00189281} };
//
//    MatrixXd returnMat;
//
//    returnMat = geocentric_triangulation2View(img22, img12, x1, x2, C, latitude_origin, longitude_origin, plat_yaw, plat_pitch, 
//        plat_roll, K1, K2, R12, t12);
//
//    std::cout << "Matrix" << std::endl;
//    std::cout << returnMat << std::endl;
//}