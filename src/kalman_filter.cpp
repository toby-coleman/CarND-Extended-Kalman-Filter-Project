#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in) {
  x_ = x_in;
  P_ = P_in;
}

void KalmanFilter::Predict(MatrixXd &F,
                           MatrixXd &Q) {
  // Predict step of Kalman filter
  x_ = F * x_;
  P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::Predict(MatrixXd (*F)(VectorXd), MatrixXd &Fj,
                           MatrixXd &Q) {
  // Predict step of Extended Kalman filter
  // Expects function F and Jacobian Fj as arguments
  x_ = F(x_);
  P_ = Fj * P_ * Fj.transpose() + Q;
}

void KalmanFilter::Update(const VectorXd &z,
                          MatrixXd &H,
                          MatrixXd &R) {
  // Update Kalman filter with new measurement
  VectorXd y = z - H * x_;
  // Pre-calculate P * H_transpose for efficiency
  MatrixXd PHt = P_ * H.transpose();
  MatrixXd S = H * PHt + R;
  MatrixXd K = PHt * S.inverse();
  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H) * P_;
}

void KalmanFilter::Update(const VectorXd &z,
                          VectorXd (*H)(VectorXd&), MatrixXd &Hj,
                          MatrixXd &R) {
  // Update Extended Kalman filter with new measurement
  // Expects function H and Jacobian Hj as arguments
  VectorXd y = z - H(x_);
  // Make sure phi is in range [-pi, +pi]
  if (y[1] < -M_PI) {
    y[1] += M_PI * 2;
  } else if (y[1] > M_PI) {
    y[1] -= M_PI * 2;
  }
  // Pre-calculate P * Hj_transpose for efficiency
  MatrixXd PHjt = P_ * Hj.transpose();
  MatrixXd S = Hj * PHjt + R;
  MatrixXd K = PHjt * S.inverse();

  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * Hj) * P_;
}
