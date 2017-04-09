#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in);

  /**
   * Predicts the state and the state covariance
   * in the case of linear Kalman filter
   * @param F Transition matrix
   * @param Q Process covariance matrix
   */
  void Predict(Eigen::MatrixXd &F, Eigen::MatrixXd &Q);

  /**
   * Predicts the state and the state covariance
   * in the case of Extended Kalman filter
   * @param F(VectorXd) Transition function
   * @param Fj Jacobian matrix of F
   * @param Q Process covariance matrix
   */
  void Predict(Eigen::MatrixXd (*F)(Eigen::VectorXd), Eigen::MatrixXd &Fj,
               Eigen::MatrixXd &Q);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   * @param H Measurement matrix
   * @param R Measurement covariance matrix
   */
  void Update(const Eigen::VectorXd &z,
              Eigen::MatrixXd &H, Eigen::MatrixXd &R);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   * @param H(VectorXd) Measurement function
   * @param Hj Jacobian matrix of H
   * @param R Measurement covariance matrix
   */
   void Update(const Eigen::VectorXd &z,
               Eigen::VectorXd (*H)(Eigen::VectorXd&), Eigen::MatrixXd &Hj,
               Eigen::MatrixXd &R);

};

#endif /* KALMAN_FILTER_H_ */
