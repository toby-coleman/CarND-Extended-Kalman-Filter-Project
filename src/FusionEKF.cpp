#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;
  dt_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  // Laser measurement matrix
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Create EKF
  ekf_ = KalmanFilter();


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

// Utility functions
MatrixXd FusionEKF::F() {
  // F matrix for state transition
  MatrixXd F_ = MatrixXd::Identity(4, 4);
  F_(0, 2) = dt_;
  F_(1, 3) = dt_;
  return F_;
}

MatrixXd FusionEKF::Q(float noise_ax, float noise_ay) {
  // Q matrix for state transition
  MatrixXd Q_ = MatrixXd(4, 4);
  float dt = dt_;
  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;
  Q_ << 0.25 * dt4 * noise_ax, 0, 0.5 * dt3 * noise_ax, 0,
        0, 0.25 * dt4 * noise_ay, 0, 0.5 * dt3 * noise_ay,
        0.5 * dt3 * noise_ax, 0, dt2 * noise_ax, 0,
        0, 0.5 * dt3 * noise_ay, 0, dt2 * noise_ay;
  return Q_;
}

VectorXd FusionEKF::H_radar(VectorXd &x) {
  // H function for radar measurements
  float rho, phi, rho_dot;
  VectorXd h_ = VectorXd(3);

  // Recover state parameters
	float px = x(0);
	float py = x(1);
	float vx = x(2);
	float vy = x(3);

  // Measurement function
  rho = sqrt(px * px + py * py);
  phi = atan2(py, px);
  rho_dot = (px * vx + py * vy) / rho;

  h_ << rho, phi, rho_dot;
  return h_;
}

MatrixXd FusionEKF::Hj_radar(VectorXd &x) {
  // Jacobian matrix for radar measurements
  MatrixXd Hj_ = MatrixXd(3, 4);

	// Recover state parameters
	float px = x(0);
	float py = x(1);
	float vx = x(2);
	float vy = x(3);

	// Pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px + py*py;
  float c2 = sqrt(c1);
	float c3 = (c1*c2);
  // Check division by zero
	if(fabs(c1) < 0.0001) {
		c1 = 0.0001; // set to a small value
    c2 = 0.0001;
    c3 = 0.0001;
	}

	// Compute the Jacobian matrix
	Hj_ << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj_;
}

void FusionEKF::UpdateTimestamp(long new_timestamp) {
  // Compute time delta in seconds
  dt_ = (new_timestamp - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = new_timestamp;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    float x, y, vx, vy;
    VectorXd x_initial(4);
    MatrixXd P_initial = MatrixXd(4, 4);

    // Create initial state
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      x = rho * cos(phi);
      y = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state, assume zero velocity
      x = measurement_pack.raw_measurements_[0];
      y = measurement_pack.raw_measurements_[1];
      vx = 0;
      vy = 0;
    }

    // Don't initialize at x=0, y=0
    if ((fabs(x) < 0.0001) & (fabs(y) < 0.0001)) {
      x = 0.0001;
    }
    x_initial << x, y, vx, vy;

    // Create initial covariance
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      P_initial << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Use more covariance on the velocity
      P_initial << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1000, 0,
                   0, 0, 0, 1000;
    }

    // Initialize
    ekf_.Init(x_initial, P_initial);

    // Store timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "EKF initialized" << endl;
    cout << "x_initial = " << x_initial << endl;
    cout << "P_initial = " << P_initial << endl;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

   UpdateTimestamp(measurement_pack.timestamp_);

   // Predict using noise_ax = 9 and noise_ay = 9
   MatrixXd F_ = F();
   MatrixXd Q_ = Q(9.0, 9.0);
   ekf_.Predict(F_, Q_);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    MatrixXd Hj_ = Hj_radar(ekf_.x_);
    ekf_.Update(measurement_pack.raw_measurements_, H_radar, Hj_, R_radar_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_, H_laser_, R_laser_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
