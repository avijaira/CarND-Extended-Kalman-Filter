#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() == ground_truth.size() || estimations.size() != 0) {
    // accumulate squared residuals
    for (int i=0; i < estimations.size(); ++i) {

      VectorXd residual = estimations[i] - ground_truth[i];

      // coefficient-wise multiplication
      residual = residual.array() * residual.array();
      rmse += residual;
    }

    // calculate the mean
    rmse = rmse / estimations.size();

    // calculate the squared root
    rmse = rmse.array().sqrt();
  }

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3, 4);

  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = c2 * c1;

  // check division by zero
  if (fabs(c1) >= 0.0001) {
    // compute the Jacobian matrix
    Hj(0, 0) = px / c2;
    Hj(0, 1) = py / c2;
    Hj(1, 0) = -py / c1;
    Hj(1, 1) = px / c1;
    Hj(2, 0) = py * (vx * py - vy * px) / c3;
    Hj(2, 1) = px * (vy * px - vx * py) / c3;
    Hj(2, 2) = px / c2;
    Hj(2, 3) = py / c2;
  }

  return Hj;
}
