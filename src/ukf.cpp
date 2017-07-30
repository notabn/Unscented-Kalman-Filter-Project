#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    
    
    //set state dimension
    n_x_ = 5;
    n_aug_ = 7;
    
    lambda_ = 3 - n_x_;
    
    
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    
    // initial state vector
    x_ = VectorXd::Zero(n_x_);
    
    // initial covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);
    P_(2,2)=10;
    P_(3,3)=10;
    P_(4,4)=10;
    
    
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5;
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 3.5;
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    
    
   
    
    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd::Zero(n_x_,2*n_aug_+1);
    

    ///* time when the state is true, in us
    time_us_ =10000;
    
    
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;
    //create matrix for sigma points in measurement space
    Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);
    
    is_initialized_ = false;
    
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
         * Initialize the state x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
         */
        // first measurement
        cout << "UKF: " << endl;
        
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
            /**
             Convert radar from polar to cartesian coordinates and initialize state.
             */
            
            float px = meas_package.raw_measurements_[0]  * cos(meas_package.raw_measurements_[1] );
            float py = meas_package.raw_measurements_[0]  * sin(meas_package.raw_measurements_[1] );
            float v = 0;
            float yaw = 0;
            float yawd = 0;
            x_ << px, py, v, yaw,yawd;
            
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
            /**
             Initialize state.
             */
            //set the state with the initial location and zero velocity
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0,0;
            
        }
        
        time_us_ = meas_package.timestamp_;
        
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }
    
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    //compute the time elapsed between the current and previous measurements
    double dt = (meas_package.timestamp_ - time_us_) / (double)1000000;
    time_us_ = meas_package.timestamp_;
    
    
    UKF::Prediction(dt);
    
    /*****************************************************************************
     *  Update
     ****************************************************************************/
    
    /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     */
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        // Radar updates<#const Eigen::VectorXd &z#>
        UKF::UpdateRadar(meas_package);
    }else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
        // Laser updates
        //measurement update
 
        UKF::UpdateLidar(meas_package);
    }
    
    // print the output
    //cout << "x_ = " << ekf_.x_ << endl;
    //cout << "P_ = " << ekf_.P_ << endl;
    
    
    
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
     TODO:
     
     Complete this function! Estimate the object's location. Modify the state
     vector, x_. Predict sigma points, the state, and the state covariance matrix.
     */
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_,2*n_x_+1);
    UKF::AugmentedSigmaPoints(&Xsig_aug);
    UKF::SigmaPointPrediction(Xsig_aug,delta_t);
    UKF::PredictMeanAndCovariance();
    
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     Complete this function! Use lidar data to update the belief about the object's
     position. Modify the state vector, x_, and covariance, P_.
     
     You'll also need to calculate the lidar NIS(Normalized Innovation Squared).
     */
    
    /**
     * update the state by using Kalman Filter equations
     */

    int n_z = 2;
    
    //set vector for weights
    VectorXd weights = VectorXd(2*n_aug_+1);
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {
        double weight = 0.5/(n_aug_+lambda_);
        weights(i) = weight;
    }
    
    
    //create example vector for mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    
    //create example matrix for predicted measurement covariance
    MatrixXd S = MatrixXd::Zero(n_z,n_z);
    
    MatrixXd ZLsig = MatrixXd::Zero(n_z,2 * n_aug_ + 1);
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        
        
        // measurement model
        ZLsig(0,i) = p_x;
        ZLsig(1,i) = p_y;
        
    }
    //std::cout << "Zsig: " << std::endl << ZLsig << std::endl;
    
    
    //mean predicted measurement
    
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights(i) * ZLsig.col(i);
    }
    
    //measurement covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = ZLsig.col(i) - z_pred;
        
        S = S + weights(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd::Zero(n_z,n_z);
    R(0,0) = std_laspx_*std_laspx_;
    R(1,1) = std_laspy_*std_laspy_;
    
    S = S + R;
    
    
    //print result
    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;
    

    
    //create example vector for incoming radar measurement
    VectorXd z =  meas_package.raw_measurements_;
    
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    
    
    //calculate cross correlation matrix
    //calculate Kalman gain K;
    //update state mean and covariance matrix
    for (int i=0;i<2*n_aug_+1;i++){
        VectorXd x_diff = Xsig_pred_.col(i)-x_;
        VectorXd z_diff = ZLsig.col(i)-z_pred;
        Tc += weights(i) *x_diff*z_diff.transpose();
    }
    
    MatrixXd K = MatrixXd::Zero(n_x_,n_z);
    K = Tc * S.inverse();
    
    x_ = x_ + K*(z-z_pred);
    P_ = P_-K*S*K.transpose();
    
    
    NIS_laser_ = (z-z_pred).transpose()*S.inverse()*(z-z_pred);
    
    //print result
    //std::cout << "Updated state lidar x: " << std::endl << x_ << std::endl;
    //std::cout << "Updated state covariance lidar P: " << std::endl << P_<< std::endl;
    
    /**
    VectorXd z = VectorXd(x_.size());
    z.fill(0.0);
    z(0) = meas_package.raw_measurements_(0);
    z(1) = meas_package.raw_measurements_(1);
    MatrixXd H_laser_ = MatrixXd::Identity(n_x_, n_x_);
    
    MatrixXd R_laser_ = MatrixXd(n_x_, n_x_);
    R_laser_ << std_laspx_*std_laspx_,0,0,0,0,
                0,std_laspy_*std_laspy_,0,0,0,
                0,0,0,0,0,
                0,0,0,0,0,
                0,0,0,0,0;
    
    VectorXd y = z - H_laser_ * x_;
    MatrixXd Ht = H_laser_.transpose();
    MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;
    
    MatrixXd I_ = MatrixXd::Identity(x_.size(), x_.size());
    //new state
    x_ = x_ + (K * y);
    P_ = (I_ - K * H_laser_) * P_;
    
    double nis = y.transpose()*Si*y;
    nis_lidar.push_back(nis);
     **/
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     TODO:
     
     Complete this function! Use radar data to update the belief about the object's
     position. Modify the state vector, x_, and covariance, P_.
     
     You'll also need to calculate the radar NIS.
     */
    
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;
    
    //set vector for weights
    VectorXd weights = VectorXd(2*n_aug_+1);
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {
        double weight = 0.5/(n_aug_+lambda_);
        weights(i) = weight;
    }
   
 
    //create example vector for mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    
    //create example matrix for predicted measurement covariance
    MatrixXd S = MatrixXd::Zero(n_z,n_z);

    
    UKF::PredictRadarMeasurement(&z_pred, &S);
    
    //create example vector for incoming radar measurement
    VectorXd z =  meas_package.raw_measurements_;

    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    
    
    //calculate cross correlation matrix
    //calculate Kalman gain K;
    //update state mean and covariance matrix
    for (int i=0;i<2*n_aug_+1;i++){
        VectorXd x_diff = Xsig_pred_.col(i)-x_;
        VectorXd z_diff = Zsig.col(i)-z_pred;
        Tc += weights(i) *x_diff*z_diff.transpose();
    }
    
    MatrixXd K = MatrixXd::Zero(n_x_,n_z);
    K = Tc * S.inverse();
    
    x_ = x_ + K*(z-z_pred);
    P_ = P_-K*S*K.transpose();
    
    
    NIS_radar_ = (z-z_pred).transpose()*S.inverse()*(z-z_pred);

    //print result
    //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P_<< std::endl;
    
    
}



void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
    
    
    //create augmented mean vector
    VectorXd x_aug = VectorXd::Zero(7);
    
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd::Zero(7, 7);
    
    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
    
    x_aug.head(n_x_) = x_;
    Xsig_aug.col(0) = x_aug;
    
    //create augmented covariance matrix
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(5,5) = std_a_ *std_a_;
    P_aug(6,6) = std_yawdd_ *std_yawdd_;
    
    MatrixXd A = P_aug.llt().matrixL();
    
    for(int i=0;i<n_aug_;i++){
        Xsig_aug.col(i+1) = x_aug + A.col(i) * sqrt(lambda_+n_aug_);
        Xsig_aug.col(n_aug_+i+1) = x_aug - A.col(i) * sqrt(lambda_+n_aug_);
    }

    //print result
    //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
    
    //write result
    *Xsig_out = Xsig_aug;
    
    
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug,double delta_t) {
    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);
        
        //predicted state values
        double px_p, py_p;
        
        
        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;
        
        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }


    //print result
    //std::cout << "Xsig_pred = " << std::endl << Xsig_pred_ << std::endl;
    
    
}

void UKF::PredictMeanAndCovariance() {
    
    //create vector for weights
    VectorXd weights = VectorXd::Zero(2*n_aug_+1);

    
    // set weights
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
        double weight = 0.5/(n_aug_+lambda_);
        weights(i) = weight;
    }
    
    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_+ weights(i) * Xsig_pred_.col(i);
    }
    
    
    P_.fill(0.0);
    //predicted state covariance matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        P_ = P_ + weights(i) * x_diff * x_diff.transpose() ;
    }
    
    //print result
    //std::cout << "Predicted state" << std::endl;
    //std::cout << x_ << std::endl;
    //std::cout << "Predicted covariance matrix" << std::endl;
    //std::cout << P_ << std::endl;
    
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) {
    
    int n_z = 3;
    
    //set vector for weights
    VectorXd weights = VectorXd::Zero(2*n_aug_+1);
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {
        double weight = 0.5/(n_aug_+lambda_);
        weights(i) = weight;
    }
    
    //std::cout << "Zsig: " << std::endl << Zsig << std::endl;
    //std::cout << "Xsig_pred_: " << std::endl << Xsig_pred_ << std::endl;
    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
    
        
        
        float rho = sqrt( p_x*p_x+p_y*p_y );
        float phi  = 0;
        float rho_dot = 0;
        
        //check division by zero
        
        // avoid division by zero
        if(fabs(p_x) < 0.0001){
            cout << "Error while converting vector x_ to polar coordinates: Division by Zero" << endl;
        }else{
            phi = atan2(p_y,p_x);
        }
        
        
        if (rho < 0.0001) {
            cout << "Error while converting vector x_ to polar coordinates: Division by Zero" << endl;
        }else{
            rho_dot = (p_x*v1 + p_y*v2) / rho;
        }
        

        
        // measurement model
        Zsig(0,i) = rho;
        Zsig(1,i) = phi;
        Zsig(2,i) = rho_dot;
        
       
        
    }


    
    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        S = S + weights(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd::Zero(n_z,n_z);
    R(0,0) = std_radr_*std_radr_;
    R(1,1) = std_radphi_*std_radphi_;
    R(2,2) = std_radrd_*std_radrd_;

    S = S + R;
    
    
    //print result
    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;
    
    //write result
    *z_out = z_pred;
    *S_out = S;
}
