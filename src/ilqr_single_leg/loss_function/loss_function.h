#ifndef LOSS_FUNCTION
#define LOSS_FUNCTION
#include<Eigen/Dense>
#include<vector>

class LossFunction{
public:
    LossFunction(int state_dim, int control_dim, double whole_time, int time_steps,
                 std::vector<Eigen::VectorXd> state_traj, 
                 const Eigen::VectorXd &state_a,
                 const Eigen::MatrixXd &Q,
                 const Eigen::MatrixXd &Qa,
                 const Eigen::MatrixXd &R,
                 const Eigen::MatrixXd &Q_final);
    
    // time_step in range [0,time_steps-1]
    double intermidiate_loss(const Eigen::VectorXd& state,const Eigen::VectorXd& control, int time_step);  
    Eigen::MatrixXd hessian_cost_to_state(const Eigen::VectorXd& current_state, const Eigen::VectorXd& control, int time_step);
    Eigen::MatrixXd hessian_cost_to_control(const Eigen::VectorXd& current_state, const Eigen::VectorXd& control, int time_step);
    Eigen::VectorXd partial_dev_cost_to_state(const Eigen::VectorXd& current_state, const Eigen::VectorXd& control, int time_step);
    Eigen::VectorXd partial_dev_cost_to_control(const Eigen::VectorXd& current_state, const Eigen::VectorXd& control, int time_step);

    double final_loss(const Eigen::VectorXd& final_state);
    Eigen::MatrixXd hessian_final_cost_to_state(const Eigen::VectorXd& final_state);
    Eigen::VectorXd dev_final_cost_to_state(const Eigen::VectorXd& final_state);
private:
    int time_steps;
    int state_dim;
    int control_dim;
    double dT;
    std::vector<Eigen::VectorXd> state_traj;
    Eigen::VectorXd state_a;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd Qa;
    Eigen::MatrixXd R;
    
    Eigen::MatrixXd Q_final;
    Eigen::MatrixXd R_final;
};
#endif