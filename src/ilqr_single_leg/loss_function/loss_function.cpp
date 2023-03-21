#include"loss_function.h"

LossFunction::LossFunction(int state_dim, int control_dim, double whole_time, int time_steps,
                           std::vector<Eigen::VectorXd> state_traj, 
                           const Eigen::VectorXd &state_a,
                           const Eigen::MatrixXd &Q,
                           const Eigen::MatrixXd &Qa,
                           const Eigen::MatrixXd &R,
                           const Eigen::MatrixXd &Q_final){
    this->state_dim=state_dim;
    this->control_dim=control_dim;

    this->time_steps=time_steps;
    this->dT=whole_time/double(time_steps);

    assert(state_traj.size()==time_steps+1);
    for(int i=0;i<state_traj.size();++i)
        this->state_traj.push_back(state_traj[i]);

    this->state_a=state_a;

    assert(Q.rows()==Q.cols());
    assert(state_dim==Q.rows());
    this->Q=Q;
    
    this->Qa=Qa;

    assert(R.rows()==R.cols());
    assert(control_dim==R.rows());
    this->R=R;

    assert(Q.rows()==Q.cols());
    assert(Q.rows()==state_dim);
    this->Q_final=Q_final;
}

double LossFunction::intermidiate_loss(const Eigen::VectorXd& state,const Eigen::VectorXd& control, int time_step){
    auto state_diff = state - state_traj[time_step];
    Eigen::MatrixXd intermidiate=state_diff.transpose()*Q*dT*state_diff+control.transpose()*R*dT*control;
    if(time_step>2000 && time_step<2100){
        auto state_diff_a = state - state_a;
        intermidiate += state_diff_a.transpose()*Qa*dT*state_diff_a;
    }
    return intermidiate(0,0);
}

Eigen::MatrixXd LossFunction::hessian_cost_to_state(const Eigen::VectorXd& current_state,const Eigen::VectorXd& control, int time_step){
    if(time_step>2000 && time_step<2100)
        return 2*Q*dT+2*Qa*dT;
    else
        return 2*Q*dT;
}

Eigen::MatrixXd LossFunction::hessian_cost_to_control(const Eigen::VectorXd& current_state,const Eigen::VectorXd& control, int time_step){
    return 2*R*dT;
}

Eigen::VectorXd LossFunction::partial_dev_cost_to_state(const Eigen::VectorXd& current_state,const Eigen::VectorXd& control, int time_step){
    if(time_step>2000 && time_step<2100)
        return 2*Q*dT*(current_state-state_traj[time_step]) + 2*Qa*dT*(current_state-state_a);
    else
        return 2*Q*dT*(current_state-state_traj[time_step]);
}

Eigen::VectorXd LossFunction::partial_dev_cost_to_control(const Eigen::VectorXd& current_state,const Eigen::VectorXd& control, int time_step){
    return 2*R*dT*control;
}

double LossFunction::final_loss(const Eigen::VectorXd& final_state){
    auto state_diff=final_state-state_traj[time_steps];
    auto final_loss_mat=(state_diff.transpose()*Q_final*dT*state_diff);
    return final_loss_mat(0,0);
}

Eigen::MatrixXd LossFunction::hessian_final_cost_to_state(const Eigen::VectorXd& final_state){
    return 2*Q_final*dT;
}

Eigen::VectorXd LossFunction::dev_final_cost_to_state(const Eigen::VectorXd& final_state){
    return 2*Q_final*dT*(final_state-state_traj[time_steps]);
}
