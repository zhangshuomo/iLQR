#include"../single_leg_without_point_trace/single_leg.h"
#include"../loss_function/loss_function.h"
#include"utils/read_param.h"
#include"utils/file_operate.h"

void iLQR(const State &initial_state, 
          std::vector<Eigen::MatrixXd> &feedback_res_gains, 
          std::vector<Eigen::VectorXd> &feedforward_res_controls){
    RobotParams robot_params;
    read_param(robot_params);
    
    int time_steps = robot_params.total_time / robot_params.step_length;
    std::vector<Eigen::VectorXd> target_state_traj(time_steps, robot_params.target_intermidiate_state);
    target_state_traj.push_back(robot_params.target_final_state);

    SingleLeg single_leg(true,
                         robot_params.k, 
                         robot_params.d, 
                         robot_params.alpha_n, 
                         robot_params.alpha_d,
                         robot_params.step_length);
    LossFunction loss_function(robot_params.state_dim, 
                               robot_params.input_dim, 
                               robot_params.total_time, 
                               time_steps, 
                               target_state_traj,
                               robot_params.state_a,
                               robot_params.Q, 
                               robot_params.Q_a,
                               robot_params.R, 
                               robot_params.Q_final);
    
    const double alpha_decay=0.8;
    const double alpha_ending_condition=1e-4;
    const int max_total_iter=500;
    double last_loss=0;

    std::vector<Control> control_traj;
    std::vector<State> state_traj;

    std::vector<Eigen::MatrixXd> state_dev_mat, control_dev_mat;
    std::vector<Eigen::VectorXd> c_vec;
    std::vector<Eigen::MatrixXd> R_mat, Q_mat;
    std::vector<Eigen::VectorXd> q_vec, s_vec;

    std::vector<Eigen::MatrixXd> feedback_gains;
    std::vector<Eigen::VectorXd> feedforward_controls;
    
    // check before simulation
    assert(feedback_res_gains.size()>=time_steps);
    assert(feedforward_res_controls.size()>=time_steps);

    // simulate the system
    State current_state=initial_state;
    state_traj.push_back(current_state);
    for(int i=0;i<time_steps;++i){
        auto current_control=feedforward_res_controls[i]+feedback_res_gains[i]*current_state;
        auto next_state=single_leg.forward_dyn(current_state, current_control);
        last_loss+=loss_function.intermidiate_loss(current_state, current_control, i);

        control_traj.push_back(current_control);
        state_traj.push_back(next_state);

        current_state=next_state;
    }
    last_loss+=loss_function.final_loss(current_state);
    std::cout<<"initial loss:\t"<<last_loss<<std::endl;
    feedback_res_gains.clear();
    feedforward_res_controls.clear();
    
    int total_iter=0;
    while(true){
        // forward iteration
        for(int i=0;i<time_steps;++i){
            auto Ak=single_leg.sensitivity_for_state(state_traj[i], control_traj[i]);
            auto Bk=single_leg.sensitivity_for_control(state_traj[i], control_traj[i]);
            auto ck=state_traj[i+1]-Ak*state_traj[i]-Bk*control_traj[i];

            auto hess_to_state=loss_function.hessian_cost_to_state(state_traj[i], control_traj[i], i);
            auto hess_to_control=loss_function.hessian_cost_to_control(state_traj[i], control_traj[i], i);
            auto grad_to_state=loss_function.partial_dev_cost_to_state(state_traj[i], control_traj[i],i);
            auto grad_to_control=loss_function.partial_dev_cost_to_control(state_traj[i], control_traj[i],i);

            auto Qk=hess_to_state / 2;
            auto Rk=hess_to_control / 2;
            auto qk=(grad_to_state-hess_to_state*state_traj[i]) / 2;
            auto sk=(grad_to_control-hess_to_control*control_traj[i]) / 2;

            // record the state trajectory and partial derivative matrices and vectors
            state_dev_mat.push_back(Ak);
            control_dev_mat.push_back(Bk);
            c_vec.push_back(ck);

            Q_mat.push_back(Qk);
            R_mat.push_back(Rk);
            q_vec.push_back(qk);
            s_vec.push_back(sk);
        }

        // backward iteration
        auto hess_to_state=loss_function.hessian_final_cost_to_state(state_traj[time_steps]);
        auto grad_to_state=loss_function.dev_final_cost_to_state(state_traj[time_steps]);
        Eigen::MatrixXd Pk_=hess_to_state / 2;
        Eigen::VectorXd pk_=(grad_to_state-hess_to_state*state_traj[time_steps]) / 2;
        for(int i=time_steps-1;i>=0;--i){
            auto Ak=state_dev_mat[i];
            auto Bk=control_dev_mat[i];
            auto ck=c_vec[i];

            auto Qk=Q_mat[i];
            auto Rk=R_mat[i];
            auto qk=q_vec[i];
            auto sk=s_vec[i];

            auto Kk=(Rk+Bk.transpose()*Pk_*Bk).inverse()*Bk.transpose()*Pk_*Ak;
            auto kk=(Rk+Bk.transpose()*Pk_*Bk).inverse()*(sk+Bk.transpose()*pk_+Bk.transpose()*Pk_*ck);
            feedback_gains.push_back(Kk);
            feedforward_controls.push_back(kk);

            auto Pk=Qk+Ak.transpose()*Pk_*Ak-(Ak.transpose()*Pk_*Bk)*Kk;
            auto pk=qk+Ak.transpose()*pk_+Ak.transpose()*Pk_*ck-(Ak.transpose()*Pk_*Bk)*kk;
            Pk_=Pk;
            pk_=pk;
        }

        std::reverse(feedback_gains.begin(), feedback_gains.end());
        std::reverse(feedforward_controls.begin(), feedforward_controls.end());
        
        // alpha line search to get the new control for this SLQ iteration
        std::vector<State> new_state_traj;
        std::vector<Control> new_control_traj;
        double total_loss=0;
        double alpha=1;
        int inner_loop_iter=0;
        const int max_inner_loop_iter=50;
        while(true){
            current_state=initial_state;
            new_state_traj.push_back(current_state);
            try{
                for(int i=0;i<time_steps;++i){
                    /* Control current_control=control_traj[i]+(-feedback_gains[i])*(current_state-state_traj[i])+
                                            alpha*(-control_traj[i]+(-feedback_gains[i])*state_traj[i]-feedforward_controls[i]); */
                    Control feedforward=(1-alpha)*(control_traj[i]+feedback_gains[i]*state_traj[i])-alpha*feedforward_controls[i];
                    Eigen::MatrixXd feedback_gain = -feedback_gains[i];
                    Control current_control = feedforward + feedback_gain * current_state;
                    
                    State next_state=single_leg.forward_dyn(current_state, current_control);
                    total_loss+=loss_function.intermidiate_loss(current_state, current_control, i);

                    feedback_res_gains.push_back(feedback_gain);
                    feedforward_res_controls.push_back(feedforward);

                    new_control_traj.push_back(current_control);
                    new_state_traj.push_back(next_state);

                    if(std::isnan(total_loss)){
                        throw(std::runtime_error("nan encountered!"));
                    }

                    current_state=next_state;
                }
            }
            catch(std::runtime_error){
                std::cerr<<"NaN encountered!"<<std::endl;
                for(int i = 0; i < feedback_res_gains.size(); ++i){                    
                    std::cout<<"time stpes:"<<i<<"\nfeedback gains:\n"<<feedback_res_gains[i]<<std::endl;
                    std::cout<<"feedforward controls:\n"<<feedforward_res_controls[i].transpose()<<std::endl;
                    std::cout<<"current state:\n"<<new_state_traj[i].transpose()<<std::endl;                    
                    std::cout<<"current control:\n"<<new_control_traj[i].transpose()<<std::endl;
                    std::cout<<"---------------------------"<<std::endl;
                }      
                throw(std::runtime_error("nan encountered!"));       
            }
            total_loss+=loss_function.final_loss(current_state);

            if(total_loss<last_loss||inner_loop_iter==max_inner_loop_iter-1){
                state_traj=new_state_traj;
                control_traj=new_control_traj;
                last_loss=total_loss;
                if(inner_loop_iter==max_inner_loop_iter-1)
                    std::cout<<"Maximum iteration has reached, break out!"<<std::endl;
                break;
            }
            feedback_res_gains.clear();
            feedforward_res_controls.clear();
            new_state_traj.clear();
            new_control_traj.clear();
            total_loss=0;

            alpha*=alpha_decay;
            inner_loop_iter++;
        }
        
        // clear the memories
        state_dev_mat.clear();
        control_dev_mat.clear();
        c_vec.clear();
        R_mat.clear();
        Q_mat.clear();
        q_vec.clear();
        s_vec.clear();
        feedback_gains.clear();
        feedforward_controls.clear();

        // iteration ending condition
        std::cout<<"iteration: "<<total_iter<<"\tloss function: "<<last_loss<<"\tstep size: "<<alpha<<std::endl;
        if(alpha<alpha_ending_condition||total_iter>=max_total_iter)    break;
        
        total_iter++;
    }
}

int main(int argc, char *argv[])
{
    srand(time(0));
    const char* load_controller_path="/home/zhangduo/test_ws/controller_parameter";
    const char* save_controller_path="/home/zhangduo/test_ws/new_controller_parameter";
    State initial_state;
        
    std::vector<Eigen::MatrixXd> feedback_gains;
    std::vector<Eigen::VectorXd> feedforward_controls;

    get_initial_state(initial_state);
    load_controller_params(feedback_gains, feedforward_controls, load_controller_path);
    iLQR(initial_state, feedback_gains, feedforward_controls);
    save_controller_params(feedback_gains, feedforward_controls, save_controller_path);
    return 0;
}
