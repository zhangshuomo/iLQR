#ifndef READ_PARAM
#define READ_PARAM
#include<string>
#include<iostream>
#include<Eigen/Dense>
#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/info_parser.hpp>

struct RobotParams{
    int state_dim;
    int input_dim;
    double total_time;
    double step_length;
    double k;
    double d;
    double alpha_n;
    double alpha_d;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd Q_a;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Q_final;

    Eigen::VectorXd target_intermidiate_state;
    Eigen::VectorXd state_a;
    Eigen::VectorXd target_final_state;
};

void read_param(RobotParams& robot_params,
                const std::string& filename="/home/zhangduo/test_ws/src/ilqr_single_leg/core/robot parameters.info", 
                const std::string& ns = "single_leg"){
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(filename, pt);
    
    robot_params.state_dim=pt.get<int>(ns+".state_dim");
    robot_params.input_dim=pt.get<int>(ns+".input_dim");
    robot_params.total_time=pt.get<double>(ns+".total_time");
    robot_params.step_length=pt.get<double>(ns+".step_length");

    robot_params.Q.resize(robot_params.state_dim, robot_params.state_dim);
    robot_params.Q_a.resize(robot_params.state_dim, robot_params.state_dim);
    robot_params.Q_final.resize(robot_params.state_dim, robot_params.state_dim);
    robot_params.R.resize(robot_params.input_dim,robot_params.input_dim);

    robot_params.target_intermidiate_state.resize(robot_params.state_dim);
    robot_params.state_a.resize(robot_params.state_dim);
    robot_params.target_final_state.resize(robot_params.state_dim);

    robot_params.k=pt.get<double>(ns+".k");
    robot_params.d=pt.get<double>(ns+".d");
    robot_params.alpha_n=pt.get<double>(ns+".alpha_n");
    robot_params.alpha_d=pt.get<double>(ns+".alpha_d");

    robot_params.Q.setZero();
    robot_params.Q_a.setZero();
    robot_params.Q_final.setZero();
    robot_params.R.setZero();
    for(int i=0;i<robot_params.state_dim;++i){
        std::stringstream ss1, ss2, ss3;
        ss1<<ns<<".Q.Q"<<i+1;
        robot_params.Q.diagonal()[i]=pt.get<double>(ss1.str());
        ss2<<ns<<".Q_final.Q"<<i+1;
        robot_params.Q_final.diagonal()[i]=pt.get<double>(ss2.str());
        ss3<<ns<<".Q_a.Q"<<i+1;
        robot_params.Q_a.diagonal()[i]=pt.get<double>(ss3.str());
    }
    for(int i=0;i<robot_params.input_dim;++i){
        std::stringstream ss;
        ss<<ns<<".R.R"<<i+1;
        robot_params.R.diagonal()[i]=pt.get<double>(ss.str());
    }
    for(int i=0;i<robot_params.state_dim;++i){
        std::stringstream ss1, ss2, ss3;
        ss1<<ns<<".int_state.state"<<i+1;
        robot_params.target_intermidiate_state[i]=pt.get<double>(ss1.str());
        ss2<<ns<<".final_state.state"<<i+1;
        robot_params.target_final_state[i]=pt.get<double>(ss2.str());
        ss3<<ns<<".state_a.state"<<i+1;
        robot_params.state_a[i]=pt.get<double>(ss3.str());
    }
}
#endif