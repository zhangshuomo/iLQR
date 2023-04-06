#include"../single_leg_without_point_trace/single_leg.h"
#include"tf2_ros/transform_broadcaster.h"
#include"geometry_msgs/TransformStamped.h"
#include"tf2/LinearMath/Quaternion.h"
#include<sensor_msgs/JointState.h>
#include"matplotlibcpp.h"
#include"utils/read_param.h"
#include"utils/file_operate.h"
#include<ros/ros.h>
namespace plt=matplotlibcpp;

void publish_state(const ros::Publisher& pub, const State& cur_state){
    // send floating base transform
    static tf2_ros::TransformBroadcaster broadcaster;
    geometry_msgs::TransformStamped tfs;
    tfs.header.frame_id = "base";
    tfs.header.stamp = ros::Time::now();

    tfs.child_frame_id = "trunk";

    tfs.transform.translation.x = cur_state[0];
    tfs.transform.translation.y = cur_state[1];
    tfs.transform.translation.z = cur_state[2];

    tf2::Quaternion qtn;
    qtn.setRPY(cur_state[3],cur_state[4],cur_state[5]);
    tfs.transform.rotation.x = qtn.getX();
    tfs.transform.rotation.y = qtn.getY();
    tfs.transform.rotation.z = qtn.getZ();
    tfs.transform.rotation.w = qtn.getW();

    broadcaster.sendTransform(tfs);

    // send joint angles
    sensor_msgs::JointState joint_state;
    joint_state.header.stamp = ros::Time::now();
    joint_state.name.resize(2);
    joint_state.position.resize(2);
    joint_state.name[0] ="thigh_joint";
    joint_state.position[0] = cur_state[6]; 
    joint_state.name[1] ="calf_joint";
    joint_state.position[1] = cur_state[7];
    pub.publish(joint_state);
}

int main(int argc, char *argv[])
{
    srand(time(0));
    ros::init(argc,argv,"single_leg_jumping_visualize");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::JointState>("/joint_states",10);

    char* load_controller_path;
    bool simulate_initial_trajectory=ros::param::param("simulate_initial_trajectory", false);
    if(simulate_initial_trajectory)
        load_controller_path="/home/zhangduo/test_ws/controller_parameter";
    else
        load_controller_path="/home/zhangduo/test_ws/new_controller_parameter";

    RobotParams robot_params;
    read_param(robot_params);

    State cur_state; 
    get_initial_state(cur_state);

    Control input;
    
    bool contact_model=ros::param::param("contact_model", true);
    SingleLeg single_leg(contact_model, robot_params.k, robot_params.d, robot_params.alpha_n, robot_params.alpha_d, robot_params.step_length);
    SingleLegTpl<double> single_leg_tpl(contact_model, robot_params.k, robot_params.d, robot_params.alpha_n, robot_params.alpha_d);
    
    double speed_to_play=ros::param::param("speed_to_play", 1.0);

    // get controller parameters
    std::vector<Eigen::MatrixXd> feedback_gains;
    std::vector<Eigen::VectorXd> feedforward_controls;
    load_controller_params(feedback_gains, feedforward_controls, load_controller_path);
   
    // collect and save initial controls
    std::vector<Control> initial_control;

    // state recorder
    std::vector<double> pos_x,pos_y,pos_z;
    std::vector<double> roll, pitch, yaw;
    std::vector<double> thigh_joint, calf_joint;

    std::vector<double> vel_x, vel_y, vel_z;
    std::vector<double> wx, wy, wz;
    std::vector<double> thigh_vel, calf_vel;

    std::vector<double> foot_height;
    std::vector<double> grf_x, grf_y, grf_z;

    ros::Rate r(1.0 / robot_params.step_length * speed_to_play);
    int counter=0;
    usleep(3e6);    // waiting for 3 seconds

    while(ros::ok()){
        if(counter >= robot_params.total_time / robot_params.step_length)    break;
        input=feedforward_controls[counter]+feedback_gains[counter]*cur_state;
        
        auto new_state = single_leg.forward_dyn(cur_state, input);
        cur_state = new_state;
        auto foot_pos=single_leg_tpl.get_foot_pos(cur_state);

        auto q=single_leg_tpl.get_joint_pos_based_on_state(cur_state);
        auto dq=single_leg_tpl.get_joint_vel_based_on_state(cur_state);
        auto grf=single_leg_tpl.compute_grf(q,dq,0);
        // visualize in rviz
        publish_state(pub, cur_state);

        // record states
        pos_x.push_back(cur_state[0]);
        pos_y.push_back(cur_state[1]);
        pos_z.push_back(cur_state[2]);

        roll.push_back(cur_state[3]);
        pitch.push_back(cur_state[4]);
        yaw.push_back(cur_state[5]);

        thigh_joint.push_back(cur_state[6]);
        calf_joint.push_back(cur_state[7]);

        vel_x.push_back(cur_state[8]);
        vel_y.push_back(cur_state[9]);
        vel_z.push_back(cur_state[10]);

        wx.push_back(cur_state[11]);
        wy.push_back(cur_state[12]);
        wz.push_back(cur_state[13]);

        thigh_vel.push_back(cur_state[14]);
        calf_vel.push_back(cur_state[15]);

        foot_height.push_back(foot_pos[2]);
        grf_x.push_back(grf[0]);
        grf_y.push_back(grf[1]);
        grf_z.push_back(grf[2]);

        r.sleep();
        counter+=1;
    }

    plt::figure();
    plt::title("position");
    plt::named_plot("pos_x", pos_x);
    plt::named_plot("pos_y", pos_y);
    plt::named_plot("pos_z", pos_z);
    plt::legend();
    plt::grid(true);

    plt::figure();
    plt::title("orientation");
    plt::named_plot("roll", roll);
    plt::named_plot("pitch", pitch);
    plt::named_plot("yaw", yaw);
    plt::legend();
    plt::grid(true);

    plt::figure();
    plt::title("joint angles");
    plt::named_plot("thigh joint",thigh_joint);
    plt::named_plot("calf joint",calf_joint);
    plt::legend();
    plt::grid(true);

    plt::figure();
    plt::title("foot height");
    plt::named_plot("foot height", foot_height);
    plt::legend();
    plt::grid(true);

    plt::figure();
    plt::title("ground reaction forces");
    plt::named_plot("grf_x",grf_x);
    plt::named_plot("grf_y",grf_y);
    plt::named_plot("grf_z",grf_z);
    plt::legend();
    plt::grid(true);
    plt::show();
    return 0;
}
