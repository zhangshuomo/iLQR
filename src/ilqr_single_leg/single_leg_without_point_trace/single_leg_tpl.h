#ifndef SINGLE_LEG_TPL
#define SINGLE_LEG_TPL
#include"../headers/pinocchio.h"

template<typename T>
Eigen::Matrix<T,3,3> rpy2rotmat(const Eigen::Matrix<T,3,1>& rpy){
    Eigen::Matrix<T,3,3> rotmat, rotZ, rotY, rotX;
    T roll=rpy[0], pitch=rpy[1], yaw=rpy[2];
    
    rotZ<<cos(yaw), -sin(yaw), T(0),
          sin(yaw), cos(yaw), T(0),
          T(0),T(0),T(1);
    rotY<<cos(pitch), T(0), sin(pitch),
          T(0), T(1), T(0),
          -sin(pitch),T(0),cos(pitch);
    rotX<<T(1),T(0),T(0),
          T(0),cos(roll),-sin(roll),
          T(0),sin(roll),cos(roll);
    rotmat=rotZ*rotY*rotX;
    return rotmat;
}

template<typename T>
Eigen::Matrix<T,4,1> rpy2qua(const Eigen::Matrix<T,3,1>& rpy){
    T roll=rpy[0], pitch=rpy[1], yaw=rpy[2];
    Eigen::Matrix<T,4,1> qua;
    T w=cos(yaw/2)*cos(pitch/2)*cos(roll/2)+sin(yaw/2)*sin(pitch/2)*sin(roll/2);
    T x=cos(yaw/2)*cos(pitch/2)*sin(roll/2)-sin(yaw/2)*sin(pitch/2)*cos(roll/2);
    T y=cos(yaw/2)*sin(pitch/2)*cos(roll/2)+sin(yaw/2)*cos(pitch/2)*sin(roll/2);
    T z=sin(yaw/2)*cos(pitch/2)*cos(roll/2)-cos(yaw/2)*sin(pitch/2)*sin(roll/2);
    qua<<x,y,z,w;
    return qua;
}

template<typename T>
Eigen::Matrix<T,3,1> euler_angluar_vel(const Eigen::Matrix<T,3,1>& rpy,const Eigen::Matrix<T,3,1>& omega){
    Eigen::Matrix<T,3,1> euler_vel;
    Eigen::Matrix<T,3,3> transform_mat;
    T roll=rpy[0], pitch=rpy[1], yaw=rpy[2];
    yaw=T(0.0); // TODO: this constrains the yaw to be zero and should be removed later!
    transform_mat<<cos(yaw)/cos(pitch), sin(yaw)/cos(pitch), 0,
                   -sin(yaw), cos(yaw), 0,
                   cos(yaw)*tan(pitch), sin(yaw)*tan(pitch), 1;
    euler_vel = transform_mat * omega;
    return euler_vel;
}

template<typename T>
class SingleLegTpl{
public:
    typedef Eigen::Matrix<T,9,1> JointPos;
    typedef Eigen::Matrix<T,8,1> JointVel;
    typedef Eigen::Matrix<T,8,1> JointAcc;
    typedef Eigen::Matrix<T,3,1> ForceVec;
    typedef Eigen::Matrix<T,3,1> ContactPos;

    SingleLegTpl(bool use_contact_model,
                 double k, 
                 double d,
                 double alpha_n,
                 double alpha_d);
    Eigen::Matrix<T,16,1> sys_continous_dyn(const Eigen::Matrix<T,16,1>& state, 
                                            const Eigen::Matrix<T,2,1>& control);

    ContactPos get_foot_pos(const Eigen::Matrix<T,16,1>& state);
    JointPos get_joint_pos_based_on_state(const Eigen::Matrix<T,16,1>& state);
    JointVel get_joint_vel_based_on_state(const Eigen::Matrix<T,16,1>& state);
    JointAcc single_leg_dynamics(const JointPos& q, 
                                 const JointVel& dq, 
                                 const Eigen::Matrix<T,2,1>& tau);
    ForceVec compute_grf(const JointPos& q, 
                         const JointVel& dq,
                         int ee_index);
private:
    pin::ModelTpl<T> model;
    pin::DataTpl<T> data;

    bool use_contact_model;
    double k, d;
    double alpha_n, alpha_d;

    std::string ee[1] = {"foot"};
};

template<typename T>
SingleLegTpl<T>::SingleLegTpl(bool use_contact_model, double k, double d, double alpha_n, double alpha_d){
    pin::Model _model;
    pin::urdf::buildModel("/home/zhangduo/test_ws/src/ilqr_single_leg/urdf/single_leg_jumping.urdf",_model);
    model=_model.cast<T>();
    data=pin::DataTpl<T>(model);

    this->use_contact_model=use_contact_model;
    this->k=k;
    this->d=d;
    
    this->alpha_n=alpha_n;
    this->alpha_d=alpha_d;
}

template<typename T>
Eigen::Matrix<T,3,1> 
SingleLegTpl<T>::get_foot_pos(const Eigen::Matrix<T,16,1>& state){
    auto frame_id=model.getFrameId(ee[0]);
    auto q=get_joint_pos_based_on_state(state);

    pin::computeJointJacobians(model, data, q);    
    pin::framesForwardKinematics(model, data, q);
    
    ContactPos p = data.oMf[frame_id].translation();      // the foot position in WF
    return p;
}

template<typename T>
Eigen::Matrix<T,16,1> 
SingleLegTpl<T>::sys_continous_dyn(const Eigen::Matrix<T,16,1>& state, 
                                   const Eigen::Matrix<T,2,1>& control){
    auto base_rpy=state.segment(3,3);
    auto wRb=rpy2rotmat<T>(base_rpy);

    auto base_lin_vel_BF=state.segment(8,3);
    auto base_ang_vel_BF=state.segment(11,3);
    auto joint_vels=state.segment(14,2);

    JointPos q=get_joint_pos_based_on_state(state);
    JointVel qd=get_joint_vel_based_on_state(state);
    JointAcc qdd=single_leg_dynamics(q, qd, control);

    Eigen::Matrix<T,16,1> state_dot;    state_dot.setZero();
    state_dot.head(3) = wRb * base_lin_vel_BF;    // base linear vel in WF
    state_dot.segment(3,3)=euler_angluar_vel<T>(base_rpy, wRb * base_ang_vel_BF);   // base euler vel
    state_dot.segment(6,2)=joint_vels;    // joint angular vel
    
    state_dot.segment(8,3)=qdd.head(3);   // base linear acc in BF
    state_dot.segment(11,3)=qdd.segment(3,3); // base angular acc in BF
    state_dot.segment(14,2)=qdd.segment(6,2);   // joint acc

    return state_dot;
}

template<typename T>
Eigen::Matrix<T,9,1>
SingleLegTpl<T>::get_joint_pos_based_on_state(const Eigen::Matrix<T,16,1>& state){
    auto base_lin_pos=state.head(3);
    auto base_rpy=state.segment(3,3);
    auto joint_angles=state.segment(6,2);
    auto qua=rpy2qua<T>(base_rpy);

    JointPos q;
    q.head(3)=base_lin_pos;
    q.segment(3,4)=qua;
    q.segment(7,2)=joint_angles;

    return q;
}

template<typename T>
Eigen::Matrix<T,8,1>
SingleLegTpl<T>::get_joint_vel_based_on_state(const Eigen::Matrix<T,16,1>& state){
    auto base_lin_vel_BF=state.segment(8,3);
    auto base_ang_vel_BF=state.segment(11,3);
    auto joint_vels=state.segment(14,2);

    JointVel dq;
    dq.head(3)=base_lin_vel_BF;
    dq.segment(3,3)=base_ang_vel_BF;
    dq.segment(6,2)=joint_vels;

    return dq;
}

template<typename T>
Eigen::Matrix<T,8,1> 
SingleLegTpl<T>::single_leg_dynamics(const JointPos& q, 
                                     const JointVel& dq, 
                                     const Eigen::Matrix<T,2,1>& tau){
    Eigen::Matrix<T,3,8> Jc;    Jc.setZero(); 
    Eigen::Matrix<T,3,1> lambda; lambda.setZero();
    pin::computeJointJacobians(model, data, q);    
    pin::framesForwardKinematics(model, data, q);

    Eigen::Matrix<T,8,1> aug_tau;  aug_tau.setZero();
    aug_tau.segment(6,2)=tau;

    if(use_contact_model){
        // compute foot jacobian
        Eigen::Matrix<T,6,8> J;     J.setZero();
        auto foot_id=model.getFrameId(ee[0]);
        pin::getFrameJacobian(model, data, foot_id, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
        Jc = J.block(0,0,3,8); // get the linear part

        // compute the grfs for all feet
        auto grf = compute_grf(q, dq, 0);  // grf represented in world frame
        lambda.segment(0,3) = grf;
        aug_tau+=Jc.transpose()*lambda;
    }

    JointAcc qdd=pin::aba(model,data,q,dq,aug_tau);    // qdd=M.inverse()*(aug_tau - nle);
    return qdd;
}

template<typename T>
Eigen::Matrix<T,3,1>
SingleLegTpl<T>::compute_grf(const JointPos& q, 
                             const JointVel& dq, 
                             int ee_index){
    ForceVec grf;   grf.setZero();
    auto frame_id=model.getFrameId(ee[ee_index]);
    pin::forwardKinematics(model, data, q, dq);
    pin::updateFramePlacements(model, data);
    auto foot_vel=pin::getFrameVelocity(model,data,frame_id,pin::ReferenceFrame::LOCAL_WORLD_ALIGNED);

    Eigen::Matrix<T,3,1> p = data.oMf[frame_id].translation();      // the foot position in WF
    Eigen::Matrix<T,3,1> dp=foot_vel.linear();                      // the foot linear vel in WF
    
    // support force
    T penetrate = -p[2];  // greater than 0 when penetrate.
    grf[2] += k * (penetrate - alpha_n/2) * (penetrate > alpha_n);
    grf[2] += k / (2*alpha_n) * (penetrate * penetrate) * (penetrate <= alpha_n) * (penetrate > 0);

    // horizontal force
    for(int i = 0; i < 3; ++i){
        // grf[i] -= d * (1.0 / (1 + exp(alpha_d * (-penetrate)))) * dp[i];
        grf[i] -= d * dp[i] * (penetrate >= 0);
    }

    return grf;
}

#endif