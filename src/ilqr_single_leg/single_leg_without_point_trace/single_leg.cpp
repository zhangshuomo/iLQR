#include"single_leg.h"

SingleLeg::SingleLeg(bool use_contact_model, double k, double d, double alpha_n, double alpha_d, double step_size):single_leg_dynamics(use_contact_model,k, d, alpha_n, alpha_d),step_size(step_size){
    initialize();   // create the cas::function obj for continuous dynamics and derivative matrices A and B.
    system_discretization();
}

void SingleLeg::initialize(){
    cas::SX cur_state=cas::SX::sym("cur_state",16);
    cas::SX cur_input=cas::SX::sym("cur_input",2);

    auto dstate_=single_leg_dynamics.sys_continous_dyn(cas_to_eig(cur_state),cas_to_eig(cur_input));
    cas::SX dstate=eig_to_cas(dstate_);
    cas::SX Amat_=jacobian(dstate, cur_state);
    cas::SX Bmat_=jacobian(dstate, cur_input);
    dyn = cas::Function("dyn",{cur_state, cur_input},{dstate});
    devA = cas::Function("devA",{cur_state, cur_input},{Amat_});
    devB = cas::Function("devB",{cur_state, cur_input},{Bmat_});
}

void SingleLeg::system_discretization(){
    // state simulate using rk4
    cas::SX state=cas::SX::sym("state", 16);
    cas::SX input=cas::SX::sym("input", 2);

    auto k1=dyn(cas::SXIList{state, input})[0];
    auto k2=dyn(cas::SXIList{state+0.5*step_size*k1,input})[0];
    auto k3=dyn(cas::SXIList{state+0.5*step_size*k2,input})[0];
    auto k4=dyn(cas::SXIList{state+step_size*k3,input})[0];
    auto next_state = state+1.0/6*step_size*(k1+2*k2+2*k3+k4);
    sys_dyn=cas::Function("sys_dyn",{state, input},{next_state});

    auto kG1=devA(cas::SXIList{state,input})[0];
    auto kG2=mtimes(devA(cas::SXIList{state+0.5*step_size*k1,input})[0],cas::SX::eye(16)+0.5*kG1*step_size);
    auto kG3=mtimes(devA(cas::SXIList{state+0.5*step_size*k2,input})[0],cas::SX::eye(16)+0.5*kG2*step_size);
    auto kG4=mtimes(devA(cas::SXIList{state+step_size*k3,input})[0],cas::SX::eye(16)+kG3*step_size);
    auto G=cas::SX::eye(16)+1.0/6*step_size*(kG1+2*kG2+2*kG3+kG4);
    A_sens=cas::Function("A_sens",{state, input},{G});

    auto kH1=devB(cas::SXIList{state,input})[0];
    auto kH2=mtimes(devA(cas::SXIList{state+0.5*step_size*k1,input})[0],0.5*step_size*kH1)+devB(cas::SXIList{state+0.5*step_size*k1,input})[0];
    auto kH3=mtimes(devA(cas::SXIList{state+0.5*step_size*k2,input})[0],0.5*step_size*kH2)+devB(cas::SXIList{state+0.5*step_size*k2,input})[0];
    auto kH4=mtimes(devA(cas::SXIList{state+step_size*k3,input})[0],step_size*kH3)+devB(cas::SXIList{state+step_size*k3,input})[0];
    auto H=1.0/6*step_size*(kH1+2*kH2+2*kH3+kH4);
    B_sens=cas::Function("B_sens",{state, input},{H});
}

State SingleLeg::forward_dyn(const State& cur_state, const Control& tau, bool use_gen_code){
    State next_state;
    cas::DM cur_state_dm(16,1);
    cas::DM tau_dm(2,1);      

    for(int i=0;i<16;++i)
        cur_state_dm(i,0)=cur_state[i];
    for(int i=0;i<2;++i)
        tau_dm(i,0)=tau[i];
    
    cas::DMVector args={cur_state_dm, tau_dm};
    auto res1=sys_dyn(args)[0].get_elements();
    for(int i=0;i<16;++i){
        next_state[i]=res1[i];
    }
    return next_state;
}

Eigen::MatrixXd SingleLeg::sensitivity_for_state(const State& cur_state,const Control& tau, bool use_gen_code){
    Eigen::Matrix<double,16,16> res;
    cas::DM cur_state_dm(16,1);
    cas::DM tau_dm(2,1);

    for(int i=0;i<16;++i)
        cur_state_dm(i,0)=cur_state[i];
    for(int i=0;i<2;++i)
        tau_dm(i,0)=tau[i];

    cas::DMVector args={cur_state_dm,tau_dm};
    auto res_vec=A_sens(args)[0].get_elements();
    for(int j=0;j<16;++j)
        for(int i=0;i<16;++i)
            res(i,j)=res_vec[16*j+i];
    return res;
}

Eigen::MatrixXd SingleLeg::sensitivity_for_control(const State& cur_state,const Control& tau, bool use_gen_code){
    Eigen::Matrix<double,16,2> res;
    cas::DM cur_state_dm(16,1);
    cas::DM tau_dm(2,1);

    for(int i=0;i<16;++i)
        cur_state_dm(i,0)=cur_state[i];
    for(int i=0;i<2;++i)
        tau_dm(i,0)=tau[i];
    
    cas::DMVector args={cur_state_dm,tau_dm};
    auto res_vec=B_sens(args)[0].get_elements();
    for(int j=0;j<2;++j)
        for(int i=0;i<16;++i)
            res(i,j)=res_vec[16*j+i];
    return res;
}

void get_initial_state(State& initial_state){
    initial_state.head(3)<<0.0,0.0,0.52;
    initial_state.segment<3>(3)<<0.0, 0.0, 0.0;
    initial_state.segment<2>(6)<<0.67,-1.3;

    initial_state.segment<3>(8)<<0.0,0.0,0.0;
    initial_state.segment<3>(11)<<0.0,0.0,0.0;
    initial_state.tail(2)<<0.0,0.0;
}

