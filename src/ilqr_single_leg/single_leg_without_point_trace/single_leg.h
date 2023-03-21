#ifndef SINGLE_LEG
#define SINGLE_LEG
#include"single_leg_tpl.h"
#include<casadi/casadi.hpp>
#include<Eigen/Dense>
#include"../headers/cas_eig_transfer.h"
#include"../headers/types.h"

class SingleLeg{
public:
    SingleLeg(bool use_contact_model, double k, double d, double alpha_n, double alpha_d, double step_size=0.001);
    State forward_dyn(const State& cur_state,const Control& tau, bool use_gen_code=true);
    Eigen::MatrixXd sensitivity_for_state(const State& cur_state,const Control& tau, bool use_gen_code=true);
    Eigen::MatrixXd sensitivity_for_control(const State& cur_state,const Control& tau, bool use_gen_code=true);
    
private:
    void initialize();
    void system_discretization();

    SingleLegTpl<cas::SX> single_leg_dynamics;
    double step_size;

    cas::Function dyn;
    cas::Function devA, devB;
    
    cas::Function sys_dyn;
    cas::Function A_sens, B_sens;
};

void get_initial_state(State& initial_state);

#endif
