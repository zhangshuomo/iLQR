#ifndef CAS_EIG_TRANSFER
#define CAS_EIG_TRANSFER
#include<Eigen/Dense>
#include<casadi/casadi.hpp>
namespace cas=casadi;

Eigen::Matrix<cas::SX,-1, 1> cas_to_eig(const cas::SX& cas){
    Eigen::Matrix<cas::SX,-1,1> eig(cas.size1());
    for(int i = 0; i < eig.size(); i++)
        eig(i) = cas(i);
    return eig;
}

cas::SX eig_to_cas(const Eigen::Matrix<cas::SX,-1,1>& eig){
    auto sx = cas::SX(cas::Sparsity::dense(eig.size()));
    for(int i = 0; i < eig.size(); i++)
        sx(i) = eig(i);
    return sx;
}

Eigen::Matrix<cas::SX,-1,-1> cas_to_eigmat(const cas::SX& cas){
    Eigen::Matrix<cas::SX,-1,-1> eig(cas.size1(),cas.size2());
    for(int i = 0; i < cas.size1(); ++i)
        for(int j = 0;j < cas.size2(); ++j)
            eig(i,j) = cas(i,j);
    return eig;
}

cas::SX eigmat_to_cas(const Eigen::Matrix<cas::SX,-1,-1> & eig){
    auto sx = cas::SX(cas::Sparsity::dense(eig.rows(), eig.cols()));
    for(int i = 0; i < eig.rows(); i++)
        for(int j = 0; j < eig.cols(); j++)
            sx(i,j) = eig(i,j);
    return sx;
}
#endif