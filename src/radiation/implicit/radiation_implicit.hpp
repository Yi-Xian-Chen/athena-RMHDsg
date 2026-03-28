#ifndef IMRADIATION_HPP_
#define IMRADIATION_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================


// C++ headers
#include <cstdint>     // int64_t
#include <functional>  // reference_wrapper
#include <string>
#include <vector>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../task_list/im_rad_task_list.hpp"



class Mesh;
class ParameterInput;
class Radiation;
class RadIntegrator;
class TimeIntegratorTaskList;

class IMRadiation {
  friend class Radiation;
  friend class RadIntegrator;
public:
  IMRadiation(Mesh *pm, ParameterInput *pin);
//  ~Radiation();

  void Iteration(Mesh *pm, 
             TimeIntegratorTaskList *ptlist, int stage);


  void CheckResidual(MeshBlock *pmb,  
        AthenaArray<Real> &ir_old, AthenaArray<Real> &ir_new);

  Real cfl_rad; // the additional CFL number. 
                // Small cfl_rad is good for convergence, 
                // but with smaller time step

  int ite_scheme;

  IMRadBDTaskList *pimradbdlist;
  IMRadHydroTaskList *pimradhylist;

private:

  Real sum_diff_;
  Real sum_full_;
  int nlimit_;       // threadhold for the number of iterations
  Real error_limit_; // 

  

};

#endif // IMRADIATION_HPP_
