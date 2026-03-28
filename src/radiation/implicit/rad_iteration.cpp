//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file rad_iteration.cpp
//  \brief iterations to solve the transport equation implicitly
//======================================================================================


// Athena++ headers
#include <sstream>    // stringstream
#include "../radiation.hpp"
#include "../../globals.hpp"
#include "../integrators/rad_integrators.hpp"
#include "radiation_implicit.hpp"
#include "../../mesh/mesh.hpp"
#include "../../hydro/hydro.hpp"
#include "../../field/field.hpp"
#include "../../defs.hpp"
#include "../../task_list/task_list.hpp"

//--------------------------------------------------------------------------------------
// \!fn void CalculateMoment()

// \brief function to create the radiation moments


// calculate the frequency integrated moments of the radiation field
// including the ghost zones



void IMRadiation::Iteration(Mesh *pm, 
             TimeIntegratorTaskList *ptlist, int stage){
   // perform Jacobi iteration including both source and flux terms
   // The iteration step is: calculate flux, calculate source term, 
   // update specific intensity, compute error
  MeshBlock *pmb = pm->my_blocks(0);
  std::stringstream msg;

  const Real wght = ptlist->stage_wghts[stage-1].beta*pm->dt;

  if(stage <= ptlist->nstages){

     // go through all the mesh blocks
    bool iteration = true;
    int niter = 0;

     // store the initial value before the iteration
     // this is always needed for the RHS

     // first save initial state
    for(int nb=0; nb<pm->nblocal; ++nb){
      pmb = pm->my_blocks(nb);
      Radiation *prad = pmb->prad;
      Hydro *ph = pmb->phydro;
      Field *pf = pmb->pfield;

      AthenaArray<Real> &ir_ini = prad->ir1;

      // prepare t_gas and vel
      if(stage == 1){
        ir_ini = prad->ir;
      }
      // use the current stage velocity for advection
      prad->pradintegrator->GetTgasVel(pmb,wght,ph->u,ph->w,pf->bcc,ir_ini);


      // Calculate advection flux due to flow velocity explicitly
      // advection velocity uses the partially updated velocity and ir from half 
      // time step 
      if(prad->pradintegrator->adv_flag_ > 0){
        if(stage == 1)
          prad->pradintegrator->CalculateFluxes(prad->ir, 1);
        else
          prad->pradintegrator->CalculateFluxes(prad->ir, prad->pradintegrator->rad_xorder);

        // store the flux divergence due to advection
        prad->pradintegrator->FluxDivergence(wght);
      }

      // prepare the coefficients
      if(ite_scheme == 0 || ite_scheme == 2)
        prad->pradintegrator->FirstOrderFluxDivergenceCoef(wght);      

      // prepare coefficients for angular fluxes
      if(prad->angle_flag == 1){
        prad->pradintegrator->ImplicitAngularFluxesCoef(wght);  
      }

      prad->ir_old = prad->ir;

    }



    while(iteration){
       // initialize the pointer
      pmb = pm->my_blocks(0);


      sum_full_ = 0.0;
      sum_diff_ = 0.0;

       // we need to go through all meshblocks for each iteration
      for(int nb=0; nb<pm->nblocal; ++nb){
        pmb = pm->my_blocks(nb);
        Hydro *ph = pmb->phydro;
        Radiation *prad = pmb->prad;
        if(ite_scheme == 0 || ite_scheme == 2)
          prad->pradintegrator->FirstOrderFluxDivergence(prad->ir);
        else if(ite_scheme == 1)
          prad->pradintegrator->FirstOrderGSFluxDivergence(wght, prad->ir);  
        else{
          msg << "### FATAL ERROR in function [Iteration]"
          << std::endl << "ite_scheme_ '" << ite_scheme << "' not allowed!";
          ATHENA_ERROR(msg);

        }             
        // calculate the coefficients and angular part before the source term
        if(prad->angle_flag == 1){
          prad->pradintegrator->ImplicitAngularFluxes(prad->ir);  
        }

     // the source term, ir1 is the ir_ini
        // the source term will combine everything together
        prad->pradintegrator->CalSourceTerms(pmb, wght, ph->u, prad->ir1, prad->ir);


      }// end nb

      // using TaskList to handle 
      // apply boundary condition, prolongation, and check residual
      pimradbdlist->DoTaskListOneStage(wght);

      for(int nb=0; nb<pm->nblocal; ++nb){
        pmb = pm->my_blocks(nb);
        Radiation *prad = pmb->prad;
         // copy the solution over
        prad->ir_old = prad->ir;
        sum_full_ += prad->sum_full;
        sum_diff_ += prad->sum_diff;

      }// end while pmb

       // MPI sum across all the cores
#ifdef MPI_PARALLEL
      Real global_sum = 0.0;
      Real global_diff = 0.0;
      MPI_Allreduce(&sum_full_, &global_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&sum_diff_, &global_diff, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD); 

      sum_full_ = global_sum;
      sum_diff_ = global_diff;

#endif  

      niter++;
      Real tot_res = sum_diff_/sum_full_;

//      if(Globals::my_rank == 0)
//        std::cout << "Iteration : " << niter
//        << " relative error: " << tot_res << std::endl;

      if((niter > nlimit_) || tot_res < error_limit_)
        iteration = false;

    }// end iteration

    if(Globals::my_rank == 0)
      std::cout << "Iteration stops at niter: " << niter
      << " relative error: " << sum_diff_/sum_full_ << std::endl;

    // After iteration,
    // add radiation source term to hydro
    // update hydro boundary
    // update opacity
    pimradhylist->DoTaskListOneStage(wght);

  }// end stage

}// end func



 void IMRadiation::CheckResidual(MeshBlock *pmb, 
             AthenaArray<Real> &ir_old, AthenaArray<Real> &ir_new)
 {

   Radiation *prad = pmb->prad;
  
   int &nang =prad->nang;
   int &nfreq=prad->nfreq;



   int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
   int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

   prad->sum_diff = 0.0;
   prad->sum_full = 0.0;
   for(int k=ks; k<=ke; ++k){
     for(int j=js; j<=je; ++j){
       for(int i=is; i<=ie; ++i){
         for(int ifr=0; ifr<nfreq; ++ifr){
           Real *iro = &(ir_old(k,j,i,ifr*nang));
           Real *irn = &(ir_new(k,j,i,ifr*nang));
           for(int n=0; n<nang; ++n){
             prad->sum_diff += fabs(iro[n] - irn[n]);
             prad->sum_full += fabs(irn[n]);
           }
         }// end ifr
       }// end i
     }// end j
   }// end k
  
 }



