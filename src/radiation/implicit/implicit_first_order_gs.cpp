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
//! \file rad_transport.cpp
//  \brief implementation of radiation integrators
//======================================================================================


// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../radiation.hpp"
#include "../../coordinates/coordinates.hpp" //
#include "../../reconstruct/reconstruction.hpp"


// class header
#include "../integrators/rad_integrators.hpp"



// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


// the flux is 
// Smax F(U_L)/(Smax-Smin) - Smin F(U_R) /(Smax-Smin)
// + Smax Smin * (U_R - U_L)/(Smax - Smin)
// Gauss-Seidel iteration, use upper triangle from last iteration
// lower triangle is from next iteration

void RadIntegrator::FirstOrderGSFluxDivergence(const Real wght, 
                                       AthenaArray<Real> &ir)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco= pmb->pcoord;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int nang = prad->nang; 
  int nfreq = prad->nfreq;
  
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
                 &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
                 &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

  AthenaArray<Real> &x1flux=prad->flux[X1DIR];
  AthenaArray<Real> &x2flux=prad->flux[X2DIR];
  AthenaArray<Real> &x3flux=prad->flux[X3DIR];


  AthenaArray<Real> &area_zeta = zeta_area_, &area_psi = psi_area_, 
                 &ang_vol = ang_vol_, &dflx_ang = dflx_ang_;
  int &nzeta = prad->nzeta, &npsi = prad->npsi;

  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) {
      pco->CenterWidth1(k,j,is-1,ie+1,dxw1_);
      for(int i=is; i<=ie+1; ++i){
        Real taul = 0.0;
        Real taur = 0.0;
        for(int ifr=0; ifr<nfreq; ++ifr){
          Real sigmal = prad->sigma_a(k,j,i-1,ifr) + prad->sigma_s(k,j,i-1,ifr);
          Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
          taul += prad->wfreq(ifr) * dxw1_(i-1) * sigmal;
          taur += prad->wfreq(ifr) * dxw1_(i) * sigmar;
        }// end ifr
        Real f_l = 1.0;
        Real f_r = 1.0;
        taul *= taufact(k,j,i-1);
        taur *= taufact(k,j,i);
        GetTaufactor(taul+taur,f_r,1);
        GetTaufactor(taul+taur,f_l,-1);

        Real *s1n = &(sfac1_x_(i,0));
        Real *s2n = &(sfac2_x_(i,0));
        Real *velxn = &(velx_(k,j,i,0));
        Real adv =  adv_vel(0,k,j,i);

        SignalSpeed(adv, f_l, f_r, velxn, s1n, s2n);

      }// end i
      // calculate x1-flux divergence 
      pco->Face1Area(k,j,is,ie+1,x1area);
      for(int i=is; i<=ie; ++i){
        Real areal = x1area(i);
        Real arear = x1area(i+1);
        Real *smax_l = &(sfac1_x_(i,0));
        Real *smax_r = &(sfac1_x_(i+1,0)); // the minimum signal speed
        Real *smin_l = &(sfac2_x_(i,0));  
        Real *smin_r = &(sfac2_x_(i+1,0)); // the maximum signal speed
        Real *vel_ln = &(velx_(k,j,i,0));
        Real *vel_rn = &(velx_(k,j,i+1,0));        

        Real *coef1n = &(const_coef_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        Real *irln = &(ir(k,j,i-1,0));
        Real *irrn = &(ir(k,j,i+1,0));
        Real *divn = &(divflx_(k,j,i,0));
        Real advl = 0.0;
        Real advr = 0.0;
        if(adv_flag_ > 0){
          advl = adv_vel(0,k,j,i);
          advr = adv_vel(0,k,j,i+1);
        }
        Real *sm_diff1 = &(sm_diff1_(0));
        Real *sm_diff2 = &(sm_diff2_(0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real diff = smax_l[n] - smin_l[n];
          if(std::fabs(diff) < TINY_NUMBER)
            sm_diff1[n] = 0.0;
          else
            sm_diff1[n] = 1.0/diff;          
        }
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real diff = smax_r[n] - smin_r[n];
          if(std::fabs(diff) < TINY_NUMBER)
            sm_diff2[n] = 0.0;
          else
            sm_diff2[n] = 1.0/diff;          
        }
        // first, consider left hand side
        if(i==is){
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vl = vel_ln[n] - advl;
            Real lflux = smax_l[n] * (vl - smin_l[n]) * sm_diff1[n] * irln[n];

            lcoefn[n] = 0.0;
            coef1n[n] = -areal * smin_l[n] * (smax_l[n] - vl) * sm_diff1[n];  
            divn[n] = areal * lflux;          
          }// end n
        }else{
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vl = vel_ln[n] - advl;
            lcoefn[n] = -areal * smax_l[n] * (vl - smin_l[n])*sm_diff1[n];
            coef1n[n] = -areal * smin_l[n] * (smax_l[n] - vl)*sm_diff1[n];  
            divn[n] = 0.0;          
          }// end n          
        }

        // now consider the right hand side, which is always from last iteration
        for(int n=0; n<prad->n_fre_ang; ++n){
          Real vr = vel_rn[n] - advr;
          Real rflux = smin_r[n] * (smax_r[n] - vr) * irrn[n] * sm_diff2[n];
          coef1n[n] += arear * smax_r[n] * (vr - smin_r[n])*sm_diff2[n];
          divn[n] += -(arear * rflux);
        }// end n
        if(adv_flag_ > 0){
          Real *flxr = &(x1flux(k,j,i+1,0));
          Real *flxl = &(x1flux(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            divn[n] += -(arear * flxr[n] - areal * flxl[n]);
          }        
        }// end adv_flag

      }// End i

    }// end j
  }// end k

     // calculate x2-flux
  if (pmb->block_size.nx2 > 1) {
    for(int k=ks; k<=ke; ++k){
      // first, calculate speed
      for(int j=js; j<=je+1; ++j){
        pco->CenterWidth2(k,j-1,is,ie,dxw1_);
        pco->CenterWidth2(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i){
          Real taul = 0.0;
          Real taur = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k,j-1,i,ifr) + prad->sigma_s(k,j-1,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            taul += prad->wfreq(ifr) * dxw1_(i) * sigmal;
            taur += prad->wfreq(ifr) * dxw2_(i) * sigmar;
          }

          Real f_l = 1.0;
          Real f_r = 1.0;
          taul *= taufact(k,j-1,i);
          taur *= taufact(k,j,i); 
          GetTaufactor(taul+taur,f_r,1);
          GetTaufactor(taul+taur,f_l,-1);

          Real *s1n = &(sfac1_y_(j,i,0));
          Real *s2n = &(sfac2_y_(j,i,0));
          Real *velyn = &(vely_(k,j,i,0));
          Real adv =  adv_vel(1,k,j,i);

          SignalSpeed(adv, f_l, f_r, velyn, s1n, s2n);

        }// end i
      }// end j

      for(int j=js; j<=je; ++j){

        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);

        for(int i=is; i<=ie; ++i){
          Real *vel_ln = &(vely_(k,j,i,0));
          Real *vel_rn = &(vely_(k,j+1,i,0));
          Real *coef2n = &(const_coef_(k,j,i,0));
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          Real *smax_l = &(sfac1_y_(j,i,0));
          Real *smax_r = &(sfac1_y_(j+1,i,0));
          Real *smin_l = &(sfac2_y_(j,i,0));
          Real *smin_r = &(sfac2_y_(j+1,i,0));

          Real *irln = &(ir(k,j-1,i,0));
          Real *irrn = &(ir(k,j+1,i,0));
          Real areal = x2area(i);
          Real arear = x2area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));
          Real advl = 0.0;
          Real advr = 0.0;
          if(adv_flag_ > 0){
            advl = adv_vel(1,k,j,i);
            advr = adv_vel(1,k,j+1,i);
          }
          Real *sm_diff1 = &(sm_diff1_(0));
          Real *sm_diff2 = &(sm_diff2_(0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real diff = smax_l[n] - smin_l[n];
            if(std::fabs(diff) < TINY_NUMBER)
              sm_diff1[n] = 0.0;
            else
              sm_diff1[n] = 1.0/diff;          
          }
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real diff = smax_r[n] - smin_r[n];
            if(std::fabs(diff) < TINY_NUMBER)
              sm_diff2[n] = 0.0;
            else
              sm_diff2[n] = 1.0/diff;          
          }
          // the left hand side
          if(j==js){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real vl = vel_ln[n] - advl;
              Real lflux = smax_l[n] * (vl - smin_l[n]) * sm_diff1[n] * irln[n];
              lcoefn[n] = 0.0;
              coef2n[n] += -areal * smin_l[n] * (smax_l[n] - vl)*sm_diff1[n];  
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real vl = vel_ln[n] - advl;
              lcoefn[n] = -areal * smax_l[n] * (vl - smin_l[n])*sm_diff1[n];
              coef2n[n] += -areal * smin_l[n] * (smax_l[n] - vl)*sm_diff1[n];  
//              divn[n] += 0.0;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vr = vel_rn[n] - advr;
            Real rflux =  smin_r[n] * (smax_r[n] - vr) * sm_diff2[n] * irrn[n];
            coef2n[n] += arear * smax_r[n] * (vr - smin_r[n])*sm_diff2[n];
            divn[n] += -(arear * rflux);  

          }// end n
          if(adv_flag_ > 0){
            Real *flxr = &(x2flux(k,j+1,i,0));
            Real *flxl = &(x2flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx2


  // calculate x3-flux divergence
  if (pmb->block_size.nx3 > 1) {
    for(int k=ks; k<=ke+1; ++k){
      for(int j=js; j<=je; ++j){
        pco->CenterWidth3(k-1,j,is,ie,dxw1_);
        pco->CenterWidth3(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i){
          Real taul = 0.0;
          Real taur = 0.0;
          for(int ifr=0; ifr<nfreq; ++ifr){
            Real sigmal = prad->sigma_a(k-1,j,i,ifr) + prad->sigma_s(k-1,j,i,ifr);
            Real sigmar = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
            taul += prad->wfreq(ifr) * dxw1_(i) * sigmal;
            taur += prad->wfreq(ifr) * dxw2_(i) * sigmar;
          }

          Real f_l = 1.0;
          Real f_r = 1.0;
          taul *= taufact(k-1,j,i);
          taur *= taufact(k,j,i);
          GetTaufactor(taul+taur,f_r,1);
          GetTaufactor(taul+taur,f_l,-1);

          Real *s1n = &(sfac1_z_(k,j,i,0));
          Real *s2n = &(sfac2_z_(k,j,i,0));
          Real *velzn = &(velz_(k,j,i,0));
          Real adv =  adv_vel(2,k,j,i);

          SignalSpeed(adv, f_l, f_r, velzn, s1n, s2n);

          
        }// end i
      }// end j
    }// end k
    for(int k=ks; k<=ke; ++k){
      for(int j=js; j<=je; ++j){
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
        for(int i=is; i<=ie; ++i){
          Real *smax_l = &(sfac1_z_(k,j,i,0));
          Real *smax_r = &(sfac1_z_(k+1,j,i,0));
          Real *smin_l = &(sfac2_z_(k,j,i,0));
          Real *smin_r = &(sfac2_z_(k+1,j,i,0));
          Real *vel_ln = &(velz_(k,j,i,0));
          Real *vel_rn = &(velz_(k+1,j,i,0));
          Real *coef3n = &(const_coef_(k,j,i,0));
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          Real *irln = &(ir(k-1,j,i,0));
          Real *irrn = &(ir(k+1,j,i,0));
          Real areal = x3area(i);
          Real arear = x3area_p1(i);
          Real *divn = &(divflx_(k,j,i,0));
          Real advl = 0.0;
          Real advr = 0.0;
          if(adv_flag_ > 0){
            advl = adv_vel(2,k,j,i);
            advr = adv_vel(2,k+1,j,i);
          }
          Real *sm_diff1 = &(sm_diff1_(0));
          Real *sm_diff2 = &(sm_diff2_(0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real diff = smax_l[n] - smin_l[n];
            if(std::fabs(diff) < TINY_NUMBER)
              sm_diff1[n] = 0.0;
            else
              sm_diff1[n] = 1.0/diff;          
          }
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real diff = smax_r[n] - smin_r[n];
            if(std::fabs(diff) < TINY_NUMBER)
              sm_diff2[n] = 0.0;
            else
              sm_diff2[n] = 1.0/diff;          
          }
          // the left hand side
          if(k==ks){
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real vl = vel_ln[n] - advl;
              Real lflux = smax_l[n] * (vl - smin_l[n]) * sm_diff1[n] * irln[n];
              lcoefn[n] = 0.0;
              coef3n[n] += -areal * smin_l[n] * (smax_l[n] - vl)*sm_diff1[n];  
              divn[n] += areal * lflux;          
            }// end n
          }else{
            for(int n=0; n<prad->n_fre_ang; ++n){
              Real vl = vel_ln[n] - advl;
              lcoefn[n] = -areal * smax_l[n] * (vl - smin_l[n])*sm_diff1[n];
              coef3n[n] += -areal * smin_l[n] * (smax_l[n] - vl)*sm_diff1[n]; 
//              divn[n] += 0.0;          
            }// end n          
          }
          // the right hand side
          for(int n=0; n<prad->n_fre_ang; ++n){
            Real vr = vel_rn[n] - advr;
            Real rflux = smin_r[n] * (smax_r[n] - vr) * sm_diff2[n] * irrn[n];
            coef3n[n] += arear * smax_r[n] * (vr - smin_r[n])*sm_diff2[n];
            divn[n] += -(arear * rflux);  

          }// end n  
          if(adv_flag_ > 0){
            Real *flxr = &(x3flux(k+1,j,i,0));
            Real *flxl = &(x3flux(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; ++n){
              divn[n] += -(arear * flxr[n] - areal * flxl[n]);
            }// end n        
          }// end adv_flag

        }// end i
      }// end j
    }// end k
  }// end nx3

  for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
      for(int i=is; i<=ie; ++i){
        Real *divn = &(divflx_(k,j,i,0));
        Real dtvol = wght/vol(i);
        for(int n=0; n<prad->n_fre_ang; ++n){
          divn[n] *= dtvol;
        }
        Real *coef1n = &(const_coef_(k,j,i,0));
        Real *lcoefn = &(left_coef1_(k,j,i,0));
        for(int n=0; n<prad->n_fre_ang; ++n){
          coef1n[n] *= dtvol;
          lcoefn[n] *= dtvol;
        }
        if(pmb->block_size.nx2 > 1){
          Real *lcoefn = &(left_coef2_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            lcoefn[n] *= dtvol;
          }          
        }// end nx2 > 1
        if(pmb->block_size.nx3 > 1){
          Real *lcoefn = &(left_coef3_(k,j,i,0));
          for(int n=0; n<prad->n_fre_ang; ++n){
            lcoefn[n] *= dtvol;
          }          
        }// end nx3 > 1

      }// end i
    }// end j
  }// end k


}// end function
