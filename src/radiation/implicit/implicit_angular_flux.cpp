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

#include <sstream>    // stringstream
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



// calculate advective flux for the implicit scheme
// only work for 1D spherical polar 
// The equation to solve is
// \partial /\partial t - c/r \partial(\sin^2 zeta I)/\sin \zeta\partial \zeta
// the output is stored in angflx

// calculate the coefficient for angular flux
void RadIntegrator::ImplicitAngularFluxesCoef(const Real wght)
{

  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco=pmb->pcoord;
  std::stringstream msg;

  int &nzeta = prad->nzeta;
  int &npsi = prad->npsi;

  AthenaArray<Real> &area_zeta = zeta_area_, &ang_vol = ang_vol_;

  
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for(int k=ks; k<=ke; ++k)
    for(int j=js; j<=je; ++j)
      for(int i=is; i<=ie; ++i){
        if(prad->npsi == 0){
        // first, related all angle to 2*nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
        // The relation is I_m = coef *I_m+1 + const

        // now, all the other angles
        //  Ir_new - ir_old + coef0 * Ir_l + coef1 ir_ new = 0.0;
          for(int n=0; n<2*nzeta; ++n){
            Real coef0 = wght * g_zeta_(n) * prad->reduced_c * 
                       area_zeta(n)/ang_vol(n);
            Real coef1 = -wght * g_zeta_(n+1) * prad->reduced_c * 
                       area_zeta(n+1)/ang_vol(n);
          // the equation is
          // ir_new - ir_old + coef0 * ir_new + coef1 * ir_new1 = 0
            if(n == 2*nzeta-1){
              imp_ang_coef_r_(k,j,i,n) = 0.0;
              imp_ang_coef_(k,j,i,n) = coef0 + coef1;
            }else{
              imp_ang_coef_r_(k,j,i,n) = coef1;
              imp_ang_coef_(k,j,i,n) = coef0;
            }

          }// end nzeta
        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0
          // first, starting from the zeta angle 2*nzeta-1
          // zeta area is only 0 at nzeta=0, not zero at nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);

          // now go from 2*nzeta-2 to 0
          for(int n=0; n<2*nzeta; ++n){
            Real zeta_coef0 =  wght * g_zeta_(n) * prad->reduced_c;
            Real zeta_coef1 = -wght * g_zeta_(n+1) * prad->reduced_c;
  
            ImplicitPsiFluxCoef(k,j,i, n, wght, zeta_coef1, zeta_coef0);

          }// end n
        }// end npsi > 0
      }// end k,j,i
}


void RadIntegrator::ImplicitPsiFluxCoef(int k, int j, int i, int n_zeta, Real wght, Real zeta_coef1, 
            Real zeta_coef)
{

  Radiation *prad=pmy_rad;
  Coordinates *pco=prad->pmy_block->pcoord;
  AthenaArray<Real> &area_psi = psi_area_, &ang_vol = ang_vol_, &zeta_area = zeta_area_;
  int &npsi = prad->npsi;


  // the equation to solve
  //(1+zeta_coef0+zeta_coef1) I + Div F_psi = 0
  pco->GetGeometryPsi(prad,k,j,i,n_zeta,g_psi_);
  // g_psi_ =sin zeta * cot \theta sin\psi/r
  // g_psi_(0) is always 0

  for(int m=0; m<2*npsi; ++m){ // all take the left state
    int ang_num = n_zeta*2*npsi+m;

    Real coef0 = wght * g_psi_(m) * prad->reduced_c * 
            area_psi(n_zeta,m)/ang_vol(ang_num);
    Real coef1 = -wght * g_psi_(m+1) * prad->reduced_c * 
            area_psi(n_zeta,m+1)/ang_vol(ang_num);
    Real z_coef1 = zeta_coef1 * zeta_area(m,n_zeta+1)/ang_vol(ang_num);
    Real z_coef = zeta_coef * zeta_area(m,n_zeta)/ang_vol(ang_num);
    Real coef_c = 0.0;
    if((g_psi_(m) < 0) || (g_psi_(m+1) < 0)){
      imp_ang_psi_l_(k,j,i,ang_num) = coef0;
      imp_ang_psi_r_(k,j,i,ang_num) = 0.0;
      coef_c = coef1;
    }else if((g_psi_(m) > 0) || (g_psi_(m+1) > 0)){
      imp_ang_psi_l_(k,j,i,ang_num) = 0.0;
      imp_ang_psi_r_(k,j,i,ang_num) = coef1;
      coef_c = coef0;       
    }
    if(n_zeta == 2*prad->nzeta-1){
      imp_ang_coef_r_(k,j,i,ang_num) = 0.0;
      imp_ang_coef_(k,j,i,ang_num) = coef_c + z_coef + z_coef1;
    }else{
      imp_ang_coef_r_(k,j,i,ang_num) = z_coef1;
      imp_ang_coef_(k,j,i,ang_num) = coef_c + z_coef;       
    }

  }
}// end psiflux_coef

void RadIntegrator::ImplicitAngularFluxes(AthenaArray<Real> &ir_ini)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco=pmb->pcoord;
  std::stringstream msg;

  int &nzeta = prad->nzeta;
  int &npsi = prad->npsi;

  AthenaArray<Real> &area_zeta = zeta_area_, &ang_vol = ang_vol_;

  
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for(int k=ks; k<=ke; ++k)
    for(int j=js; j<=je; ++j)
      for(int i=is; i<=ie; ++i){
        if(prad->npsi == 0){
          // the angle 2*nzeta-1 does not contribute to ang_flx_
          Real *p_angflx = &(ang_flx_(k,j,i,0));
          Real *coef_r = &(imp_ang_coef_r_(k,j,i,0));
          Real *p_ir = &(ir_ini(k,j,i,0));
          for(int n=0; n<2*nzeta-1; ++n){
            p_angflx[n] = -coef_r[n] * p_ir[n+1];
          }// end nzeta
        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0
          // now go from 2*nzeta-2 to 0
          for(int n=0; n<2*nzeta; ++n){
            ImplicitPsiFlux(k,j,i, n, ir_ini);
          }// end n
        }// end npsi > 0
      }// end k,j,i

}// end calculate_flux


void RadIntegrator::ImplicitPsiFlux(int k, int j, int i, int n_zeta, AthenaArray<Real> &ir_ini)
{
  
  Radiation *prad=pmy_rad;
  int &npsi = prad->npsi;
  //m=0
  int ang_num = n_zeta*2*npsi;

  Real *psi_l = &(imp_ang_psi_l_(k,j,i,ang_num));
  Real *psi_r = &(imp_ang_psi_r_(k,j,i,ang_num));
  Real *p_ir = &(ir_ini(k,j,i,ang_num));
  Real *p_angflx = &(ang_flx_(k,j,i,ang_num));



  if(n_zeta == 2*prad->nzeta-1){
    // m=0
    p_angflx[0] = -(psi_l[0] * p_ir[2*npsi-1] + psi_r[0] * p_ir[1]);
    for(int m=1; m<2*npsi-1; ++m){ // all take the left state
      p_angflx[m] = -(psi_l[m] * p_ir[m-1] + psi_r[m] * p_ir[m+1]);
    }
    //m=2*npsi-1
    p_angflx[2*npsi-1] = -(psi_l[2*npsi-1] * p_ir[2*npsi-2] + psi_r[2*npsi-1] * p_ir[0]);
  }else{//end nzeta=2*prad->nzeta-1
    Real *p_ir_zetar = &(ir_ini(k,j,i,(n_zeta+1)*2*npsi));
    Real *zeta_r = &(imp_ang_coef_r_(k,j,i,ang_num));

    p_angflx[0] = -(psi_l[0] * p_ir[2*npsi-1] + psi_r[0] * p_ir[1])
                  -zeta_r[0] * p_ir_zetar[0];
    for(int m=1; m<2*npsi-1; ++m){ // all take the left state
      p_angflx[m] = -(psi_l[m] * p_ir[m-1] + psi_r[m] * p_ir[m+1])
                    -zeta_r[m] * p_ir_zetar[m];
    }
    //m=2*npsi-1
    p_angflx[2*npsi-1] = -(psi_l[2*npsi-1] * p_ir[2*npsi-2] + psi_r[2*npsi-1] * p_ir[0])
                         -zeta_r[2*npsi-1] * p_ir_zetar[2*npsi-1];
  }

}


//switch to centered difference in optically thick regime
// the flux is 
// (Smax v_f I_l-Smin v_f I_r) + Smin*Smax(Ir - Il)/(Smax-Smin)

void RadIntegrator::ImplicitAngularFluxesCenter(const Real wght, 
                                     AthenaArray<Real> &ir_ini)
{
  Radiation *prad=pmy_rad;
  MeshBlock *pmb=prad->pmy_block;
  Coordinates *pco=pmb->pcoord;
  std::stringstream msg;

  int &nzeta = prad->nzeta;
  int &npsi = prad->npsi;

  AthenaArray<Real> &area_zeta = zeta_area_, &ang_vol = ang_vol_;

  
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;


  for(int k=ks; k<=ke; ++k)
    for(int j=js; j<=je; ++j){
       // get the optical depth per cell
      pco->CenterWidth1(k,j,is,ie,dxw1_);
      if(je > js){
        pco->CenterWidth2(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i)
          dxw1_(i) += dxw2_(i);
      }
      if(ke > ks){
        pco->CenterWidth3(k,j,is,ie,dxw2_);
        for(int i=is; i<=ie; ++i)
          dxw1_(i) += dxw2_(i);
      }
      for(int i=is; i<=ie; ++i){
        // get the optical depth per cell
        Real tau_c= 0.0;
        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          Real sigma = prad->sigma_a(k,j,i,ifr) + prad->sigma_s(k,j,i,ifr);
          tau_c += prad->wfreq(ifr) * dxw1_(i) * sigma;
        }
        tau_c *= taufact(k,j,i);
        Real f_l = 1.0;
        Real f_r = 1.0;
        GetTaufactor(tau_c,f_r,1);
        GetTaufactor(tau_c,f_l,-1);   



        if(prad->npsi == 0){
        // first, related all angle to 2*nzeta
          pco->GetGeometryZeta(prad,k,j,i,g_zeta_);
        // The relation is I_m = coef *I_m+1 + const
          // the interface velocity is always negative
          for(int n=0; n<2*nzeta; ++n){
            int nl = n-1;
            int nr = n+1;
            if(n==0)
              nl = 0;
            
            if(nr == 2*nzeta-1)
              nr = n;

            // the left hand side
            Real vl = -g_zeta_(n) * prad->reduced_c * area_zeta(n);
            Real smax = -vl * f_l;
            Real smin = vl * f_r;

            Real ratio = smax - smin;
            if(ratio > TINY_NUMBER)
              ratio = 1.0/ratio;
            Real coef_l = -wght * (vl - smin) * smax * ratio/ang_vol(n);
            Real coef_c = -wght * (smax - vl) * smin * ratio/ang_vol(n);  
            if(n == 0){
              coef_l = 0.0;
              coef_c = -wght * vl/ang_vol(0);
            }         

            // the right hand side
            Real vr = -g_zeta_(n+1) * prad->reduced_c * area_zeta(n+1);
            smax = -vr * f_l;
            smin = vr * f_r;
            ratio = smax - smin;
            if(ratio > TINY_NUMBER)
              ratio = 1.0/ratio;
            Real coef_r = wght * (smax - vr) * smin * ratio/ang_vol(n);
            if(n == 2*nzeta-1){
              coef_r = 0.0;
              coef_c +=  wght * vr/ang_vol(n);
            }else
             coef_c += wght * (vr - smin) * smax * ratio/ang_vol(n);    

          // the equation is
          // ir_new - ir_old + coef0 * ir_new + coef1 * ir_new1 = 0
            ang_flx_(k,j,i,n) = -coef_r * ir_ini(k,j,i,nr)
                                -coef_l * ir_ini(k,j,i,nl);
            imp_ang_coef_(k,j,i,n) = coef_c;
          }// end nzeta

        ///////////////////////////////////////////////////////////////////////////////
        }else{//end npsi ==0

          for(int n=0; n<2*nzeta; ++n){
            Real g_zeta_coef_l = g_zeta_(n);
            Real g_zeta_coef_r = g_zeta_(n+1);
            Real zeta_coef0 =  g_zeta_coef_l * prad->reduced_c;
            Real zeta_coef1 =  g_zeta_coef_r * prad->reduced_c;
  
            ImplicitPsiFluxCenter(k,j,i, n, wght, zeta_coef1, 
                              zeta_coef0, f_l, f_r, ir_ini);
          }// end n

        }// end npsi > 0
      }
    }// end k,j,i

}// end calculate_flux




void RadIntegrator::ImplicitPsiFluxCenter(int k, int j, int i, int n_zeta, Real wght, Real zeta_coef1, 
            Real zeta_coef, Real f_l, Real f_r, AthenaArray<Real> &ir_ini)
{

  Radiation *prad=pmy_rad;
  Coordinates *pco=prad->pmy_block->pcoord;
  AthenaArray<Real> &area_psi = psi_area_, &ang_vol = ang_vol_, &area_zeta = zeta_area_;
  int &npsi = prad->npsi;
  int nzeta_l = n_zeta-1;
  if(n_zeta == 0)
    nzeta_l = 0;
  int nzeta_r = n_zeta+1;
  if(n_zeta == 2*prad->nzeta-1)
    nzeta_r = n_zeta;

  // the equation to solve
  //(1+zeta_coef0+zeta_coef1) I + Div F_psi = 0
  pco->GetGeometryPsi(prad,k,j,i,n_zeta,g_psi_);


    // in this case, psi velocity positive for m=0 to npsi
    // psi velocity negative for m=npsi to 2*npsi
  for(int m=0; m<2*npsi; ++m){
    int ang_num = n_zeta*2*npsi+m;
    int ang_psi_r = n_zeta*2*npsi+m+1;
    int ang_psi_l = n_zeta*2*npsi+m-1;
    int ang_zeta_l = nzeta_l*2*npsi+m;
    int ang_zeta_r = nzeta_r*2*npsi+m;
    if(m==0)
      ang_psi_l = n_zeta*2*npsi+m;
    if(m==2*npsi-1)
      ang_psi_r = n_zeta*2*npsi+m;

      // the left hand side of zeta
    Real v_zeta_l =  -zeta_coef * area_zeta(m,n_zeta);
    Real smax = -v_zeta_l * f_l;
    Real smin = v_zeta_l * f_r;
    Real ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;
    Real coef_zeta_l = -wght * (v_zeta_l - smin) * smax * ratio/ang_vol(ang_num);
    Real coef_zeta_c = -wght * (smax - v_zeta_l) * smin * ratio/ang_vol(ang_num);  

    if(n_zeta == 0){
      coef_zeta_l = 0.0;
      coef_zeta_c = -wght * v_zeta_l/ang_vol(ang_num);
    }

      // the right hand side of zeta

    Real v_zeta_r = -zeta_coef1 * area_zeta(m,n_zeta+1);
    smax = -v_zeta_r * f_l;
    smin = v_zeta_r * f_r;

    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;
    Real coef_zeta_r = wght * (smax - v_zeta_r) * smin * ratio/ang_vol(ang_num);
    if(n_zeta == 2*prad->nzeta-1){
      coef_zeta_c += wght * v_zeta_r/ang_vol(ang_num);
      coef_zeta_r = 0.0;
    }else{
      coef_zeta_c += wght * (v_zeta_r - smin) * smax * ratio/ang_vol(ang_num);   
    }

      // the left hand side of Psi
    Real v_psi_l = -g_psi_(m) * prad->reduced_c * area_psi(n_zeta,m);
    smax = v_psi_l * f_r;
    smin = -v_psi_l * f_l;
    if(v_psi_l < 0){
      smax = -v_psi_l * f_l;
      smin = v_psi_l * f_r;
    }
    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;    
    Real coef_psi_l = -wght * (v_psi_l - smin) * smax * ratio/ang_vol(ang_num);
    Real coef_psi_c = -wght * (smax - v_psi_l) * smin * ratio/ang_vol(ang_num);   
    if(m == 0){
      coef_psi_l = 0.0;
    }

    //the right hand side of psi
    Real v_psi_r = -g_psi_(m+1) * prad->reduced_c * area_psi(n_zeta,m+1);
    smax = v_psi_r * f_r;
    smin = -v_psi_r * f_l;    
    if(v_psi_r < 0){
      smax = -v_psi_r * f_l;
      smin = v_psi_r * f_r;
    }
    ratio = smax - smin;
    if(ratio > TINY_NUMBER)
      ratio = 1.0/ratio;    
    Real coef_psi_r = wght * (smax - v_psi_r) * smin * ratio/ang_vol(ang_num);
    if(m == 2*npsi-1){
      coef_psi_r = 0.0;
    }else{
      coef_psi_c += wght * (v_psi_r - smin) * smax * ratio/ang_vol(ang_num);   
    }
    
        

    ang_flx_(k,j,i,ang_num) = -coef_zeta_l * ir_ini(k,j,i,ang_zeta_l) 
                              -coef_zeta_r * ir_ini(k,j,i,ang_zeta_r)
                              -coef_psi_l * ir_ini(k,j,i,ang_psi_l) 
                              -coef_psi_r * ir_ini(k,j,i,ang_psi_r);
    imp_ang_coef_(k,j,i,ang_num) = coef_zeta_c + coef_psi_c;   

  }// end m=0 to 2*npsi

}
