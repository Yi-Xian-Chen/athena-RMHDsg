#ifndef RADINTEGRATORS_HPP
#define RADINTEGRATORS_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class
//======================================================================================

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../radiation.hpp" // radiation
#include "../../task_list/task_list.hpp"

class MeshBlock;
class ParameterInput;
class Radiation;

//! \class RadIntegrator
//  \brief integrate algorithm for radiative transfer


class RadIntegrator {
  friend class Radiation;
  friend class IMRadiation;
public:
  RadIntegrator(Radiation *prad, ParameterInput *pin);
  ~RadIntegrator();
  
  Radiation *pmy_rad;
  
  void FluxDivergence(const Real wght, AthenaArray<Real> &ir_in, 
                                       AthenaArray<Real> &ir_out);
  void FluxDivergence(const Real wght);


  void FirstOrderFluxDivergenceCoef(const Real wght);
  void FirstOrderFluxDivergence(AthenaArray<Real> &ir);


  void FirstOrderGSFluxDivergence(const Real wght, 
                                AthenaArray<Real> &ir);

  void ImplicitAngularFluxesCoef(const Real wght);
  void ImplicitAngularFluxes(AthenaArray<Real> &ir);
  void ImplicitAngularFluxesCenter(const Real wght, AthenaArray<Real> &ir);

  void ImplicitPsiFluxCoef(int k, int j, int i, int n_zeta, Real wght, 
            Real zeta_coefr, Real zeta_coefl);
  void ImplicitPsiFlux(int k, int j, int i, int n_zeta, AthenaArray<Real> &ir);
  void ImplicitPsiFluxCenter(int k, int j, int i, int n_zeta, Real wght, 
            Real zeta_coefr, Real zeta_coefl, Real f_l, Real f_r, 
            AthenaArray<Real> &ir);
    
  void CalculateFluxes(AthenaArray<Real> &w,
                       AthenaArray<Real> &ir, const int order);
  void CalculateFluxes(AthenaArray<Real> &ir, const int order);
  
  void CalSourceTerms(MeshBlock *pmb, const Real dt, AthenaArray<Real> &u,
                      AthenaArray<Real> &ir_ini, AthenaArray<Real> &ir);

  void AddSourceTerms(MeshBlock *pmb, AthenaArray<Real> &u,  
       AthenaArray<Real> &ir_ini, AthenaArray<Real> &ir);
  void AddIMSourceTerms(MeshBlock *pmb, AthenaArray<Real> &u, 
       AthenaArray<Real> &rad_source);

  Real AbsorptionScattering(AthenaArray<Real> &wmu_cm,
          AthenaArray<Real> &tran_coef, Real *sigma_a, Real *sigma_p,
          Real *sigma_ae, Real *sigma_s, Real dt, Real rho, Real &tgas,
          AthenaArray<Real> &implicit_coef_, AthenaArray<Real> &ir_cm);

  void GetTgasVel(MeshBlock *pmb, const Real dt,
    AthenaArray<Real> &u, AthenaArray<Real> &w, 
    AthenaArray<Real> &bcc, AthenaArray<Real> &ir);

  
  void Compton(AthenaArray<Real> &wmu_cm,
          AthenaArray<Real> &tran_coef, Real *sigma_s,
          Real dt, Real rho, Real &tgas, AthenaArray<Real> &ir_cm);
  
  void LabToCom(const Real vx, const Real vy, const Real vz,
                          Real *mux, Real *muy, Real *muz,
                          Real *ir_lab, AthenaArray<Real> &ir_cm);
  
  void ComToLab(const Real vx, const Real vy, const Real vz,
                          Real *mux, Real *muy, Real *muz,
                          AthenaArray<Real> &ir_cm, Real *ir_lab);
  
  void ComAngle(const Real vx, const Real vy, const Real vz,
          Real mux, Real muy, Real muz, Real *mux0, Real *muy0, Real *muz0);

  void ComFrequency(const Real vx, const Real vy, const Real vz, 
                                 const int k, const int j, const int i);

  
  void GetTaufactor(const Real tau, Real &factor1, int dir);
  void GetTaufactorAdv(const Real tau, Real &factor);

  void PredictVel(AthenaArray<Real> &ir, int k, int j, int i, Real dt, Real rho,
                  Real *vx, Real *vy, Real *vz);

  void SignalSpeed(const Real adv, const Real factor1, 
                 const Real factor2, Real *vel, Real *smax, Real *smin);

  void SplitVelocity(Real *vel_l, Real *vel_r, const Real advl, 
            const Real advr, Real *smax_l, Real *smin_l, Real *smax_r, Real *smin_r);


  int rad_xorder; 
  AthenaArray<Real> adv_vel; // the advectioin velocity that we separate
  AthenaArray<Real> taufact;
  AthenaArray<Real> rad_source; // store the radiation source terms

  
private:
  AthenaArray<Real> vel_, velx_,vely_,velz_;
  AthenaArray<Real> il_, ilb_, ir_;// for recontruction
                          // temporary array to store the flux, velocity
  AthenaArray<Real> vncsigma_, vncsigma2_, wmu_cm_, tran_coef_, ir_cm_;
  AthenaArray<Real> cm_to_lab_;
  AthenaArray<Real> g_zeta_, q_zeta_, ql_zeta_, qr_zeta_, zeta_flux_, zeta_area_;
  AthenaArray<Real> g_psi_, q_psi_, ql_psi_, qr_psi_, psi_flux_, psi_area_;
  AthenaArray<Real> dflx_ang_, ang_vol_;
  AthenaArray<Real> tgas_, vel_source_, tgas_new_; // array to store gas temperature, 
                                        // velocity for source term

                                    
 // temporary 1D array with size of nang

  int tau_flag_;
  int compton_flag_; // flag to add simple Compton scattering
  int planck_flag_; // flag to add additional Planck absorption opacity
  int adv_flag_; // flag used to indicate whether separate
                 // advection flux from diffustion flux or not.

  int flux_correct_flag_; // flag to do second order flux crrection or not.
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_, dflx_, cwidth2_, cwidth3_;

  AthenaArray<Real> adv_flx_;

  AthenaArray<Real> const_coef_, exp_coef_;
  AthenaArray<Real> const_coef1_l_, const_coef1_r_;
  AthenaArray<Real> const_coef2_l_, const_coef2_r_;
  AthenaArray<Real> const_coef3_l_, const_coef3_r_;
  AthenaArray<Real> divflx_, implicit_coef_, ang_flx_, imp_ang_coef_;
  AthenaArray<Real> imp_ang_coef_r_;
  AthenaArray<Real> imp_ang_psi_l_, imp_ang_psi_r_;
  AthenaArray<Real> left_coef1_, left_coef2_, left_coef3_;
  AthenaArray<Real> limiter_, limiterj_, limiterk_, dql_, dqr_;
  AthenaArray<Real> sfac1_x_, sfac2_x_;
  AthenaArray<Real> sfac1_y_, sfac2_y_;
  AthenaArray<Real> sfac1_z_, sfac2_z_;
  AthenaArray<Real> sm_diff1_, sm_diff2_;
  AthenaArray<Real> vel_ex_l_, vel_im_l_, vel_ex_r_, vel_im_r_;
  AthenaArray<Real> dxw1_, dxw2_;


};

#endif // RADINTEGRATORS_HPP
