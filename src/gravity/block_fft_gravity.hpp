#ifndef GRAVITY_BLOCK_FFT_GRAVITY_HPP_
#define GRAVITY_BLOCK_FFT_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file block_fft_gravity.hpp
//! \brief defines BlockFFTGravity class

// C headers

// C++ headers
#include <complex>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/block_fft.hpp"
#include "../hydro/hydro.hpp"
#include "../task_list/fft_grav_task_list.hpp"
#include "gravity.hpp"

//! identifiers for gravity boundary conditions
enum class GravityBoundaryFlag {periodic, disk, open};

//! flag for the Green's function choice
enum class GreenFuncFlag {point_mass, cell_averaged};

//! free functions to return boundary flag given input string
GravityBoundaryFlag GetGravityBoundaryFlag(const std::string& input_string);

//! free functions to return Green's function flag given input string
GreenFuncFlag GetGreenFuncFlag(const std::string& input_string);

//! indefinite integral for the cell-averaged Green's function
Real _GetIGF(Real x, Real y, Real z);

//! \class BlockFFTGravity
//! \brief minimalist FFT gravity solver for each block

class BlockFFTGravity : public BlockFFT {
 public:
  BlockFFTGravity(MeshBlock *pmb, ParameterInput *pin);
  ~BlockFFTGravity();

  // data
  bool SHEAR_PERIODIC; // flag for shear periodic boundary (true w/ , false w/o)
  bool PHASE_SHIFT; // False if using roll-unroll method
  GravityBoundaryFlag gbflag; // flag for the gravity boundary condition
  GreenFuncFlag grfflag; // flag for the Green's function

  // functions
  void ExecuteForward() final;
  void ApplyKernel() final;
  void ExecuteBackward() final;
  void Solve(int stage);
  void InitGreen();
  void LoadOBCSource(const AthenaArray<Real> &src, int px, int py, int pz);
  void RetrieveOBCResult(AthenaArray<Real> &dst, int px, int py, int pz);
  void MultiplyGreen(int px, int py, int pz);
  void RollUnroll(AthenaArray<Real> &dat, Real dt);
  void SetPhysicalBoundaries();

 private:
  FFTGravitySolverTaskList *gtlist_;
  Real Omega_0_,qshear_,rshear_;
  Real dx1_,dx2_,dx3_;
  Real dx1sq_,dx2sq_,dx3sq_;
  Real Lx1_,Lx2_,Lx3_;
  const std::complex<Real> I_; // sqrt(-1)
  std::complex<Real> *in2_,*in_e_,*in_o_;
  std::complex<Real> *grf_; // Green's function for open BC
#ifdef MPI_PARALLEL
#ifdef FFT
  FFTMPI_NS::FFT3d *pf3dgrf_; // FFT3d instance for FFT'ing open BC Green's func.
#endif
#endif
  bool is_particle_gravity;
  // buffers for roll-unroll method
  AthenaArray<Real> roll_var, roll_buf, send_buf, recv_buf, pflux;
  AthenaArray<Real> send_gbuf, recv_gbuf; // ghost zone buffers at y boundaries
};

#endif // GRAVITY_BLOCK_FFT_GRAVITY_HPP_
