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
// Problem Generator History:
// March-29-2014, Zhaohuan Zhu, support various disk density, velocity,
//                temperature, magnetic fields configurations,and polar boundary
// April-1-2015, Zhaohuan Zhu & Wenhua Ju, add binary/planet in inertial or corotating
//                frame, add stratified x2 boundary condition.             
//
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <cstdlib>    // srand
#include <cfloat>     // FLT_MIN
#include <stdio.h>
#include <math.h>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

//----------------------------------------
// class for planetary system including mass, position, velocity

class PlanetarySystem
{
public:
  int np;
  double *mass;
  double *xp, *yp, *zp;         // position in Cartesian coord.
  double *vxp, *vyp, *vzp;      // velocity in Cartesian coord.
  int *FeelOthers;
  PlanetarySystem(int np);
  ~PlanetarySystem();
private:
  double *xpn, *ypn, *zpn;       // intermediate position for leap-frog integrator
  double *vxpn, *vypn, *vzpn;
public:
  void integrate(double dt);     // integrate planetary orbit
  void disktoplanet(double dt);  // to be added to allow planet migration
  void fixorbit(double dt);      // circular planetary orbit
  void Rotframe(double dt);      // for frame rotating at omegarot
};

//------------------------------------------
// constructor for planetary system for np planets

PlanetarySystem::PlanetarySystem(int np0)
{
  np   = np0;
  mass = new double[np];
  xp   = new double[np];
  yp   = new double[np];
  zp   = new double[np];
  vxp  = new double[np];
  vyp  = new double[np];
  vzp  = new double[np];
  xpn  = new double[np];
  ypn  = new double[np];
  zpn  = new double[np];
  vxpn = new double[np];
  vypn = new double[np];
  vzpn = new double[np];
  FeelOthers=new int[np];
}

//---------------------------------------------
// destructor for planetary system

PlanetarySystem::~PlanetarySystem()
{
  delete[] mass;
  delete[] xp;
  delete[] yp;
  delete[] zp;
  delete[] vxp;
  delete[] vyp;
  delete[] vzp;
  delete[] xpn;
  delete[] ypn;
  delete[] zpn;
  delete[] vxpn;
  delete[] vypn;
  delete[] vzpn;
  delete[] FeelOthers;
};

// File scope global variables
// initial condition
static Real gm0=0.0, gms=0.0, gm1=0.0, r0 = 1.0, omegarot=0.0;
static int dflag, vflag, tflag, per;
static Real rho0, vy0, mm, rrigid, origid, rmagsph, denstar;
// floor
static Real dfloor, pfloor, rho_floor0, slope_rho_floor;
// different problem
static int fuori;
static Real dslope, pslope, p0_over_r0, amp;
static Real ifield,b0,beta;
static Real rcut, rs;
static Real firsttime;
// readin table
static AthenaArray<Real> rtable, ztable, dentable, portable, tdusttable, tgrid; 
static int nrtable, nztable;
// convert unit
static Real timeunit, lunit, rhounit, tempunit, presunit, massunit, mu, tfloor; 
// radiation start
static Real radstart;

// planet center CPD
static Real sl, sh, wtran, gapw, rstart, rtrunc;
// scalar setup
static Real rscal;
static Real rprotect;
// boundary condition
static std::string ix1_bc, ox1_bc, ix2_bc, ox2_bc;
static int hbc_ix1, hbc_ox1, mbc_ix1, mbc_ox1;
static int hbc_ix2, hbc_ox2, mbc_ix2, mbc_ox2;
// energy related
static double gamma_gas;
static Real tlow, thigh, tcool;
// grid related
static Real x1min, x1max, nx2coarse, xcut;
static Real smoothin, smoothtr;
static Real xc, yc, zc;
static Real tdamp;
static int halfstep=1;
// density jump
static Real djump,rinjump,routjump;
static Real routfield;
// planetary system
std::ofstream myfile; 
static PlanetarySystem *psys;
static int fixorb;
static int cylpot;
static Real insert_start,insert_time;
static Real rsoft2=0.0;
static int ind;
// planetary system: output
static Real timeout=0.0,dtorbit;
// planetary system: circumplanetary disk depletion
static Real rcird, tcird, dcird,rocird;
// boundary radiation influx
static Real radinfluxinnerx1, radinfluxouterx1, influxix1, influxox1;
static Real consFr;
static Real heatrate = 0.;
static int heatflag;
static int iRadInner;
// fixed emission
static Real nuemission;
static int Opacityflag;
static Real opaacgs, opapcgs, opascgs;
// output quantities to ifov
static int nuseroutvar;
static int ifov_flag=0;
AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol;
// alpha viscosity
static Real alpha, aslope, alphalow, astoptime;
// the opacity table
static AthenaArray<Real> opacitytableross, opacitytableplanck;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logptable;

// Functions for initial condition
static Real rho_floor(const Real x1, const Real x2, const Real x3);
static Real rho_floorsf(const Real x1, const Real x2, const Real x3);
static Real A3(const Real x1, const Real x2, const Real x3);
static Real A2(const Real x1, const Real x2, const Real x3);
static Real A1(const Real x1, const Real x2, const Real x3);
static Real DenProfile(const Real x1, const Real x2, const Real x3);
static Real DenProfilesf(const Real x1, const Real x2, const Real x3);
static Real PoverR(const Real x1, const Real x2, const Real x3);
static Real PoverRsf(const Real x1, const Real x2, const Real x3);
static void VelProfile(const Real x1, const Real x2, const Real x3, const Real den, 
		       Real &v1, Real &v2, Real &v3);
static void VelProfilesf(const Real x1, const Real x2, const Real x3, const Real den,
                       Real &v1, Real &v2, Real &v3);
static Real Interp(const Real r, const Real z, const int nxaxis, const int nyaxis, AthenaArray<Real> &xaxis, 
                   AthenaArray<Real> &yaxis, AthenaArray<Real> &datatable );
// Functions for coordinate conversion
void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi);
void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z);
void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp);
void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz);

// viscosity
void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke);

// radiation
void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);
void DiskOpacityDust(MeshBlock *pmb, AthenaArray<Real> &prim);
void FixedOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

void LoadRadVariable(MeshBlock *pmb);

// Functions for boundary conditions
// Summary function for InnerX1, OuterX1, InnerX2, OuterX2
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void Heating(MeshBlock *pmb, const Real time, const Real dt,
     const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Individual BC functions to be called determined by 
// hbc_ix1, hbc_ox1, mbc_ix1, mbc_ox1
// hbc_ix2, hbc_ox2, mbc_ix2, mbc_ox2
static void SteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, 
  		  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void SteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, 
		  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void UserOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void UserOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void InflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldSteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldSteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldOutflowInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldOutflowOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldDivFreeInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldDivFreeOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldInflowOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldInflowAdvectOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldInflowVerticalOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void SteadyInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void SteadyOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void StratInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void StratOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldOutflowInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void FieldOutflowOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void InfluxRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
                  const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void InconstfluxRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
                  const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
static void InfluxRadOuterX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
                  const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

// Check density floor
void Checkfloor(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Functions for Planetary Source terms
void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt, 
	const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void Damp(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void DepleteCir(MeshBlock *pmb,const Real dt, const AthenaArray<Real> &prim, 
		AthenaArray<Real> &cons);
void Cooling(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  int fixedemission;

  firsttime=time;

  x1min=mesh_size.x1min;
  x1max=mesh_size.x1max;

  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  gms=gm0;
  r0 = pin->GetOrAddReal("problem","r0",1.0);
  omegarot = pin->GetOrAddReal("problem","omegarot",0.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  rho_floor0 = pin->GetReal("problem","rho_floor0"); 
  slope_rho_floor = pin->GetOrAddReal("problem","slope_rho_floor",0.0);
  dflag = pin->GetInteger("problem","dflag");
  vflag = pin->GetInteger("problem","vflag");
  rrigid = pin->GetOrAddReal("problem","rrigid",0.0);
  origid = pin->GetOrAddReal("problem","origid",0.0);
  rmagsph = pin->GetOrAddReal("problem","rmagsph",0.0);
  denstar = pin->GetOrAddReal("problem","denstar",0.0);
  vy0 = pin->GetOrAddReal("problem","vy0",0.0);
  dslope = pin->GetOrAddReal("problem","dslope",0.0);
  per = pin->GetOrAddInteger("problem","per",0);
  amp = pin->GetOrAddReal("problem","amp",0.0);

  // density transition
  djump = pin->GetOrAddReal("problem","djump",0.0);
  rinjump = pin->GetOrAddReal("problem","rinjump",0.0);
  routjump = pin->GetOrAddReal("problem","routjump",1.0);

  // viscosity
  alpha = pin->GetOrAddReal("problem","nu_iso",0.0);
  aslope = pin->GetOrAddReal("problem","aslope",0.0);
  alphalow = pin->GetOrAddReal("problem","alphalow",0.0);
  astoptime = pin->GetOrAddReal("problem","astoptime",HUGE_NUMBER);

  // CPD study
  sl = pin->GetOrAddReal("problem","sl",1.);
  sh = pin->GetOrAddReal("problem","sh",1.);
  wtran = pin->GetOrAddReal("problem","wtran",1.);
  gapw = pin->GetOrAddReal("problem","gapw",0.1);

  // scalar radius
  rscal = pin->GetOrAddReal("problem","rscal",HUGE_NUMBER);
  rprotect = pin->GetOrAddReal("problem","rprotect",HUGE_NUMBER);

  if(dflag==4){
    rstart = pin->GetOrAddReal("problem","rstart",x1min);
    rtrunc = pin->GetOrAddReal("problem","rtrunc",x1max);
  }

  // Get parameters of initial pressure and cooling parameters
  if(NON_BAROTROPIC_EOS){
    tflag = pin->GetOrAddInteger("problem","tflag",0);
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    tlow = pin->GetOrAddReal("problem","tlow",0.0);
    thigh = pin->GetOrAddReal("problem","thigh",0.0);
    tcool = pin->GetOrAddReal("problem","tcool",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  }else{
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  dfloor = pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN))); 
  pfloor = pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

  xcut = pin->GetOrAddReal("problem","xcut",x1min);

  // damp quantities
  tdamp = pin->GetOrAddReal("problem","tdamp",0.0);

  // Get loop center for field loop tests;
  xc = pin->GetOrAddReal("problem","xc",1.0);
  yc = pin->GetOrAddReal("problem","yc",0.0);
  zc = pin->GetOrAddReal("problem","zc",0.0);

  // Initialize the magnetic fields
  if (MAGNETIC_FIELDS_ENABLED){
    // Get parameters of inital magnetic fields
    rcut =  pin->GetOrAddReal("problem","rcut",0.0);
    rs =  pin->GetOrAddReal("problem","rs",0.0);
    ifield = pin->GetInteger("problem","ifield");
    routfield =  pin->GetOrAddReal("problem","routfield",HUGE_NUMBER);
    beta = pin->GetReal("problem","beta");
    mm = pin->GetOrAddReal("problem","mm",0.0);
    b0=sqrt(2.*p0_over_r0*rho0/beta);

    nx2coarse = pin->GetInteger("mesh","nx2");
  }

  nuseroutvar=pin->GetOrAddInteger("mesh","nuser_out_var",0);

  // Get IFOV choice
  if(nuseroutvar >=1){
    ifov_flag = pin->GetInteger("problem","ifov_flag");
  }

  // Get boundary condition flags
  ix1_bc = pin->GetOrAddString("mesh","ix1_bc","none");
  ox1_bc = pin->GetOrAddString("mesh","ox1_bc","none");
  ix2_bc = pin->GetOrAddString("mesh","ix2_bc","none");
  ox2_bc = pin->GetOrAddString("mesh","ox2_bc","none");

  if(ix1_bc == "user") {
    hbc_ix1 = pin->GetReal("problem","hbc_ix1");
    if(MAGNETIC_FIELDS_ENABLED)
      mbc_ix1 = pin->GetReal("problem","mbc_ix1");
  }
  if(ox1_bc == "user"){
    hbc_ox1 = pin->GetReal("problem","hbc_ox1");
    if(MAGNETIC_FIELDS_ENABLED)
      mbc_ox1 = pin->GetReal("problem","mbc_ox1");
  }
  if(ix2_bc == "user") {
    hbc_ix2 = pin->GetReal("problem","hbc_ix2");
    if(MAGNETIC_FIELDS_ENABLED)
      mbc_ix2 = pin->GetReal("problem","mbc_ix2");
  }
  if(ox2_bc == "user"){
    hbc_ox2 = pin->GetReal("problem","hbc_ox2");
    if(MAGNETIC_FIELDS_ENABLED)
      mbc_ox2 = pin->GetReal("problem","mbc_ox2");
  }

  // Get circumplanetary disk density depletion
  rcird = pin->GetOrAddReal("problem","rcird",0.0);
  rocird = pin->GetOrAddReal("problem","rocird",1.0e10);
  tcird = pin->GetOrAddReal("problem","tcird",0.0);
  dcird = pin->GetOrAddReal("problem","dcird",0.0);

  // For FU Ori problem
  fuori = pin->GetOrAddInteger("problem","fuori",0);
  std::string opacityfile ;
  if(fuori==1&&(RADIATION_ENABLED|| IM_RADIATION_ENABLED)){
    heatrate = pin->GetOrAddReal("problem","heatrate",0.);
    heatflag = pin->GetOrAddInteger("problem","heatflag",0);
    tfloor = pin->GetOrAddReal("radiation", "tfloor", 0.01);
    rhounit = pin->GetOrAddReal("radiation", "rhounit", 1.e-8);
    timeunit = pin->GetOrAddReal("radiation", "timeunit", 289977.36); // 0.3 solar mass at 0.1 AU
    lunit = pin->GetOrAddReal("radiation", "lunit", 1.496e12); // 0.1 AU
    mu = pin->GetOrAddReal("radiation", "mu", 1.);
    presunit=rhounit*lunit*lunit/timeunit/timeunit;
    tempunit=lunit*lunit*mu/8.3144598e7/timeunit/timeunit;

    radstart = pin->GetOrAddReal("radiation", "radstart", 0.0);
    influxix1 = pin->GetOrAddReal("radiation", "influxix1", 1.0);
    influxox1 = pin->GetOrAddReal("radiation", "influxox1", 1.0);
    iRadInner = pin->GetOrAddInteger("radiation", "iRadInner", 0);
    consFr = pin->GetOrAddReal("radiation", "consFr", 0.0);
    fixedemission = pin->GetOrAddInteger("hydro","fixed_emission_flag",0);
    opacityfile =  pin->GetOrAddString("radiation", "OpacityFile","none");
    nuemission = pin->GetOrAddReal("radiation","nuemission",0.0);
    Opacityflag = pin->GetOrAddInteger("radiation","Opacityflag",0);
    opaacgs = pin->GetOrAddReal("radiation","opaacgs",0.0);
    opapcgs = pin->GetOrAddReal("radiation","opapcgs",0.0);
    opascgs = pin->GetOrAddReal("radiation","opascgs",0.0);

    radinfluxinnerx1 = p0_over_r0 * p0_over_r0 * p0_over_r0 * p0_over_r0 * influxix1;
    radinfluxouterx1 = p0_over_r0 * p0_over_r0 * p0_over_r0 * p0_over_r0 * influxox1;
  }

  if(ifov_flag==1 or ifov_flag==2){
    int nx1=pin->GetInteger("meshblock","nx1");
    nx1+=2*(NGHOST);
    x1area.NewAthenaArray(nx1);
    x2area.NewAthenaArray(nx1);
    x2area_p1.NewAthenaArray(nx1);
    x3area.NewAthenaArray(nx1);
    x3area_p1.NewAthenaArray(nx1);
    vol.NewAthenaArray(nx1);
  }

  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED){
   if (Opacityflag==0){
    opacitytableross.NewAthenaArray(100,50);
    opacitytableplanck.NewAthenaArray(100,50);
    logttable.NewAthenaArray(100);
    logptable.NewAthenaArray(50);

    FILE *fkappaross, *fkappaplanck, *flogt, *flogp;
    if ( (fkappaross=fopen("./rosscalcdust.dat","r"))==NULL )
    {
      printf("Open Rossland mean opacity input file error");
      return;
    }

    if ( (fkappaplanck=fopen("./planckcalcdust.dat","r"))==NULL )
    {
      printf("Open Planck mean opacity input file error");
      return;
    }

    if ( (flogt=fopen("./logT.dat","r"))==NULL )
    {
      printf("Open logT input file error");
      return;
    }

    if ( (flogp=fopen("./logP.dat","r"))==NULL )
    {
      printf("Open logP input file error");
      return;
    }

    int i, j;
    for(j=0; j<100; j++){
      for(i=0; i<50; i++){
          fscanf(fkappaross,"%lf",&(opacitytableross(j,i)));
      }
    }

    for(j=0; j<100; j++){
      for(i=0; i<50; i++){
          fscanf(fkappaplanck,"%lf",&(opacitytableplanck(j,i)));
      }
    }

    for(i=0; i<50; i++){
      fscanf(flogp,"%lf",&(logptable(i)));
    }

    for(i=0; i<100; i++){
      fscanf(flogt,"%lf",&(logttable(i)));
    }

    fclose(fkappaross);
    fclose(fkappaplanck);
    fclose(flogt);
    fclose(flogp);
   } else if (Opacityflag==3) {
      opacitytableross.NewAthenaArray(70,140);
      opacitytableplanck.NewAthenaArray(70,140);
      logttable.NewAthenaArray(70);
      logptable.NewAthenaArray(140);

      FILE *fileopa;
      if ( (fileopa=fopen(opacityfile.c_str(),"r"))==NULL )
      {
        printf("Open input file error");
        return;
      }

      int i, j;
      Real rhoread,tread,rossread,planckread;
      char * line = NULL;
      size_t len = 0;
      getline(&line, &len, fileopa);
      printf ("I have read: %s \n",line);
      for(j=0; j<140; j++){
        for(i=0; i<70; i++){
          fscanf(fileopa,"%lf %lf %lf %lf",&rhoread,&tread,&rossread,&planckread);
          logptable(j)=log10(rhoread);
          logttable(i)=log10(tread);
          opacitytableross(i,j)=rossread;
          opacitytableplanck(i,j)=planckread;
        }
      }
    }
  }

  //read in table if necessary
  if(dflag==5 || tflag==5){
    lunit=1.496e13;   // 1 AU
    timeunit=6003209.3; // 0.7 solar mass 1 AU velocity,  1 solar mass velocity 5022635.6  1 year/2pi
    massunit=1.341914e30;  // 0.10271e15(number density at 1 AU)*lunit^3*2.35(mean weight)*1.66054e-24(mole mass)
//    presunit=massunit/pow(lunit,4); // GM*massunit/lunit^4
    presunit=massunit/lunit/timeunit/timeunit;
    tempunit=presunit/8.3144598e7/(massunit/pow(lunit,3)); //Pcode/rhocode=Tcode/mu
    Real tdust, tgas, den, pre;
    nrtable = 96;
    nztable = 1779;
    rtable.NewAthenaArray(nrtable);
    ztable.NewAthenaArray(nztable);
    dentable.NewAthenaArray(nztable,nrtable);
    portable.NewAthenaArray(nztable,nrtable);
    tdusttable.NewAthenaArray(nztable,nrtable);
    std::ifstream infile("./Original_data.out");
    if (infile.is_open()){
      for (int i = 0; i < nrtable; i++) {
        for (int j = 0; j < nztable; j++) {
	  infile >> rtable(i) >> ztable(j) >> tgas >> tdust >> den;
	  rtable(i)=rtable(i)/lunit;
          ztable(j)=ztable(j)/lunit;
	  den=den*2.35*1.66054e-24; // in gram/cm^3
	  dentable(j,i)=den/(massunit/pow(lunit,3)); // code unit
	  portable(j,i) = tgas/tempunit/2.35; 
	  tdusttable(j,i) = tdust;
        }
      }
    }else{
      std::cout<<"Cannot open table input file"<<std::endl;
    }
    infile.close();
  }
  // open planetary system and set up variables
  ind = pin->GetOrAddInteger("planets","ind",1);
  rsoft2 = pin->GetOrAddReal("planets","rsoft2",0.0);
  Real np = pin->GetOrAddInteger("planets","np",0);
  psys = new PlanetarySystem(np);
  fixorb = pin->GetOrAddInteger("planets","fixorb",0);
  insert_start = pin->GetOrAddReal("planets","insert_start",0.0);
  insert_time = pin->GetOrAddReal("planets","insert_time",0.0);
  cylpot = pin->GetOrAddInteger("planets","cylpot",0);

  // for planetary orbit output
  if(psys->np>0 && Globals::my_rank==0) myfile.open("orbit.txt",std::ios_base::app);
  dtorbit = pin->GetOrAddReal("planets","dtorbit",0.0);

  // set initial planet properties
  for(int ip=0; ip<psys->np; ++ip){
    char pname[10];
    sprintf(pname,"mass%d",ip);
    psys->mass[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"x%d",ip);
    psys->xp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"y%d",ip);
    psys->yp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"z%d",ip);
    psys->zp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"vx%d",ip);
    psys->vxp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"vy%d",ip);
    psys->vyp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"vz%d",ip);
    psys->vzp[ip]=pin->GetOrAddReal("planets",pname,0.0);
    sprintf(pname,"feel%d",ip);
    psys->FeelOthers[ip]=pin->GetOrAddInteger("planets",pname,0);
  }
  if(psys->np>0){
    gm1 = psys->mass[0];
    if(dflag==4) gms=gm1;
  }

  EnrollViscosityCoefficient(AlphaVis);

  // setup boundary condition
  if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
    if (RADIATION_ENABLED|| IM_RADIATION_ENABLED) EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, DiskRadInnerX1);
  }
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
    if (RADIATION_ENABLED|| IM_RADIATION_ENABLED) EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, DiskRadOuterX1);
  }
  if(mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if(mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
 
  // Enroll User Source terms
  if (((RADIATION_ENABLED|| IM_RADIATION_ENABLED)&&fixedemission == 0)||!(RADIATION_ENABLED|| IM_RADIATION_ENABLED)){
    EnrollUserExplicitSourceFunction(PlanetarySourceTerms);
  }
  return;
}

//======================================================================================
//////! \fn void Mesh::TerminateUserMeshProperties(void)
//////  \brief Clean up the Mesh properties
//////======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  // free memory

  if(ifov_flag==1 or ifov_flag==2){
    x1area.DeleteAthenaArray();
    x2area.DeleteAthenaArray();
    x2area_p1.DeleteAthenaArray();
    x3area.DeleteAthenaArray();
    x3area_p1.DeleteAthenaArray();
    vol.DeleteAthenaArray();
  }

  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED){

    opacitytableross.DeleteAthenaArray();
    opacitytableplanck.DeleteAthenaArray();
    logttable.DeleteAthenaArray();
    logptable.DeleteAthenaArray();

  }

  return;
}

//======================================================================================
//! \file disk.cpp
//  \brief Initializes Keplerian accretion disk in spherical polar coords
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  std::srand(gid);

  AthenaArray<Real> ir_cm;
  Real *ir_lab;

  smoothin = pin->GetOrAddReal("problem","smoothin",0.0);
  smoothtr = pin->GetOrAddReal("problem","smoothtr",0.0);
  phydro->hsrc.smoothin = smoothin;
  phydro->hsrc.smoothtr = smoothtr;


  Real crat, prat;
  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED){
    ir_cm.NewAthenaArray(prad->n_fre_ang);
    crat = prad->crat;
    prat = prad->prat;
  }
  

  // Set initial magnetic fields
  if (MAGNETIC_FIELDS_ENABLED){
    if(ifield==1||ifield==2||ifield==3||ifield==4||ifield==6||ifield==7||ifield==8){
      // Compute vector potential
      AthenaArray<Real> a1,a2,a3;
      int nx1 = (ie-is)+1 + 2*(NGHOST);
      int nx2 = (je-js)+1 + 2*(NGHOST);
      int nx3 = (ke-ks)+1 + 2*(NGHOST);
      a1.NewAthenaArray(nx3,nx2,nx1);
      a2.NewAthenaArray(nx3,nx2,nx1);
      a3.NewAthenaArray(nx3,nx2,nx1);

      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie+1; i++) {
            a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
            a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
            a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
          }
        }
      }

      // Initialize interface fields
      AthenaArray<Real> area,len,len_p1;
      area.NewAthenaArray(nx1);
      len.NewAthenaArray(nx1);
      len_p1.NewAthenaArray(nx1);

      // for 1,2,3-D
      for (int k=ks; k<=ke; ++k) {
        // reset loop limits for polar boundary
        int jl=js; int ju=je+1;
        if (pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar") || pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar_wedge")) jl=js+1; 
        if (pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar") || pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar_wedge")) ju=je;
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,is,ie,area);
          pcoord->Edge3Length(k,j,is,ie+1,len);
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = -1.0*(len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
          }
        }
      }
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face3Area(k,j,is,ie,area);
          pcoord->Edge2Length(k,j,is,ie+1,len);
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = (len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
          }
        }
      }
      // for 2D and 3D
      if (block_size.nx2 > 1) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            pcoord->Face1Area(k,j,is,ie+1,area);
            pcoord->Edge3Length(k,j  ,is,ie+1,len);
            pcoord->Edge3Length(k,j+1,is,ie+1,len_p1);
            for (int i=is; i<=ie+1; ++i) {
              pfield->b.x1f(k,j,i) = (len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
            }
          }
        }
        for (int k=ks; k<=ke+1; ++k) {
          for (int j=js; j<=je; ++j) {
            pcoord->Face3Area(k,j,is,ie,area);
            pcoord->Edge1Length(k,j  ,is,ie,len);
            pcoord->Edge1Length(k,j+1,is,ie,len_p1);
            for (int i=is; i<=ie; ++i) {
              pfield->b.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);
            }
          }
        }
      }
      // for 3D only
      if (block_size.nx3 > 1) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            pcoord->Face1Area(k,j,is,ie+1,area);
            pcoord->Edge2Length(k  ,j,is,ie+1,len);
            pcoord->Edge2Length(k+1,j,is,ie+1,len_p1);
            for (int i=is; i<=ie+1; ++i) {
              pfield->b.x1f(k,j,i) -= (len_p1(i)*a2(k+1,j,i) - len(i)*a2(k,j,i))/area(i);
            }
          }
        }
        for (int k=ks; k<=ke; ++k) {
          // reset loop limits for polar boundary
          int jl=js; int ju=je+1;
          if (pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar") || pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar_wedge")) jl=js+1; 
          if (pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar") || pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar_wedge")) ju=je;
          for (int j=jl; j<=ju; ++j) {
            pcoord->Face2Area(k,j,is,ie,area);
            pcoord->Edge1Length(k  ,j,is,ie,len);
            pcoord->Edge1Length(k+1,j,is,ie,len_p1);
            for (int i=is; i<=ie; ++i) {
              pfield->b.x2f(k,j,i) += (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
            }
          }
        }
      }
  
      a1.DeleteAthenaArray();
      a2.DeleteAthenaArray();
      a3.DeleteAthenaArray();
      area.DeleteAthenaArray();
      len.DeleteAthenaArray();
      len_p1.DeleteAthenaArray();
    }
  }
  //  Initialize density
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN,k,j,i) = DenProfile(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k));
        if (NSCALARS > 0) {
          if (pcoord->x1v(i)<rscal) pscalars->s(0,k,j,i) = phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  // ifield 5: toroidal fields within 2 MIDPLANE disk scale height 
  if (MAGNETIC_FIELDS_ENABLED){
   if(ifield==5){
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          Real r = std::max(fabs(x1*sin(x2)),xcut);
          Real z = fabs(x1*cos(x2));
          Real p_over_r = p0_over_r0;
	  Real p_over_r_mid = p_over_r;
          if (NON_BAROTROPIC_EOS) {
	    p_over_r_mid = PoverR(x1*sin(x2), PI/2., pcoord->x3v(k)); 
	    p_over_r = PoverR(x1, x2, pcoord->x3v(k)); 
	  }
          if (fabs(z)<2.*sqrt(p_over_r_mid*r*r*r/gms)) 
	    pfield->b.x3f(k,j,i) = sqrt(2.*p_over_r*phydro->u(IDN,ks,j,i)/beta);
        }
      }
    }
   }
  }

  // add density perturbation, needs to be after ifield 5 which assumes the disk is axisymmetric in the x3 direction so that divB=0.
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (MAGNETIC_FIELDS_ENABLED){
          if(pcoord->x1v(i)>=routfield){
            pfield->b.x1f(k,j,i)=0.0;
            pfield->b.x2f(k,j,i)=0.0;
            pfield->b.x3f(k,j,i)=0.0;  
          }
        }
	if(per==0){
 	  phydro->u(IDN,k,j,i) = std::max(phydro->u(IDN,k,j,i)*
		               (1+amp*((double)rand()/(double)RAND_MAX-0.5)), 
		               rho_floor(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k)));
	}
      }
    }
  }

  //  Initialize velocity
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        Real v1, v2, v3;
        VelProfile(x1, x2, x3, phydro->u(IDN,k,j,i), v1, v2, v3);
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
	phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
 	phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
      }
    }
  }

  int il = is; int jl = js; int kl = ks;
  int iu = ie; int ju = je; int ku = ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }
  if (NON_BAROTROPIC_EOS&&(RADIATION_ENABLED|| IM_RADIATION_ENABLED)){
    for(int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          prad->t_floor_(k,j,i)=tfloor;
//          prad->t_ceiling_(k,j,i)=1000.*tfloor;
        }
      }
    }
  }
  //  Initialize pressure
  if (NON_BAROTROPIC_EOS){
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          Real r = std::max(fabs(x1*sin(x2)),xcut);
          Real p_over_r = PoverR(x1, x2, pcoord->x3v(k)); 
	  Real rho = phydro->u(IDN,k,j,i);
          Real press = p_over_r*rho;
          Real gast;
	  if (RADIATION_ENABLED|| IM_RADIATION_ENABLED) {
            Real temp0 = press/rho;
            Real coef1 = prat/3.0;
            Real coef2 = rho;
            Real coef3 = -press;
            gast = temp0;
//            gast = Rtsafe(Tequilibrium, 0.0, temp0, 1.e-12, coef1, coef2, coef3, 0.0);
            if(gast < tfloor) gast = tfloor;
            // initialize radiation quantity
            for(int n=0; n<prad->n_fre_ang; ++n)
              ir_cm(n) = gast * gast * gast * gast;
            Real *mux = &(prad->mu(0,k,j,i,0));
            Real *muy = &(prad->mu(1,k,j,i,0));
            Real *muz = &(prad->mu(2,k,j,i,0));

            ir_lab = &(prad->ir(k,j,i,0));
            prad->pradintegrator->ComToLab(0,0,phydro->u(IM3,k,j,i)/phydro->u(IDN,k,j,i),mux,muy,muz,ir_cm,ir_lab);
          }else{
            gast = press/rho;
          }
          press = gast*rho;

	  phydro->u(IEN,k,j,i) = press/(gamma_gas - 1.0);

          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+
				       SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);

	}
      }
    }

    if (MAGNETIC_FIELDS_ENABLED){
      pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
      for(int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) +=
              0.5*(SQR((pfield->bcc(IB1,k,j,i)))
               + SQR((pfield->bcc(IB2,k,j,i)))
               + SQR((pfield->bcc(IB3,k,j,i))));
          }
        }
      }
    }
  }

  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED)
    ir_cm.DeleteAthenaArray();

  return;
}

//--------------------------------------------------------------------------------------
//! \fn static Real rho_floor
//  \brief density floor
//  dflag==4 CPD centered on the planet

static Real rho_floor(const Real x1, const Real x2, const Real x3)
{
  Real rhof;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    rhof=rho_floorsf(rsphsf,thetasf,phisf);
  }else{
    rhof=rho_floorsf(x1,x2,x3); 
  }
  return rhof;
}


static Real rho_floorsf(const Real x1, const Real x2, const Real x3)
{
  Real r = fabs(x1*sin(x2));
  Real zmod = std::max(fabs(x1*cos(x2)),xcut);
  Real rhofloor=0.0;
  if (r<xcut) {
    rhofloor=rho_floor0*pow(xcut/r0, slope_rho_floor);//*((x1min-r)/x1min*19.+1.);
    if(x1<3.*xcut) rhofloor=rhofloor*(5.-(x1-xcut)/xcut*2.)*((xcut-r)/xcut*4.+1.);
  }else{
    rhofloor=rho_floor0*pow(r/r0, slope_rho_floor);
  }
  rhofloor=rhofloor/(zmod/r0)/(zmod/r0)*x1min/x1;
//  if(x1<rmagsph) rhofloor=rhofloor*pow((rmagsph/x1),3);
  if(x1<rmagsph) rhofloor=rho0*mm*mm/beta/1.e6*pow((r0/x1),7)*(9.*cos(x2)*cos(x2)+1.);
  return std::max(rhofloor,dfloor);
}

void AlphaVis(HydroDiffusion *phdif, MeshBlock *pmb,const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,int is, int ie, int js, int je, int ks, int ke) {
  Real rad,phi,z,poverr=p0_over_r0;
  poverr=p0_over_r0;
  Coordinates *pcoord = pmb->pcoord;
  if (phdif->nu_iso > 0.0) {
    Real alphad=alpha;
    if(pmb->pmy_mesh->time>astoptime) alphad=alphalow;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          rad = std::max(fabs(pcoord->x1v(i)*sin(pcoord->x2v(j))),xcut);
          if (NON_BAROTROPIC_EOS) poverr=w(IEN,k,j,i)/w(IDN,k,j,i);
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alphad*std::pow(rad/r0,aslope)*poverr/(sqrt(gm0/rad)/rad);
        }
      }
    }
  }

  return;
}


//---------------------------------------------------------------------------------------
//! \f static Real DenProfile
//  dflag  1: uniform,     2: step function,    31: disk structure with numerical integration assuming hydrostatic equilibrium
//         3: normal disk structure with the anayltically derived hydrostatic equilibrium, tflag has to be 0
//         4: CPD setup
//
static Real DenProfile(const Real x1, const Real x2, const Real x3)
{
  Real den;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    den=DenProfilesf(rsphsf,thetasf,phisf);
  }else{
    den=DenProfilesf(x1,x2,x3);
  }
  return den;
}

static Real DenProfilesf(const Real x1, const Real x2, const Real x3)
{  
  Real den;
  std::stringstream msg;
  if (dflag == 1) {
    den = rho0;
  } else if(dflag == 2) {
    Real y = x1*fabs(sin(x2))*sin(x3);
    if (y<0.2*x1max)
      den = 0.5*rho0;
    else
      den = rho0;
  } else if (dflag == 31) {
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real denmid = rho0*pow(r/r0,dslope);
    if (r>=rinjump && r<=routjump) denmid=denmid*(djump*pow(sin((r-rinjump)/(routjump-rinjump)*PI/2.),2)+1.);
    if (r>=routjump) denmid=denmid*(djump+1.);
    Real zo = 0.0;
    Real zn = zo;
    den=denmid;
    Real x1o,x2o,x3o,x1n,x2n,x3n,coe,h,dz,poverro,poverrn; 
    while (zn <= z){
      coe = gms*0.5*(1./sqrt(r*r+zn*zn)-1./sqrt(r*r+zo*zo));
      x1o = sqrt(r*r+zo*zo);
      x2o = atan(r/zo);
      x3o = 0.0;
      poverro=PoverRsf(x1o,x2o,x3o);
      h = sqrt(poverro)/sqrt(gms/r/r/r);
      dz = h/32.;
      x1n = sqrt(r*r+zn*zn);
      x2n = atan(r/zn);
      x3n = 0.0;
      poverrn=PoverRsf(x1n,x2n,x3n);
      den = den*(coe+poverro)/(poverrn-coe);
      zo = zn;
      zn = zo+dz;
    }
  } else if (dflag==3){
    if(tflag!=0){
      msg <<"### FATAL ERROR in Problem Generator"  << std::endl
          <<"tflag has to be zero when dflag is 3" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    Real denmid = rho0*pow(r/r0,dslope);
    if (r>=rinjump && r<=routjump) denmid=denmid*(djump*pow(sin((r-rinjump)/(routjump-rinjump)*PI/2.),2)+1.);
    if (r>=routjump) denmid=denmid*(djump+1.);
    den = denmid*exp(gms/p_over_r*(1./sqrt(SQR(r)+SQR(z))-1./r));
  } else if (dflag==32){
    if(tflag!=0){
      msg <<"### FATAL ERROR in Problem Generator"  << std::endl
          <<"tflag has to be zero when dflag is 3" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    Real denmid = rho0*pow(r/r0,dslope);
    if (r>=rinjump && r<=routjump) denmid=denmid*(djump*pow(sin((r-rinjump)/(routjump-rinjump)*PI/2.),2)+1.);
    if (r>=routjump) denmid=denmid*(djump+1.);
    den = denmid*exp(gms/p_over_r*(1./sqrt(SQR(r)+SQR(z))-1./r));
    if (x1<rmagsph){
      Real rint=x1min;
      Real p_over_r = p0_over_r0;
      if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1min, x2, x3);
      Real pre=denstar*p_over_r;
      Real dr = rint/100.;
      while(rint<x1){
        pre = pre - dr*gms/rint/rint*(rint-smoothin)*(rint-smoothin)/((rint-smoothin)*(rint-smoothin)+smoothtr*smoothtr)*pre/p_over_r; 
        rint = rint + dr;
        if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(rint, x2, x3);
      }
      den = pre/p_over_r;
    }
  } else if (dflag==4){ //CPD density
    Real xsf,ysf,zsf,xpf,ypf,zpf,rcylsf,rcyld,den0;
    ConvSphCar(x1, x2, x3, xsf, ysf, zsf);  
    rcylsf=std::max(sqrt(xsf*xsf+ysf*ysf),xcut); 
    rcyld=r0-rcylsf;
    Real dp=sqrt((xsf+r0)*(xsf+r0)+ysf*ysf+zsf*zsf);
    if(dp>rprotect){
      if(rcyld<-gapw) den0=(2.-exp((rcyld+gapw)/wtran))*(sh-sl)/2.+sl;
      if(rcyld>-gapw&&rcyld<0.0) den0=exp((-rcyld-gapw)/wtran)*(sh-sl)/2.+sl;
      if(rcyld<gapw&&rcyld>0.0)  den0=exp((rcyld-gapw)/wtran)*(sh-sl)/2.+sl;
      if(rcyld>gapw)  den0=(2.-exp((gapw-rcyld)/wtran))*(sh-sl)/2.+sl;
    }else{
      den0=sh;
    }
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    Real denmid = den0*pow(rcylsf/r0,dslope);
    den = denmid*exp(gms/p_over_r*(1./sqrt(SQR(rcylsf)+SQR(zsf))-1./rcylsf));
  } else if (dflag==5){
    Real r = fabs(x1*sin(x2));
    Real z = fabs(x1*cos(x2));
    den = Interp(r, z, nrtable, nztable, rtable, ztable, dentable);
  }
  return(std::max(den,rho_floorsf(x1, x2, x3)));
}

static Real Interp(const Real r, const Real z, const int nxaxis, const int nyaxis, AthenaArray<Real> &xaxis, AthenaArray<Real> &yaxis, AthenaArray<Real> &datatable )
{
  Real drf, dzf, data;
  int i=0;
  if(r<xaxis(0)){
    i=0;
    drf=0.0;
  }else if(r>xaxis(nxaxis-1)){
    i=nxaxis-2;
    drf=1.0;
  }else{
    while(i< nxaxis-1 && (r-xaxis(i))*(r-xaxis(i+1))>=0.0){i++;}
    drf=(r-xaxis(i))/(xaxis(i+1)-xaxis(i));
  }
  int j=0;
  if(z<yaxis(0)){
    j=0;
    dzf=0.0;
  }else if(z>yaxis(nyaxis-1)){
    j=nyaxis-2;
    dzf=1.0;
  }else{
    while(j< nyaxis-1 && (z-yaxis(j))*(z-yaxis(j+1))>=0.0){j++;}
    dzf=(z-yaxis(j))/(yaxis(j+1)-yaxis(j));
  }
  data = datatable(j,i)+(datatable(j+1,i)-datatable(j,i))*dzf+
                       (datatable(j,i+1)-datatable(j,i))*drf;
  return data; 
}
//---------------------------------------------------------------------------------------
//! \f static Real PoverR
//  tflag   0:  radial power law, vertical isothermal
//          1:  radial power law, vertical within h, T, beyond 4 h, 50*T, between power law
//          dflag==4 CPD centered on the planet
static Real PoverR(const Real x1, const Real x2, const Real x3)
{
  Real por;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    por=PoverRsf(rsphsf,thetasf,phisf);
  }else{
    por=PoverRsf(x1,x2,x3);
  }
  return por;
}


static Real PoverRsf(const Real x1, const Real x2, const Real x3)
{  
  Real poverr;
  if (tflag == 0) {
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    poverr = p0_over_r0*pow(r/r0, pslope);
  } else if(tflag == 1){
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real z = fabs(x1*cos(x2));
    Real h = sqrt(poverrmid)/sqrt(gms/r/r/r);
    if (z<h) {
      poverr=poverrmid;
    } else if (z>4*h){
      poverr=50.*poverrmid;
    } else {
      poverr=poverrmid*pow(3.684,(z-h)/h);
    }
  } else if(tflag == 5){
    Real r = fabs(x1*sin(x2));
    Real z = fabs(x1*cos(x2));
    poverr = Interp(r, z, nrtable, nztable, rtable, ztable, portable);
  }
  return(poverr);
}

//------------------------------------------------------------------------------------
////! \f horseshoe velocity profile for CPD
//

static void VelProfile(const Real x1, const Real x2, const Real x3,
                                const Real den, Real &v1, Real &v2, Real &v3)
{
  std::stringstream msg;
  Real xsf,ysf,zsf,xpf,ypf,zpf,rcylsf,rcyld,den0, rsphsf, thetasf, phisf;
  Real vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf;
  if(dflag==4){
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf, ysf, zsf, rsphsf, thetasf, phisf);
    VelProfilesf(rsphsf, thetasf, phisf, den, vrsphsf, vthetasf, vphisf); 
    ConvVSphCar(rsphsf, thetasf, phisf, vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf);
    ConvVCarSph(xpf, ypf, zpf, vxsf, vysf, vzsf, v1, v2, v3);
  }else{
    VelProfilesf(x1,x2,x3,den,v1,v2,v3);
  }
  return;
}

void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi){
  rad=sqrt(x*x+y*y+z*z);
  theta=acos(z/rad);
  phi=atan2(y,x);
  return;
}

void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z){
  x=rad*sin(theta)*cos(phi);
  y=rad*sin(theta)*sin(phi);
  z=rad*cos(theta);
  return;
}

void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp){
  Real rads=sqrt(x*x+y*y+z*z);
  Real radc=sqrt(x*x+y*y);
  vr=vx*x/rads+vy*y/rads+vz*z/rads;
  vt=((x*vx+y*vy)*z-radc*radc*vz)/rads/radc;
  vp=vy*x/radc-vx*y/radc;
  return;
}

void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz){
  vx=vr*sin(theta)*cos(phi)+vt*cos(theta)*cos(phi)-vp*sin(phi);
  vy=vr*sin(theta)*sin(phi)+vt*cos(theta)*sin(phi)+vp*cos(phi);
  vz=vr*cos(theta)-vt*sin(theta);
  return;
}


//------------------------------------------------------------------------------------
//! \f velocity profile
// vflag  1: uniform in cartesion coordinate with vy0 along +y direction
//        2: disk velocity with the analytically derived profile, dflag has to be 3 or 4
//        21: similar to 2, but within rigid it is a solid body rotation, dflag has to be 3 or 4
//        22: disk velocity derived by numericlally solving radial pressure gradient
//        3: solid body rotation
//	  4: 2D Keplerian velocity 
//
static void VelProfilesf(const Real x1, const Real x2, const Real x3, 
		       const Real den, Real &v1, Real &v2, Real &v3)
{  
  std::stringstream msg;
  if (vflag == 1) {
    v1 = vy0*sin(x2)*sin(x3);
    v2 = vy0*cos(x2)*sin(x3);
    v3 = vy0*cos(x3);
  } else if (vflag == 2 || vflag == 21 || vflag == 22) {       
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real vel;
    if (den <= (1.+amp)*rho_floorsf(x1, x2, x3)) {
      vel = sqrt(gms*SQR(fabs(x1*sin(x2)))/(SQR(fabs(x1*sin(x2)))+SQR(z))
		 /sqrt(SQR(fabs(x1*sin(x2)))+SQR(z)));
    } else {
      if (NON_BAROTROPIC_EOS){
        if(vflag==2 || vflag==21){
	  if(dflag!=3&&dflag!=4&&dflag!=32){
            msg <<"### FATAL ERROR in Problem Generator"  << std::endl
            <<"dflag has to be 3 or 4 when vflag is 2 or 21" << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
 	  Real p_over_r = PoverRsf(x1, x2, x3);
          vel = (dslope+pslope)*p_over_r/(gms/r) + (1.+pslope) - pslope*r/x1;
          vel = sqrt(gms/r)*sqrt(vel);
        }
        if(vflag==22){
          Real dx1=0.001*x1;
          Real dx2=PI*0.001;
	  Real dpdR= (PoverRsf(x1+dx1, x2, x3)*DenProfilesf(x1+dx1, x2, x3)
                    -PoverRsf(x1-dx1, x2, x3)*DenProfilesf(x1-dx1, x2, x3))/2./dx1*sin(x2)+
		   (PoverRsf(x1, x2+dx2, x3)*DenProfilesf(x1, x2+dx2, x3)
                    -PoverRsf(x1, x2-dx2, x3)*DenProfilesf(x1, x2-dx2, x3))/2./dx2*cos(x2)/x1;
	  vel = sqrt(std::max(gms*r*r/sqrt(r*r+z*z)/(r*r+z*z)
		     +r/DenProfilesf(x1, x2, x3)*dpdR,0.0));
        }
      } else {
        vel = dslope*p0_over_r0/(gms/r)+1.0;
        vel = sqrt(gms/r)*sqrt(vel);
      }
    }
    if (vflag == 21) {
      if (x1 <= rrigid) {
	vel=origid*fabs(x1*sin(x2));
      }
    }
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  } else if (vflag ==3) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = gms*x1*sin(x2);
  } 
  if(omegarot!=0.0) v3-=omegarot*fabs(x1*sin(x2));
  return;
}
//--------------------------------------------------------------------------------------
//! \fn static Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential
/*
   ifield  1:  constant B field parallel to z xaxis over Z direciton, constatnt over radius beyond rcut, smooth transition to 0 within rs
           6:  constatn B field parallel to z xaxis over Z direction, plasma beta at the midplane is constant over radius
	   7:  similar to 6, but the field becomes a constant over the radius within x1min to avoid the singularity
           8:  7+dipole field
   per     1:  perturbed along phi direction as |sin(phi)|
 * */
static Real A3(const Real x1, const Real x2, const Real x3)
{  
  Real a3=0.0;
  if(ifield==1) {
    Real r = fabs(x1*sin(x2));
    if(r>=rcut) {
      a3 = r*b0/2.0;
    }else if(r<=rs) {
      a3 = 0.0;
    }else{
      Real a=rs/rcut;
      Real adenom=(-1.+a)*(-1.+a)*(-1.+a);
      Real b1=-3.*(1.+a*a)*b0/adenom;
      Real b2=2.*(1.+a+a*a)*b0/adenom/(1.+a);
      Real b3=(3.+a+a*a+a*a*a)*a*b0/adenom/(1.+a);
      Real c=-a*a*a*b0*rcut*rcut/2./adenom/(1.+a);
      a3 = b1/rcut*r*r/3.+b2/rcut/rcut*r*r*r/4.+b3*r/2.+c/r;
    }    
  }
  if(ifield==6) {
    Real dx2coarse=PI/nx2coarse;
    Real r1 = x1*dx2coarse/2.;
    Real r = fabs(x1*sin(x2))+1.e-6;
    Real a=(pslope+dslope)/2.;
    a3 = b0/pow(r0,a)*pow(r,a+1.)/(a+2.)+b0/pow(r0,a)*pow(r1,a+2.)/r*(1.-1./(a+2.));
  }
  if(ifield==7) {
    Real r = fabs(x1*sin(x2));
    Real a=(pslope+dslope)/2.;
    if (r<=xcut){
      a3 = r/2.0*b0*pow(xcut/r0,a);
    }else{
      a3 = b0/pow(r0,a)*pow(r,a+1.)/(a+2.)+b0*(pow(xcut,a+2)/pow(r0,a)*(1./2.-1./(a+2.)))/r;
    }
  }
  if(ifield==8) {
    Real r = fabs(x1*sin(x2));
    Real a=(pslope+dslope)/2.;
    if (r<=xcut){
      a3 = r/2.0*b0*pow(xcut/r0,a);
    }else{
      a3 = b0/pow(r0,a)*pow(r,a+1.)/(a+2.)+b0*(pow(xcut,a+2)/pow(r0,a)*(1./2.-1./(a+2.)))/r;
    }
    a3 += mm*b0*r/x1/x1/x1*(1.-exp(-(sin(x2)/0.2)*(sin(x2)/0.2))); 
  }
  if(per==1){
    a3 = a3*fabs(sin(x3));
  }
  return(a3);
}

//--------------------------------------------------------------------------------------
//! \fn static Real A2(const Real x1,const Real x2,const Real x3)
/*  \brief A2: 2-component of vector potential
    ifield:2  field loop with 1.e-6 amplitude
    ifield:3  uniform field parallel to x xaxis
    ifield:4  uniform field parallel to x xaxis, but when y<1.0, field become half
*/
static Real A2(const Real x1, const Real x2, const Real x3)
 { 
  Real a2=0.0;
  Real az=0.0;
  if(ifield==2) {
    Real x=x1*fabs(sin(x2))*cos(x3);
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI){
     x=-x;
     y=-y;
    }
    Real z=x1*cos(x2);
    if(sqrt(SQR(x-xc)+SQR(y-yc))<=0.5 && fabs(z-zc)<0.2){
      az=1.0e-6*(0.5-sqrt(SQR(x-xc)+SQR(y-yc)));
    }
    a2=-az*fabs(sin(x2));
  }else if(ifield==3){
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a2=-b0*y*fabs(sin(x2));
  }else if(ifield==4){
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a2=-b0*y*fabs(sin(x2));
    a2=a2*sin(y);
  }
  return(a2);
}

//--------------------------------------------------------------------------------------
//! \fn static Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential
/*
    ifield:2  field loop with 1.e-6 amplitude
    ifield:3  uniform field parallel to x xaxis
    ifield:4  uniform field parallel to x xaxis, but the amplitude is sin(y) 
  */
static Real A1(const Real x1, const Real x2, const Real x3)
{
  Real a1=0.0;
  Real az=0.0;
  if(ifield==2) {
    Real x=x1*fabs(sin(x2))*cos(x3);
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI){
     x=-x;
     y=-y;
    }
    Real z=x1*cos(x2);
    if(sqrt(SQR(x-xc)+SQR(y-yc))<=0.5 && fabs(z-zc)<0.2){
      az=1.e-6*(0.5-sqrt(SQR(x-xc)+SQR(y-yc)));
    }
    a1=az*cos(x2);
  }else if(ifield==3) {
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a1=b0*y*cos(x2);
  }else if(ifield==4) {
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a1=b0*y*cos(x2);
    a1=a1*sin(y);
  }
  return(a1);
}

//------------------------------------------------------------
// f: User-defined boundary Condition
// 
// Summary of all BCs for inner X1 
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // inner x1 hydro BCs
  if(hbc_ix1 == 1)
    SteadyInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ix1 == 2)
    DiodeInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ix1 == 4)
    UserOutflowInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ix1 == 5)
    ReflectInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);

  // inner x1 field BCs
  if (MAGNETIC_FIELDS_ENABLED) {
    if(mbc_ix1 == 1)
      FieldSteadyInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ix1 == 2)
      FieldOutflowInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ix1 == 3)
      FieldDivFreeInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  }
}

// Summary of all BCs for outer X1 
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // outer x1 hydro BCs
  if(hbc_ox1 == 1)
    SteadyOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ox1 == 2)
    DiodeOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ox1 == 3)
    InflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ox1 == 4)
    UserOutflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);

  // outer x1 field BCs
  if (MAGNETIC_FIELDS_ENABLED) {
    if(mbc_ox1 == 1)
      FieldSteadyOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ox1 == 2)
      FieldOutflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ox1 == 3)
      FieldDivFreeOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ox1 == 4)
      FieldInflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ox1 == 5)
      FieldInflowAdvectOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
    else if(mbc_ox1 == 6)
      FieldInflowVerticalOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  }
}

// Summary of all BCs for Inner X2
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // inner X2 hydro BCs
  if(hbc_ix2 == 1)
    SteadyInnerX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ix2 == 2)
    StratInnerX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  // inner X2 field BCs
  if (MAGNETIC_FIELDS_ENABLED) {
    if(mbc_ix2 == 1)
      FieldOutflowInnerX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh); 
  }
}

// Summary of all BCs for Outer X2
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // outer X2 hydro BCs
  if(hbc_ox2 == 1)
    SteadyOuterX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  else if(hbc_ox2 == 2)
    StratOuterX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);

  // outer X2 field BCs
  if (MAGNETIC_FIELDS_ENABLED) {
    if(mbc_ox2 == 1)
      FieldOutflowOuterX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke,ngh);
  }
}

// Summary of all radiation BCs for inner X1 
void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // inner x1 hydro BCs
  if(iRadInner==0) InfluxRadInnerX1(pmb, pco, prad, w, bc, ir, time, dt, is,ie,js,je,ks,ke,ngh);
  if(iRadInner==1) InconstfluxRadInnerX1(pmb, pco, prad, w, bc, ir, time, dt, is,ie,js,je,ks,ke,ngh);
}

// Summary of all radiation BCs for outer X1 
void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // outer x1 hydro BCs
  InfluxRadOuterX1(pmb, pco, prad, w, bc, ir, time, dt, is,ie,js,je,ks,ke,ngh);
}

void InfluxRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  int &nang = prad->nang;
  int &nfreq = prad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
  for (int i=1; i<=ngh; ++i) {
  for (int ifr=0; ifr<nfreq; ++ifr){
    for(int n=0; n<nang; ++n){
      int ang=ifr*nang+n;
      Real& miux=prad->mu(0,k,j,is,n);
      if(miux < 0){
        ir(k,j,is-i,ang) = ir(k,j,is,ang);
      }else{
        ir(k,j,is-i,ang) = radinfluxinnerx1;
      }
    }

  }
  }}}
  return;
}

void InconstfluxRadInnerX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  int &nang = prad->nang;
  int &nfreq = prad->nfreq; // number of frequency bands

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real rho = w(IDN,k,j,is);
      Real tgasi = w(IEN,k,j,is)/rho;
      for (int i=1; i<=ngh; ++i) {
      for (int ifr=0; ifr<nfreq; ++ifr){
        Real jlocal=tgasi*tgasi*tgasi*tgasi+3.*(prad->sigma_s(k,j,is,ifr)+prad->sigma_a(k,j,is,ifr))*consFr*(pco->x1v(is)-pco->x1v(is-i));
        Real coefa = 0.0, coefb = 0.0;
        Real coefa1 = 0.0, coefb1 = 0.0;
        for(int n=0; n<nang; ++n){
          Real &miux = prad->mu(0,k,j,is-i,n);
          Real &weight = prad->wmu(n);
          if(miux > 0.0){
            coefa += weight;
            coefb += (miux * weight);
          }else{
            coefa1 += weight;
            coefb1 += (miux * weight);
          }
        }

        for(int n=0; n<nang; ++n){
          Real &miux = prad->mu(0,k,j,i,n);
          int ang=ifr*nang+n;
          if(miux > 0.0){
            ir(k,j,is-i,ang) = (jlocal/coefa1-consFr/coefb1)/(coefa/coefa1-coefb/coefb1);
	  }else{
            ir(k,j,is-i,ang) = (jlocal/coefa-consFr/coefb)/(coefa1/coefa-coefb1/coefb);
	  }
	}
      }
      }
    }
  }
  return;
}

void InfluxRadOuterX1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, const AthenaArray<Real> &bc, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  int &nang = prad->nang;
  int &nfreq = prad->nfreq; // number of frequency bands

  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
  for (int i=1; i<=ngh; ++i) {
  for (int ifr=0; ifr<nfreq; ++ifr){
    for(int n=0; n<nang; ++n){
      int ang=ifr*nang+n;
      Real& miux=prad->mu(0,k,j,ie,n);
      if(miux > 0){
        ir(k,j,ie+i,ang) = ir(k,j,ie,ang);
      }else{
        ir(k,j,ie+i,ang) = radinfluxouterx1;
      }
    }


  }
  }}}

  return;
}




// Hydro BC at inner X1: reset to initial condition
void SteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
 		   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){ //cannot use time since time here is temperaory and can be the halfstep and fullstep time, first time is meshtime which will not change during one whole integration including predict and correct
  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) { 
      for (int i=1; i<=ngh; ++i) {
        Real x1 = pcoord->x1v(is-i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        prim(IDN,k,j,is-i) = DenProfile(x1, x2, x3);

        Real v1, v2, v3;
        VelProfile(x1, x2, x3, prim(IDN,k,j,is-i), v1, v2, v3);
        
        prim(IM1,k,j,is-i) = v1;
        prim(IM2,k,j,is-i) = v2;
        prim(IM3,k,j,is-i) = v3;
        if (NON_BAROTROPIC_EOS) 
          prim(IEN,k,j,is-i) = PoverR(x1, x2, x3)*prim(IDN,k,j,is-i);
      }
    }
  }
 }
}

//  Hydro BC at outer X1: reset to initial condition
void SteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real x1 = pcoord->x1v(ie+i);
          Real x2 = pcoord->x2v(j);
          Real x3 = pcoord->x3v(k);
          prim(IDN,k,j,ie+i) = DenProfile(x1, x2, x3);

          Real v1, v2, v3;
          VelProfile(x1, x2, x3, prim(IDN,k,j,ie+i), v1, v2, v3);

          prim(IM1,k,j,ie+i) = v1;
          prim(IM2,k,j,ie+i) = v2;
          prim(IM3,k,j,ie+i) = v3;
          if (NON_BAROTROPIC_EOS) prim(IEN,k,j,ie+i) = PoverR(x1, x2, x3)*prim(IDN,k,j,ie+i);
        }
      }
    }
  }
 }
}

// Hydro BC at inner X1: copy density and pressure, diode Vr, copy Vphi, Vtheta=0
void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);

	Real v1, v2, v3;
        Real x1 = pco->x1v(is-i);
        Real x2 = pco->x2v(j);
        Real x3 = pco->x3v(k);
	VelProfile(x1, x2, x3, prim(IDN,k,j,is-i), v1, v2, v3);
        prim(IM1,k,j,is-i) = std::min(prim(IM1,k,j,is), 0.0);
        prim(IM2,k,j,is-i) = 0.0;
        prim(IM3,k,j,is-i) = v3;

        if(NON_BAROTROPIC_EOS) 
          prim(IEN,k,j,is-i) = prim(IEN,k,j,is);
      }
    }
  }
}


// Hydro BC at outer X1: copy density and pressure, diode Vr, copy Vphi, Vtheta=0 
void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);

        Real v1, v2, v3;
        Real x1 = pco->x1v(ie+i);
        Real x2 = pco->x2v(j);
        Real x3 = pco->x3v(k);
        VelProfile(x1, x2, x3, prim(IDN,k,j,ie+i), v1, v2, v3);
        prim(IM1,k,j,ie+i) = std::max(prim(IM1,k,j,ie), 0.0);
        prim(IM2,k,j,ie+i) = 0.0;
        prim(IM3,k,j,ie+i) = v3;
        if(NON_BAROTROPIC_EOS)
          prim(IEN,k,j,ie+i) = prim(IEN,k,j,ie);
      }
    }
  }
}

// Hydro BC at inner X1: similar to built-in outflow except do not allow mass flow in to the active region
void UserOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is);
          if(n==IVX&&prim(n,k,j,is-i)>0.0) prim(n,k,j,is-i)=0.0;
        }
      }
    }
  }
}

// Hydro BC at outer X1: similar to built-in outflow except do not allow mass flow in to the active region
void UserOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie);
          if(n==IVX&&prim(n,k,j,ie+i)<0.0)prim(n,k,j,ie+i)=0.0;
        }
      }
    }
  }
}

//  Hydro BC at outer X1: L1 inflow
void InflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{

}


void ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb,
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is+i-1);
        prim(IM1,k,j,is-i) = -prim(IM1,k,j,is+i-1);
        prim(IM2,k,j,is-i) = prim(IM2,k,j,is+i-1);
        prim(IM3,k,j,is-i) = prim(IM3,k,j,is+i-1);
        if (NON_BAROTROPIC_EOS){
          prim(IEN,k,j,is-i) = prim(IEN,k,j,is+i-1);
        }
      }
    }
  }
}


// Hydro BC at inner X2: reset to initial condition
void SteadyInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
 		   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int k=ks; k<=ke; ++k) { 
    for (int j=1; j<=ngh; ++j) { 
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(js-j);
        Real x3 = pcoord->x3v(k);
        prim(IDN,k,js-j,i) = DenProfile(x1, x2, x3);

        Real v1, v2, v3;
        VelProfile(x1, x2, x3, prim(IDN,k,js-j,i), v1, v2, v3);
        
        prim(IM1,k,js-j,i) = v1;
        prim(IM2,k,js-j,i) = v2;
        prim(IM3,k,js-j,i) = v3;
        if (NON_BAROTROPIC_EOS) 
          prim(IEN,k,js-j,i) = PoverR(x1, x2, x3)*prim(IDN,k,js-j,i);
      }
    }
  }
 }
}

//  Hydro BC at outer X1: reset to initial condition
void SteadyOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(je+j);
          Real x3 = pcoord->x3v(k);
          prim(IDN,k,je+j,i) = DenProfile(x1, x2, x3);

          Real v1, v2, v3;
          VelProfile(x1, x2, x3, prim(IDN,k,je+j,i), v1, v2, v3);

          prim(IM1,k,je+j,i) = v1;
          prim(IM2,k,je+j,i) = v2;
          prim(IM3,k,je+j,i) = v3;
          if (NON_BAROTROPIC_EOS) prim(IEN,k,je+j,i) = PoverR(x1, x2, x3)*prim(IDN,k,je+j,i);
        }
      }
    }
  }
 }
}


// Hydro BC at inner X2
void StratInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
	  Real this_rho_floor = rho_floor(pco->x1v(k,js,i),pco->x2v(k,js,i),pco->x3v(k,js,i));
	  Real tempa = prim(IEN,k,js,i) / prim(IDN,k,js,i);
          Real vpa = prim(IM3,k,js,i);
          prim(IDN,k,js-j,i) = prim(IDN,k,js,i)*pow(fabs(sin(pco->x2v(js-j))/sin(pco->x2v(js))),vpa*vpa/tempa);
          if (prim(IDN,k,js-j,i) < this_rho_floor) 
	    prim(IDN,k,js-j,i) = this_rho_floor;
          prim(IM1,k,js-j,i) = 0.0;
          prim(IM2,k,js-j,i) = 0.0;
          prim(IM3,k,js-j,i) = prim(IM3,k,js,i);
          if(NON_BAROTROPIC_EOS) 
	    prim(IEN,k,js-j,i) = tempa * prim(IDN,k,js-j,i);
        }
      }
    }
}

// Hydro BC at outer X2
void StratOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real this_rho_floor = rho_floor(pco->x1v(k,je,i),pco->x2v(k,je,i),pco->x3v(k,je,i));
	  Real tempa = prim(IEN,k,je,i) / prim(IDN,k,je,i);
          Real vpa = prim(IM3,k,je,i);
          prim(IDN,k,je+j,i) = prim(IDN,k,je,i)*pow(fabs(sin(pco->x2v(je+j))/sin(pco->x2v(je))),vpa*vpa/tempa);
          if (prim(IDN,k,je+j,i) < this_rho_floor) 
	    prim(IDN,k,je+j,i) = this_rho_floor;
          prim(IM1,k,je+j,i) = 0.0;
          prim(IM2,k,je+j,i) = 0.0;
          prim(IM3,k,je+j,i) = prim(IM3,k,je,i);
          if(NON_BAROTROPIC_EOS)
	    prim(IEN,k,je+j,i) = tempa * prim(IDN,k,je+j,i);
        }
      }
    }
}


// Field BC at inner X1: reset ghost zone to initial condition
void FieldSteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){
  if (MAGNETIC_FIELDS_ENABLED){
    Real eps=pcoord->dx2f(0)/100.;

    // Compute vector potential
    AthenaArray<Real> a1,a2,a3;
    int nx1 = (pmb->ie-pmb->is)+2 + 2*ngh;
    int nx2 = (pmb->je-pmb->js)+2 + 2*ngh;
    int nx3 = (pmb->ke-pmb->ks)+2 + 2*ngh;
    a1.NewAthenaArray(nx3,nx2,nx1);
    a2.NewAthenaArray(nx3,nx2,nx1);
    a3.NewAthenaArray(nx3,nx2,nx1);

    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is-ngh; i<=is-1; i++) {
          a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
        }
      }
    }

    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is-ngh; i<=is; i++) {
          a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
        } 
      }   
    } 

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is-ngh; i<=is; i++) {
          a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
        } 
      }   
    } 

    // Initialize interface fields
    AthenaArray<Real> area,len,len_p1;
    area.NewAthenaArray(nx1);
    len.NewAthenaArray(nx1);
    len_p1.NewAthenaArray(nx1);

    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,is-ngh,is-1,area);
        pcoord->Edge3Length(k,j,is-ngh,is,len);
        for (int i=is-ngh; i<=is-1; ++i) {
	  // x2f should flip sign across the pole
          if(pcoord->x2f(j)<0.0-eps||pcoord->x2f(j)>PI+eps) {
            bb.x2f(k,j,i) = (len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
          }else{
            if(pcoord->x2f(j)>0.0+eps&&pcoord->x2f(j)<PI-eps) 
	      bb.x2f(k,j,i) = -(len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
          }
	}
      }
    }

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face3Area(k,j,is-ngh,is-1,area);
      pcoord->Edge2Length(k,j,is-ngh,is,len);
      for (int i=is-ngh; i<=is-1; ++i) {
	// x3f should flip sign across the pole 
        if(pcoord->x2v(j)<0.0||pcoord->x2v(j)>PI) { 
          bb.x3f(k,j,i) = -(len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
        }else{
          bb.x3f(k,j,i) = (len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
        }
      }
    }}

    if (pmb->block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face1Area(k,j,is-ngh,is-1,area);
        pcoord->Edge3Length(k,j  ,is-ngh,is-1,len);
        pcoord->Edge3Length(k,j+1,is-ngh,is-1,len_p1);
        for (int i=is-ngh; i<=is-1; ++i) {
	  // across the pole, we should use j index quantities to minus j+1 index quantities
          if(pcoord->x2v(j)<0.0||pcoord->x2v(j)>PI) {  
            bb.x1f(k,j,i) = -(len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
          }else{
            bb.x1f(k,j,i) = (len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
          }
        }
      }}

      for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face3Area(k,j,is-ngh,is-1,area);
        pcoord->Edge1Length(k,j  ,is-ngh,is-1,len);
        pcoord->Edge1Length(k,j+1,is-ngh,is-1,len_p1);
        for (int i=is-ngh; i<=is-1; ++i) {
	  // across the pole, we should use j index quantities to minus j+1 index quantities, but x3f should also flip sign
          if(pcoord->x2v(j)<0.0||pcoord->x2v(j)>PI) {  
            bb.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);
          }else{
            bb.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);
          }
        }
      }}
    }

    if (pmb->block_size.nx3 > 1) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face1Area(k,j,is-ngh,is-1,area);
        pcoord->Edge2Length(k  ,j,is-ngh,is-1,len);
        pcoord->Edge2Length(k+1,j,is-ngh,is-1,len_p1);
        for (int i=is-ngh; i<=is-1; ++i) {
          bb.x1f(k,j,i) -= (len_p1(i)*a2(k+1,j,i) - len(i)*a2(k,j,i))/area(i);
        }
      }}

      for (int k=ks; k<=ke; ++k) {
        // reset loop limits for polar boundary
        int jl=js; int ju=je+1;
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,is-ngh,is-1,area);
          pcoord->Edge1Length(k  ,j,is-ngh,is-1,len);
          pcoord->Edge1Length(k+1,j,is-ngh,is-1,len_p1);
          for (int i=is-ngh; i<=is-1; ++i) {
	    // x2f should flip sign across the pole
            if(pcoord->x2f(j)<0.0-eps||pcoord->x2f(j)>PI+eps) { 
              bb.x2f(k,j,i) -= (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
            }else{
              if(pcoord->x2f(j)>0.0+eps&&pcoord->x2f(j)<PI-eps) {
                bb.x2f(k,j,i) += (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
              }
            }
          }
        }
      }
    }

// calculate the pole Btheta magnetic flux at the pole by averaging two adjcent cells
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is-ngh; i<=is-1; ++i) {
          if(pcoord->x2f(j)>0-eps && pcoord->x2f(j)<0.0+eps) 
	    bb.x2f(k,j,i) = 0.5*(bb.x2f(k,j-1,i)+bb.x2f(k,j+1,i));
          if(pcoord->x2f(j)>PI-eps && pcoord->x2f(j)<PI+eps) 
	    bb.x2f(k,j,i) = 0.5*(bb.x2f(k,j-1,i)+bb.x2f(k,j+1,i));
        }
      }
    }

    a1.DeleteAthenaArray();
    a2.DeleteAthenaArray();
    a3.DeleteAthenaArray();
    area.DeleteAthenaArray();
    len.DeleteAthenaArray();
    len_p1.DeleteAthenaArray();

  }
 }
  return;
}

// Field BC at inner X1: copy 
void FieldOutflowInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x1f(k,j,(is-i)) = bb.x1f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x2f(k,j,(is-i)) = bb.x2f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x3f(k,j,(is-i)) = bb.x3f(k,j,is);
      }
    }}
  }

  return;
}


// Field BC at outer X1: reset field to initial condition
void FieldSteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 if(pmb->pmy_mesh->time==firsttime){
  if (MAGNETIC_FIELDS_ENABLED){
    Real eps=pcoord->dx2f(0)/100.;

    // Compute vector potential
    AthenaArray<Real> a1,a2,a3;
    int nx1 = (pmb->ie-pmb->is)+2 + 2*ngh;
    int nx2 = (pmb->je-pmb->js)+2 + 2*ngh;
    int nx3 = (pmb->ke-pmb->ks)+2 + 2*ngh;
    a1.NewAthenaArray(nx3,nx2,nx1);
    a2.NewAthenaArray(nx3,nx2,nx1);
    a3.NewAthenaArray(nx3,nx2,nx1);

    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=ie+1; i<=ie+ngh; i++) {
          a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
        }
      }
    }

    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=ie+1; i<=ie+ngh+1; i++) {
          a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
	}
      }
    }
    
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=ie+1; i<=ie+ngh+1; i++) {
          a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
	}
      }
    }

    // Initialize interface fields
    AthenaArray<Real> area,len,len_p1;
    area.NewAthenaArray(nx1);
    len.NewAthenaArray(nx1+1);
    len_p1.NewAthenaArray(nx1+1);

    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,ie+1,ie+ngh,area);
        pcoord->Edge3Length(k,j,ie+1,ie+1+ngh,len);
        for (int i=ie+1; i<=ie+ngh; ++i) {
	  /* x2f should flip sign across the pole */
          if(pcoord->x2f(j)<0.0-eps || pcoord->x2f(j)>PI+eps) { 
            bb.x2f(k,j,i) = (len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
          }else{
            if(pcoord->x2f(j)>0.0+eps && pcoord->x2f(j)<PI-eps) 
	      bb.x2f(k,j,i) = -(len(i+1)*a3(k,j,i+1) - len(i)*a3(k,j,i))/area(i);
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->Face3Area(k,j,ie+1,ie+ngh,area);
      pcoord->Edge2Length(k,j,ie+1,ie+1+ngh,len);
      for (int i=ie+1; i<=ie+ngh; ++i) {
	 /* x3f should flip sign across the pole */
        if(pcoord->x2v(j)<0.0 || pcoord->x2v(j)>PI) { 
          bb.x3f(k,j,i) = -(len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
        }else{
          bb.x3f(k,j,i) = (len(i+1)*a2(k,j,i+1) - len(i)*a2(k,j,i))/area(i);
        }
      }
    }}
    // 2D and 3D
    if (pmb->block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face1Area(k,j,ie+2,ie+1+ngh,area);
        pcoord->Edge3Length(k,j  ,ie+2,ie+1+ngh,len);
        pcoord->Edge3Length(k,j+1,ie+2,ie+1+ngh,len_p1);
        for (int i=ie+2; i<=ie+1+ngh; ++i) {
	  /* should use j index quantities to minus j+1 index quantities */
          if(pcoord->x2v(j)<0.0 || pcoord->x2v(j)>PI) { 
            bb.x1f(k,j,i) = -(len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
          }else{
            bb.x1f(k,j,i) = (len_p1(i)*a3(k,j+1,i) - len(i)*a3(k,j,i))/area(i);
          }
        }
      }}
      for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) { 
        pcoord->Face3Area(k,j,ie+1,ie+ngh,area);
        pcoord->Edge1Length(k,j  ,ie+1,ie+ngh,len);
        pcoord->Edge1Length(k,j+1,ie+1,ie+ngh,len_p1);
        for (int i=ie+1; i<=ie+ngh; ++i) {
	  /* across the pole, we should use j index quantities to minus j+1 index quantities, but x3f should also flip sign */
          if(pcoord->x2v(j)<0.0||pcoord->x2v(j)>PI) { 
            bb.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);          
          }else{
            bb.x3f(k,j,i) -= (len_p1(i)*a1(k,j+1,i) - len(i)*a1(k,j,i))/area(i);
          }
        } 
      }}
    } 
    // 3D only
    if (pmb->block_size.nx3 > 1) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face1Area(k,j,ie+2,ie+1+ngh,area);
        pcoord->Edge2Length(k  ,j,ie+2,ie+1+ngh,len);
        pcoord->Edge2Length(k+1,j,ie+2,ie+1+ngh,len_p1);
        for (int i=ie+2; i<=ie+1+ngh; ++i) {
          bb.x1f(k,j,i) -= (len_p1(i)*a2(k+1,j,i) - len(i)*a2(k,j,i))/area(i);
        }
      }}
      for (int k=ks; k<=ke; ++k) {
        // reset loop limits for polar boundary
        int jl=js; int ju=je+1;
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,ie+1,ie+ngh,area);
          pcoord->Edge1Length(k  ,j,ie+1,ie+ngh,len);
          pcoord->Edge1Length(k+1,j,ie+1,ie+ngh,len_p1);
          for (int i=ie+1; i<=ie+ngh; ++i) {
	    /* x2f should flip sign across the pole */
            if(pcoord->x2f(j)<0.0-eps||pcoord->x2f(j)>PI+eps) {
              bb.x2f(k,j,i) -= (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
            }else{
              if(pcoord->x2f(j)>0.0+eps&&pcoord->x2f(j)<PI-eps) {
                bb.x2f(k,j,i) += (len_p1(i)*a1(k+1,j,i) - len(i)*a1(k,j,i))/area(i);
              }
            }
          }
        }
      }
    }

    // calculate the pole Btheta magnetic flux at the pole by averaging two adjcent cells
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=ie+1; i<=ie+ngh; ++i) {
          if(pcoord->x2f(j)>0-eps&&pcoord->x2f(j)<0.0+eps) 
	    bb.x2f(k,j,i) = 0.5*(bb.x2f(k,j-1,i)+bb.x2f(k,j+1,i));
          if(pcoord->x2f(j)>PI-eps&&pcoord->x2f(j)<PI+eps) 
	    bb.x2f(k,j,i) = 0.5*(bb.x2f(k,j-1,i)+bb.x2f(k,j+1,i));
        }
      }
    }

    a1.DeleteAthenaArray();
    a2.DeleteAthenaArray();
    a3.DeleteAthenaArray();
    area.DeleteAthenaArray();
    len.DeleteAthenaArray();
    len_p1.DeleteAthenaArray();
  }
 }

  return;
}

// Field BC at outer X1: copy
void FieldOutflowOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x1f(k,j,(ie+i+1)) = bb.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x2f(k,j,(ie+i)) = bb.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(ngh); ++i) {
        bb.x3f(k,j,(ie+i)) = bb.x3f(k,j,ie);
      }
    }}
  }
}

// Field BC at inner X2: outflow
void FieldOutflowInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie+1; ++i) {
        bb.x1f(k,(js-j),i) = bb.x1f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        bb.x2f(k,(js-j),i) = bb.x2f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        bb.x3f(k,(js-j),i) = bb.x3f(k,js,i);
      }
    }}
  }

  return;
}

// Field BC at outer X2: outflow
void FieldOutflowOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie+1; ++i) {
        bb.x1f(k,(je+j  ),i) = bb.x1f(k,(je  ),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        bb.x2f(k,(je+j+1),i) = bb.x2f(k,(je+1),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=(ngh); ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        bb.x3f(k,(je+j  ),i) = bb.x3f(k,(je  ),i);
      }
    }}
  }
}

void FieldDivFreeInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
}

void FieldDivFreeOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                         Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
}

void FieldInflowOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
}

void FieldInflowAdvectOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                              Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
}

void FieldInflowVerticalOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                                Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  AllocateUserOutputVariables(nuseroutvar);

  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED){
//      if(prad->ir_output > 0){
//        prad->ir_index(0) = 0;
//        prad->ir_index(1) = 1;
//      }
      if (Opacityflag==0 || Opacityflag==3) prad->EnrollOpacityFunction(DiskOpacity);

      if (Opacityflag==1) prad->EnrollOpacityFunction(DiskOpacityDust);

      if (Opacityflag==2) prad->EnrollOpacityFunction(FixedOpacity);

            
      if(nuser_out_var >= 26) {
        int nx1 = (ie-is)+1 + 2*(NGHOST);
        int nx2 = (je-js)+1 + 2*(NGHOST);
        int nx3 = (ke-ks)+1;
        if(block_size.nx3 > 1) nx3 = nx3+2*(NGHOST);
        AllocateRealUserMeshBlockDataField(5);
        ruser_meshblock_data[0].NewAthenaArray(nx3,nx2,nx1); // radiation source term for MX3 
        ruser_meshblock_data[1].NewAthenaArray(nx3,nx2,nx1); // radiation source term for energy
        ruser_meshblock_data[2].NewAthenaArray(nx3,nx2,nx1);
        ruser_meshblock_data[3].NewAthenaArray(nx3,nx2,nx1);
        ruser_meshblock_data[4].NewAthenaArray(nx3,nx2,nx1);
      }
  }

  return;
} 

void MeshBlock::UserWorkInLoop(void)
{
  if(pmy_mesh->time==firsttime){
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          if (x1>rprotect){
            Real x2 = pcoord->x2v(j);
            Real x3 = pcoord->x3v(k);
            phydro->u(IDN,k,j,i) = DenProfile(x1, x2, x3);
  
            Real v1, v2, v3;
            VelProfile(x1, x2, x3, phydro->u(IDN,k,j,i), v1, v2, v3);

            phydro->u(IM1,k,j,i) = v1*phydro->u(IDN,k,j,i);
            phydro->u(IM2,k,j,i) = v2*phydro->u(IDN,k,j,i);
            phydro->u(IM3,k,j,i) = v3*phydro->u(IDN,k,j,i);
            if (NON_BAROTROPIC_EOS){
              phydro->u(IEN,k,j,i) = PoverR(x1, x2, x3)*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0)+0.5*(v1*v1+v2*v2+v3*v3)*phydro->u(IDN,k,j,i);
            }
          }
          if (NSCALARS > 0) {
            if(rscal!=0.0){
              if (pcoord->x1v(i)<rscal) {
                pscalars->s(0,k,j,i) = phydro->u(IDN,k,j,i) ;
              }else{
                pscalars->s(0,k,j,i) = 0.0 ; 
              }
            }
          }
        }
      }
    }
  }


/*
    if (phydro->fixed_emission_flag == 1 && pmy_mesh->time==firsttime){
      std::cout<<" Fixed_emission "<<std::endl;
      Real hokb=4.79924466e-11;
      Real *ir_lab; 
      for(int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) { 
          for (int i=is; i<=ie; ++i) {
            ir_lab = &(prad->ir(k,j,i,0));
            for(int n=0; n<prad->n_fre_ang; n++){
              ir_lab[n] = 0.;
            }
            Real tgas=(phydro->u(IEN,k,j,i)-0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i)-0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i)-0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i))/phydro->u(IDN,k,j,i)*(gamma_gas-1.0);
            if (nuemission >0.0) {
              prad->fixed_emission(k,j,i)=1./(exp(hokb*nuemission/(tgas*tempunit))-1.);
            } else {
              prad->fixed_emission(k,j,i)=tgas*tgas*tgas*tgas;
            }
          }
        }
      }    
    }
*/
// Initialize Kepler velocity for User_in_loop
  if(ifov_flag==1 or ifov_flag==2){
    AthenaArray<Real> out_Vkepc, out_Vkepf1, out_Vkepf2;

    out_Vkepc.InitWithShallowSlice(user_out_var,4,0,1);
    out_Vkepf1.InitWithShallowSlice(user_out_var,4,1,1);
    out_Vkepf2.InitWithShallowSlice(user_out_var,4,2,1);
    Real x3c = pcoord->x3v(ks);
    Real v1, v2, v3;
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1c = pcoord->x1v(i);
	Real x1f = pcoord->x1f(i);
        Real x2c = pcoord->x2v(j);
	Real x2f = pcoord->x2f(j);
	if(out_Vkepc(ks,j,i)==0.){
          VelProfile(x1c, x2c, x3c, phydro->u(IDN,ks,j,i), v1, v2, v3);
          out_Vkepc(ks,j,i) = v3;
        }
	if(out_Vkepf1(ks,j,i)==0.){
          VelProfile(x1f, x2c, x3c, phydro->u(IDN,ks,j,i), v1, v2, v3);	
	  out_Vkepf1(ks,j,i) = v3;
	}
	if(out_Vkepf2(ks,j,i)==0.){
	  VelProfile(x1c, x2f, x3c, phydro->u(IDN,ks,j,i), v1, v2, v3);
          out_Vkepf2(ks,j,i) = v3;
        }
      }
    }
    for (int j=js; j<=je; ++j) {
      Real x1f = pcoord->x1f(ie+1);
      Real x2c = pcoord->x2v(j);
      if(out_Vkepf1(ks,j,ie+1)==0.){
	VelProfile(x1f, x2c, x3c, phydro->u(IDN,ks,j,ie), v1, v2, v3);
        out_Vkepf1(ks,j,ie+1) = v3;	
      }
    }
    for (int i=is; i<=ie; ++i) {
      Real x1c = pcoord->x1v(i);
      Real x2f = pcoord->x2f(je+1);
      if(out_Vkepf2(ks,je+1,i)==0.){
   	VelProfile(x1c, x2f, x3c, phydro->u(IDN,ks,je,i), v1, v2, v3);
	out_Vkepf2(ks,je+1,i) = v3;
      }
    }
  }


 if(ifov_flag==1 or ifov_flag==2){

  AthenaArray<Real> &x1flux=phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux=phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux=phydro->flux[X3DIR];
  
  // output quantities to ifov

  if(ifov_flag==1){

    AthenaArray<Real> out_mass, out_AM;
    AthenaArray<Real> out_massflux1, out_massflux2, out_massflux3;
    AthenaArray<Real> out_AMflux1, out_AMflux2, out_AMflux3;
    AthenaArray<Real> out_Vkepc, out_Vkepf1, out_Vkepf2, out_dAM, out_dAMflux1, out_dAMflux2, out_rvk, out_tvk, out_pmass;

    if(nuser_out_var >= 17) {
      out_Vkepc.InitWithShallowSlice(user_out_var,4,0,1);
      out_Vkepf1.InitWithShallowSlice(user_out_var,4,1,1);
      out_Vkepf2.InitWithShallowSlice(user_out_var,4,2,1);
      out_mass.InitWithShallowSlice(user_out_var,4,3,1);
      out_AM.InitWithShallowSlice(user_out_var,4,4,1);
      out_massflux1.InitWithShallowSlice(user_out_var,4,5,1);
      out_massflux2.InitWithShallowSlice(user_out_var,4,6,1);
      out_massflux3.InitWithShallowSlice(user_out_var,4,7,1);
      out_AMflux1.InitWithShallowSlice(user_out_var,4,8,1);
      out_AMflux2.InitWithShallowSlice(user_out_var,4,9,1);
      out_AMflux3.InitWithShallowSlice(user_out_var,4,10,1);
      out_dAM.InitWithShallowSlice(user_out_var,4,11,1);
      out_dAMflux1.InitWithShallowSlice(user_out_var,4,12,1);
      out_dAMflux2.InitWithShallowSlice(user_out_var,4,13,1);
      out_rvk.InitWithShallowSlice(user_out_var,4,14,1);
      out_tvk.InitWithShallowSlice(user_out_var,4,15,1);
      out_pmass.InitWithShallowSlice(user_out_var,4,16,1);
    }

    Real dt = pmy_mesh->dt;
  
    if(nuser_out_var >= 17) {
      if (block_size.nx3 > 1) {
#pragma omp for schedule(static)
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            pcoord->CellVolume(k,j,is,ie,vol);
            pcoord->Face1Area(k,j,is,ie+1,x1area);
            pcoord->Face2Area(k,j  ,is,ie,x2area   );
            pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
            pcoord->Face3Area(k  ,j,is,ie,x3area   );
            pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
            for (int n=0; n<NHYDRO; ++n) {
	      for (int i=is; i<=ie; ++i) { 
	        if(n==IDN) {
                  out_mass(k,j,i) = phydro->u(n,k,j,i);
                  out_massflux1(k,j,i) -= dt*(x1area(i+1) *x1flux(n,k,j,i+1)
                                            - x1area(i)   *x1flux(n,k,j,i))/vol(i);
                  out_massflux2(k,j,i) -= dt*(x2area_p1(i)*x2flux(n,k,j+1,i)
                                            - x2area(i)   *x2flux(n,k,j,i))/vol(i);
                  out_massflux3(k,j,i) -= dt*(x3area_p1(i)*x3flux(n,k+1,j,i)
                                            - x3area(i)   *x3flux(n,k,j,i))/vol(i);
	    	  out_pmass(k,j,i) += dt*(x3area_p1(i)*x3flux(n,k+1,j,i)
                                            - x3area(i)   *x3flux(n,k,j,i))/vol(i)*out_Vkepc(ks,j,i);
                }
                if(n==IM3) {
                  Real x1f=pcoord->x1f(i);
                  Real x1fp=pcoord->x1f(i+1);
                  Real x2f=pcoord->x2f(j);
                  Real x2fp=pcoord->x2f(j+1);
                  Real x1v=0.5*(x1f+x1fp);
                  Real sinx2c=0.5*(sin(pcoord->x2f(j))+sin(pcoord->x2f(j+1)));
                  out_AM(k,j,i) = phydro->u(n,k,j,i);
	  	  out_dAM(k,j,i) = phydro->u(n,k,j,i)-phydro->u(IDN,k,j,i)*out_Vkepc(ks,j,i);
                  out_AMflux1(k,j,i) -= dt*(x1fp* x1area(i+1) *x1flux(n,k,j,i+1)
                                          - x1f * x1area(i)   *x1flux(n,k,j,i))/vol(i)/x1v;
                  out_AMflux2(k,j,i) -= dt*(sin(x2fp)*x2area_p1(i)*x2flux(n,k,j+1,i)
                                          - sin(x2f) *x2area(i)   *x2flux(n,k,j,i))/vol(i)/sinx2c;
                  out_AMflux3(k,j,i) -= dt*(x3area_p1(i)*x3flux(n,k+1,j,i)
                                          - x3area(i)   *x3flux(n,k,j,i))/vol(i);
	  	  out_dAMflux1(k,j,i) -= dt*(x1fp* x1area(i+1) *(x1flux(n,k,j,i+1)-x1flux(IDN,k,j,i+1)*out_Vkepf1(ks,j,i+1))
                                          -  x1f * x1area(i)   *(x1flux(n,k,j,i)-x1flux(IDN,k,j,i)*out_Vkepf1(ks,j,i)))/vol(i)/x1v;
                  out_dAMflux2(k,j,i) -= dt*(sin(x2fp)*x2area_p1(i)*(x2flux(n,k,j+1,i)-x2flux(IDN,k,j+1,i)*out_Vkepf2(ks,j+1,i))
                                          - sin(x2f) *x2area(i)    *(x2flux(n,k,j,i)-x2flux(IDN,k,j,i)*out_Vkepf2(ks,j,i)))/vol(i)/sinx2c;
  		  out_rvk(k,j,i) -= dt*(x1area(i+1) *out_Vkepf1(ks,j,i+1)/x1fp
                                      - x1area(i)   *out_Vkepf1(ks,j,i)/x1f)*0.5*(x1flux(IDN,k,j,i+1)*x1fp*x1fp+x1flux(IDN,k,j,i)*x1f*x1f)/vol(i)/x1v;
	          out_tvk(k,j,i) -= dt*(x2area_p1(i)*out_Vkepf2(ks,j+1,i)
                                      - x2area(i)   *out_Vkepf2(ks,j,i))*0.5*(x2flux(IDN,k,j+1,i)*sin(x2fp)+x2flux(IDN,k,j,i)*sin(x2f))/vol(i)/sinx2c;
                }
	      }
            }
          }
        }
      }
    }
  }

  if(ifov_flag==2){

    AthenaArray<Real> out_mflux1, out_mflux2, out_vflux1, out_vflux2, out_vflux1r, out_vflux2r, out_vflux1b, out_vflux2b;
    AthenaArray<Real> out_Vkepc, out_Vkepf1, out_Vkepf2, out_dAM, out_dAMflux1, out_dAMflux2, out_rvk, out_tvk;
    AthenaArray<Real> out_sigmas, out_sigmaa, out_sigmaae, out_sigmaplanck;
    AthenaArray<Real> out_radmx3, out_dEfluxx1, out_dEfluxx2, out_dEPot, out_cooling, out_radE; 

    if(nuser_out_var >= 16) {
      out_Vkepc.InitWithShallowSlice(user_out_var,4,0,1);
      out_Vkepf1.InitWithShallowSlice(user_out_var,4,1,1);
      out_Vkepf2.InitWithShallowSlice(user_out_var,4,2,1);
      out_mflux1.InitWithShallowSlice(user_out_var,4,3,1);
      out_mflux2.InitWithShallowSlice(user_out_var,4,4,1);
      out_dAM.InitWithShallowSlice(user_out_var,4,5,1);
      out_dAMflux1.InitWithShallowSlice(user_out_var,4,6,1);
      out_dAMflux2.InitWithShallowSlice(user_out_var,4,7,1);
      out_rvk.InitWithShallowSlice(user_out_var,4,8,1);
      out_tvk.InitWithShallowSlice(user_out_var,4,9,1);
      out_vflux1.InitWithShallowSlice(user_out_var,4,10,1);
      out_vflux2.InitWithShallowSlice(user_out_var,4,11,1);
      out_vflux1r.InitWithShallowSlice(user_out_var,4,12,1);
      out_vflux2r.InitWithShallowSlice(user_out_var,4,13,1);
      out_vflux1b.InitWithShallowSlice(user_out_var,4,14,1);
      out_vflux2b.InitWithShallowSlice(user_out_var,4,15,1);
    }
    if((RADIATION_ENABLED || IM_RADIATION_ENABLED )&&nuser_out_var >= 26) {
      out_sigmas.InitWithShallowSlice(user_out_var,4,16,1);
      out_sigmaa.InitWithShallowSlice(user_out_var,4,17,1);
      out_sigmaae.InitWithShallowSlice(user_out_var,4,18,1);
      out_sigmaplanck.InitWithShallowSlice(user_out_var,4,19,1);
      out_radmx3.InitWithShallowSlice(user_out_var,4,20,1);
      out_dEfluxx1.InitWithShallowSlice(user_out_var,4,21,1);
      out_dEfluxx2.InitWithShallowSlice(user_out_var,4,22,1);
      out_dEPot.InitWithShallowSlice(user_out_var,4,23,1);
      out_cooling.InitWithShallowSlice(user_out_var,4,24,1);
      out_radE.InitWithShallowSlice(user_out_var,4,25,1);
    }

    Real dt = pmy_mesh->dt;
  
    if(nuser_out_var >= 16) {
      if (block_size.nx3 > 1) {
#pragma omp for schedule(static)
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            pcoord->CellVolume(k,j,is,ie,vol);
            pcoord->Face1Area(k,j,is,ie+1,x1area);
            pcoord->Face2Area(k,j  ,is,ie,x2area   );
            pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
            pcoord->Face3Area(k  ,j,is,ie,x3area   );
            pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
            for (int n=0; n<NHYDRO; ++n) {
	      for (int i=is; i<=ie; ++i) { 
	        if(n==IDN) {
                  out_mflux1(k,j,i) += dt*x1flux(n,k,j,i);
                  out_mflux2(k,j,i) += dt*x2flux(n,k,j,i);
                }
                if(n==IM3) {
                  Real x1f=pcoord->x1f(i);
                  Real x1fp=pcoord->x1f(i+1);
                  Real x2f=pcoord->x2f(j);
                  Real x2fp=pcoord->x2f(j+1);
                  Real x1v=0.5*(x1f+x1fp);
                  Real sinx2c=0.5*(sin(pcoord->x2f(j))+sin(pcoord->x2f(j+1)));
	  	  out_dAM(k,j,i) = phydro->u(n,k,j,i)-phydro->u(IDN,k,j,i)*out_Vkepc(ks,j,i);
	  	  out_dAMflux1(k,j,i) -= dt*(x1fp* x1area(i+1) *(x1flux(n,k,j,i+1)-x1flux(IDN,k,j,i+1)*out_Vkepf1(ks,j,i+1))
                                          -  x1f * x1area(i)   *(x1flux(n,k,j,i)-x1flux(IDN,k,j,i)*out_Vkepf1(ks,j,i)))/vol(i)/x1v;
                  out_dAMflux2(k,j,i) -= dt*(sin(x2fp)*x2area_p1(i)*(x2flux(n,k,j+1,i)-x2flux(IDN,k,j+1,i)*out_Vkepf2(ks,j+1,i))
                                          - sin(x2f) *x2area(i)    *(x2flux(n,k,j,i)-x2flux(IDN,k,j,i)*out_Vkepf2(ks,j,i)))/vol(i)/sinx2c;
  		  out_rvk(k,j,i) -= dt*(x1area(i+1) *out_Vkepf1(ks,j,i+1)/x1fp
                                      - x1area(i)   *out_Vkepf1(ks,j,i)/x1f)*0.5*(x1flux(IDN,k,j,i+1)*x1fp*x1fp+x1flux(IDN,k,j,i)*x1f*x1f)/vol(i)/x1v;
	          out_tvk(k,j,i) -= dt*(x2area_p1(i)*out_Vkepf2(ks,j+1,i)
                                      - x2area(i)   *out_Vkepf2(ks,j,i))*0.5*(x2flux(IDN,k,j+1,i)*sin(x2fp)+x2flux(IDN,k,j,i)*sin(x2f))/vol(i)/sinx2c;
		  out_vflux1(k,j,i) += dt*(x1flux(n,k,j,i)-x1flux(IDN,k,j,i)*out_Vkepf1(ks,j,i));
		  out_vflux2(k,j,i) += dt*(x2flux(n,k,j,i)-x2flux(IDN,k,j,i)*out_Vkepf2(ks,j,i));
		  out_vflux1r(k,j,i) += dt*phydro->u(IM1,k,j,i)*(0.25*(phydro->u(IM3,k,j,i)+phydro->u(IM3,k+1,j,i)+phydro->u(IM3,k,j,i-1)+phydro->u(IM3,k+1,j,i-1))/phydro->u(IDN,k,j,i)-out_Vkepf1(ks,j,i));
		  out_vflux2r(k,j,i) += dt*phydro->u(IM2,k,j,i)*(0.25*(phydro->u(IM3,k,j,i)+phydro->u(IM3,k+1,j,i)+phydro->u(IM3,k,j-1,i)+phydro->u(IM3,k+1,j-1,i))/phydro->u(IDN,k,j,i)-out_Vkepf2(ks,j,i));
		  if (MAGNETIC_FIELDS_ENABLED){
	            out_vflux1b(k,j,i) -= dt*pfield->b.x1f(k,j,i)*0.25*(pfield->b.x3f(k,j,i)+pfield->b.x3f(k+1,j,i)+pfield->b.x3f(k,j,i-1)+pfield->b.x3f(k+1,j,i-1));
		    out_vflux2b(k,j,i) -= dt*pfield->b.x2f(k,j,i)*0.25*(pfield->b.x3f(k,j,i)+pfield->b.x3f(k+1,j,i)+pfield->b.x3f(k,j-1,i)+pfield->b.x3f(k+1,j-1,i));
		  }
                }
	      }
            }
          }
        }
      }
    }
    if((RADIATION_ENABLED || IM_RADIATION_ENABLED)&&nuser_out_var >= 26) {
      int ifr=0;
      if (block_size.nx3 > 1) {
#pragma omp for schedule(static)
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
	    pcoord->CellVolume(k,j,is,ie,vol);
            pcoord->Face1Area(k,j,is,ie+1,x1area);
            pcoord->Face2Area(k,j  ,is,ie,x2area   );
            pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
            pcoord->Face3Area(k  ,j,is,ie,x3area   );
            pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
            for (int i=is; i<=ie; ++i) {
              out_sigmas(k,j,i) = prad->sigma_s(k,j,i,ifr)/phydro->u(IDN,k,j,i)/rhounit/lunit;
              out_sigmaa(k,j,i) = prad->sigma_a(k,j,i,ifr)/phydro->u(IDN,k,j,i)/rhounit/lunit;
              out_sigmaae(k,j,i) = prad->sigma_ae(k,j,i,ifr)/phydro->u(IDN,k,j,i)/rhounit/lunit;
              out_sigmaplanck(k,j,i) = prad->sigma_planck(k,j,i,ifr)/phydro->u(IDN,k,j,i)/rhounit/lunit;            
	      out_radmx3(k,j,i) += ruser_meshblock_data[0](k,j,i);
	      out_dEfluxx1(k,j,i) -= dt*(x1area(i+1) * x1flux(IEN,k,j,i+1)
                                        -x1area(i) * x1flux(IEN,k,j,i))/vol(i); 
	      out_dEfluxx2(k,j,i) -= dt*(x2area_p1(i) * x2flux(IEN,k,j+1,i)
                                        -x2area(i) * x2flux(IEN,k,j,i))/vol(i);
	      out_dEPot(k,j,i) -= dt*0.5*(1.0/SQR(pcoord->x1v(i))*x1flux(IDN,k,j,i)*gm0
                                         +1.0/SQR(pcoord->x1v(i))*x1flux(IDN,k,j,i+1)*gm0);
	      out_radE(k,j,i) += ruser_meshblock_data[1](k,j,i);
	    }
          }
        }
      }
    }    
  }
 }

 return;
}

// check density floor
void Checkfloor(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Coordinates *pco = pmb->pcoord;
  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real sinx2=sin(pco->x2v(j));
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (cons(IDN,k,j,i)<rho_floor(pco->x1v(i),pco->x2v(j),pco->x3v(k))) {
          cons(IDN,k,j,i)=rho_floor(pco->x1v(i),pco->x2v(j),pco->x3v(k));
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------
// f: cooling function with damping boundary
// tcool: orbital cooling time
//
void Cooling(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim,  
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
 if(tcool>0.0) {
   Coordinates *pco = pmb->pcoord;
   for(int k=pmb->ks; k<=pmb->ke; ++k){
     for (int j=pmb->js; j<=pmb->je; ++j) {
       Real sinx2=sin(pco->x2v(j));
       for (int i=pmb->is; i<=pmb->ie; ++i) {
	 // to avoid the divergence at 0 for both Keplerian motion and p_over_r
         Real r = std::max(fabs(pco->x1v(i)*sinx2),xcut);
         Real eint = cons(IEN,k,j,i)-0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
	 	                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
	 if (MAGNETIC_FIELDS_ENABLED){
	   eint = eint-0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
	 }
         Real pres_over_r=eint*(gamma_gas-1.0)/cons(IDN,k,j,i);
         Real p_over_r = PoverR(pco->x1v(i),pco->x2v(j),pco->x3v(k)); 
	 // reset temperature when the temperature is below tlow times the initial temperature
         if(tlow>0 && pres_over_r<tlow*p_over_r){
           eint=tlow*p_over_r*cons(IDN,k,j,i)/(gamma_gas-1.0);
	   cons(IEN,k,j,i)=eint+0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
			             +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
           if (MAGNETIC_FIELDS_ENABLED){
             cons(IEN,k,j,i)+=0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
           }
         }
	 // reset temperature when the temperature is above thigh times the initial temperature
         if(thigh>0 && pres_over_r>thigh*p_over_r){
           eint=thigh*p_over_r*cons(IDN,k,j,i)/(gamma_gas-1.0);
	   cons(IEN,k,j,i)=eint+0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
				     +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
           if (MAGNETIC_FIELDS_ENABLED){
             cons(IEN,k,j,i)+=0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
           }
         }
         Real dtr = std::max(tcool*2.*PI/sqrt(gm0/r/r/r),dt);
         Real dfrac=dt/dtr;
         Real dE=eint-p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
         cons(IEN,k,j,i) -= dE*dfrac;
       }
     }
   }
 }
 return;
}

//--------------------------------------------------------------------------------
// f: damp to the initital condition close to the inner boundary
//
//

void Damp(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
 Real rdi1=1.25*xcut;
 Real ramp = 0.0, tau=0.0, lambda=0.0, e_nomag=0.0;
 if(tdamp>0.0) {
  Coordinates *pco = pmb->pcoord;
  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real sinx2=sin(pco->x2v(j));
      for (int i=pmb->is; i<=pmb->ie; ++i) {
	Real x1 = pco->x1v(i);
	Real x2 = pco->x2v(j);
	Real x3 = pco->x3v(k);
        // to avoid the divergence at 0 for both Keplerian motion and p_over_r
	Real r = std::max(fabs(x1*sinx2),xcut);
        // damp timescale
	ramp = 0.0;
	if(x1 < rdi1){
          ramp = (x1-rdi1)/(rdi1-xcut);
          ramp = ramp*ramp;
          tau = 2.0*PI*sqrt(r*r*r/gm0)*tdamp;
        }
        // desired quantities
	Real den, v1, v2, v3, m1, m2, m3, eint;
        den = DenProfile(x1, x2, x3);

        VelProfile(x1, x2, x3, den, v1, v2, v3);
	m1 = den*v1;
	m2 = den*v2;
	m3 = den*v3;
	if (NON_BAROTROPIC_EOS){
	  Real p_over_r = PoverR(x1, x2, x3); 
	  eint = p_over_r*den/(gamma_gas - 1.0);
	}
        // damp quantities 
	if(ramp>0.0){
          lambda = ramp/tau*dt;
	  cons(IDN,k,j,i)=(cons(IDN,k,j,i)+lambda*den)/(1.+lambda);
          cons(IM1,k,j,i)=(cons(IM1,k,j,i)+lambda*m1)/(1.+lambda);
          cons(IM2,k,j,i)=(cons(IM2,k,j,i)+lambda*m2)/(1.+lambda);     
          cons(IM3,k,j,i)=(cons(IM3,k,j,i)+lambda*m3)/(1.+lambda);
          if(NON_BAROTROPIC_EOS) {
	    if (MAGNETIC_FIELDS_ENABLED){
              e_nomag = cons(IEN,k,j,i) - 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
            }else{
	      e_nomag = cons(IEN,k,j,i); 
	    }
	    e_nomag=(e_nomag+lambda*(eint+0.5*(m1*m1+m2*m2+m3*m3)/den))/(1.+lambda);
	    if (MAGNETIC_FIELDS_ENABLED){
              cons(IEN,k,j,i) = e_nomag + 0.5*(SQR(bcc(IB1,k,j,i))+SQR(bcc(IB2,k,j,i))+SQR(bcc(IB3,k,j,i)));
            }else{
              cons(IEN,k,j,i) = e_nomag;
            }
	  }	  
	}
      }
    }
  }
 }
}

//************************************************
//////* Additional Physical Source Terms 
//////************************************************
////
//////****** Use grav potential to calculate forces 

Real grav_pot_car(const Real xca, const Real yca, const Real zca, 
	const Real xpp, const Real ypp, const Real zpp, const Real gmp, const int ip) 
{
  Real dist2;
  if(cylpot==1) dist2 = (xca-xpp)*(xca-xpp) + (yca-ypp)*(yca-ypp);
    else dist2 = (xca-xpp)*(xca-xpp) + (yca-ypp)*(yca-ypp) + (zca-zpp)*(zca-zpp);
  Real pot = -gmp*(dist2+1.5*rsoft2)/(dist2+rsoft2)/sqrt(dist2+rsoft2);
  if(ind!=0){
    Real pdist=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
    pot+=gmp/pdist/pdist/pdist*(xca*xpp+yca*ypp+zca*zpp);
  }
  return(pot);
}

Real grav_pot_car_cen(const Real xca, const Real yca, const Real zca) {
// Centrifugal force
  Real pot;
  if(omegarot!=0.0) pot = -0.5*omegarot*omegarot*(xca*xca+yca*yca);
  return(pot);
}

void DepleteCir(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, AthenaArray<Real> &cons)
{
  Coordinates *pco = pmb->pcoord;

  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real R=pco->x1v(i);
        Real th=pco->x2v(j);
        Real phi=pco->x3v(k);
        Real xg = R*sin(th)*cos(phi);
        Real yg = R*sin(th)*sin(phi);
        Real zg = R*cos(th);
        Real rdist = sqrt(SQR(xg-psys->xp[0])+SQR(yg-psys->yp[0])+SQR(zg-psys->zp[0]));
        Real v1=0.0;
        Real v2=0.0;
        Real v3=0.0;
        if(rdist>rocird){
          cons(IDN,k,j,i) = dcird;
          cons(IM1,k,j,i) = dcird*v1;
          cons(IM2,k,j,i) = dcird*v2;
          cons(IM3,k,j,i) = dcird*v3;
        }
        if(rdist<3.*rcird){
          Real dtr = std::max(tcird, dt);
          Real dfrac = dt/dtr;
          cons(IDN,k,j,i) -= (cons(IDN,k,j,i) - dcird)*dfrac;
          cons(IM1,k,j,i) -= (cons(IM1,k,j,i) - dcird*v1)*dfrac;
          cons(IM2,k,j,i) -= (cons(IM2,k,j,i) - dcird*v2)*dfrac;
          cons(IM3,k,j,i) -= (cons(IM3,k,j,i) - dcird*v3)*dfrac;
          if(NON_BAROTROPIC_EOS) {
            Real ende=0.5*dcird*(SQR(v1)+SQR(v2)+SQR(v3))+PoverR(R,th,phi)*dcird/(gamma_gas - 1.0);
            cons(IEN,k,j,i) -= (cons(IEN,k,j,i) - ende)*dfrac;
          }
        }
      }
    }
  }
}

//******** Grav force from GM1, and indirect term
void PlanetarySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  const AthenaArray<Real> *flux=pmb->phydro->flux;
  Real src[NHYDRO];
  Coordinates *pco = pmb->pcoord;
  
  // integrate planet orbit
  if (psys->np > 0||omegarot!=0.0) {
    if(myfile.is_open()&&Globals::my_rank==0&&time>=timeout)
    {
      myfile<<time+dt<<' ';
      for (int i=0; i<psys->np; ++i){
        Real th=atan(psys->yp[i]/psys->xp[i]);
        if(psys->xp[i]<0.0) th+=PI;
        myfile<<psys->xp[i]<<' '<<psys->yp[i]<<' '<<psys->zp[i]
        <<' '<<psys->vxp[i]<<' '<<psys->vyp[i]<<' '<<psys->vzp[i]<<' ';
      }
      myfile<<'\n'<<std::flush;

      timeout+=dtorbit;
    }
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real x3=pco->x3v(k);
      Real cosx3=cos(x3);
      Real sinx3=sin(x3);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real x2=pco->x2v(j);
        Real cosx2=cos(x2);
        Real sinx2=sin(x2);
        Real sm = std::fabs(std::sin(pco->x2f(j  )));
        Real sp = std::fabs(std::sin(pco->x2f(j+1)));
        Real cmmcp = std::fabs(std::cos(pco->x2f(j  )) - std::cos(pco->x2f(j+1)));
        Real coord_src1_j=(sp-sm)/cmmcp;
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real rm = pco->x1f(i  );
          Real rp = pco->x1f(i+1);
          Real coord_src1_i=1.5*(rp*rp-rm*rm)/(rp*rp*rp-rm*rm*rm);
          Real drs = pco->dx1v(i) / 10000.;
          Real xcar = pco->x1v(i)*sinx2*cosx3;
          Real ycar = pco->x1v(i)*sinx2*sinx3;
          Real zcar = pco->x1v(i)*cosx2;
          Real f_x1 = 0.0;
          Real f_x2 = 0.0;
          Real f_x3 = 0.0;
          for (int ip=0; ip< psys->np; ++ip){
            Real xpp=psys->xp[ip];
            Real ypp=psys->yp[ip];
            Real zpp=psys->zp[ip];
            Real mp;
  	    /* insert the planet at insert_start and gradually increase its mass during insert_time */
            if(time<insert_start){
              mp = 0.0;
            }else{
              mp = 1.0*std::min(1.0,((time-insert_start+1.e-10)/(insert_time+1.e-10)))*psys->mass[ip];
            }
            /* forces calculated using gradient of potential */
            Real f_xca = -1.0* (grav_pot_car(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp,ip)
  			        -grav_pot_car(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp,ip))/(2.0*drs);
            Real f_yca = -1.0* (grav_pot_car(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp,ip)
  	 		        -grav_pot_car(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp,ip))/(2.0*drs);
            Real f_zca = -1.0* (grav_pot_car(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp,ip)
			        -grav_pot_car(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp,ip))/(2.0*drs);
            f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
            f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
            f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
          }
          if(omegarot!=0.0) {
//            Real omegar=omegarot*cosx2;
//            Real omegat=-omegarot*sinx2;
            /* centrifugal force */
/*
            Real f_xca = -1.0* (grav_pot_car_cen(xcar+drs, ycar, zcar)
 			        -grav_pot_car_cen(xcar-drs, ycar, zcar))/(2.0*drs);
            Real f_yca = -1.0* (grav_pot_car_cen(xcar, ycar+drs, zcar)
			        -grav_pot_car_cen(xcar, ycar-drs, zcar))/(2.0*drs);
            Real f_zca = -1.0* (grav_pot_car_cen(xcar, ycar, zcar+drs)
			        -grav_pot_car_cen(xcar, ycar, zcar-drs))/(2.0*drs);
            f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
            f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
            f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
*/
            f_x1 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i;
            f_x2 += omegarot*omegarot*pco->x1v(i)*pco->x1v(i)*sinx2*sinx2*coord_src1_i*coord_src1_j;
//            Real f_co1=omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))/prim(IDN,k,j,i)*pco->x1v(i)*coord_src1_i;
//            Real f_co2=omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))/prim(IDN,k,j,i)*pco->x1v(i)*coord_src1_i*coord_src1_j;
          }
            /* Coriolis force */
//            f_x1 -= 2.0*omegat*prim(IM3,k,j,i);
//            f_x2 += 2.0*omegar*prim(IM3,k,j,i);
//            f_x3 -= 2.0*omegar*prim(IM2,k,j,i)-2.0*omegat*prim(IM1,k,j,i);
//            Real fc1 = 2.0*omegar*prim(IM2,k,j,i)-2.0*omegat*prim(IM1,k,j,i); 
//            Real fc2 = (omegarot*3.*(sp+sm)/(rp+rm)/(rp*rp+rp*rm+rm*rm)*(rp*rp*(3.*rp+rm)/4.*flux[X1DIR](IDN,k,j,i+1)+rm*rm*(3.*rm+rp)/4.*flux[X1DIR](IDN,k,j,i))+omegarot*3.*(rp+rm)*(rp+rm)*(sp-sm)/2./(sp+sm)/(rp*rp+rp*rm+rm*rm)/cmmcp*((3.*sp+sm)/4.*sp*flux[X2DIR](IDN,k,j+1,i)+(3.*sm+sp)/4.*sm*flux[X2DIR](IDN,k,j,i)))/prim(IDN,k,j,i);
//            f_x1 -= omegat*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))/prim(IDN,k,j,i);
//            f_x2 += omegar*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))/prim(IDN,k,j,i);
//            f_x3 -= (omegarot*3.*(sp+sm)/(rp+rm)/(rp*rp+rp*rm+rm*rm)*(rp*rp*(3.*rp+rm)/4.*flux[X1DIR](IDN,k,j,i+1)+rm*rm*(3.*rm+rp)/4.*flux[X1DIR](IDN,k,j,i))+omegarot*3.*(rp+rm)*(rp+rm)*(sp-sm)/2./(sp+sm)/(rp*rp+rp*rm+rm*rm)/cmmcp*((3.*sp+sm)/4.*sp*flux[X2DIR](IDN,k,j+1,i)+(3.*sm+sp)/4.*sm*flux[X2DIR](IDN,k,j,i)))/prim(IDN,k,j,i);

	  src[IM1] = dt*prim(IDN,k,j,i)*f_x1;
          src[IM2] = dt*prim(IDN,k,j,i)*f_x2;
          src[IM3] = dt*prim(IDN,k,j,i)*f_x3;

          cons(IM1,k,j,i) += src[IM1];
          cons(IM2,k,j,i) += src[IM2];
          cons(IM3,k,j,i) += src[IM3];

          if(NON_BAROTROPIC_EOS) {
            src[IEN] = f_x1*dt*0.5*(flux[X1DIR](IDN,k,j,i)+flux[X1DIR](IDN,k,j,i+1))+f_x2*dt*0.5*(flux[X2DIR](IDN,k,j,i)+flux[X2DIR](IDN,k,j+1,i))+f_x3*dt*0.5*(flux[X3DIR](IDN,k,j,i)+flux[X3DIR](IDN,k+1,j,i));
//            src[IEN] = src[IM1]*prim(IM1,k,j,i)+ src[IM2]*prim(IM2,k,j,i) 
//	               + src[IM3]*prim(IM3,k,j,i);
            cons(IEN,k,j,i) += src[IEN];
          }

          if(omegarot!=0.0) {
	  /* Coriolis force */
            cons(IM1,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*dt;
            cons(IM2,k,j,i) += omegarot*sinx2*(flux[X3DIR](IDN,k+1,j,i)+flux[X3DIR](IDN,k,j,i))*pco->x1v(i)*coord_src1_i*coord_src1_j*dt;
            cons(IM3,k,j,i) -= (omegarot*3.*(sp+sm)/(rp+rm)/(rp*rp+rp*rm+rm*rm)*(rp*rp*(3.*rp+rm)/4.*flux[X1DIR](IDN,k,j,i+1)+rm*rm*(3.*rm+rp)/4.*flux[X1DIR](IDN,k,j,i))+omegarot*3.*(rp+rm)*(rp+rm)*(sp-sm)/2./(sp+sm)/(rp*rp+rp*rm+rm*rm)/cmmcp*((3.*sp+sm)/4.*sp*flux[X2DIR](IDN,k,j+1,i)+(3.*sm+sp)/4.*sm*flux[X2DIR](IDN,k,j,i)))*dt;
          }          
        }
      }
    }
    // integrate planet orbit
    if(halfstep==1){
      psys->disktoplanet(dt);
      if(fixorb==1) psys->fixorbit(dt);
        else psys->integrate(dt);
      if(omegarot!=0.0) psys->Rotframe(dt);
      halfstep=0;
      if(rcird>0.0) DepleteCir(pmb,dt,prim,cons);
    }else{
      psys->disktoplanet(dt/2.);
      if(fixorb==1) psys->fixorbit(dt/2.);
        else psys->integrate(dt/2.); //integrate another half timestep in the full dt update
      if(omegarot!=0.0) psys->Rotframe(dt/2.);
      halfstep=1;
      if(rcird>0.0) DepleteCir(pmb,dt,prim,cons);
    }
  }
  if(tdamp>0.0) Damp(pmb,dt,prim,bcc,cons);
  Checkfloor(pmb,dt,prim,bcc,cons);
  if(RADIATION_ENABLED|| IM_RADIATION_ENABLED){
    if(tcool>0.0&&time<radstart) Cooling(pmb,dt,prim,bcc,cons);
  }else{
    if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,dt,prim,bcc,cons);
  }
  
  Heating(pmb, time,dt,prim,bcc,cons);
}

void PlanetarySystem::disktoplanet(double dt)
{

}

//------------------------------------------
// f: circular planet orbit
//
void PlanetarySystem::fixorbit(double dt)
{
  int i;
  for(i=0; i<np; ++i){
    double dis=sqrt(xp[i]*xp[i]+yp[i]*yp[i]);
    double ome=sqrt((gm0+mass[i])/dis/dis/dis);
    double ang=acos(xp[i]/dis);
    if(yp[i]<0.0) ang=2*PI-ang;
    ang += ome*dt;
    xp[i]=dis*cos(ang);
    yp[i]=dis*sin(ang);
  }
  return;
}

//------------------------------------------
//  f: planet position in the frame rotating at omegarot
//
void PlanetarySystem::Rotframe(double dt)
{
  int i;
  for(i=0; i<np; ++i){
    double dis=sqrt(xp[i]*xp[i]+yp[i]*yp[i]);
    double ang=acos(xp[i]/dis);
    if(yp[i]<0.0) ang=2*PI-ang;
    ang -= omegarot*dt;
    xp[i]=dis*cos(ang);
    yp[i]=dis*sin(ang);
  }
  return;
}

//----------------------------------------------------
// f: planetary orbit integrator
// 
void PlanetarySystem::integrate(double dt)
{
  int i,j;
  double forcex,forcey,forcez;
  double forcexi=0., forceyi=0., forcezi=0.;
  double *dist;
  dist=new double[np];
  for(i=0; i<np; ++i){
    xpn[i]=xp[i]+vxp[i]*dt/2.;
    ypn[i]=yp[i]+vyp[i]*dt/2.;
    zpn[i]=zp[i]+vzp[i]*dt/2.;
  }
  for(i=0; i<np; ++i) dist[i]=sqrt(xpn[i]*xpn[i]+ypn[i]*ypn[i]+zpn[i]*zpn[i]);
  // indirect term (acceleration of the central star) from the gravity of the planets themselves.
  // will be added to direct force for each planet
  for(j=0; j<np; ++j){
    forcexi -= mass[j]/dist[j]/dist[j]/dist[j]*xpn[j];
    forceyi -= mass[j]/dist[j]/dist[j]/dist[j]*ypn[j];
    forcezi -= mass[j]/dist[j]/dist[j]/dist[j]*zpn[j];
  }
  for(i=0; i<np; ++i){
    // direct term from the central star
    forcex= -gm0/dist[i]/dist[i]/dist[i]*xpn[i];
    forcey= -gm0/dist[i]/dist[i]/dist[i]*ypn[i];
    forcez= -gm0/dist[i]/dist[i]/dist[i]*zpn[i];
    forcex += forcexi;
    forcey += forceyi;
    forcez += forcezi;
    // gravity from other planets
    for(j=0; j<np; ++j){
      if(j!=i){
        double dis=(xpn[i]-xpn[j])*(xpn[i]-xpn[j])+(ypn[i]-ypn[j])*
		   (ypn[i]-ypn[j])+(zpn[i]-zpn[j])*(zpn[i]-zpn[j]);
        dis=sqrt(dis);
        forcex += mass[j]/dis/dis/dis*(xpn[j]-xpn[i]);
        forcey += mass[j]/dis/dis/dis*(ypn[j]-ypn[i]);
        forcez += mass[j]/dis/dis/dis*(zpn[j]-zpn[i]);
      }
    }
    vxpn[i] = vxp[i] + forcex*dt;
    vypn[i] = vyp[i] + forcey*dt;
    vzpn[i] = vzp[i] + forcez*dt;
  }
  for(i=0; i<np; ++i){
    xpn[i]=xpn[i]+vxpn[i]*dt/2.;
    ypn[i]=ypn[i]+vypn[i]*dt/2.;
    zpn[i]=zpn[i]+vzpn[i]*dt/2.;
  }
  for(i=0; i<np; ++i){
    xp[i]=xpn[i];
    yp[i]=ypn[i];
    zp[i]=zpn[i];
    vxp[i]=vxpn[i];
    vyp[i]=vypn[i];
    vzp[i]=vzpn[i];
  }
  delete[] dist;
  return;
}

void Heating(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is, iu=pmb->ie;


  for(int k=kl; k<=ku; ++k){
    for(int j=jl; j<=ju; ++j){
#pragma omp simd
      for(int i=il; i<=iu; ++i){
        Real rho = cons(IDN,k,j,i);
        if (heatflag == 0){
          cons(IEN,k,j,i) += heatrate*rho * dt;
        } else if (heatflag==1){
          Real rrat=x1min/pmb->pcoord->x1v(i);
          cons(IEN,k,j,i) += consFr*rrat*rrat*rrat/pmb->pcoord->x1v(i)*dt*pmb->prad->prat*pmb->prad->crat; 
        }
      }
    }
  }

}

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  Radiation *prad = pmb->prad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real press = prim(IEN,k,j,i)*(gamma_gas-1);
    Real gast = prim(IEN,k,j,i)/rho;
    Real logt = log10(gast * tempunit);
    Real logp;
    int np1 = 0, np2 = 0, nt1=0, nt2=0;
    Real kappa, kappaP;
    Real kappas = 0.0;

    if (Opacityflag == 0 ){
      logp = log10(press* presunit);

      while((logp > logptable(np2)) && (np2 < 50)){
        np1 = np2;
        np2++;
      }
      if(np2==50) np2=np1;

  /* The data point should between NrhoT1 and NrhoT2 */
      while((logt > logttable(nt2)) && (nt2 < 100)){
        nt1 = nt2;
        nt2++;
      }
      if(nt2==100) nt2=nt1;

      if(gast > 2.e4 ) kappas = 0.2 * (1.0 + 0.6);
    } else if (Opacityflag==3){
      logp = log10(rho* rhounit);

      while((logp > logptable(np2)) && (np2 < 140)){
        np1 = np2;
        np2++;
      }
      if(np2==140) np2=np1;

  /* The data point should between NrhoT1 and NrhoT2 */
      while((logt > logttable(nt2)) && (nt2 < 70)){
        nt1 = nt2;
        nt2++;
      }
      if(nt2==70) nt2=nt1;
    }

    Real kappaross_t1_p1=opacitytableross(nt1,np1);
    Real kappaross_t1_p2=opacitytableross(nt1,np2);
    Real kappaross_t2_p1=opacitytableross(nt2,np1);
    Real kappaross_t2_p2=opacitytableross(nt2,np2);
    Real kappaplanck_t1_p1=opacitytableplanck(nt1,np1);
    Real kappaplanck_t1_p2=opacitytableplanck(nt1,np2);
    Real kappaplanck_t2_p1=opacitytableplanck(nt2,np1);
    Real kappaplanck_t2_p2=opacitytableplanck(nt2,np2);

    Real p_1 = logptable(np1);
    Real p_2 = logptable(np2);
    Real t_1 = logttable(nt1);
    Real t_2 = logttable(nt2);
    if(np1 == np2){
      if(nt1 == nt2){
        kappa = kappaross_t1_p1;
        kappaP = kappaplanck_t1_p1;
      }else{
        kappa = kappaross_t1_p1 + (kappaross_t2_p1 - kappaross_t1_p1) *
                                (logt - t_1)/(t_2 - t_1);
        kappaP = kappaplanck_t1_p1 + (kappaplanck_t2_p1 - kappaplanck_t1_p1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1 == nt2){
        kappa = kappaross_t1_p1 + (kappaross_t1_p2 - kappaross_t1_p1) *
                                (logp - p_1)/(p_2 - p_1);
        kappaP = kappaplanck_t1_p1 + (kappaplanck_t1_p2 - kappaplanck_t1_p1) *
                                (logp - p_1)/(p_2 - p_1);
      }else{
        kappa = kappaross_t1_p1 * (t_2 - logt) * (p_2 - logp)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t2_p1 * (logt - t_1) * (p_2 - logp)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t1_p2 * (t_2 - logt) * (logp - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t2_p2 * (logt - t_1) * (logp - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1));
        kappaP = kappaplanck_t1_p1 * (t_2 - logt) * (p_2 - logp)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t2_p1 * (logt - t_1) * (p_2 - logp)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t1_p2 * (t_2 - logt) * (logp - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t2_p2 * (logt - t_1) * (logp - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1));
      }
    }/* end same p */

    prad->sigma_s(k,j,i,ifr) = kappas * rho * rhounit * lunit;
    prad->sigma_a(k,j,i,ifr) = kappa * rho * rhounit * lunit;
    prad->sigma_ae(k,j,i,ifr) = prad->sigma_a(k,j,i,ifr);
    prad->sigma_planck(k,j,i,ifr) = (kappaP-kappa) * rho * rhounit * lunit;

  }
  }}}
}

void FixedOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  Radiation *prad = pmb->prad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;

  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    prad->sigma_s(k,j,i,ifr) = opascgs * rho * rhounit * lunit;
    prad->sigma_a(k,j,i,ifr) = opaacgs * rho * rhounit * lunit;
    prad->sigma_ae(k,j,i,ifr)= prad->sigma_a(k,j,i,ifr);
    prad->sigma_planck(k,j,i,ifr) = (opapcgs-opaacgs) * rho * rhounit * lunit;
  }
  }
  }
  }
}


void DiskOpacityDust(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  Radiation *prad = pmb->prad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  Real rossabs[6]={0.02847576, -0.16854279, 0.30235122, -0.67387853, 2.65703615, -3.27695587};
  Real rosssca[6]={-0.09305515, 0.74358988, -1.80067725, 0.6050232, 3.01726376, -3.09687495};
  Real planckabs[6]={0.05032147, -0.38562233, 1.04702911, -1.58181518, 2.55057924, -2.61389748};
  Real rossabsISM[6]={0.14568893, -1.14278005, 2.93327461, -2.91803535, 2.84325336, -3.51464483};
  Real rossscaISM[6]={-2.12197785e-03, 1.60875361e-02, -4.66724374e-02, 7.31932974e-02,
                      2.93131758e+00, -1.34511597e+01};
  Real planckabsISM[6]={0.11478382, -0.83039063, 1.83708756, -1.42737681, 2.21618023, -3.19473119};
  Real kappas, kappa, kappaP;

  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);
    if (gast*tempunit < 1500.){
      Real logt = log10(gast * tempunit);
      Real logt2 = logt*logt;
      Real logt3 = logt2*logt;
      Real logt4 = logt2*logt2;
      Real logt5 = logt3*logt2;

      kappas=pow(10.,rosssca[0]*logt5+rosssca[1]*logt4+rosssca[2]*logt3+rosssca[3]*logt2+rosssca[4]*logt+rosssca[5]);
      kappa =pow(10.,rossabs[0]*logt5+rossabs[1]*logt4+rossabs[2]*logt3+rossabs[3]*logt2+rossabs[4]*logt+rossabs[5]);
      kappaP=pow(10.,planckabs[0]*logt5+planckabs[1]*logt4+planckabs[2]*logt3+planckabs[3]*logt2+planckabs[4]*logt+planckabs[5]);
    } else {
      kappas = 0.0;
      kappa = 0.01;
      kappaP = 0.01;
    }
    prad->sigma_s(k,j,i,ifr) = kappas * rho * rhounit * lunit;
    prad->sigma_a(k,j,i,ifr) = kappa * rho * rhounit * lunit;
    prad->sigma_ae(k,j,i,ifr)= prad->sigma_a(k,j,i,ifr);
    prad->sigma_planck(k,j,i,ifr) = (kappaP-kappa) * rho * rhounit * lunit;
  }
  }
  }
  }
}
