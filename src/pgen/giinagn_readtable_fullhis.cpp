//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//radiative hydro simulation of GI in AGN
//======================================================================================


//This configuration will use Omega, Sigma and the Jeans length c_s^2/G\Sigma as units

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <fstream>    // ofstream
#include <iomanip>    // setprecision
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../gravity/gravity.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/integrators/rad_integrators.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

// this unit system uses Sigmaunit (with consistent Hunit and rhounit), Q and Tunit as input

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


static Real tunit;
static Real tfloor;
static Real rhofloor;

static AthenaArray<Real> opacitytable;
static AthenaArray<Real> planckopacity;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logrhottable;
static AthenaArray<Real> logttable_planck;
static AthenaArray<Real> logrhottable_planck;
static AthenaArray<Real> ztable;
static AthenaArray<Real> densitytable;
static AthenaArray<Real> fluxtable;
static AthenaArray<Real> ini_profile;
static AthenaArray<Real> bd_data;



void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Outflow_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Inflow_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad, 
     const AthenaArray<Real> &w, FaceField &b,  AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

void analyticalopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck);

Real finddensity(const Real x3);

Real findflux(const Real x3);

void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  Real ga = pmb->peos->GetGamma();
  Real tau = 3.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        cons(IEN,k,j,i) -= dt / tau * prim(IPR,k,j,i) / (ga - 1.0);
      }
    }
  }
  return;
}


namespace {

    
Real HistoryEr(MeshBlock *pmb, int iout);   
Real HistoryFztot(MeshBlock *pmb, int iout);   
Real HistoryGrav(MeshBlock *pmb, int iout);
Real HistoryReynolds(MeshBlock *pmb, int iout);
Real HistorymidQ(MeshBlock *pmb, int iout);
Real Historypressure(MeshBlock *pmb, int iout);
Real HistoryQT(MeshBlock *pmb, int iout);
Real HistorydVx2(MeshBlock *pmb, int iout);
Real HistorydVy2(MeshBlock *pmb, int iout);
Real HistorydVz2(MeshBlock *pmb, int iout);
Real HistoryEtot(MeshBlock *pmb, int iout);


Real cs, gm1, d0, p0, gconst, K, Sigma0, accumulateddens, Sigma0direct, I3Q, temper, flag, kopp, previoustemper, rideal, rhounit, lunit;
Real Q, scaleH, beta, amp, temp, Sigmacgs, Omegacgs, Tcgs, rhocgs, cscgs,hcgs, arad, pratmidplane;
int nwx, nwy; // wavenumbers
Real x1size,x2size,x3size;
Real qshear, Omega0; // shear parameters
bool strat;
} // namespace


//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) { 
    
  turb_flag = pin->GetInteger("problem","turb_flag");

  tfloor = pin->GetReal("radiation", "tfloor");

  rhofloor = pin->GetReal("hydro", "dfloor");
  
  AllocateUserHistoryOutput(9);
  EnrollUserHistoryOutput(0, HistoryEr, "Er"); //integrated gravitational stress, 
  EnrollUserHistoryOutput(1, HistoryFztot, "Fztot"); //integrated gravitational stress, 
  EnrollUserHistoryOutput(2, HistoryReynolds, "vxvy"); //integrated gravitational stress, 
  EnrollUserHistoryOutput(3, HistoryGrav, "gxgy");  //integrated Reynolds stress, clear
  EnrollUserHistoryOutput(4, Historypressure, "press"); //integrated pressure, clear
  EnrollUserHistoryOutput(5, HistorymidQ, "midrho");
  EnrollUserHistoryOutput(6, HistoryQT, "QT");
  EnrollUserHistoryOutput(7, HistorydVx2, "vx2"); //integrated 0.5*rho vx^2, check if its same as 1KE output by history.cpp
  EnrollUserHistoryOutput(8, HistorydVy2, "vy2"); // integrated 0.5*rho (delta vy)^2, will be clear if vx2 is good
  //EnrollUserHistoryOutput(7, HistorydVz2, "vz2");
  //EnrollUserHistoryOutput(8, HistoryEtot, "Etot");
    

  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, Outflow_X2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, Inflow_X1);
  
  if(RADIATION_ENABLED||IM_RADIATION_ENABLED){

  EnrollUserRadBoundaryFunction(BoundaryFace::inner_x3, Inflow_rad_X1);
  EnrollUserRadBoundaryFunction(BoundaryFace::outer_x3, Outflow_rad_X2);
      
      
        ztable.NewAthenaArray(128);
        densitytable.NewAthenaArray(128);
        fluxtable.NewAthenaArray(128);
            
            // read in the opacity table
            FILE *fdens, *fflux, *fx3;
            if ( (fx3=fopen("./zcoord.txt","r"))==NULL )
            {
              printf("Open input file error: zcoord");
              return;
            }
            
            if ( (fdens=fopen("./density.txt","r"))==NULL )
            {
              printf("Open input file error: density");
              return;
            }
            
            if ( (fflux=fopen("./flux.txt","r"))==NULL )
            {
              printf("Open input file error: flux");
              return;
            }
            int i;
              for(i=0; i<128; i++){
                fscanf(fx3,"%lf",&(ztable(i)));
              }
            
              for(i=0; i<128; i++){
                fscanf(fdens,"%lf",&(densitytable(i)));
              }
            
             for(i=0; i<128; i++){
               fscanf(fflux,"%lf",&(fluxtable(i)));
             }
            
            fclose(fx3);
            fclose(fdens);
            fclose(fflux);

  }

  EnrollUserExplicitSourceFunction(VertGrav); //we  need vertgrav


  if (MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in msa.cpp ProblemGenerator" << std::endl
        << "Magnetic field is not yet included." << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_size.nx2 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in msa.cpp ProblemGenerator" << std::endl
        << "This problem does NOT work on a 1D grid." << std::endl;
    ATHENA_ERROR(msg);
  }

  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // shearing box parameters
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega0 = pin->GetReal("orbital_advection","Omega0"); //which will always be one

  // hydro parameters
  if (NON_BAROTROPIC_EOS) {
    gm1 = (pin->GetReal("hydro","gamma") - 1.0);
  }

  // MSA parameters
  rideal = 8.314462618e7/0.6;
  arad = 7.5657332500e-15;

  lunit  = pin->GetReal("radiation", "length_unit");
  rhounit  = pin->GetReal("radiation", "density_unit");



  Sigmacgs = pin->GetReal("problem", "Sigmacgs"); //the surface density in cgs, which will be used as a unit
  Omegacgs = pin->GetReal("problem", "Omegacgs"); //the angular frequency in cgs, which will be used as a unit
  Tcgs = pin->GetReal("problem", "Tcgs");
  cscgs = std::sqrt(rideal*Tcgs);
  rhocgs = pin->GetReal("problem", "rhocgs");// midplane density
  hcgs = pin->GetReal("problem", "hcgs");

  Sigma0 = 1.00757;

  scaleH = Sigma0*hcgs/lunit;
  d0 = rhocgs/rhounit;//something is wrong about the temperature
  cs = cscgs/lunit/Omegacgs; //sound speed in code unit
  Q = pin->GetReal("problem","Q"); //the 3D Q value
  beta = pin->GetReal("problem","beta");
  amp = pin->GetReal("problem","amp");
  nwx = pin->GetInteger("problem","nwx");
  nwy = pin->GetInteger("problem","nwy");
  strat = pin->GetBoolean("problem","strat");
  pratmidplane = arad*std::pow(Tcgs,3.0)/(rhocgs*rideal);
  I3Q = 0.323/std::sqrt(Q+1.72);


  if (NON_BAROTROPIC_EOS) {
    p0 = SQR(cs)*d0; // midplane pressure, total
  }


  //if (SELF_GRAVITY_ENABLED) {
    gconst = 1/TWO_PI/d0/Q;
    SetGravitationalConstant(gconst);
    Real eps = pin->GetOrAddReal("self_gravity","grav_eps", 0.0);
    SetGravityThreshold(eps);
  //}
  K = SQR(cs);
return;
}


//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  ini_profile.DeleteAthenaArray();
  bd_data.DeleteAthenaArray();

  // free memory
  if(RADIATION_ENABLED){
      ztable.DeleteAthenaArray();
      densitytable.DeleteAthenaArray();
      fluxtable.DeleteAthenaArray();
    opacitytable.DeleteAthenaArray();
    logttable.DeleteAthenaArray();
    logrhottable.DeleteAthenaArray();
    planckopacity.DeleteAthenaArray();
    logttable_planck.DeleteAthenaArray();
    logrhottable_planck.DeleteAthenaArray();
  }

  return;
}



void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  if(RADIATION_ENABLED||IM_RADIATION_ENABLED){
    
      prad->EnrollOpacityFunction(DiskOpacity); //update opacity
  
  }

  return;
}



//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================


void MeshBlock::UserWorkInLoop(void) 
{
if(RADIATION_ENABLED||IM_RADIATION_ENABLED){
  
    int il=is, iu=ie, jl=js, ju=je, kl=ks, ku=ke;
    il -= NGHOST;
    iu += NGHOST;
    if(ju>jl){
       jl -= NGHOST;
       ju += NGHOST;
    }
    if(ku>kl){
      kl -= NGHOST;
      ku += NGHOST;
    }
    Real gamma1 = peos->GetGamma() - 1.0;
    AthenaArray<Real> ir_cm;
    ir_cm.NewAthenaArray(prad->n_fre_ang);
      
     for (int k=kl; k<=ku; ++k){
      for (int j=jl; j<=ju; ++j){
       for (int i=il; i<=iu; ++i){
         
          Real& vx=phydro->w(IVX,k,j,i);
          Real& vy=phydro->w(IVY,k,j,i);
          Real& vz=phydro->w(IVZ,k,j,i);
         
          Real& rho=phydro->w(IDN,k,j,i);
          Real& pgas=phydro->w(IEN,k,j,i);
         
          Real vel = sqrt(vx*vx+vy*vy+vz*vz);

          if(vel > prad->vmax * prad->crat){
            Real ratio = prad->vmax * prad->crat / vel;
            vx *= ratio;
            vy *= ratio;
            vz *= ratio;
            
            phydro->u(IM1,k,j,i) = rho*vx;
            phydro->u(IM2,k,j,i) = rho*vy;
            phydro->u(IM3,k,j,i) = rho*vz;

            Real ke = 0.5 * rho * (vx*vx+vy*vy+vz*vz);
            
            Real pb=0.0;
            if(MAGNETIC_FIELDS_ENABLED){
               pb = 0.5*(SQR(pfield->bcc(IB1,k,j,i))+SQR(pfield->bcc(IB2,k,j,i))
                     +SQR(pfield->bcc(IB3,k,j,i)));
            }
            
            Real  eint = phydro->w(IEN,k,j,i)/gamma1;
            
            phydro->u(IEN,k,j,i) = eint + ke + pb;

          }

      }}}
      
      ir_cm.DeleteAthenaArray();
           
    }

}


//========================================================================================
//! \fn void MeshBlock::PGen()
//  \brief 
//========================================================================================


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    std::cout << "In code units, cs = " << cs << std::endl;
    std::cout << "scaleH = " << scaleH << std::endl;
    std::cout << "rho_0 = " << d0 << std::endl;
    //std::cout << "sigma0 = " << d0*scaleH*4*std::sqrt(2)*(I3Q) << std::endl;
    std::cout << "G = " << gconst << std::endl;
    std::cout << "Q_Toomre = " << cs/PI/gconst << std::endl;
    std::cout << "[msa.cpp]: [Lx,Ly,Lz] = [" << x1size <<","<<x2size
              <<","<<x3size<<"]"<<std::endl;
    AthenaArray<Real> ir_cm;
    if(RADIATION_ENABLED||IM_RADIATION_ENABLED){
    std::cout << "in cgs units, tunit = " << prad->tunit << std::endl;
    std::cout << "rhounit = " << prad->rhounit << std::endl;
    std::cout << "lunit = " << prad->lunit << std::endl;
    std::cout << "sigmaunit = " << rhounit*lunit << std::endl;
    std::cout << "midplane prat = " << pratmidplane << std::endl;
    std::cout << "unit prat = " << prad->prat << std::endl;
    std::cout << "midplane cgasrat = " << 3.0e10/cscgs << std::endl;
    std::cout << "unit crat = " << prad->crat << std::endl;
//  std::cout << "tfloor = " << prad->t_floor_<<std::endl;
   //std::cout << "calculated prat = " << prad->prat << std::endl;
//Radiation
  
  //if(RADIATION_ENABLED || IM_RADIATION_ENABLED)
    ir_cm.NewAthenaArray(prad->n_fre_ang);
}
  // set wavenumbers of initial shearing perturbation
  Real kx = (TWO_PI/x1size)*(static_cast<Real>(nwx));
  Real ky = (TWO_PI/x2size)*(static_cast<Real>(nwy));

  Real x1, x2, x3, rd, rp, rvx, rvy;
  Real den, prs;
  // update the hydro variables as initial conditions
  for (int k=ks; k<=ke; k++) {
        x3 = pcoord->x3v(k);
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            x1 = pcoord->x1v(i);
            x2 = pcoord->x2v(j);
            den=0.0;

        if (strat){
          den= finddensity(x3); //we have found our density through theta
          rd = amp*std::cos(kx*x1 + ky*x2); //stratified disk is without initial perturbations
        }
        else {
        den = d0;
        rd = amp*std::cos(kx*x1 + ky*x2); //unstratified disk, with initial perturbation
        }
        
        if (den < rhofloor){  //set initial density floor
            den = rhofloor;
          }

        rvx = amp*kx/ky*std::sin(kx*x1 + ky*x2);
        rvy = amp*std::sin(kx*x1 + ky*x2);

        if (NON_BAROTROPIC_EOS) {
          prs = p0*std::pow(den/d0, 1.3333333333);
          rp = SQR(cs)*rd; //initial pressure perturbation
        }
        
        phydro->u(IDN,k,j,i) = (den+rd);
        phydro->u(IM1,k,j,i) = (den+rd)*rvx;
        phydro->u(IM2,k,j,i) = (den+rd)*rvy;
        if (pmy_mesh->shear_periodic) {
          phydro->u(IM2,k,j,i) -= (den+rd)*(qshear*Omega0*x1);
        }
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) 
        {
          phydro->u(IEN,k,j,i) = (prs+rp)/gm1 + 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                                SQR(phydro->u(IM2,k,j,i)) +
                                                SQR(phydro->u(IM3,k,j,i))
                                                ) / phydro->u(IDN,k,j,i);
        }

              
        if(RADIATION_ENABLED || IM_RADIATION_ENABLED){

          x3 = pcoord->x3v(k);
          Real gast = std::max((prs+rp)/(den+rd),tfloor);
          Real rhounit = prad->rhounit;
          Real lunit = prad->lunit;
          Real prat = prad->prat; //This ratio is calculated at the midplane but assumed to be universal
          Real kappa, kappa_planck;
          analyticalopacity(den, gast, kappa, kappa_planck);
          Real radflx = findflux(x3);
            if (radflx> 1.4997190e-05){
            radflx=1.4997190e-05;}
            if (radflx<- 1.4997190e-05){
            radflx=-1.4997190e-05;}
          Real er = std::max(gast * gast * gast * gast, std::abs(radflx));

          for(int ifr=0; ifr<prad->nfreq; ++ifr){
            Real coefa = 0.0, coefb = 0.0;
            for(int n=0; n<prad->nang; ++n){
              Real &miuz = prad->mu(2,k,j,i,n); // 2 is the z direction
              Real &weight = prad->wmu(n);
              if(miuz > 0.0){
                coefa += weight;
                coefb += (miuz * weight);   
              }
            }
            
            for(int n=0; n<prad->nang; ++n){
              Real &miuz = prad->mu(2,k,j,i,n);
            
              if(miuz > 0.0){
                prad->ir(k,j,i,ifr*prad->nang+n) = 0.5 *
                                       (er/coefa + radflx/coefb);
              }else{
                prad->ir(k,j,i,ifr*prad->nang+n) = 0.5 *
                                       (er/coefa - radflx/coefb);
              }

              ir_cm(n) = prad->ir(k,j,i,ifr*prad->nang+n); //nfreq=0

            }
        }
          
          
          Real vx = phydro->u(IM1,k,j,i)/(den+rd);
          Real vy = phydro->u(IM2,k,j,i)/(den+rd);
          Real vz = phydro->u(IM3,k,j,i)/(den+rd);

          //for(int n=0; n<prad->n_fre_ang; ++n)
            //ir_cm(n) = gast * gast * gast * gast;

          Real *ir_lab = &(prad->ir(k,j,i,0));
          Real *mux = &(prad->mu(0,k,j,i,0));
          Real *muy = &(prad->mu(1,k,j,i,0));
          Real *muz = &(prad->mu(2,k,j,i,0));

          prad->pradintegrator->ComToLab(vx,vy,vz,mux,muy,muz,ir_cm,ir_lab); //Lorentz transformation
        }
        
      } //i
    } //j
  } //k

//if(RADIATION_ENABLED || IM_RADIATION_ENABLED)
    ir_cm.DeleteAthenaArray();

  return;
}


Real finddensity(const Real x3)
{
    int znum = 0;
    while(x3 > ztable(znum)-0.1){
        znum++;
    }
    
     Real dens = densitytable(znum);
     if (dens>1){dens=1.0e-06;}
    return dens;
    
}


Real findflux(const Real x3)
{
    int znum = 0;
    while(x3 > ztable(znum)-0.1){
        znum++;
    }
    
     Real flux = fluxtable(znum);
    if (std::abs(flux)<1.49971829e-10){flux = 1.49971829e-05;}    
return flux;
    
}


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real fsmooth, xi, sign;
  Real Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real z0 = Lz/2.0;
  Real lambda = 0.1/z0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);
        // smoothing function
        if (x3 >= 0) {
          sign = -1.0;
        } else {
          sign = 1.0;
        }
        xi = z0/x3;
        fsmooth = SQR(std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign);
        fsmooth = 1.0;
        // multiply gravitational potential by smoothing function, which we will set to be 1, using the full expression seems to cause numerical issues
        cons(IM3,k,j,i) -= dt*den*SQR(Omega0)*x3*fsmooth;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= dt*den*SQR(Omega0)*prim(IVZ,k,j,i)*x3*fsmooth;
        }
      } 
    }
  }
  return;
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
  Real rhounit = prad->rhounit;
  Real lunit = prad->lunit;

      // electron scattering opacity

  Real kappas = 0.2 * (1.0 + 1.0);
  Real kappaa = 0.0;
  
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);

    Real kappa, kappa_planck;

    //rossopacity(rho, gast, kappa, kappa_planck); 
    // we will first use a completely analytical formula
    //below to approximate kappa and kappa_planck, e.g.

     analyticalopacity(rho, gast, kappa, kappa_planck);


    prad->sigma_s(k,j,i,ifr) = kappas * rho * rhounit * lunit; //unit of rho and length, shouldn't it be only the rhounit?
    prad->sigma_a(k,j,i,ifr) = kappa * rho * rhounit * lunit; //unit of rho and length
    prad->sigma_ae(k,j,i,ifr) = prad->sigma_a(k,j,i,ifr);
    if(kappaa < kappa_planck)
      prad->sigma_planck(k,j,i,ifr) = (kappa_planck-kappaa)*rho*rhounit*lunit; //still need to read rhounit and lunit
    else
      prad->sigma_planck(k,j,i,ifr) = 0.0;

    //printf("sigma_s: %e sigma_a: %e sigma_ae: %e\n",prad->sigma_s(k,j,i,ifr),prad->sigma_a(k,j,i,ifr),prad->sigma_planck(k,j,i,ifr));
  }    


 }}}

}


void analyticalopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck)
{
  
  kappa=0.04;

  kappa_planck=0.04; //just as an example, set Rossland mean opacity to 0.04 cm^2/g

  return;
}


void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
      {
for (int i=is; i<=ie; ++i) {
    for (int j=js; j<=je; ++j) {
      for (int k=1; k<=ngh; ++k) {
          prim(IDN,ke+k,j,i) = prim(IDN,ke,j,i);
          prim(IVX,ke+k,j,i) = prim(IVX,ke,j,i);
          prim(IVY,ke+k,j,i) = prim(IVY,ke,j,i);
          prim(IVZ,ke+k,j,i) = std::max(prim(IVZ,ke,j,i),0.0);
          prim(IEN,ke+k,j,i) = prim(IEN,ke,j,i);
        
      }
    }
  }
  return;
      }


void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{

  
  for (int i=is; i<=ie; ++i) {
    for (int j=js; j<=je; ++j) {
      for (int k=1; k<=ngh; ++k) {
          prim(IDN,ks-k,j,i) = prim(IDN,ks,j,i);
          prim(IVX,ks-k,j,i) = prim(IVX,ks,j,i);
          prim(IVY,ks-k,j,i) = prim(IVY,ks,j,i);
          prim(IVZ,ks-k,j,i) = std::min(prim(IVZ,ks,j,i),0.0);
          prim(IEN,ks-k,j,i) = prim(IEN,ks,j,i);
        
      }
    }
  }


  return;
}



void Outflow_rad_X2(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{

  for (int i=is; i<=ie; ++i) {
    for (int j=js; j<=je; ++j) {
      for (int k=1; k<=ngh; ++k) {

         for(int ifr=0; ifr<prad->nfreq; ++ifr){
            for(int n=0; n<prad->nang; ++n){
              Real miuz = prad->mu(2,ke+k,j,i,ifr*prad->nang+n); //question, is muz mu(0, k, j, i) or mu(2, k, j, i)??
              if(miuz > 0.0){
                ir(ke+k,j,i,ifr*prad->nang+n)
                              = ir(ke+k-1,j,i,ifr*prad->nang+n);
              }else{
                ir(ke+k,j,i,ifr*prad->nang+n) = 0.0;
              }

          }

         }

      }//k
    }//j
  }//i

  return;
}
  



void Inflow_rad_X1(MeshBlock *pmb, Coordinates *pco, Radiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{


  for (int i=is; i<=ie; ++i) {
    for (int j=js; j<=je; ++j) {
      for (int k=1; k<=ngh; ++k) {
 
         for(int ifr=0; ifr<prad->nfreq; ++ifr){
            for(int n=0; n<prad->nang; ++n){
              Real miuz = prad->mu(2,ks-k,j,i,ifr*prad->nang+n);
              if(miuz < 0.0){
                ir(ks-k,j,i,ifr*prad->nang+n)
                              = ir(ks-k+1,j,i,ifr*prad->nang+n);
              }else{
                ir(ks-k,j,i,ifr*prad->nang+n) = 0.0;
              }
          }
        }   

      }//k
    }//j
  }//i


  return;
}


namespace {

    
Real HistoryEr(MeshBlock *pmb, int iout) {
 Real dEr = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
     Real gastemp = w(IPR,k,j,i)/w(IDN,k,j,i);
     dEr +=volume(i)*gastemp*gastemp*gastemp*gastemp;
   }
  }
 }
    Real Er = dEr;
 return Er;
}

Real HistoryFztot(MeshBlock *pmb, int iout) {
 Real Fztot = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->prad->rad_mom_cm;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
    for (int k=ks; k<=ke; k++) {
      if (std::abs(pmb->pcoord->x3v(k)-10) < ((pmb->pcoord->x3v(k))-(pmb->pcoord->x3v(k-1))))
      {
     for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        Fztot += w(IFR3,k,j,i)*volume(i);
     }
    }
  }
 }
 return Fztot;
}
   
    
    
Real HistoryReynolds(MeshBlock *pmb, int iout) {
 Real drhorhovxvy = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 Real vshear = 0.0;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
    if(!pmb->porb->orbital_advection_defined) {
     vshear = -qshear*Omega0*pmb->pcoord->x1v(i);
    } else {
     vshear = 0.0;
    }
     drhorhovxvy += volume(i)*(w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) - vshear));
   }
  }
 }
  Real Reynolds = (drhorhovxvy);
 return Reynolds;
}
 // namespace


//Reynolds stress
Real HistoryGrav(MeshBlock *pmb, int iout) {
 Real dgxgy = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 AthenaArray<Real> &phi = pmb->pgrav->phi;
 Real vshear = 0.0;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
     Real gx= (phi(k,j,i+1)-phi(k,j,i-1))/((pmb->pcoord->x1v(i+1))-(pmb->pcoord->x1v(i-1)));
     Real gy= (phi(k,j+1,i)-phi(k,j-1,i))/((pmb->pcoord->x2v(j+1))-(pmb->pcoord->x2v(j-1)));
     dgxgy += volume(i)*gx*gy/(pmb->pgrav->four_pi_G);
   }
  }
 }
  Real Grav = dgxgy;
 return Grav;
}
 // namespace


//averaged pressure

Real Historypressure(MeshBlock *pmb, int iout) {
 Real drhoP = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 Real vshear = 0.0;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
     drhoP +=volume(i)*w(IPR,k,j,i);
   }
  }
 }
  Real Reynolds = drhoP;
 return Reynolds;
}
 // namespace

Real HistorymidQ(MeshBlock *pmb, int iout) {
 Real rhomid = 0.0;
 Real vol = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
   if (std::abs(pmb->pcoord->x3v(k)) < ((pmb->pcoord->x3v(k))-(pmb->pcoord->x3v(k-1))))
   {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
     rhomid += w(IDN,k,j,i)*volume(i);
     vol += volume(i);
   }
  }

   }
 }
 return rhomid;
}
 
Real HistoryQT(MeshBlock *pmb, int iout) {
 Real Qt = 0.0;
 Real vol = 0.0;
 int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
 AthenaArray<Real> &w = pmb->phydro->w;
 Real vshear = 0.0;
 AthenaArray<Real> volume; // 1D array of volumes
 // allocate 1D array for cell volume used in usr def history
 volume.NewAthenaArray(pmb->ncells1);
 for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
   pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
   for (int i=is; i<=ie; i++) {
     Qt += w(IDN,k,j,i)*volume(i)*std::sqrt(w(IPR,j,k,i)/w(IDN,j,k,i))/PI;
     vol += volume(i);
   }
  }
 }
 return Qt; //needs to be divided by gconst*totalmass in code units
}

Real HistorydVx2(MeshBlock *pmb, int iout) {
  Real dvx2 = 0.0;
  Real vol = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        dvx2 += 0.5*volume(i)*u(IM1,k,j,i)*u(IM1,k,j,i)/u(IDN,k,j,i);
        vol += volume(i);
      }
    }
  }
  return dvx2;
}


    //total

Real HistorydVy2(MeshBlock *pmb, int iout) { 
  Real dvy2 = 0.0;
  Real vol = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        if(!pmb->porb->orbital_advection_defined) {
          vshear = -qshear*Omega0*pmb->pcoord->x1v(i);
        } else {
          vshear = 0.0;
        }
        dvy2 += 0.5*volume(i)*(u(IM2,k,j,i)- u(IDN,k,j,i)*vshear)*(u(IM2,k,j,i) - u(IDN,k,j,i)*vshear)/u(IDN,k,j,i);
        vol += volume(i);
      }
    }
  }
  return dvy2;
}

/*
Real HistorydVz2(MeshBlock *pmb, int iout) {
  Real dvz2 = 0.0;
  Real vol = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        dvz2 += 0.5*volume(i)*w(IDN,k,j,i)*(w(IVZ,k,j,i))*(w(IVZ,k,j,i));
        vol += volume(i);
      }
    }
  }
  return dvz2/vol;
}


Real HistoryEtot(MeshBlock *pmb, int iout) {
  Real dE = 0.0;
  Real vol = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        dE += volume(i)*w(IEN,k,j,i)/gm1;
        vol += volume(i);
      }
    }
  }
  return dE/vol;
}



*/
}
