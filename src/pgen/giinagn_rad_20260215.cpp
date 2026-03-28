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
//#include "../nr_radiation/radiation.hpp"
//#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif



// The space for opacity table

static AthenaArray<Real> opacitytable;
static AthenaArray<Real> planckopacity;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logrhottable;
static AthenaArray<Real> logttable_planck;
static AthenaArray<Real> logrhottable_planck;


static Real rhofloor;


static AthenaArray<Real> ztable;
static AthenaArray<Real> densitytable;
static AthenaArray<Real> energytable;




void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Outflow_rad_X2(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Inflow_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck);


Real findznum(const Real x3);


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);


void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  Real tau = 3.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        cons(IEN,k,j,i) -= dt / tau * prim(IPR,k,j,i) / 0.66667; //(ga - 1.0);
      }
    }
  }
  return;
}


namespace {

      
Real HistoryGrav(MeshBlock *pmb, int iout);
Real HistoryReynolds(MeshBlock *pmb, int iout);
Real HistorymidQ(MeshBlock *pmb, int iout);
Real Historypressure(MeshBlock *pmb, int iout);
Real HistoryQT(MeshBlock *pmb, int iout);
Real HistorydVx2(MeshBlock *pmb, int iout);
Real HistorydVy2(MeshBlock *pmb, int iout);
Real HistorydVz2(MeshBlock *pmb, int iout);
Real HistoryEtot(MeshBlock *pmb, int iout);


Real p0, gconst;
Real G = 6.67e-8;
Real Q, beta, amp, Sigmacgs, Omegacgs, rhocgs, cscgs, hcgs, Tcgs;
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

  rhofloor = pin->GetReal("hydro", "dfloor");
  
  AllocateUserHistoryOutput(8);
  EnrollUserHistoryOutput(0, HistoryReynolds, "vxvy"); //integrated gravitational stress, 
  EnrollUserHistoryOutput(1, HistoryGrav, "gxgy");  //integrated Reynolds stress, clear
  EnrollUserHistoryOutput(2, Historypressure, "press"); //integrated pressure, clear
  EnrollUserHistoryOutput(3, HistorymidQ, "midrho");
  EnrollUserHistoryOutput(4, HistoryQT, "QT");
  EnrollUserHistoryOutput(5, HistorydVx2, "vx2"); //integrated 0.5*rho vx^2, check if its same as 1KE output by history.cpp
  EnrollUserHistoryOutput(6, HistorydVy2, "vy2"); // integrated 0.5*rho (delta vy)^2, will be clear if vx2 is good

  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, Outflow_X2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, Inflow_X1);
  
      
      
     
        ztable.NewAthenaArray(256);
        densitytable.NewAthenaArray(256);
            
            // read in the structure table
            FILE *fdens, *fenergy, *fx3;
            if ( (fx3=fopen("./zcoord.txt","r"))==NULL )
            {
              printf("Open input file error");
              return;
            }
            
            if ( (fdens=fopen("./density.txt","r"))==NULL )
            {
              printf("Open input file error");
              return;
            }

            int i;
              for(i=0; i<256; i++){
                fscanf(fx3,"%lf",&(ztable(i)));
              }
            
              for(i=0; i<256; i++){
                fscanf(fdens,"%lf",&(densitytable(i)));
              }

            fclose(fx3);
            fclose(fdens);

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
  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0", 1.0); //which will always be the time unit

  // hydro parameters

  Tcgs = pin->GetOrAddReal("problem", "midtemp", 1.e4); //the midplane temperature in cgs, which will be used as a unit, should be same as T_unit in radiation
  rhocgs = pin->GetOrAddReal("problem", "midrho", 1.e-10);// midplane density


  Omegacgs = pin->GetOrAddReal("problem", "Omegacgs", 1.e-9); //the angular frequency in cgs, which will be used as a unit
  //cscgs = std::sqrt(Pcgs/rhocgs); //midplane sound speed
  //hcgs = cscgs/Omegacgs; //cs/omega
  Q = SQR(Omegacgs)/TWO_PI/rhocgs/G; //calculate Q as sanity check
  
  beta = pin->GetReal("problem","beta");
  amp = pin->GetReal("problem","amp");
  nwx = pin->GetInteger("problem","nwx");
  nwy = pin->GetInteger("problem","nwy");
  strat = pin->GetBoolean("problem","strat");


  if (NON_BAROTROPIC_EOS) {
    T0 = Tcgs; // midplane temperature
  }


  if (SELF_GRAVITY_ENABLED) {
    gconst = 1/TWO_PI/Q; //in code units
    SetGravitationalConstant(gconst);
    Real eps = pin->GetOrAddReal("self_gravity","grav_eps", 0.0);
    SetGravityThreshold(eps);
  }
return;
}


//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

  // free memory
      ztable.DeleteAthenaArray();
      densitytable.DeleteAthenaArray();
  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  if(RADIATION_ENABLED||IM_RADIATION_ENABLED){
    
      pnrrad->EnrollOpacityFunction(DiskOpacity); //update opacity
  }
  return;
}


//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================


void MeshBlock::UserWorkInLoop(void) 
{
}


//========================================================================================
//! \fn void MeshBlock::PGen()
//  \brief 
//========================================================================================


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    //std::cout << "rho_0 in unit = " << d0 << std::endl;
    //std::cout << "G = in unit " << gconst << std::endl;
    //std::cout << "pratmid " << pratmidplane << std::endl;
  // set wavenumbers of initial shearing perturbation
  Real kx = (TWO_PI/x1size)*(static_cast<Real>(nwx));
  Real ky = (TWO_PI/x2size)*(static_cast<Real>(nwy));

  Real x1, x2, x3, rd, rp, rvx, rvy;
  Real den, energy;
  // update the hydro variables as initial conditions
  for (int k=ks; k<=ke; k++) {
        x3 = pcoord->x3v(k);
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            x1 = pcoord->x1v(i);
            x2 = pcoord->x2v(j);
            den= 1.0;

        if (strat){
          int z_id = findznum(x3);
          //den= finddensity(x3); //we have found our density through theta, normalized already
          den = densitytable(z_id);
          rd = amp*std::cos(kx*x1 + ky*x2); //stratified disk is without initial perturbations
        }
        
        if (den < rhofloor){  //set initial density floor
            den = rhofloor;
          }
        
        Real temp = den * den * den;
        Real energy = 1.5 * den * temp; //gas energy!

        rvx = amp*kx/ky*std::sin(kx*x1 + ky*x2);
        rvy = amp*std::sin(kx*x1 + ky*x2);
        
        phydro->u(IDN,k,j,i) = (den+rd);
        phydro->u(IM1,k,j,i) = (den+rd)*rvx;
        phydro->u(IM2,k,j,i) = (den+rd)*rvy;
        if (pmy_mesh->shear_periodic) {
          phydro->u(IM2,k,j,i) -= (den+rd)*(qshear*Omega0*x1);
        }
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) 
        {
          phydro->u(IEN,k,j,i) = energy + 0.5*(SQR(phydro->u(IM1,k,j,i)) +SQR(phydro->u(IM2,k,j,i)) +
                                                SQR(phydro->u(IM3,k,j,i))
                                                ) / phydro->u(IDN,k,j,i); //calculate U from general EoS
        }
        if(RADIATION_ENABLED || IM_RADIATION_ENABLED){

          x3 = pcoord->x3v(k);
          //Real rhounit = pnrrad->rhounit;
          //Real tunit = pnrrad->tunit;
          //Real lunit = pnrrad->lunit; 
          Real prat = pnrrad->prat; //This ratio is calculated at the midplane but assumed to be universal
          Real radflx = 0.0;
          Real gast = temp; 
          Real er = std::max(gast * gast * gast * gast, 1.e-8); //set some floor value initially   std::abs(radflx)

          for(int ifr=0; ifr<pnrrad->nfreq; ++ifr){
            Real coefa = 0.0, coefb = 0.0;
            for(int n=0; n<pnrrad->nang; ++n){
              Real &miuz = pnrrad->mu(2,k,j,i,n); // 2 is the z direction
              Real &weight = pnrrad->wmu(n);
              if(miuz > 0.0){
                coefa += weight;
                coefb += (miuz * weight);   
              }
            }
            for(int n=0; n<pnrrad->nang; ++n){
              Real &miuz = pnrrad->mu(2,k,j,i,n);
            
              if(miuz > 0.0){
                pnrrad->ir(k,j,i,ifr*pnrrad->nang+n) = 0.5 *
                                       (er/coefa + radflx/coefb);
              }else{
                pnrrad->ir(k,j,i,ifr*pnrrad->nang+n) = 0.5 *
                                       (er/coefa - radflx/coefb);
              }
              ir_cm(n) = pnrrad->ir(k,j,i,ifr*pnrrad->nang+n); //nfreq=0

            }
        }
        
          Real vx = phydro->u(IM1,k,j,i)/(den+rd);
          Real vy = phydro->u(IM2,k,j,i)/(den+rd);
          Real vz = phydro->u(IM3,k,j,i)/(den+rd);

          //for(int n=0; n<prad->n_fre_ang; ++n)
            //ir_cm(n) = gast * gast * gast * gast;

          Real *ir_lab = &(pnrrad->ir(k,j,i,0));
          Real *mux = &(pnrrad->mu(0,k,j,i,0));
          Real *muy = &(pnrrad->mu(1,k,j,i,0));
          Real *muz = &(pnrrad->mu(2,k,j,i,0));
          pnrrad->pradintegrator->ComToLab(vx,vy,vz,mux,muy,muz,ir_cm,ir_lab); //Lorentz transformation
        }
      } //i
    } //j
  } //k

  return;
}


Real findznum(const Real x3)
{
    int znum = 0;
    while(x3 > ztable(znum)){
        znum++;
    }
     return znum;
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



void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  NRRadiation *pnrrad = pmb->pnrrad;
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
      // electron scattering opacity
  Real kappas = 0.2 * (1.0 + 0.7);
  Real kappaa = 0.0;
  
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<pnrrad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);
    Real kappa, kappa_planck;
    rossopacity(rho, gast, kappa, kappa_planck);

    if(kappa < kappas){
      if(gast < 1.0){
        kappaa = kappa;
        kappa = 0.0;
      }else{
        kappaa = 0.0;
      }
    }else{
      kappaa = kappa - kappas;
      kappa = kappas;
    }
    Real factor = 1.0;
    pnrrad->sigma_s(k,j,i,ifr) = kappa * rho * rhounit * lunit * factor;
    pnrrad->sigma_a(k,j,i,ifr) = kappaa * rho * rhounit * lunit* factor;
    pnrrad->sigma_p(k,j,i,ifr) = kappaa * rho * rhounit * lunit* factor; //kappa_planck * rho * rhounit * lunit* factor;
    pnrrad->sigma_pe(k,j,i,ifr) = pnrrad->sigma_p(k,j,i,ifr);

   // pnrrad->sigma_s(k,j,i,ifr) = kappa * rho * rhounit * lunit;
   // pnrrad->sigma_a(k,j,i,ifr) = kappaa * rho * rhounit * lunit;
   // pnrrad->sigma_ae(k,j,i,ifr) = pnrrad->sigma_a(k,j,i,ifr);
   // if(kappaa < kappa_planck)
   //   pnrrad->sigma_planck(k,j,i,ifr) = (kappa_planck-kappaa)*rho*rhounit*lunit;
   // else
   //   pnrrad->sigma_planck(k,j,i,ifr) = 0.0;
  }    
 }}}

}



void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck)
{
    Real tunit = Tcgs;
    Real rhounit = rhocgs;
    Real logt = log10(tgas * tunit);
    Real logrhot = log10(rho* rhounit) - 3.0* logt + 18.0;
    int nrhot1_planck = 0;
    int nrhot2_planck = 0;
    
    int nrhot1 = 0;
    int nrhot2 = 0;

    while((logrhot > logrhottable_planck(nrhot2_planck)) && (nrhot2_planck < 36)){
      nrhot1_planck = nrhot2_planck;
      nrhot2_planck++;
    }
    if(nrhot2_planck==36 && (logrhot > logrhottable_planck(nrhot2_planck)))
      nrhot1_planck=nrhot2_planck;

    while((logrhot > logrhottable(nrhot2)) && (nrhot2 < 45)){
      nrhot1 = nrhot2;
      nrhot2++;
    }
    if(nrhot2==45 && (logrhot > logrhottable(nrhot2)))
      nrhot1=nrhot2;
  
  /* The data point should between NrhoT1 and NrhoT2 */
    int nt1_planck = 0;
    int nt2_planck = 0;
    int nt1 = 0;
    int nt2 = 0;
    while((logt > logttable_planck(nt2_planck)) && (nt2_planck < 137)){
      nt1_planck = nt2_planck;
      nt2_planck++;
    }
    if(nt2_planck==137 && (logt > logttable_planck(nt2_planck)))
      nt1_planck=nt2_planck;

    while((logt > logttable(nt2)) && (nt2 < 211)){
      nt1 = nt2;
      nt2++;
    }
    if(nt2==211 && (logt > logttable(nt2)))
      nt1=nt2;

  

    Real kappa_t1_rho1=opacitytable(nt1,nrhot1);
    Real kappa_t1_rho2=opacitytable(nt1,nrhot2);
    Real kappa_t2_rho1=opacitytable(nt2,nrhot1);
    Real kappa_t2_rho2=opacitytable(nt2,nrhot2);

    Real planck_t1_rho1=planckopacity(nt1_planck,nrhot1_planck);
    Real planck_t1_rho2=planckopacity(nt1_planck,nrhot2_planck);
    Real planck_t2_rho1=planckopacity(nt2_planck,nrhot1_planck);
    Real planck_t2_rho2=planckopacity(nt2_planck,nrhot2_planck);


    // in the case the temperature is out of range
    // the planck opacity should be smaller by the 
    // ratio T^-3.5
    if(nt2_planck == 137 && (logt > logttable_planck(nt2_planck))){
       Real scaling = pow(10.0, -3.5*(logt - logttable_planck(137)));
       planck_t1_rho1 *= scaling;
       planck_t1_rho2 *= scaling;
       planck_t2_rho1 *= scaling;
       planck_t2_rho2 *= scaling;
    }


    Real rho_1 = logrhottable(nrhot1);
    Real rho_2 = logrhottable(nrhot2);
    Real t_1 = logttable(nt1);
    Real t_2 = logttable(nt2);

    
    if(nrhot1 == nrhot2){
      if(nt1 == nt2){
        kappa = kappa_t1_rho1;
      }else{
        kappa = kappa_t1_rho1 + (kappa_t2_rho1 - kappa_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1 == nt2){
        kappa = kappa_t1_rho1 + (kappa_t1_rho2 - kappa_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);
      }else{
        kappa = kappa_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1));
      }
    }/* end same rhoT */

    rho_1 = logrhottable_planck(nrhot1_planck);
    rho_2 = logrhottable_planck(nrhot2_planck);
    t_1 = logttable_planck(nt1_planck);
    t_2 = logttable_planck(nt2_planck);
 
  /* Now do the same thing for Planck mean opacity */
    if(nrhot1_planck == nrhot2_planck){
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1;
      }else{
        kappa_planck = planck_t1_rho1 + (planck_t2_rho1 - planck_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1 + (planck_t1_rho2 - planck_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);

      }else{        
        kappa_planck = planck_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1));
      }
    }/* end same rhoT */

    return;

}


namespace {

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





Real HistoryCooling(MeshBlock *pmb, int iout) {
  Real coolTotal = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &prim = pmb->phydro->w;  // primitive variables
  Real coolingr = 0.0;  // cooling rate parameter
  
  // Allocate 1D array for cell volume
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);
  
  // Iterate through the mesh
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
          coolingrate(prim(IDN,k,j,i), prim(IEN,k,j,i), coolingr);
        coolTotal += volume(i) * prim(IDN,k,j,i) * coolingr;
      }
    }
  }
  return coolTotal;
}

}


