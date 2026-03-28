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

// this unit system uses Sigmaunit (with consistent Hunit and rhounit), Q and Tunit as input

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif




static Real rhofloor;

static AthenaArray<Real> coolingtable;
static AthenaArray<Real> logPtable;
static AthenaArray<Real> logrhotable;


static AthenaArray<Real> ztable;
static AthenaArray<Real> densitytable;
static AthenaArray<Real> energytable;




void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


Real findznum(const Real x3);


void coolingrate(const Real rho, const Real press, Real &edot);


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


void TableCooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real coolingr; 
        coolingrate(prim(IDN,k,j,i), prim(IEN,k,j,i), coolingr);
        cons(IEN,k,j,i) -=  prim(IDN,k,j,i) * coolingr * dt; //(ga - 1.0);
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
Real Q, beta, amp, Sigmacgs, Omegacgs, rhocgs, cscgs, hcgs,   Pcgs;
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
  
  AllocateUserHistoryOutput(9);
  EnrollUserHistoryOutput(0, HistoryReynolds, "vxvy"); //integrated gravitational stress, 
  EnrollUserHistoryOutput(1, HistoryGrav, "gxgy");  //integrated Reynolds stress, clear
  EnrollUserHistoryOutput(2, Historypressure, "press"); //integrated pressure, clear
  EnrollUserHistoryOutput(3, HistorymidQ, "midrho");
  EnrollUserHistoryOutput(4, HistoryQT, "QT");
  EnrollUserHistoryOutput(5, HistorydVx2, "vx2"); //integrated 0.5*rho vx^2, check if its same as 1KE output by history.cpp
  EnrollUserHistoryOutput(6, HistorydVy2, "vy2"); // integrated 0.5*rho (delta vy)^2, will be clear if vx2 is good


  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, Outflow_X2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, Inflow_X1);
  
      
      
      //************************
        ztable.NewAthenaArray(256);
        densitytable.NewAthenaArray(256);
        energytable.NewAthenaArray(256);

        coolingtable.NewAthenaArray(212,46);//example
        logPtable.NewAthenaArray(212);
        logrhotable.NewAthenaArray(46);
            
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

            if ( (fenergy=fopen("./energy.txt","r"))==NULL )
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

              for(i=0; i<256; i++){
                fscanf(fenergy,"%lf",&(energytable(i)));
              }
            
            fclose(fx3);
            fclose(fdens);
            fclose(fenergy);
            //read in the cooling function table

            FILE *fcooling, *flogP, *flogrho;

            fcooling = fopen("./cooling.txt", "r");
            flogP = fopen("./logT.txt", "r");
            flogrho = fopen("./logrho.txt", "r");

            for(int j=0; j<212; j++){
                for(int i=0; i<46; i++){
                  fscanf(fcooling,"%lf",&(coolingtable(j,i)));
                }
             }
                 for(int i=0; i<46; i++){
                fscanf(flogrho,"%lf",&(logrhotable(i)));
             }

                for(int i=0; i<212; i++){
                    fscanf(flogP,"%lf",&(logPtable(i)));
             }
             fclose(fcooling);
             fclose(flogrho);
             fclose(flogP);


//************************

  EnrollUserExplicitSourceFunction(VertGrav); //we  need vertgrav
    
  EnrollUserExplicitSourceFunction(TableCooling); //test cooling function

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

  Pcgs = pin->GetOrAddReal("problem", "midpressure", 1.e27); //the surface density in cgs, which will be used as a unit
  rhocgs = pin->GetOrAddReal("problem", "midrho", 1.e10);// midplane density
  Omegacgs = pin->GetOrAddReal("problem", "Omegacgs", 10.0); //the angular frequency in cgs, which will be used as a unit
  cscgs = std::sqrt(Pcgs/rhocgs); //midplane sound speed, also unit
  hcgs = cscgs/Omegacgs; //cs/omega, also length unit
  Q = SQR(Omegacgs)/TWO_PI/rhocgs/G; //calculate Q as sanity check
  
  beta = pin->GetReal("problem","beta");
  amp = pin->GetReal("problem","amp");
  nwx = pin->GetInteger("problem","nwx");
  nwy = pin->GetInteger("problem","nwy");
  strat = pin->GetBoolean("problem","strat");


  if (NON_BAROTROPIC_EOS) {
    p0 = Pcgs; // midplane pressure, total
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
      energytable.DeleteAthenaArray();
      coolingtable.DeleteAthenaArray();
      logrhotable.DeleteAthenaArray();
      logPtable.DeleteAthenaArray();


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
  Real den,  energy;
  // update the hydro variables as initial conditions
  for (int k=ks; k<=ke; k++) {
        x3 = pcoord->x3v(k);
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            x1 = pcoord->x1v(i);
            x2 = pcoord->x2v(j);
            den=1.0;

        if (strat){
          int z_id = findznum(x3);
          //den= finddensity(x3); //we have found our density through theta, normalized already
          den = densitytable(z_id);
          energy = energytable(z_id);
          rd = amp*std::cos(kx*x1 + ky*x2); //stratified disk is without initial perturbations
        }
        
        if (den < rhofloor){  //set initial density floor
            den = rhofloor;
          }

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


void coolingrate(const Real rho, const Real press, Real &edot) //interpolate cooling rate from table
{
    // Convert the pressure and density into logarithmic space for table lookups
    Real logP = log10(press * Pcgs);   // Logarithmic pressure (adjusted by Pcgs)
    Real logrho = log10(rho * rhocgs); // Logarithmic density (adjusted by rhocgs)

    int rho_idx1 = 0; // Lower index for the density
    int rho_idx2 = 0; // Upper index for the density

    // Find the appropriate indices (rho_idx1, rho_idx2) in the logrho table for interpolation
    while ((logrho > logrhotable(rho_idx2)) && (rho_idx2 < 99)) {
        rho_idx1 = rho_idx2; // Update the lower bound
        rho_idx2++;          // Move to the next upper bound
    }
    // If the input logrho is beyond the table's upper limit, fix to the last interval
    if (rho_idx2 == 99 && (logrho > logrhotable(rho_idx2)))
        rho_idx1 = rho_idx2;

    /* The data point should now be between rho_idx1 and rho_idx2 */

    int P_idx1 = 0; // Lower index for the pressure
    int P_idx2 = 0; // Upper index for the pressure

    // Find the appropriate indices (P_idx1, P_idx2) in the logP table for interpolation
    while ((logP > logPtable(P_idx2)) && (P_idx2 < 99)) {
        P_idx1 = P_idx2; // Update the lower bound
        P_idx2++;        // Move to the next upper bound
    }
    // If the input logP is beyond the table's upper limit, fix to the last interval
    if (P_idx2 == 99 && (logP > logPtable(P_idx2)))
        P_idx1 = P_idx2;

    // Retrieve the cooling rate values at the four corner points of the interpolation rectangle
    Real cool_P1_rho1 = coolingtable(P_idx1, rho_idx1); // Cooling rate at (P1, rho1)
    Real cool_P1_rho2 = coolingtable(P_idx1, rho_idx2); // Cooling rate at (P1, rho2)
    Real cool_P2_rho1 = coolingtable(P_idx2, rho_idx1); // Cooling rate at (P2, rho1)
    Real cool_P2_rho2 = coolingtable(P_idx2, rho_idx2); // Cooling rate at (P2, rho2)

    // Retrieve the logarithmic density and pressure bounds for interpolation
    Real rho_1 = logrhotable(rho_idx1); // Lower density bound
    Real rho_2 = logrhotable(rho_idx2); // Upper density bound
    Real P_1 = logPtable(P_idx1);       // Lower pressure bound
    Real P_2 = logPtable(P_idx2);       // Upper pressure bound

    // Perform interpolation based on the indices
    if (rho_idx1 == rho_idx2) { // Density bounds are the same
        if (P_idx1 == P_idx2) { // Pressure bounds are the same
            edot = cool_P1_rho1; // Direct value, no interpolation needed
        } else {
            // Linear interpolation along the pressure axis
            edot = cool_P1_rho1 + (cool_P2_rho1 - cool_P1_rho1) *
                                   (logP - P_1) / (P_2 - P_1);
        }
    } else { // Density bounds are different
        if (P_idx1 == P_idx2) { // Pressure bounds are the same
            // Linear interpolation along the density axis
            edot = cool_P1_rho1 + (cool_P1_rho2 - cool_P1_rho1) *
                                   (logrho - rho_1) / (rho_2 - rho_1);
        } else {
            // Bilinear interpolation across both density and pressure
            edot = cool_P1_rho1 * (P_2 - logP) * (rho_2 - logrho) /
                                   ((P_2 - P_1) * (rho_2 - rho_1))
                 + cool_P2_rho1 * (logP - P_1) * (rho_2 - logrho) /
                                   ((P_2 - P_1) * (rho_2 - rho_1))
                 + cool_P1_rho2 * (P_2 - logP) * (logrho - rho_1) /
                                   ((P_2 - P_1) * (rho_2 - rho_1))
                 + cool_P2_rho2 * (logP - P_1) * (logrho - rho_1) /
                                   ((P_2 - P_1) * (rho_2 - rho_1));
        }
    }
    edot = edot *rhocgs/Pcgs/Omegacgs; //normalization
    /* End of interpolation logic */
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


}




