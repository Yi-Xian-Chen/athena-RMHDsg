//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_grav_task_list.cpp
//! \brief function implementation for FFTGravitySolverTaskList

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/implicit/radiation_implicit.hpp"
#include "im_rad_task_list.hpp"

//----------------------------------------------------------------------------------------
//! IMRadTaskList constructor

IMRadBDTaskList::IMRadBDTaskList(Mesh *pm) {
  pmy_mesh = pm;
  // Now assemble list of tasks for each stage of time integrator
  {using namespace IMRadBDTaskNames; // NOLINT (build/namespace)
    // compute hydro fluxes, integrate hydro variables
    AddTask(SEND_RAD_BND,NONE);
    AddTask(RECV_RAD_BND,NONE);
    AddTask(SETB_RAD_BND,(RECV_RAD_BND|SEND_RAD_BND));
    if(pm->shear_periodic){
      AddTask(SEND_RAD_SH,SETB_RAD_BND);
      AddTask(RECV_RAD_SH,SEND_RAD_SH|RECV_RAD_BND);
    }
    TaskID setb = SETB_RAD_BND;
    if(pm->shear_periodic)
      setb=(setb|RECV_RAD_SH);
    if(pm->multilevel){
      AddTask(PRLN_RAD_BND,setb);
      AddTask(RAD_PHYS_BND,PRLN_RAD_BND);
    }else{
      AddTask(RAD_PHYS_BND,setb);
    }
    AddTask(CLEAR_RAD, RAD_PHYS_BND);
    // check residual does not need ghost zones
    AddTask(CHK_RAD_RES,NONE);
  } // end of using namespace block

}

//----------------------------------------------------------------------------------------
//! \fn void FFTGravitySolverTaskList::AddTask(const TaskID& id, const TaskID& dep)
//! \brief Sets id and dependency for "ntask" member of task_list_ array, then iterates
//! value of ntask.

void IMRadBDTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace IMRadBDTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_RAD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::ClearRadBoundary);
  } else if (id == SEND_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::SendRadBoundary);
  } else if (id == RECV_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::ReceiveRadBoundary);
  } else if (id == SETB_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::SetRadBoundary);
  } else if (id == RAD_PHYS_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::PhysicalBoundary);
   }else if (id == PRLN_RAD_BND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::ProlongateBoundary);
  } else if (id == SEND_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::SendRadBoundaryShear);
  } else if (id == RECV_RAD_SH) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::ReceiveRadBoundaryShear);
  } else if (id == CHK_RAD_RES) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (IMRadTaskList::*)(MeshBlock*)>
        (&IMRadBDTaskList::CheckResidual);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in IMRadTaskList::AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}


TaskStatus IMRadBDTaskList::ClearRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.ClearBoundary(BoundaryCommSubset::radiation);
  return TaskStatus::success;
}

TaskStatus IMRadBDTaskList::SendRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus IMRadBDTaskList::SendRadBoundaryShear(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SendShearingBoxBoundaryBuffers();
  return TaskStatus::success;
}


TaskStatus IMRadBDTaskList::ReceiveRadBoundary(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveBoundaryBuffers();
  if (!ret)
    return TaskStatus::fail;
  return TaskStatus::success;
}

TaskStatus IMRadBDTaskList::ReceiveRadBoundaryShear(MeshBlock *pmb) {
  bool ret = pmb->prad->rad_bvar.ReceiveShearingBoxBoundaryBuffers();
  if(ret){
    pmb->prad->rad_bvar.SetShearingBoxBoundaryBuffers();
    return TaskStatus::success;
  }else{
    return TaskStatus::fail;
  }
}


TaskStatus IMRadBDTaskList::SetRadBoundary(MeshBlock *pmb) {
  pmb->prad->rad_bvar.SetBoundaries();
  return TaskStatus::success;
}


TaskStatus IMRadBDTaskList::CheckResidual(MeshBlock *pmb) {
  pmy_mesh->pimrad->CheckResidual(pmb, pmb->prad->ir_old,pmb->prad->ir);
  return TaskStatus::success;
}

void IMRadBDTaskList::StartupTaskList(MeshBlock *pmb) {
  pmb->prad->rad_bvar.StartReceiving(BoundaryCommSubset::radiation);
  if(pmy_mesh->shear_periodic)
    pmb->prad->rad_bvar.StartReceivingShear(BoundaryCommSubset::radiation);
  return;
}


