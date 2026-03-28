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
//! \file radiation.cpp
//  \brief implementation of functions in class Radiation
//======================================================================================


#include <sstream>  // msg
#include <iostream>  // cout
#include <stdexcept> // runtime erro
#include <stdio.h>  // fopen and fwrite


// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp" 
#include "../radiation.hpp"
#include "radiation_implicit.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../../globals.hpp"
#include "../integrators/rad_integrators.hpp"



IMRadiation::IMRadiation(Mesh *pm, ParameterInput *pin){
  // read in the parameters
  // maximum number of iterations
  nlimit_ = pin->GetOrAddInteger("radiation","nlimit",100);
  error_limit_ =  pin->GetOrAddReal("radiation","error_limit",1.e-3);
  cfl_rad = pin->GetOrAddReal("radiation","cfl_rad",1.0);
  ite_scheme = pin->GetOrAddInteger("radiation","iteration",2);

  pimradbdlist = new IMRadBDTaskList(pm);
  pimradhylist = new IMRadHydroTaskList(pm);
}




