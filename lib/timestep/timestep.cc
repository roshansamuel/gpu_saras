/********************************************************************************************************************************************
 * Saras
 * 
 * Copyright (C) 2019, Mahendra K. Verma
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     3. Neither the name of the copyright holder nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************************************************************************
 */
/*! \file timestep.cc
 *
 *  \brief Definitions for functions of class timestep
 *  \sa timestep.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "timestep.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the timestep class
 *
 *          The empty constructer merely initializes the local reference to the global mesh variable.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   dt is a const reference to the variable dt used by invoking solver
 * \param   V is a reference to the velocity field and is used merely to initialize local objects
 * \param   P is a reference to the pressure field and is used merely to initialize local objects
 ********************************************************************************************************************************************
 */
timestep::timestep(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P):
    solTime(sTime),
    dt(dt),
    mesh(mesh),
    Pp(mesh),
    mgRHS(mesh),
    tsWriter(tsIO),
    pressureGradient(mesh)
{
    // Below flags may be turned on for debugging/dignostic runs only
    bool viscSwitch = false;
    bool diffSwitch = false;

    if (mesh.inputParams.probType <= 4) {
        // For hydrodynamics simulation, set value of kinematic viscosity only
        nu = 1.0/mesh.inputParams.Re;

    } else if (mesh.inputParams.probType <= 7) {
        // For scalar simulation, set values of kinematic viscosity and thermal diffusion
        if (mesh.inputParams.rbcType == 1) {
            nu = mesh.inputParams.Pr;
            kappa = 1.0;
        } else if (mesh.inputParams.rbcType == 2) {
            nu = sqrt(mesh.inputParams.Pr/mesh.inputParams.Ra);
            kappa = 1.0/sqrt(mesh.inputParams.Pr*mesh.inputParams.Ra);
        } else if (mesh.inputParams.rbcType == 3) {
            nu = 1.0;
            kappa = 1.0/mesh.inputParams.Pr;
        } else if (mesh.inputParams.rbcType == 4) {
            nu = sqrt(mesh.inputParams.Pr/mesh.inputParams.Ra);
            kappa = 1.0/sqrt(mesh.inputParams.Pr*mesh.inputParams.Ra);
        } else {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Invalid RBC non-dimensionalization type. Aborting" << std::endl;
            }
            exit(0);
        }
    }

    // Additional options to turn off diffusion for debugging/diagnostics only
    if (viscSwitch) {
        nu = 0.0;
    }

    if (diffSwitch) {
        kappa = 0.0;
    }

    core = mesh.coreDomain;

    // UPPER AND LOWER LIMITS FOR LOOPS OVER CORE DOMAIN
    xSt = core.lbound(0);        xEn = core.ubound(0);
#ifndef PLANAR
    ySt = core.lbound(1);        yEn = core.ubound(1);
#endif
    zSt = core.lbound(2);        zEn = core.ubound(2);

    tsWriter.mDiff = nu;
    tsWriter.tDiff = kappa;
}


/**
 ********************************************************************************************************************************************
 * \brief   Prototype overloaded function to time-advance the solution by one time-step
 *
 * \param   V is a reference to the velocity vector field to be advanced
 * \param   P is a reference to the pressure scalar field to be advanced
 ********************************************************************************************************************************************
 */
void timestep::timeAdvance(vfield &V, sfield &P) { };


/**
 ********************************************************************************************************************************************
 * \brief   Prototype overloaded function to time-advance the solution by one time-step
 *
 * \param   V is a reference to the velocity vector field to be advanced
 * \param   P is a reference to the pressure scalar field to be advanced
 * \param   T is a reference to the temperature scalar field to be advanced
 ********************************************************************************************************************************************
 */
void timestep::timeAdvance(vfield &V, sfield &P, sfield &T) { };
