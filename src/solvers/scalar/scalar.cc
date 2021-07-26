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
/*! \file scalar.cc
 *
 *  \brief Definitions of common functions for both 2D and 3D runs of the solver class scalar - this class solves the basic Navier-Stokes equation.
 *  \sa scalar.h
 *  \author Roshan Samuel, Shashwat Bhattacharya
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "scalar.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the base scalar class
 *
 *          The short base constructor of the scalar class merely assigns the const references to the grid and parser
 *          class instances being used in the solver.
 *          Also, the maximum allowable number of iterations for the Jacobi iterative solver being used to solve for the
 *          velocities implicitly is set as \f$ N_{max} = N_x \times N_y \times N_z \f$, where \f$N_x\f$, \f$N_y\f$ and \f$N_z\f$
 *          are the number of grid points in the collocated grid at the local sub-domains along x, y and z directions
 *          respectively.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 * \param   mpiParam is a reference to the object of parallel class containing the necessary rank data
 ********************************************************************************************************************************************
 */
scalar::scalar(const grid &mesh, const parser &solParam, parallel &mpiParam):
            hydro(mesh, solParam, mpiParam),
            T(mesh, "T") { }


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the forcing terms for velocity
 *
 *          The forcing terms for the velocity field are initialized here.
 *          Out of the different forcings available in the force class,
 *          the appropriate forcing is chosen according to the parameters set by the user.
 ********************************************************************************************************************************************
 */
void scalar::initVForcing() {
    switch (inputParams.forceType) {
        case 0:
            if (mpiData.rank == 0) std::cout << "WARNING: Running scalar simulation with zero velocity forcing" << std::endl << std::endl;
            V.vForcing = new zeroForcing(mesh, V);
            break;
        case 1:
            if (mpiData.rank == 0) std::cout << "WARNING: Running scalar simulation with random velocity forcing" << std::endl << std::endl;
            V.vForcing = new randomForcing(mesh, V);
            break;
        case 2:
            if (mpiData.rank == 0) std::cout << "WARNING: Running scalar simulation with pure rotation" << std::endl << std::endl;
            V.vForcing = new coriolisForce(mesh, V);
            break;
        case 3:
            if (mpiData.rank == 0) std::cout << "Running convection simulation with pure buoyancy forcing for velocity" << std::endl << std::endl;
            V.vForcing = new buoyantForce(mesh, V, T);
            break;
        case 4:
            if (mpiData.rank == 0) std::cout << "Running rotating convection simulation with both buoyancy and Coriolis forcing for velocity" << std::endl << std::endl;
            V.vForcing = new rotatingConv(mesh, V, T);
            break;
        default:
            if (mpiData.rank == 0) std::cout << "WARNING: Chosen velocity forcing is incompatible with scalar runs. Defaulting to buoyant forcing" << std::endl << std::endl;
            V.vForcing = new buoyantForce(mesh, V, T);
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the forcing terms for temperature
 *
 *          The forcing terms for the temperature field are initialized here.
 *          Out of the different forcings available in the force class,
 *          the appropriate forcing is chosen according to the parameters set by the user.
 ********************************************************************************************************************************************
 */
void scalar::initTForcing() {
    // Currently no forcing for scalar terms are available
    if (mpiData.rank == 0) std::cout << "Running scalar simulation with zero scalar forcing" << std::endl << std::endl;
    T.tForcing = new zeroForcing(mesh, V);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the boundary conditions for temperature
 *
 *          The temperature boundary conditions for all the 6 walls (4 in case of 2D simulations) are initialized here.
 *          Out of the different boundary conditions available in the boundary class,
 *          the appropriate BCs are chosen according to the type of problem being solved.
 ********************************************************************************************************************************************
 */
void scalar::initTBCs() {
    // ADIABATIC BC FOR RBC, SST AND RRBC
    if (inputParams.probType == 5 || inputParams.probType == 6 || inputParams.probType == 8) {
        T.tLft = new neumann(mesh, T.F, 0, 0.0);
        T.tRgt = new neumann(mesh, T.F, 1, 0.0);

    // CONDUCTING BC FOR VERTICAL CONVECTION
    } else if (inputParams.probType == 7) {
        T.tLft = new dirichlet(mesh, T.F, 0, 1.0);
        T.tRgt = new dirichlet(mesh, T.F, 1, 0.0);
    }

#ifndef PLANAR
    T.tFrn = new neumann(mesh, T.F, 2, 0.0);
    T.tBak = new neumann(mesh, T.F, 3, 0.0);
#endif

    if (inputParams.zPer) {
        T.tBot = new periodic(mesh, T.F, 4);
        T.tTop = new periodic(mesh, T.F, 5);
    } else {
        // HOT PLATE AT BOTTOM AND COLD PLATE AT TOP FOR RBC AND RRBC
        if (inputParams.probType == 5 || inputParams.probType == 8) {
            // CREATE HEATING PATCH IF THE USER SET PARAMETER FOR HEATING PLATE IS TRUE
            if (inputParams.nonHgBC) {
#ifndef PLANAR
                if (mpiData.rank == 0) std::cout << "Using non-homogeneous boundary condition (heating plate) on bottom wall" << std::endl << std::endl;
                T.tBot = new hotPlate(mesh, T.F, 4, inputParams.patchRadius);
#else
                if (mpiData.rank == 0) std::cout << "WARNING: Non-homogenous BC flag is set to true in input paramters for 2D simulation. IGNORING" << std::endl << std::endl;
#endif
            } else {
                T.tBot = new dirichlet(mesh, T.F, 4, 1.0);
            }
            T.tTop = new dirichlet(mesh, T.F, 5, 0.0);

        // COLD PLATE AT BOTTOM AND HOT PLATE AT TOP FOR SST
        } else if (mesh.inputParams.probType == 6) {
            T.tBot = new dirichlet(mesh, T.F, 4, 0.0);
            T.tTop = new dirichlet(mesh, T.F, 5, 1.0);
        }
    }
};

