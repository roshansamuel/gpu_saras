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
/*! \file eulerCN_d3.cc
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
 *          The empty constructor merely initializes the local reference to the global mesh variable.
 *          Also, the maximum allowable number of iterations for the Jacobi iterative solver being used to solve for the
 *          velocities implicitly is set as \f$ N_{max} = N_x \times N_y \times N_z \f$, where \f$N_x\f$, \f$N_y\f$ and \f$N_z\f$
 *          are the number of grid points in the collocated grid at the local sub-domains along x, y and z directions
 *          respectively.
 *
 * \param   mesh is a const reference to the global data contained in the grid class.
 ********************************************************************************************************************************************
 */
eulerCN_d3::eulerCN_d3(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P):
    timestep(mesh, sTime, dt, tsIO, V, P),
    mgSolver(mesh, mesh.inputParams)
{
    setCoefficients();

    // This upper limit on max iterations is an arbitrarily chosen function.
    // Using Nx x Ny x Nz as the upper limit may cause the run to freeze for very long time.
    // This can eat away a lot of core hours unnecessarily.
    // It remains to be seen if this upper limit is safe.
    maxIterations = int(std::pow(std::log(mesh.coreSize(0)*mesh.coreSize(1)*mesh.coreSize(2)), 3));

    // If LES switch is enabled, initialize LES model
    if (mesh.inputParams.lesModel) {
        if (mesh.rankData.rank == 0) {
            std::cout << "LES Switch is ON. Using stretched spiral vortex LES Model\n" << std::endl;
        }

        sgsLES = new spiral(mesh, nu);
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to advance the solution using Euler method and Implicit Crank-Nicholson method
 *
 *          The non-linear terms are advanced using explicit Euler method, while the duffusion terms are
 *          advanced by semi-implicit Crank-Nicholson method.
 *          This overloaded function advances velocity and pressure fields for hydrodynamics simulations.
 *
 ********************************************************************************************************************************************
 */
void eulerCN_d3::timeAdvance(vfield &V, sfield &P) {
    static plainvf nseRHS(mesh);
    real subgridKE;

    nseRHS = 0.0;

    // Compute the diffusion term of momentum equation
    V.computeDiff(nseRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    nseRHS *= nu/2;

    // Compute the non-linear term and subtract it from the RHS
    V.computeNLin(V, nseRHS);

    // Add the velocity forcing term
    V.vForcing->addForcing(nseRHS);

    // Add sub-grid stress contribution from LES Model, if enabled
    if (mesh.inputParams.lesModel and solTime > 5*mesh.inputParams.tStp) {
        subgridKE = sgsLES->computeSG(nseRHS, V);
        tsWriter.subgridEnergy = subgridKE;
    }

    // Subtract the pressure gradient term
    pressureGradient = 0.0;
    P.gradient(pressureGradient);
    nseRHS -= pressureGradient;

    // Multiply the entire RHS with dt and add the velocity of previous time-step to advance by explicit Euler method
    nseRHS *= dt;
    nseRHS += V;

    // Synchronize the RHS term across all processors by updating its sub-domain pads
    nseRHS.syncData();

    // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
    solveVx(V, nseRHS);
    solveVy(V, nseRHS);
    solveVz(V, nseRHS);

    // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
    V.divergence(mgRHS);
    mgRHS *= 1.0/dt;

    // IF THE POISSON SOLVER IS BEING TESTED, THE RHS IS SET TO ONE.
    // THIS IS FOR TESTING ONLY AND A SINGLE TIME ADVANCE IS PERFORMED IN THIS TEST
#ifdef TEST_POISSON
    mgRHS.F = 1.0;
#endif

    // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
    mgSolver.mgSolve(Pp, mgRHS);

    // Synchronise the pressure correction term across processors
    Pp.syncData();

    // IF THE POISSON SOLVER IS BEING TESTED, THE PRESSURE IS SET TO ZERO.
    // THIS WAY, AFTER THE SOLUTION OF MG SOLVER, Pp, IS DIRECTLY WRITTEN INTO P AND AVAILABLE FOR PLOTTING
    // THIS IS FOR TESTING ONLY AND A SINGLE TIME ADVANCE IS PERFORMED IN THIS TEST
#ifdef TEST_POISSON
    P.F = 0.0;
#endif

    // Add the pressure correction term to the pressure field of previous time-step, P
    P += Pp;

    // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
    Pp.gradient(pressureGradient);
    pressureGradient *= dt;
    V -= pressureGradient;

    // Impose boundary conditions on the updated velocity field, V
    V.imposeBCs();

    // Impose boundary conditions on the updated pressure field, P
    P.imposeBCs();
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to advance the solution using Euler method and Implicit Crank-Nicholson method
 *
 *          The non-linear terms are advanced using explicit Euler method, while the duffusion terms are
 *          advanced by semi-implicit Crank-Nicholson method.
 *          This overloaded function advances velocity, temperature and pressure fields for scalar simulations.
 *
 ********************************************************************************************************************************************
 */
void eulerCN_d3::timeAdvance(vfield &V, sfield &P, sfield &T) {
    static plainvf nseRHS(mesh);
    static plainsf tmpRHS(mesh);
    real subgridKE;

    nseRHS = 0.0;
    tmpRHS = 0.0;

    // Compute the diffusion term of momentum equation
    V.computeDiff(nseRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    nseRHS *= nu/2;

    // Compute the diffusion term of scalar equation
    T.computeDiff(tmpRHS);
    // Split the diffusion term and multiply by diffusion coefficient
    tmpRHS *= kappa/2;

    // Compute the non-linear term and subtract it from the RHS of momentum equation
    V.computeNLin(V, nseRHS);

    // Compute the non-linear term and subtract it from the RHS of scalar equation
    T.computeNLin(V, tmpRHS);

    // Add the velocity forcing term
    V.vForcing->addForcing(nseRHS);

    // Add the scalar forcing term
    T.tForcing->addForcing(tmpRHS);

    // Add sub-grid stress contribution from LES Model, if enabled
    if (mesh.inputParams.lesModel and solTime > 5*mesh.inputParams.tStp) {
        subgridKE = 0.0;

        if (mesh.inputParams.lesModel == 1)
            subgridKE = sgsLES->computeSG(nseRHS, V);
        else if (mesh.inputParams.lesModel == 2)
            subgridKE = sgsLES->computeSG(nseRHS, tmpRHS, V, T);

        tsWriter.subgridEnergy = subgridKE;
    }

    // Subtract the pressure gradient term from momentum equation
    pressureGradient = 0.0;
    P.gradient(pressureGradient);
    nseRHS -= pressureGradient;

    // Multiply the entire RHS with dt and add the velocity of previous time-step to advance by explicit Euler method
    nseRHS *= dt;
    nseRHS += V;

    // Multiply the entire RHS with dt and add the temperature of previous time-step to advance by explicit Euler method
    tmpRHS *= dt;
    tmpRHS += T;

    // Synchronize both the RHS terms across all processors by updating their sub-domain pads
    nseRHS.syncData();
    tmpRHS.syncData();

    // Using the RHS term computed, compute the guessed velocity of CN method iteratively (and store it in V)
    solveVx(V, nseRHS);
    solveVy(V, nseRHS);
    solveVz(V, nseRHS);

    // Using the RHS term computed, compute the temperature at next time-step iteratively (and store it in T)
    solveT(T, tmpRHS);

    // Calculate the rhs for the poisson solver (mgRHS) using the divergence of guessed velocity in V
    V.divergence(mgRHS);
    mgRHS *= 1.0/dt;

    // Using the calculated mgRHS, evaluate pressure correction (Pp) using multi-grid method
    mgSolver.mgSolve(Pp, mgRHS);

    // Synchronise the pressure correction term across processors
    Pp.syncData();

    // Add the pressure correction term to the pressure field of previous time-step, P
    P += Pp;

    // Finally get the velocity field at end of time-step by subtracting the gradient of pressure correction from V
    Pp.gradient(pressureGradient);
    pressureGradient *= dt;
    V -= pressureGradient;

    // Impose boundary conditions on the updated velocity field, V
    V.imposeBCs();

    // Impose boundary conditions on the updated pressure field, P
    P.imposeBCs();

    // Impose boundary conditions on the updated temperature field, T
    T.imposeBCs();
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for x-velocity
 *
 *          The implicit equation for \f$ u_x' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void eulerCN_d3::solveVx(vfield &V, plainvf &nseRHS) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVx(V.Vx.F.lbound(), V.Vx.F.shape());

    while (true) {
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(tempVx)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVx(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) +
                                           i2hx * mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) +
                                           ihy2 * mesh.ety2(iY) * (V.Vx.F(iX, iY+1, iZ) + V.Vx.F(iX, iY-1, iZ)) +
                                           i2hy * mesh.etyy(iY) * (V.Vx.F(iX, iY+1, iZ) - V.Vx.F(iX, iY-1, iZ)) +
                                           ihz2 * mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) + V.Vx.F(iX, iY, iZ-1)) +
                                           i2hz * mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1))) *
                            dt * nu / 2.0 + nseRHS.Vx(iX, iY, iZ)) /
                     (1.0 + dt * nu * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }

        V.Vx.F = tempVx;

        V.imposeVxBC();

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(tempVx)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVx(iX, iY, iZ) = V.Vx.F(iX, iY, iZ) - 0.5 * dt * nu * (
                              mesh.xix2(iX) * (V.Vx.F(iX+1, iY, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vx.F(iX+1, iY, iZ) - V.Vx.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vx.F(iX, iY+1, iZ) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vx.F(iX, iY+1, iZ) - V.Vx.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vx.F(iX, iY, iZ+1) - 2.0 * V.Vx.F(iX, iY, iZ) + V.Vx.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vx.F(iX, iY, iZ+1) - V.Vx.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }

        tempVx(core) = abs(tempVx(core) - nseRHS.Vx(core));

        locMax = blitz::max(tempVx(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of Vx not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for y-velocity
 *
 *          The implicit equation for \f$ u_y' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void eulerCN_d3::solveVy(vfield &V, plainvf &nseRHS) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVy(V.Vy.F.lbound(), V.Vy.F.shape());

    while (true) {
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(tempVy)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVy(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vy.F(iX+1, iY, iZ) + V.Vy.F(iX-1, iY, iZ)) +
                                           i2hx * mesh.xixx(iX) * (V.Vy.F(iX+1, iY, iZ) - V.Vy.F(iX-1, iY, iZ)) +
                                           ihy2 * mesh.ety2(iY) * (V.Vy.F(iX, iY+1, iZ) + V.Vy.F(iX, iY-1, iZ)) +
                                           i2hy * mesh.etyy(iY) * (V.Vy.F(iX, iY+1, iZ) - V.Vy.F(iX, iY-1, iZ)) +
                                           ihz2 * mesh.ztz2(iZ) * (V.Vy.F(iX, iY, iZ+1) + V.Vy.F(iX, iY, iZ-1)) +
                                           i2hz * mesh.ztzz(iZ) * (V.Vy.F(iX, iY, iZ+1) - V.Vy.F(iX, iY, iZ-1))) *
                            dt * nu / 2.0 + nseRHS.Vy(iX, iY, iZ)) /
                     (1.0 + dt * nu * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }

        V.Vy.F = tempVy;

        V.imposeVyBC();

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(tempVy)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVy(iX, iY, iZ) = V.Vy.F(iX, iY, iZ) - 0.5 * dt * nu * (
                              mesh.xix2(iX) * (V.Vy.F(iX+1, iY, iZ) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vy.F(iX+1, iY, iZ) - V.Vy.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vy.F(iX, iY+1, iZ) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vy.F(iX, iY+1, iZ) - V.Vy.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vy.F(iX, iY, iZ+1) - 2.0 * V.Vy.F(iX, iY, iZ) + V.Vy.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vy.F(iX, iY, iZ+1) - V.Vy.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }

        tempVy(core) = abs(tempVy(core) - nseRHS.Vy(core));

        locMax = blitz::max(tempVy(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of Vy not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to solve the implicit equation for z-velocity
 *
 *          The implicit equation for \f$ u_z' \f$ of the implicit Crank-Nicholson method is solved using the Jacobi
 *          iterative method here.
 *
 *          The loop exits when the global maximum of the error in computed solution falls below the specified tolerance.
 *          If the solution doesn't converge even after an internally assigned maximum number for iterations, the solver
 *          aborts with an error message.
 *
 ********************************************************************************************************************************************
 */
void eulerCN_d3::solveVz(vfield &V, plainvf &nseRHS) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempVz(V.Vz.F.lbound(), V.Vz.F.shape());

    while (true) {
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(nseRHS) shared(tempVz)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVz(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) +
                                           i2hx * mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) +
                                           ihy2 * mesh.ety2(iY) * (V.Vz.F(iX, iY+1, iZ) + V.Vz.F(iX, iY-1, iZ)) +
                                           i2hy * mesh.etyy(iY) * (V.Vz.F(iX, iY+1, iZ) - V.Vz.F(iX, iY-1, iZ)) +
                                           ihz2 * mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) + V.Vz.F(iX, iY, iZ-1)) +
                                           i2hz * mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1))) *
                            dt * nu / 2.0 + nseRHS.Vz(iX, iY, iZ)) /
                     (1.0 + dt * nu * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }

        V.Vz.F = tempVz;

        V.imposeVzBC();

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(V) shared(tempVz)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempVz(iX, iY, iZ) = V.Vz.F(iX, iY, iZ) - 0.5 * dt * nu * (
                              mesh.xix2(iX) * (V.Vz.F(iX+1, iY, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX-1, iY, iZ)) * ihx2 +
                              mesh.xixx(iX) * (V.Vz.F(iX+1, iY, iZ) - V.Vz.F(iX-1, iY, iZ)) * i2hx +
                              mesh.ety2(iY) * (V.Vz.F(iX, iY+1, iZ) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY-1, iZ)) * ihy2 +
                              mesh.etyy(iY) * (V.Vz.F(iX, iY+1, iZ) - V.Vz.F(iX, iY-1, iZ)) * i2hy +
                              mesh.ztz2(iZ) * (V.Vz.F(iX, iY, iZ+1) - 2.0 * V.Vz.F(iX, iY, iZ) + V.Vz.F(iX, iY, iZ-1)) * ihz2 +
                              mesh.ztzz(iZ) * (V.Vz.F(iX, iY, iZ+1) - V.Vz.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }

        tempVz(core) = abs(tempVz(core) - nseRHS.Vz(core));

        locMax = blitz::max(tempVz(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of Vz not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


void eulerCN_d3::solveT(sfield &T, plainsf &tmpRHS) {
    int iterCount = 0;
    real locMax = 0.0;
    real gloMax = 0.0;
    static blitz::Array<real, 3> tempT(T.F.F.lbound(), T.F.F.shape());

    while (true) {
#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tmpRHS) shared(tempT)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempT(iX, iY, iZ) = ((ihx2 * mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) + T.F.F(iX-1, iY, iZ)) +
                                          i2hx * mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) +
                                          ihy2 * mesh.ety2(iY) * (T.F.F(iX, iY+1, iZ) + T.F.F(iX, iY-1, iZ)) +
                                          i2hy * mesh.etyy(iY) * (T.F.F(iX, iY+1, iZ) - T.F.F(iX, iY-1, iZ)) +
                                          ihz2 * mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) + T.F.F(iX, iY, iZ-1)) +
                                          i2hz * mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1))) *
                        dt * kappa / 2.0 + tmpRHS.F(iX, iY, iZ)) /
                 (1.0 + dt * kappa * (ihx2 * mesh.xix2(iX) + ihy2 * mesh.ety2(iY) + ihz2 * mesh.ztz2(iZ)));
                }
            }
        }

        T.F.F = tempT;

        T.imposeBCs();

#pragma omp parallel for num_threads(mesh.inputParams.nThreads) default(none) shared(T) shared(tempT)
        for (int iX = xSt; iX <= xEn; iX++) {
            for (int iY = ySt; iY <= yEn; iY++) {
                for (int iZ = zSt; iZ <= zEn; iZ++) {
                    tempT(iX, iY, iZ) = T.F.F(iX, iY, iZ) - 0.5 * dt * kappa * (
                           mesh.xix2(iX) * (T.F.F(iX+1, iY, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX-1, iY, iZ)) * ihx2 +
                           mesh.xixx(iX) * (T.F.F(iX+1, iY, iZ) - T.F.F(iX-1, iY, iZ)) * i2hx +
                           mesh.ety2(iY) * (T.F.F(iX, iY+1, iZ) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY-1, iZ)) * ihy2 +
                           mesh.etyy(iY) * (T.F.F(iX, iY+1, iZ) - T.F.F(iX, iY-1, iZ)) * i2hy +
                           mesh.ztz2(iZ) * (T.F.F(iX, iY, iZ+1) - 2.0 * T.F.F(iX, iY, iZ) + T.F.F(iX, iY, iZ-1)) * ihz2 +
                           mesh.ztzz(iZ) * (T.F.F(iX, iY, iZ+1) - T.F.F(iX, iY, iZ-1)) * i2hz);
                }
            }
        }

        tempT(core) = abs(tempT(core) - tmpRHS.F(core));

        locMax = blitz::max(tempT(core));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (gloMax < mesh.inputParams.cnTolerance) break;

        iterCount += 1;

        if (iterCount > maxIterations) {
            if (mesh.rankData.rank == 0) {
                std::cout << "ERROR: Jacobi iterations for solution of T not converging. Aborting" << std::endl;
            }
            MPI_Finalize();
            exit(0);
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the coefficients used for solving the implicit equations of U, V and W
 *
 *          The function assigns values to the variables \ref hx, \ref hy, etc.
 *          These coefficients are repeatedly used at many places in the iterative solver for implicit calculation of velocities.
 ********************************************************************************************************************************************
 */
void eulerCN_d3::setCoefficients() {
    real hx2 = pow(mesh.dXi, 2.0);
    real hy2 = pow(mesh.dEt, 2.0);
    real hz2 = pow(mesh.dZt, 2.0);

    i2hx = 0.5/mesh.dXi;
    i2hy = 0.5/mesh.dEt;
    i2hz = 0.5/mesh.dZt;

    ihx2 = 1.0/hx2;
    ihy2 = 1.0/hy2;
    ihz2 = 1.0/hz2;
};
