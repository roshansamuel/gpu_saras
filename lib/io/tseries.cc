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
/*! \file tseries.cc
 *
 *  \brief Definitions for functions of class tseries
 *  \sa tseries.h
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <iostream>
#include "tseries.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the tseries class
 *
 *          The constructor initializes the variables and parameters for writing time-series data
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solverV is a reference to the velocity vector field whose data is used in calculating global quantities
 * \param   solverP is a const reference to the pressure scalar field whose dimensions are used to set array limits
 * \param   solverTime is a const reference to the real variable holding the value of current solver time
 * \param   timeStep is a const reference to the real variable holding the value of current time-step
 *
 ********************************************************************************************************************************************
 */
tseries::tseries(const grid &mesh, vfield &solverV, const real &solverTime, const real &timeStep):
                 time(solverTime), tStp(timeStep), mesh(mesh), V(solverV), divV(mesh)
{
    blitz::RectDomain<3> core = mesh.coreDomain;

    // Open TimeSeries file
    if (mesh.inputParams.restartFlag) {
        ofFile.open("output/TimeSeries.dat", std::fstream::out | std::fstream::app);
    } else {
        ofFile.open("output/TimeSeries.dat", std::fstream::out);
    }

    // UPPER AND LOWER LIMITS WHEN COMPUTING ENERGY IN STAGGERED GRID
    xLow = core.lbound(0);        xTop = core.ubound(0);
#ifndef PLANAR
    yLow = core.lbound(1);        yTop = core.ubound(1);
#endif
    zLow = core.lbound(2);        zTop = core.ubound(2);

    // TOTAL VOLUME FOR AVERAGING THE RESULT OF VOLUMETRIC INTEGRATION
    real localVol = 0.0;

    totalVol = 0.0;
#ifdef PLANAR
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iZ = zLow; iZ <= zTop; iZ++) {
            localVol += (mesh.dXi/mesh.xi_x(iX))*(mesh.dZt/mesh.zt_z(iZ));
        }
    }
#else
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iY = yLow; iY <= yTop; iY++) {
            for (int iZ = zLow; iZ <= zTop; iZ++) {
                localVol += (mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }
#endif
    MPI_Allreduce(&localVol, &totalVol, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

    // This switch decides if mean or maximum of divergence has to be printed.
    // Ideally maximum has to be tracked, but mean is a less strict metric.
    // By default, the mean is computed. To enable a stricter check, the below flag
    // must be turned on.
    maxSwitch = true;

    if (mesh.inputParams.lesModel) subgridEnergy = 0.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to write the the headers for time-series data.
 *
 *          The header for the time-series file, as well as for the I/O are written
 *          by this function.
 *          This was originally a part of the constructor, but was later shifted to
 *          a separate function so that the tseries object can be passed to other
 *          functions which need to write to the I/O before the header is printed.
 *          Only the root rank (rank 0) writes the output.
 *          One line is written to the standard I/O, while another is written to the time series dat file.
 *
 ********************************************************************************************************************************************
 */
void tseries::writeTSHeader() {
    // WRITE THE HEADERS FOR BOTH STANDARD I/O AS WELL AS THE OUTPUT TIME-SERIES FILE
    if (mesh.rankData.rank == 0) {
        if (mesh.inputParams.probType <= 4) {
            std::cout << std::setw(9)  << "Time" <<
                         std::setw(20) << "Total KE" <<
                         std::setw(20) << "Divergence" << std::endl;

            if (mesh.inputParams.lesModel) {
                ofFile << "#VARIABLES = Time, Total KE, U_rms, Divergence, Subgrid KE, dt\n";
            } else {
                ofFile << "#VARIABLES = Time, Total KE, U_rms, Divergence, dt\n";
            }
        } else {
            std::cout << std::setw(9)  << "Time" <<
                         std::setw(20) << "Re (Urms)" <<
                         std::setw(20) << "Nusselt No" <<
                         std::setw(20) << "Divergence" << std::endl;

            if (mesh.inputParams.lesModel) {
                ofFile << "#VARIABLES = Time, Reynolds No., Nusselt No., Total KE, Total TE, Divergence, Subgrid KE, dt\n";
            } else {
                ofFile << "#VARIABLES = Time, Reynolds No., Nusselt No., Total KE, Total TE, Divergence, dt\n";
            }
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded function to write the time-series data for hydro solver.
 *
 *          The function computes total energy and divergence for hydro solver runs.
 *          Only the root rank (rank 0) writes the output.
 *          One line is written to the standard I/O, while another is written to the time series dat file.
 *
 ********************************************************************************************************************************************
 */
void tseries::writeTSData() {
    V.divergence(divV);
    divValue = maxSwitch? divV.fxMaxAbs(): divV.fxMean();

    if (divValue > 1.0e5) {
        if (mesh.rankData.rank == 0) std::cout << std::endl << "ERROR: Divergence exceeds permissible limits. ABORTING" << std::endl << std::endl;
        MPI_Finalize();
        exit(0);
    }

    localKineticEnergy = 0.0;
    totalKineticEnergy = 0.0;
#ifdef PLANAR
    int iY = 0;
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iZ = zLow; iZ <= zTop; iZ++) {
            localKineticEnergy += (pow(V.Vx.F(iX, iY, iZ), 2.0) +
                                   pow(V.Vz.F(iX, iY, iZ), 2.0))*0.5*(mesh.dXi/mesh.xi_x(iX))*(mesh.dZt/mesh.zt_z(iZ));
        }
    }
#else
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iY = yLow; iY <= yTop; iY++) {
            for (int iZ = zLow; iZ <= zTop; iZ++) {
                localKineticEnergy += (pow(V.Vx.F(iX, iY, iZ), 2.0) + pow(V.Vy.F(iX, iY, iZ), 2.0) +
                                       pow(V.Vz.F(iX, iY, iZ), 2.0))*0.5*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }
#endif
    MPI_Allreduce(&localKineticEnergy, &totalKineticEnergy, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    totalKineticEnergy /= totalVol;
    if (mesh.inputParams.lesModel) subgridEnergy /= totalVol;

    if (mesh.rankData.rank == 0) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                   std::setprecision(8) << std::setw(20) << totalKineticEnergy <<
                                                           std::setw(20) << divValue << std::endl;

        if (mesh.inputParams.lesModel) {
            ofFile << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                    std::setprecision(8) << std::setw(20) << totalKineticEnergy <<
                                                            std::setw(20) << sqrt(2.0*totalKineticEnergy) <<
                                                            std::setw(20) << divValue <<
                                                            std::setw(20) << subgridEnergy <<
                                                            std::setw(20) << tStp << std::endl;
        } else {
            ofFile << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                    std::setprecision(8) << std::setw(20) << totalKineticEnergy <<
                                                            std::setw(20) << sqrt(2.0*totalKineticEnergy) <<
                                                            std::setw(20) << divValue <<
                                                            std::setw(20) << tStp << std::endl;
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded function to write the time-series data for scalar solver.
 *
 *          The function computes total energy, Nusselt number and divergence for scalar solver runs.
 *          Only the root rank (rank 0) writes the output.
 *          One line is written to the standard I/O, while another is written to the time series dat file.
 *
 * \param   T is a const reference to the temperature scalar field whose data is used to compute Nusselt number
 *
 ********************************************************************************************************************************************
 */
void tseries::writeTSData(const sfield &T) {
    real theta = 0.0;

    // COMPUTE ENERGY AND DIVERGENCE FOR THE INITIAL CONDITION
    V.divergence(divV);
    divValue = maxSwitch? divV.fxMaxAbs(): divV.fxMean();

    if (divValue > 1.0e5) {
        if (mesh.rankData.rank == 0) std::cout << std::endl << "ERROR: Divergence exceeds permissible limits. ABORTING" << std::endl << std::endl;
        MPI_Finalize();
        exit(0);
    }

    localKineticEnergy = 0.0;
    totalKineticEnergy = 0.0;

    localThermalEnergy = 0.0;
    totalThermalEnergy = 0.0;

    localUzT = 0.0;

#ifdef PLANAR
    int iY = 0;
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iZ = zLow; iZ <= zTop; iZ++) {
            // Check if the following value of theta is valid for all scalar runs
            theta = T.F.F(iX, iY, iZ) + mesh.z(iZ) - 1.0;

            localKineticEnergy += (pow(V.Vx.F(iX, iY, iZ), 2.0) + pow(V.Vz.F(iX, iY, iZ), 2.0))*0.5*
                                    (mesh.dXi/mesh.xi_x(iX))*(mesh.dZt/mesh.zt_z(iZ));

            localThermalEnergy += (pow(theta, 2.0))*0.5*(mesh.dXi/mesh.xi_x(iX))*(mesh.dZt/mesh.zt_z(iZ));

            localUzT += V.Vz.F(iX, iY, iZ)*T.F.F(iX, iY, iZ)*(mesh.dXi/mesh.xi_x(iX))*(mesh.dZt/mesh.zt_z(iZ));
        }
    }
#else
    for (int iX = xLow; iX <= xTop; iX++) {
        for (int iY = yLow; iY <= yTop; iY++) {
            for (int iZ = zLow; iZ <= zTop; iZ++) {
                // Check if the following value of theta is valid for all scalar runs
                theta = T.F.F(iX, iY, iZ) + mesh.z(iZ) - 1.0;

                localKineticEnergy += (pow(V.Vx.F(iX, iY, iZ), 2.0) + pow(V.Vy.F(iX, iY, iZ), 2.0) + pow(V.Vz.F(iX, iY, iZ), 2.0))*0.5*
                                    (mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));

                localThermalEnergy += (pow(theta, 2.0))*0.5*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));

                localUzT += V.Vz.F(iX, iY, iZ)*T.F.F(iX, iY, iZ)*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }
#endif

    MPI_Allreduce(&localKineticEnergy, &totalKineticEnergy, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localThermalEnergy, &totalThermalEnergy, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localUzT, &totalUzT, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
    totalKineticEnergy /= totalVol;
    totalThermalEnergy /= totalVol;
    NusseltNo = 1.0 + (totalUzT/totalVol)/tDiff;
    ReynoldsNo = sqrt(2.0*totalKineticEnergy)/mDiff;
    if (mesh.inputParams.lesModel) subgridEnergy /= totalVol;

    if (mesh.rankData.rank == 0) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                   std::setprecision(8) << std::setw(20) << ReynoldsNo <<
                                                           std::setw(20) << NusseltNo <<
                                                           std::setw(20) << divValue << std::endl;

        if (mesh.inputParams.lesModel) {
            ofFile << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                    std::setprecision(8) << std::setw(20) << ReynoldsNo <<
                                                            std::setw(20) << NusseltNo <<
                                                            std::setw(20) << totalKineticEnergy <<
                                                            std::setw(20) << totalThermalEnergy <<
                                                            std::setw(20) << divValue <<
                                                            std::setw(20) << subgridEnergy <<
                                                            std::setw(20) << tStp << std::endl;
        } else {
            ofFile << std::fixed << std::setprecision(4) << std::setw(9)  << time <<
                                    std::setprecision(8) << std::setw(20) << ReynoldsNo <<
                                                            std::setw(20) << NusseltNo <<
                                                            std::setw(20) << totalKineticEnergy <<
                                                            std::setw(20) << totalThermalEnergy <<
                                                            std::setw(20) << divValue <<
                                                            std::setw(20) << tStp << std::endl;
        }
    }
}


tseries::~tseries() {
    ofFile.close();
}
