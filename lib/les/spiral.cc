/********************************************************************************************************************************************
 * Spiral LES
 * 
 * Copyright (C) 2009, California Institute of Technology (Caltech)
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
 * THIS SOFTWARE IS PROVIDED BY Caltech "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Caltech BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************************************************************************
 */
/*! \file spiral.cc
 *
 *  \brief Definitions for functions of class spiral
 *  \sa les.h
 *  \date Sep 2020
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "les.h"

// The unusually high value of EPS defined below is taken from the KAUST LES code.
// The original spiral solver used a value of 2e-15, which is too low.
#define EPS (2e-3)

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the spiral class
 *
 *          The constructor initializes the base les class using part of the arguments supplied to it.
 *          The value of viscous diffusion coefficient, denoted by nu, is also set in the initialization list.
 *          All the blitz arrays and scalar fields necessary to compute velocity gradient tensor, stress tensor, etc.
 *          are initialized in the constructor.
 *
 * \param   mesh is a const reference to the global data contained in the grid class.
 * \param   kDiff is a const reference to scalar value denoting viscous dissipation
 ********************************************************************************************************************************************
 */
spiral::spiral(const grid &mesh, const real &kDiff): les(mesh), nu(kDiff) {
    blitz::TinyVector<int, 3> dSize = mesh.fullDomain.ubound() - mesh.fullDomain.lbound() + 1;
    blitz::TinyVector<int, 3> dlBnd = mesh.fullDomain.lbound();

    // Scalar fields used to store components of the sub-grid stress tensor field
    Txx = new sfield(mesh, "Txx");
    Tyy = new sfield(mesh, "Tyy");
    Tzz = new sfield(mesh, "Tzz");
    Txy = new sfield(mesh, "Txy");
    Tyz = new sfield(mesh, "Tyz");
    Tzx = new sfield(mesh, "Tzx");

    // Scalar fields used to store components of the sub-grid scalar flux vector
    qX = new sfield(mesh, "qX");
    qY = new sfield(mesh, "qY");
    qZ = new sfield(mesh, "qZ");

    // 3x3x3 arrays which store local velocity field when computing structure function
    u.resize(3, 3, 3);
    v.resize(3, 3, 3);
    w.resize(3, 3, 3);

    // 3x3 matrix to store the velocity gradient tensor
    dudx.resize(3, 3);

    // The 9 blitz arrays of tensor components have the same dimensions and limits as the cell centered variable
    A11.resize(dSize);      A11.reindexSelf(dlBnd);
    A12.resize(dSize);      A12.reindexSelf(dlBnd);
    A13.resize(dSize);      A13.reindexSelf(dlBnd);
    A21.resize(dSize);      A21.reindexSelf(dlBnd);
    A22.resize(dSize);      A22.reindexSelf(dlBnd);
    A23.resize(dSize);      A23.reindexSelf(dlBnd);
    A31.resize(dSize);      A31.reindexSelf(dlBnd);
    A32.resize(dSize);      A32.reindexSelf(dlBnd);
    A33.resize(dSize);      A33.reindexSelf(dlBnd);

    // The 9 arrays of vector components have the same dimensions and limits as the cell centered variable.
    // These arrays are needed only when computing the sub-grid scalar flux.
    // Hence they are initialized only when the scalar flux model must also be included in the LES.
    if (mesh.inputParams.lesModel == 2) {
        B1.resize(dSize);       B1.reindexSelf(dlBnd);
        B2.resize(dSize);       B2.reindexSelf(dlBnd);
        B3.resize(dSize);       B3.reindexSelf(dlBnd);

        // If true, compute sub-grid thermal diffusion using the sub-grid scalar flux function.
        // If false, use sub-grid momentum diffusion and Prandtl number to compute sub-grid thermal diffusion.
        sgfFlag = true;
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute and add contribution from LES model to the RHS of NSE
 *
 *          This function is called when only the hydrodynamics equations are being solved.
 *          It calls the sgsStress function repeatedly for every point in the domain to get the
 *          sub-grid stress tensor field.
 *          The divergence of this tensor field is then calculated to obtain the spiral vortex model's
 *          contribution to the NSE.
 *
 * \param   nseRHS is a reference to the plain vector field denoting the RHS of the NSE
 * \param   V is a reference the vector field denoting the velocity field
 ********************************************************************************************************************************************
 */
real spiral::computeSG(plainvf &nseRHS, vfield &V) {
    real localSGKE, totalSGKE;

    V.syncData();

    // Compute the x, y and z derivatives of the interpolated velocity field and store them into
    // the arrays A11, A12, A13, ... A33. These arrays will be later accessed when constructing
    // the velocity gradient tensor at each point in the domain.
    V.derVx.calcDerivative1_x(A11);
    V.derVx.calcDerivative1_y(A12);
    V.derVx.calcDerivative1_z(A13);
    V.derVy.calcDerivative1_x(A21);
    V.derVy.calcDerivative1_y(A22);
    V.derVy.calcDerivative1_z(A23);
    V.derVz.calcDerivative1_x(A31);
    V.derVz.calcDerivative1_y(A32);
    V.derVz.calcDerivative1_z(A33);

    // Set the array limits when looping over the domain to compute SG contribution.
    // Since correct U, V, and W data is available only in the core,
    // the limits of core are used so that the boundary points are excluded
    // while computing derivatives and structure functions.
    xS = core.lbound(0);       xE = core.ubound(0);
    yS = core.lbound(1);       yE = core.ubound(1);
    zS = core.lbound(2);       zE = core.ubound(2);

    localSGKE = 0.0;
    for (int iX = xS; iX <= xE; iX++) {
        real dx = mesh.x(iX) - mesh.x(iX - 1);
        for (int iY = yS; iY <= yE; iY++) {
            real dy = mesh.y(iY) - mesh.y(iY - 1);
            for (int iZ = zS; iZ <= zE; iZ++) {
                real dz = mesh.z(iZ) - mesh.z(iZ - 1);

                // Specify the values of all the quantities necessary for sgsFlux function to
                // compute the sub-grid stress correctly. These are all member variables of the les
                // class and hence globally available to all the member functions of the class.

                // The required quantities are:
                // 1. Cutoff wavelength
                del = std::pow(dx*dy*dz, 1.0/3.0);

                // 2. Velocities at the 3 x 3 x 3 points over which structure function will be calculated
                u = V.Vx.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                v = V.Vy.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                w = V.Vz.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));

                // 3. The x, y and z coordinates of the 3 x 3 x 3 points over which u, v and w have been specified
                x = mesh.x(blitz::Range(iX-1, iX+1));
                y = mesh.y(blitz::Range(iY-1, iY+1));
                z = mesh.z(blitz::Range(iZ-1, iZ+1));

                // 4. The velocity gradient tensor specified as a 3 x 3 matrix
                dudx = A11(iX, iY, iZ), A12(iX, iY, iZ), A13(iX, iY, iZ),
                       A21(iX, iY, iZ), A22(iX, iY, iZ), A23(iX, iY, iZ),
                       A31(iX, iY, iZ), A32(iX, iY, iZ), A33(iX, iY, iZ);

                // Now the sub-grid stress can be calculated
                sgsStress(&sTxx, &sTyy, &sTzz, &sTxy, &sTyz, &sTzx);

                // Copy the calculated values to the sub-grid stress tensor field
                Txx->F.F(iX, iY, iZ) = sTxx;
                Tyy->F.F(iX, iY, iZ) = sTyy;
                Tzz->F.F(iX, iY, iZ) = sTzz;
                Txy->F.F(iX, iY, iZ) = sTxy;
                Tyz->F.F(iX, iY, iZ) = sTyz;
                Tzx->F.F(iX, iY, iZ) = sTzx;

                localSGKE += abs(K)*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }

    MPI_Allreduce(&localSGKE, &totalSGKE, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

    // Synchronize the sub-grid stress tensor field data across MPI processors
    Txx->syncData();
    Tyy->syncData();
    Tzz->syncData();
    Txy->syncData();
    Tyz->syncData();
    Tzx->syncData();

    A11 = 0.0;  A12 = 0.0;  A13 = 0.0;
    A21 = 0.0;  A22 = 0.0;  A23 = 0.0;
    A31 = 0.0;  A32 = 0.0;  A33 = 0.0;

    // Compute the components of the divergence of sub-grid stress tensor field
    Txx->derS.calcDerivative1_x(A11);
    Txy->derS.calcDerivative1_x(A12);
    Tzx->derS.calcDerivative1_x(A13);
    Txy->derS.calcDerivative1_y(A21);
    Tyy->derS.calcDerivative1_y(A22);
    Tyz->derS.calcDerivative1_y(A23);
    Tzx->derS.calcDerivative1_z(A31);
    Tyz->derS.calcDerivative1_z(A32);
    Tzz->derS.calcDerivative1_z(A33);

    // Add the divergence of the stress tensor field to the RHS of NSE provided as argument to the function
    nseRHS.Vx(core) = nseRHS.Vx(core) + A11(core) + A21(core) + A31(core);
    nseRHS.Vy(core) = nseRHS.Vy(core) + A12(core) + A22(core) + A32(core);
    nseRHS.Vz(core) = nseRHS.Vz(core) + A13(core) + A23(core) + A33(core);

    return totalSGKE;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute and add contribution from LES model to the RHS of NSE and temperature equation
 *
 *          This function is called when both hydrodynamics and scalar equations are being solved.
 *          It calles the sgsStress and sgsFlux functions repeatedly for every point in the domain to get the
 *          sub-grid stress tensor field sub-grid scalar flux vector field repsectively.
 *          The divergences of these fields are then calculated to obtain the spiral vortex model contribution
 *          to the NSE and scalar equations
 *
 * \param   nseRHS is a reference to the plain vector field denoting the RHS of the NSE
 * \param   tmpRHS is a reference to the plain scalar field denoting the RHS of the scalar equation
 * \param   V is a reference the vector field denoting the velocity field
 * \param   T is a reference the scalar field denoting the scalar equation
 ********************************************************************************************************************************************
 */
real spiral::computeSG(plainvf &nseRHS, plainsf &tmpRHS, vfield &V, sfield &T) {
    real localSGKE, totalSGKE;
    real sQx, sQy, sQz;

    V.syncData();

    // Compute the x, y and z derivatives of the interpolated velocity field and store them into
    // the arrays A11, A12, A13, ... A33. These arrays will be later accessed when constructing
    // the velocity gradient tensor at each point in the domain.
    V.derVx.calcDerivative1_x(A11);
    V.derVx.calcDerivative1_y(A12);
    V.derVx.calcDerivative1_z(A13);
    V.derVy.calcDerivative1_x(A21);
    V.derVy.calcDerivative1_y(A22);
    V.derVy.calcDerivative1_z(A23);
    V.derVz.calcDerivative1_x(A31);
    V.derVz.calcDerivative1_y(A32);
    V.derVz.calcDerivative1_z(A33);

    // Compute the x, y and z derivatives of the temperature field and store them into
    // the arrays B1, B2, and B3. These arrays will be later accessed when constructing
    // the temperature gradient tensor at each point in the domain.
    if (sgfFlag) {
        T.derS.calcDerivative1_x(B1);
        T.derS.calcDerivative1_y(B2);
        T.derS.calcDerivative1_z(B3);
    }

    // Set the array limits when looping over the domain to compute SG contribution.
    // Since correct U, V, and W data is available only in the core,
    // the limits of core are used so that the boundary points are excluded
    // to compute derivatives and structure functions correctly.
    xS = core.lbound(0);       xE = core.ubound(0);
    yS = core.lbound(1);       yE = core.ubound(1);
    zS = core.lbound(2);       zE = core.ubound(2);

    localSGKE = 0.0;
    for (int iX = xS; iX <= xE; iX++) {
        real dx = mesh.x(iX) - mesh.x(iX - 1);
        for (int iY = yS; iY <= yE; iY++) {
            real dy = mesh.y(iY) - mesh.y(iY - 1);
            for (int iZ = zS; iZ <= zE; iZ++) {
                real dz = mesh.z(iZ) - mesh.z(iZ - 1);

                // Specify the values of all the quantities necessary for sgsFlux function to
                // compute the sub-grid stress correctly. These are all member variables of the les
                // class and hence globally available to all the member functions of the class.

                // The required quantities are:
                // 1. Cutoff wavelength
                del = std::pow(dx*dy*dz, 1.0/3.0);

                // 2. Velocities at the 3 x 3 x 3 points over which structure function will be calculated
                u = V.Vx.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                v = V.Vy.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));
                w = V.Vz.F(blitz::Range(iX-1, iX+1), blitz::Range(iY-1, iY+1), blitz::Range(iZ-1, iZ+1));

                // 3. The x, y and z coordinates of the 3 x 3 x 3 points over which u, v and w have been specified
                x = mesh.x(blitz::Range(iX-1, iX+1));
                y = mesh.y(blitz::Range(iY-1, iY+1));
                z = mesh.z(blitz::Range(iZ-1, iZ+1));

                // 4. The velocity gradient tensor specified as a 3 x 3 matrix
                dudx = A11(iX, iY, iZ), A12(iX, iY, iZ), A13(iX, iY, iZ),
                       A21(iX, iY, iZ), A22(iX, iY, iZ), A23(iX, iY, iZ),
                       A31(iX, iY, iZ), A32(iX, iY, iZ), A33(iX, iY, iZ);

                // Now the sub-grid stress can be calculated
                sgsStress(&sTxx, &sTyy, &sTzz, &sTxy, &sTyz, &sTzx);

                // Copy the calculated values to the sub-grid stress tensor field
                Txx->F.F(iX, iY, iZ) = sTxx;
                Tyy->F.F(iX, iY, iZ) = sTyy;
                Tzz->F.F(iX, iY, iZ) = sTzz;
                Txy->F.F(iX, iY, iZ) = sTxy;
                Tyz->F.F(iX, iY, iZ) = sTyz;
                Tzx->F.F(iX, iY, iZ) = sTzx;

                if (sgfFlag) {
                    // To compute sub-grid scalar flux, the sgsStress calculations have already provided
                    // most of the necessary values. Only an additional temperature gradient vector is needed
                    // 5. The temperature gradient vector specified as a 3 component tiny vector
                    dsdx = B1(iX, iY, iZ), B2(iX, iY, iZ), B3(iX, iY, iZ);

                    // Now the sub-grid scalar flus can be calculated
                    sgsFlux(&sQx, &sQy, &sQz);

                    // Copy the calculated values to the sub-grid scalar flux vector field
                    qX->F.F(iX, iY, iZ) = sQx;
                    qY->F.F(iX, iY, iZ) = sQy;
                    qZ->F.F(iX, iY, iZ) = sQz;
                }

                localSGKE += abs(K)*(mesh.dXi/mesh.xi_x(iX))*(mesh.dEt/mesh.et_y(iY))*(mesh.dZt/mesh.zt_z(iZ));
            }
        }
    }

    MPI_Allreduce(&localSGKE, &totalSGKE, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

    // Synchronize the sub-grid stress tensor field data across MPI processors
    Txx->syncData();
    Tyy->syncData();
    Tzz->syncData();
    Txy->syncData();
    Tyz->syncData();
    Tzx->syncData();

    // Compute the components of the divergence of sub-grid stress tensor field
    Txx->derS.calcDerivative1_x(A11);
    Txy->derS.calcDerivative1_x(A12);
    Tzx->derS.calcDerivative1_x(A13);
    Txy->derS.calcDerivative1_y(A21);
    Tyy->derS.calcDerivative1_y(A22);
    Tyz->derS.calcDerivative1_y(A23);
    Tzx->derS.calcDerivative1_z(A31);
    Tyz->derS.calcDerivative1_z(A32);
    Tzz->derS.calcDerivative1_z(A33);

    // Compute the divergence of the sub-grid stress tensor field
    B1 = A11 + A21 + A31;
    B2 = A12 + A22 + A32;
    B3 = A13 + A23 + A33;

    // Add the divergence to the RHS of NSE provided as argument to the function
    nseRHS.Vx(core) = nseRHS.Vx(core) + B1(core);
    nseRHS.Vy(core) = nseRHS.Vy(core) + B2(core);
    nseRHS.Vz(core) = nseRHS.Vz(core) + B3(core);

    if (sgfFlag) {
        // Synchronize the sub-grid scalar flux vector field data across MPI processors
        qX->syncData();
        qY->syncData();
        qZ->syncData();

        // Compute the components of the divergence of sub-grid scalar flux vector field
        qX->derS.calcDerivative1_x(B1);
        qY->derS.calcDerivative1_y(B2);
        qZ->derS.calcDerivative1_z(B3);

        // Sum the components to get the divergence of scalar flux, and add its contribution
        // to the RHS of the temperature field equation provided as argument to the function
        tmpRHS.F(core) = tmpRHS.F(core) + B1(core) + B2(core) + B3(core);

    } else {
        V.derVx.calcDerivative2xx(A11);
        V.derVx.calcDerivative2yy(A12);
        V.derVx.calcDerivative2zz(A13);
        V.derVy.calcDerivative2xx(A21);
        V.derVy.calcDerivative2yy(A22);
        V.derVy.calcDerivative2zz(A23);
        V.derVz.calcDerivative2xx(A31);
        V.derVz.calcDerivative2yy(A32);
        V.derVz.calcDerivative2zz(A33);

        A11 = A11 + A12 + A13;
        A22 = A21 + A22 + A23;
        A33 = A31 + A32 + A33;

        B1 /= A11;
        B2 /= A22;
        B3 /= A33;

        B1 = (B1 + B2 + B3)/(mesh.inputParams.Pr*3.0);

        T.derS.calcDerivative2xx(A11);
        T.derS.calcDerivative2yy(A22);
        T.derS.calcDerivative2zz(A33);

        tmpRHS.F(core) = tmpRHS.F(core) + B1(core)*(A11(core) + A22(core) + A33(core));
    }

    return totalSGKE;
}


/**
 ********************************************************************************************************************************************
 * \brief   Main function to calculate the sub-grid stress tensor using stretched vortex model
 *
 *          The six components of the subgrid stress tensor - Txx, Tyy, Tzz, Txy, Tyz, Tzx are calculated at x[0], y[0], z[0].
 *          It needs the resolved velocity gradient tensor dudx[3][3], LES cutoff scale del, and kinematic viscosity nu.
 *          It first computes the alignment of the subgrid vortex, e, by calculating the eigenvectors of S_ij.
 *          To compute the structure function, it needs 3x3x3 samples of the local resolved velocity field,
 *          (u[0,0,0], v[0,0,0], w[0,0,0]) at (x[0], y[0], z[0]) to (u[2,2,2], v[2,2,2], w[2,2,2]) at (x[2], y[2], z[2]).
 *          Finally \mathcal{K}_0 \epsilon^{2/3} k_c^{-2/3}, where a = e_i^v e_j^v S_{ij} is the axial stretching.
 *
 ********************************************************************************************************************************************
 */
void spiral::sgsStress(
    real *Txx, real *Tyy, real *Tzz,
    real *Txy, real *Tyz, real *Tzx)
{
    // lv = Sqrt[2 nu / (3 Abs[a])]
    real lv = 0.0;

    {
        // Strain-rate tensor
        Sxx = 0.5 * (dudx(0, 0) + dudx(0, 0));
        Syy = 0.5 * (dudx(1, 1) + dudx(1, 1));
        Szz = 0.5 * (dudx(2, 2) + dudx(2, 2));
        Sxy = 0.5 * (dudx(0, 1) + dudx(1, 0));
        Syz = 0.5 * (dudx(1, 2) + dudx(2, 1));
        Szx = 0.5 * (dudx(2, 0) + dudx(0, 2));

        // By default, eigenvalue corresponding to most extensive eigenvector is returned
        real eigval = eigenvalueSymm();

        // Default alignment: most extensive eigenvector
        e = eigenvectorSymm(eigval);

        // Make e[3] a unit vector
        real length = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        e /= length;

        // Strain along vortex axis
        real a = e[0] * e[0] * Sxx + e[0] * e[1] * Sxy + e[0] * e[2] * Szx
               + e[1] * e[0] * Sxy + e[1] * e[1] * Syy + e[1] * e[2] * Syz
               + e[2] * e[0] * Szx + e[2] * e[1] * Syz + e[2] * e[2] * Szz;
        lv = sqrt(2.0 * nu / (3.0 * (fabs(a) + EPS)));
    }

    // Structure function calculation
    real F2 = 0.0;
    real Qd = 0.0; 
    {    
        // Average over neighboring points
        int sfCount = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    if (i or j or k) {
                        real du = u(i+1, j+1, k+1) - u(1, 1, 1);
                        real dv = v(i+1, j+1, k+1) - v(1, 1, 1);
                        real dw = w(i+1, j+1, k+1) - w(1, 1, 1);
                        F2 += du * du + dv * dv + dw * dw;
                        real dx = x[i+1] - x[1];
                        real dy = y[j+1] - y[1];
                        real dz = z[k+1] - z[1];
                        real dx2 = dx * dx   + dy * dy   + dz * dz;
                        real dxe = dx * e[0] + dy * e[1] + dz * e[2];
                        real d = sqrt(dx2 - dxe * dxe) / del;
                        Qd += sfIntegral(d);
                        sfCount++;
                    }
                }
            }
        }
        F2 /= (real) (sfCount);
        Qd /= (real) (sfCount);
    }
    // prefac is the group prefactor
    real prefac = F2 / Qd; // \mathcal{K}_0 \epsilon^{2/3} k_c^{-2/3}
    real kc = M_PI / del;

    K = prefac * keIntegral(kc * lv);

    // T_{ij} = (\delta_{ij} - e_i^v e_j^v) K
    *Txx = (1.0 - e[0] * e[0]) * K;
    *Tyy = (1.0 - e[1] * e[1]) * K;
    *Tzz = (1.0 - e[2] * e[2]) * K;
    *Txy = (    - e[0] * e[1]) * K;
    *Tyz = (    - e[1] * e[2]) * K;
    *Tzx = (    - e[2] * e[0]) * K;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the sub-grid scalar flux vector using stretched vortex model
 *
 *          The three components of the subgrid scalar flux vector - qx, qy, qz are calculated at x[0], y[0], z[0].
 *          It needs the resolved scalar gradient tensor dsdx[3], sub-grid vortex alignment e[3] (a unit vector),
 *          LES cutoff scale del, and the precalculated SGS kinetic energy K.
 *          WARNING: For this function to work, the values of member variables e and K must be pre-calculated through
 *          a call to the sgsStress function.
 *
 ********************************************************************************************************************************************
 */
void spiral::sgsFlux(real *qx, real *qy, real *qz) {
    real gam = 1.0; // Universal model constant
    real P = -0.5 * gam * del * sqrt(K);

    // q_i = P (\delta_{ij} - e_i^v e_j^v) ds/dx_j
    *qx = P * ((1.0 - e[0] * e[0]) * dsdx[0]
             + (    - e[0] * e[1]) * dsdx[1]
             + (    - e[0] * e[2]) * dsdx[2]);
    *qy = P * ((    - e[1] * e[0]) * dsdx[0]
             + (1.0 - e[1] * e[1]) * dsdx[1]
             + (    - e[1] * e[2]) * dsdx[2]);
    *qz = P * ((    - e[2] * e[0]) * dsdx[0]
             + (    - e[2] * e[1]) * dsdx[1]
             + (1.0 - e[2] * e[2]) * dsdx[2]);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to approximately evaluate the integral for sub-grid energy
 *
 *          The sub-grid energy is calculated as an approximation of (1/2) k^(2/3) Gamma[-1/3, k^2]
 *          with maximum relative error of 0.17% at k=2.42806.
 *          The only input for the function is the non-dimensionalized cut-off wavenumber k.
 *          It returns the sub-grid energy as a real valued number.
 *
 ********************************************************************************************************************************************
 */
real spiral::keIntegral(real k) {
    real k2 = k * k;
    if (k2 < 2.42806) {
        real pade = (3.0 +   2.5107 * k2 +  0.330357 * k2 * k2
                    +  0.0295481 * k2 * k2 * k2)
                    / (1.0 + 0.336901 * k2 + 0.0416684 * k2 * k2
                    + 0.00187191 * k2 * k2 * k2);
        return 0.5 * (pade - 4.06235 * pow(k2, 1.0 / 3.0));
    }
    else {
        real pade = (1.26429 + 0.835714 * k2 + 0.0964286 * k2 * k2)
                    / (1.0     +   2.25   * k2 +  0.964286 * k2 * k2
                    + 0.0964286 * k2 * k2 * k2);
        return 0.5 * pade * exp(-k2);
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to approximately evaluate the structure function integral
 *
 *          The structure function integral is calculated as an approximation of
 *          Integrate[4 x^(-5/3) (1 - BesselJ[0, x Pi d]), {x, 0, 1}]
 *          with maximum relative error of 2.71% at d=0.873469.
 *          It returns the structure function integral as a real valued number.
 *
 ********************************************************************************************************************************************
 */
real spiral::sfIntegral(real d) {
    // Uncomment if spherical averaging and d=1.
    // if (d == 1.0) return 4.09047;

    real d2 = d * d;
    if (d < 0.873469)
        return 7.4022 * d2 - 1.82642 * d2 * d2;
    else
        return 12.2946 * pow(d, 2.0 / 3.0) - 6.0
            - 0.573159 * pow(d, -1.5) * sin(3.14159 * d - 0.785398);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the eigenvalues of the strain-rate tensor
 *
 *          The function claculates the eigenvalues, eigval[0] < eigval[1] < eigval[2],
 *          of the 3 x 3 symmetric matrix, { { Sxx, Sxy, Szx }, { Sxy, Syy, Syz }, { Szx, Syz, Szz } },
 *          assuming distinct eigenvalues.
 *          It returns the eigenvalue corresponding to the most extensive eigenvector.
 *
 ********************************************************************************************************************************************
 */
real spiral::eigenvalueSymm() {
    real eigval[3];

    // x^3 + a * x^2 + b * x + c = 0, where x is the eigenvalue
    real a = - (Sxx + Syy + Szz);
    real b = Sxx * Syy - Sxy * Sxy + Syy * Szz
             - Syz * Syz + Szz * Sxx - Szx * Szx;
    real c = - (Sxx * (Syy * Szz - Syz * Syz)
                + Sxy * (Syz * Szx - Sxy * Szz)
                + Szx * (Sxy * Syz - Syy * Szx));

    real q = (3.0 * b - a * a) / 9.0;
    real r = (9.0 * a * b - 27.0 * c - 2.0 * a * a * a) / 54.0;

    if (q >= 0.0) {
        if (mesh.rankData.rank == 0) {
            std::cout << "The value of q is greater than or equal to 0 in Spiral Eigenvalue calculation. Aborting" << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    real costheta = r / sqrt(-q * q * q);

    // |costheta| > 1 should not occur, except from round-off errors
    real theta;
    theta = costheta > 1.0 ? 0.0 :
            costheta < -1.0 ? M_PI :
            acos(costheta);

    eigval[0] = 2.0 * sqrt(-q) * cos((theta             ) / 3.0) - a / 3.0;
    eigval[1] = 2.0 * sqrt(-q) * cos((theta + 2.0 * M_PI) / 3.0) - a / 3.0;
    eigval[2] = 2.0 * sqrt(-q) * cos((theta + 4.0 * M_PI) / 3.0) - a / 3.0;

    // Sort eigenvalues: eigval[0] < eigval[1] < eigval[2]
    if (eigval[0] > eigval[1]) {
        real tmp = eigval[0]; eigval[0] = eigval[1]; eigval[1] = tmp;
    }
    if (eigval[1] > eigval[2]) {
        real tmp = eigval[1]; eigval[1] = eigval[2]; eigval[2] = tmp;
    }
    if (eigval[0] > eigval[1]) {
        real tmp = eigval[0]; eigval[0] = eigval[1]; eigval[1] = tmp;
    }

    return eigval[2];
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to calculate the eigenvectors of the strain-rate tensor
 *
 *          The function claculates the eigenvector (not normalized), eigvec[3],
 *          corresponding to the precalculated eigenvalue, eigval, of the 3 x 3 symmetric matrix,
 *          { { Sxx, Sxy, Szx }, { Sxy, Syy, Syz }, { Szx, Syz, Szz } }, assuming distinct eigenvalues.
 *          It returns the eigenvector corresponding to the eigenvalue supplied, as a blitz TinyVector.
 *
 ********************************************************************************************************************************************
 */
blitz::TinyVector<real, 3> spiral::eigenvectorSymm(real eigval) {
    blitz::TinyVector<real, 3> eigvec;

    // Frobenius norm for normalization
    real fNorm = std::sqrt(Sxx*Sxx + Syy*Syy + Szz*Szz +
                           Sxy*Sxy + Syz*Syz + Szx*Szx);

    // Check if the given value is indeed an eigenvalue of the matrix
    if (fabs((Sxx - eigval) * ((Syy - eigval) * (Szz - eigval) - Syz * Syz)
            + Sxy * (Syz * Szx - Sxy * (Szz - eigval))
            + Szx * (Sxy * Syz - (Syy - eigval) * Szx))/fabs(fNorm) > EPS) {
        if (mesh.rankData.rank == 0) {
            std::cout << "Invalid eigenvalue in Spiral Eigenvector calculation. Aborting" << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    real det[3] = { (Syy - eigval) * (Szz - eigval) - Syz * Syz,
                    (Szz - eigval) * (Sxx - eigval) - Szx * Szx,
                    (Sxx - eigval) * (Syy - eigval) - Sxy * Sxy };

    real fabsdet[3] = { fabs(det[0]), fabs(det[1]), fabs(det[2]) };

    if (fabsdet[0] >= fabsdet[1] && fabsdet[0] >= fabsdet[2]) {
        eigvec = 1.0, (-Sxy*(Szz - eigval) + Szx*Syz)/det[0], (-Szx*(Syy - eigval) + Sxy*Syz)/det[0];
    }
    else if (fabsdet[1] >= fabsdet[2] && fabsdet[1] >= fabsdet[0]) {
        eigvec = (-Sxy*(Szz - eigval) + Syz*Szx)/det[1], 1.0, (-Syz*(Sxx - eigval) + Sxy*Szx)/det[1];
    }
    else if (fabsdet[2] >= fabsdet[0] && fabsdet[2] >= fabsdet[1]) {
        eigvec = (-Szx*(Syy - eigval) + Syz*Sxy)/det[2], (-Syz*(Sxx - eigval) + Szx*Sxy)/det[2], 1.0;
    }
    else {
        if (mesh.rankData.rank == 0) {
            std::cout << "Eigenvalues are not distinct in Spiral Eigenvector calculation. Aborting" << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    return eigvec;
}
