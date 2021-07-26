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
/*! \file poisson3.cc
 *
 *  \brief Definitions for functions of class poisson for 3D
 *  \sa poisson.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "poisson.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the multigrid_d3 class derived from the poisson class
 *
 *          The constructor of the derived multigrid_d3 class frst calls the base poisson class with the arguments passed to it.
 *          Most of the array initializations are performed in the constructor of the base class.
 *          Initializations unique to 3D implementation of multi-grid solver are then done here.
 *          Primarily this involves generating the MPI datatypes for data transfer between sub-domain boundaries.
 *          When testing the Poisson solver with Dirichlet BC, the analytical 3D solution for the test is also calculated here
 *          through a call to initDirichlet() function.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
multigrid_d3::multigrid_d3(const grid &mesh, const parser &solParam): poisson(mesh, solParam) {
    // CREATE THE MPI SUB-ARRAYS NECESSARY TO TRANSFER DATA ACROSS SUB-DOMAINS AT ALL MESH LEVELS
    createMGSubArrays();

    // INITIALIZE DIRICHLET BCs WHEN TESTING THE POISSON SOLVER
#ifdef TEST_POISSON
    initDirichlet();
#endif
}


void multigrid_d3::computeResidual() {
    tmp(vLevel) = 0.0;

    // Compute Laplacian of the pressure field and subtract it from the RHS of Poisson equation to obtain the residual
    // This residual is temporarily stored into tmp, from which it will be coarsened into rhs array.
    // Needed update: Substitute the below OpenMP parallel loop with vectorized Blitz operation and check for speed increase.
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
    for (int i = 0; i <= xEnd(vLevel); ++i) {
        for (int j = 0; j <= yEnd(vLevel); ++j) {
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                tmp(vLevel)(i, j, k) =  rhs(vLevel)(i, j, k) -
                             (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i - 1, j, k)) +
                              xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                              ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j - 1, k)) +
                              etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                              ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j, k - 1)) +
                              ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)));
            }
        }
    }

    updatePads(tmp);
}


void multigrid_d3::smooth(const int smoothCount) {
    tmp(vLevel) = 0.0;

    for(int n=0; n<smoothCount; ++n) {
        imposeBC();

        // WARNING: When using the gauss-seidel smoothing as written below, the edges of interior sub-domains after MPI decomposition will not have the updated values
        // As a result, the serial and parallel results will not match when using gauss-seidel smoothing
        if (inputParams.gsSmooth) {
            // GAUSS-SEIDEL ITERATIVE SMOOTHING
            for (int i = 0; i <= xEnd(vLevel); ++i) {
                for (int j = 0; j <= yEnd(vLevel); ++j) {
                    for (int k = 0; k <= zEnd(vLevel); ++k) {
                        lhs(vLevel)(i, j, k) = (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                                xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                                ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                                etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                                ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                                ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                 rhs(vLevel)(i, j, k)) / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                    }
                }
            }
        } else {
            // JACOBI ITERATIVE SMOOTHING
#pragma omp parallel for num_threads(inputParams.nThreads) default(none)
            for (int i = 0; i <= xEnd(vLevel); ++i) {
                for (int j = 0; j <= yEnd(vLevel); ++j) {
                    for (int k = 0; k <= zEnd(vLevel); ++k) {
                        tmp(vLevel)(i, j, k) = (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                                xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                                ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                                etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                                ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                                ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                                 rhs(vLevel)(i, j, k)) / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                    }
                }
            }

            swap(tmp, lhs);
        }
    }

    imposeBC();
}


void multigrid_d3::solve() {
    int iterCount = 0;
    real tempValue, localMax, globalMax;

    while (true) {
        imposeBC();

        // GAUSS-SEIDEL ITERATIVE SOLVER
        for (int i = 0; i <= xEnd(vLevel); ++i) {
            for (int j = 0; j <= yEnd(vLevel); ++j) {
                for (int k = 0; k <= zEnd(vLevel); ++k) {
                    lhs(vLevel)(i, j, k) = (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                            xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                            ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) + lhs(vLevel)(i, j - 1, k)) +
                                            etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                            ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) + lhs(vLevel)(i, j, k - 1)) +
                                            ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1)) -
                                             rhs(vLevel)(i, j, k)) / (2.0 * (ihx2(vLevel)*xix2(vLevel)(i) + ihy2(vLevel)*ety2(vLevel)(j) + ihz2(vLevel)*ztz2(vLevel)(k)));
                }
            }
        }

        tempValue = 0.0;
        localMax = -1.0e-10;
        for (int i = 0; i <= xEnd(vLevel); ++i) {
            for (int j = 0; j <= yEnd(vLevel); ++j) {
                for (int k = 0; k <= zEnd(vLevel); ++k) {
                    tempValue = fabs(rhs(vLevel)(i, j, k) -
                               (xix2(vLevel)(i) * ihx2(vLevel) * (lhs(vLevel)(i + 1, j, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i - 1, j, k)) +
                                xixx(vLevel)(i) * i2hx(vLevel) * (lhs(vLevel)(i + 1, j, k) - lhs(vLevel)(i - 1, j, k)) +
                                ety2(vLevel)(j) * ihy2(vLevel) * (lhs(vLevel)(i, j + 1, k) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j - 1, k)) +
                                etyy(vLevel)(j) * i2hy(vLevel) * (lhs(vLevel)(i, j + 1, k) - lhs(vLevel)(i, j - 1, k)) +
                                ztz2(vLevel)(k) * ihz2(vLevel) * (lhs(vLevel)(i, j, k + 1) - 2.0*lhs(vLevel)(i, j, k) + lhs(vLevel)(i, j, k - 1)) +
                                ztzz(vLevel)(k) * i2hz(vLevel) * (lhs(vLevel)(i, j, k + 1) - lhs(vLevel)(i, j, k - 1))));

                    if (tempValue > localMax)
                        localMax = tempValue;
                }
            }
        }

        MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (globalMax < inputParams.mgTolerance) break;

        iterCount += 1;
        if (iterCount > maxCount) {
            if (inputParams.printResidual)
                if (mesh.rankData.rank == 0)
                    std::cout << "WARNING: Iterations for solution at coarsest level not converging." << std::endl;

            break;
        }
    }

    imposeBC();
}


void multigrid_d3::coarsen() {
    int pLevel;
    int i2, j2, k2;

    pLevel = vLevel;
    vLevel += 1;

    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i*2;
        for (int j = 0; j <= yEnd(vLevel); ++j) {
            j2 = j*2;
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                k2 = k*2;
                rhs(vLevel)(i, j, k) = (tmp(pLevel)(i2, j2, k2) + tmp(pLevel)(i2 + 1, j2 + 1, k2 + 1) +
                                        tmp(pLevel)(i2, j2 + 1, k2 + 1) + tmp(pLevel)(i2 + 1, j2, k2) +
                                        tmp(pLevel)(i2 + 1, j2, k2 + 1) + tmp(pLevel)(i2, j2 + 1, k2) +
                                        tmp(pLevel)(i2 + 1, j2 + 1, k2) + tmp(pLevel)(i2, j2, k2 + 1))/8;
            }
        }
    }
}


void multigrid_d3::prolong() {
    int pLevel;
    int i2, j2, k2;

    pLevel = vLevel;
    vLevel -= 1;

    lhs(vLevel) = 0.0;

    for (int i = 0; i <= xEnd(vLevel); ++i) {
        i2 = i/2;
        for (int j = 0; j <= yEnd(vLevel); j++) {
            j2 = j/2;
            for (int k = 0; k <= zEnd(vLevel); ++k) {
                k2 = k/2;
                lhs(vLevel)(i, j, k) = lhs(pLevel)(i2, j2, k2);
            }
        }
    }
}


real multigrid_d3::computeError(const int normOrder) {
    real residualVal = 0.0;
    real numValLoc = 0.0;
    real denValLoc = 0.0;
    real tempNum = 0.0;
    real tempDen = 0.0;

    // This function is called at the finest grid level only.

    // Problem with Koenig lookup is that when using the function abs with blitz arrays, it automatically computes
    // the absolute of the float values without hitch.
    // When replacing with computing absolute of individual array elements in a loop, ADL chooses a version of
    // abs in the STL which **rounds off** the number.
    // In this case, abs has to be replaced with fabs.
    for (int i = 0; i <= xEnd(0); ++i) {
        for (int j = 0; j <= yEnd(0); ++j) {
            for (int k = 0; k <= zEnd(0); ++k) {
                tempNum = fabs(rhs(0)(i, j, k) -
                           (xix2(0)(i) * ihx2(0) * (lhs(0)(i + 1, j, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i - 1, j, k)) +
                            xixx(0)(i) * i2hx(0) * (lhs(0)(i + 1, j, k) - lhs(0)(i - 1, j, k)) +
                            ety2(0)(j) * ihy2(0) * (lhs(0)(i, j + 1, k) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j - 1, k)) +
                            etyy(0)(j) * i2hy(0) * (lhs(0)(i, j + 1, k) - lhs(0)(i, j - 1, k)) +
                            ztz2(0)(k) * ihz2(0) * (lhs(0)(i, j, k + 1) - 2.0*lhs(0)(i, j, k) + lhs(0)(i, j, k - 1)) +
                            ztzz(0)(k) * i2hz(0) * (lhs(0)(i, j, k + 1) - lhs(0)(i, j, k - 1))));

                tempDen = fabs(rhs(0)(i, j, k));

                switch (normOrder) {
                    case 0: // L-Infinity Norm
                        if (tempNum > numValLoc) numValLoc = tempNum;
                        if (tempDen > denValLoc) denValLoc = tempDen;
                        break;
                    case 1: // L-1 Norm
                        numValLoc += tempNum;
                        denValLoc += tempDen;
                        break;
                    case 2: // L-2 Norm
                        numValLoc += tempNum*tempNum;
                        denValLoc += tempDen*tempDen;
                        break;
                }
            }
        }
    }

    real numValGlo = 0.0;
    real denValGlo = 0.0;
    int pointCount = mesh.totalPoints;
    switch (normOrder) {
        case 0:     // L-Infinity Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo;
            }
            break;
        case 1:     // L-1 Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = numValGlo/denValGlo;
            } else {
                residualVal = numValGlo/pointCount;
            }
            break;
        case 2:     // L-2 Norm
            MPI_Allreduce(&numValLoc, &numValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&denValLoc, &denValGlo, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            if (denValGlo) {
                residualVal = sqrt(numValGlo/pointCount)/sqrt(denValGlo/pointCount);
            } else {
                residualVal = sqrt(numValGlo/pointCount);
            }
            break;
    }

    return residualVal;
}


void multigrid_d3::createMGSubArrays() {
    int count, length, stride;

    recvStatus.resize(4);
    recvRequest.resize(4);

    xMGArray.resize(inputParams.vcDepth + 1);
    yMGArray.resize(inputParams.vcDepth + 1);
    zMGArray.resize(inputParams.vcDepth + 1);

    mgSendLft.resize(inputParams.vcDepth + 1);          mgSendRgt.resize(inputParams.vcDepth + 1);
    mgRecvLft.resize(inputParams.vcDepth + 1);          mgRecvRgt.resize(inputParams.vcDepth + 1);
    mgSendFrn.resize(inputParams.vcDepth + 1);          mgSendBak.resize(inputParams.vcDepth + 1);
    mgRecvFrn.resize(inputParams.vcDepth + 1);          mgRecvBak.resize(inputParams.vcDepth + 1);

    mgSendLftFrn.resize(inputParams.vcDepth + 1);       mgSendRgtBak.resize(inputParams.vcDepth + 1);
    mgRecvLftFrn.resize(inputParams.vcDepth + 1);       mgRecvRgtBak.resize(inputParams.vcDepth + 1);
    mgSendRgtFrn.resize(inputParams.vcDepth + 1);       mgSendLftBak.resize(inputParams.vcDepth + 1);
    mgRecvRgtFrn.resize(inputParams.vcDepth + 1);       mgRecvLftBak.resize(inputParams.vcDepth + 1);

    /***************************************************************************************************
    * Previously xMGArray and yMGArray were defined only if npX > 1 or npY > 1 respectively.
    * This condition remained as a hidden bug in the code for the long time
    * Because for periodic cases, it was implicitly assumed that periodic data transfer will serve
    * But for a sequential case with npX = 1 and npY = 1, this transfer will not happen
    * Now xMGArray and yMGArray are defined irrespective of npX and npY
    \**************************************************************************************************/

    for(int n=0; n<=inputParams.vcDepth; n++) {
        // CREATE X_MG_ARRAY DATATYPE
        count = (stagFull(n).ubound(2) + 2)*(stagFull(n).ubound(1) + 2);

        MPI_Type_contiguous(count, MPI_FP_REAL, &xMGArray(n));
        MPI_Type_commit(&xMGArray(n));

        // CREATE Y_MG_ARRAY DATATYPE
        count = stagFull(n).ubound(0) + 2;
        length = stagFull(n).ubound(2) + 2;
        stride = (stagFull(n).ubound(2) + 2)*(stagFull(n).ubound(1) + 2);

        MPI_Type_vector(count, length, stride, MPI_FP_REAL, &yMGArray(n));
        MPI_Type_commit(&yMGArray(n));

        // CREATE Z_MG_ARRAY DATATYPE - FOR DATA-TRANSFER ACROSS SUB-DOMAIN EDGES
        count = stagFull(n).ubound(2) + 2;

        MPI_Type_contiguous(count, MPI_FP_REAL, &zMGArray(n));
        MPI_Type_commit(&zMGArray(n));

        // SET STARTING INDICES OF MEMORY LOCATIONS FROM WHERE TO READ (SEND) AND WRITE (RECEIVE) DATA
        mgSendLft(n) =  0, -1, -1;
        mgRecvLft(n) = -1, -1, -1;
        mgSendRgt(n) = stagCore(n).ubound(0), -1, -1;
        mgRecvRgt(n) = stagCore(n).ubound(0) + 1, -1, -1;

        mgSendFrn(n) = -1,  0, -1;
        mgRecvFrn(n) = -1, -1, -1;
        mgSendBak(n) = -1, stagCore(n).ubound(1), -1;
        mgRecvBak(n) = -1, stagCore(n).ubound(1) + 1, -1;

        mgSendLftFrn(n) =  0,  0, -1;
        mgRecvLftFrn(n) = -1, -1, -1;
        mgSendRgtBak(n) = stagCore(n).ubound(0), stagCore(n).ubound(1), -1;
        mgRecvRgtBak(n) = stagCore(n).ubound(0) + 1, stagCore(n).ubound(1) + 1, -1;

        mgSendRgtFrn(n) = stagCore(n).ubound(0),  0, -1;
        mgRecvRgtFrn(n) = stagCore(n).ubound(0) + 1, -1, -1;
        mgSendLftBak(n) =  0, stagCore(n).ubound(1), -1;
        mgRecvLftBak(n) = -1, stagCore(n).ubound(1) + 1, -1;
    }
}


void multigrid_d3::initDirichlet() {
    real xDist, yDist, zDist;

    // Generate the walls as 2D Blitz arrays
    xWall.resize(stagFull(0).ubound(1) - stagFull(0).lbound(1) + 1, stagFull(0).ubound(2) - stagFull(0).lbound(2) + 1);
    xWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(1), stagFull(0).lbound(2)));
    xWall = 0.0;

    yWall.resize(stagFull(0).ubound(0) - stagFull(0).lbound(0) + 1, stagFull(0).ubound(2) - stagFull(0).lbound(2) + 1);
    yWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(0), stagFull(0).lbound(2)));
    yWall = 0.0;

    zWall.resize(stagFull(0).ubound(0) - stagFull(0).lbound(0) + 1, stagFull(0).ubound(1) - stagFull(0).lbound(1) + 1);
    zWall.reindexSelf(blitz::TinyVector<int, 2>(stagFull(0).lbound(0), stagFull(0).lbound(1)));
    zWall = 0.0;

    // Compute values at the walls using the (r^2)/6 formula
    // Along X-direction - Left and Right Walls
    xDist = mesh.inputParams.Lx/2.0;
    for (int j=0; j<=stagCore(0).ubound(1); ++j) {
        yDist = mesh.y(j) - 0.5;
        for (int k=0; k<=stagCore(0).ubound(2); ++k) {
            zDist = mesh.z(k) - 0.5;

            xWall(j, k) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }

    // Along Y-direction - Front and Rear Walls
    yDist = mesh.inputParams.Ly/2.0;
    for (int i=0; i<=stagCore(0).ubound(0); ++i) {
        xDist = mesh.x(i) - 0.5;
        for (int k=0; k<=stagCore(0).ubound(2); ++k) {
            zDist = mesh.z(k) - 0.5;

            yWall(i, k) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }

    // Along Z-direction - Top and Bottom Walls
    zDist = mesh.inputParams.Lz/2.0;
    for (int i=0; i<=stagCore(0).ubound(0); ++i) {
        xDist = mesh.x(i) - 0.5;
        for (int j=0; j<=stagCore(0).ubound(1); ++j) {
            yDist = mesh.y(j) - 0.5;

            zWall(i, j) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
        }
    }
}


void multigrid_d3::imposeBC() {
    // FOR PARALLEL RUNS, FIRST UPDATE GHOST POINTS OF MPI SUB-DOMAINS
    updatePads(lhs);

    if (not inputParams.xPer) {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
        if (zeroBC) {
            if (xfr) lhs(vLevel)(-1, all, all) = -lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = -lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);

            //if (xfr) lhs(vLevel)(-1, all, all) = 0.0;
            //if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = 0.0;

        } else {
            if (xfr) lhs(vLevel)(-1, all, all) = 2.0*xWall(all, all) - lhs(vLevel)(0, all, all);
            if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = 2.0*xWall(all, all) - lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);

            //if (xfr) lhs(vLevel)(-1, all, all) = xWall(all, all);
            //if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = xWall(all, all);
        }
#else
        // NEUMANN BOUNDARY CONDITION AT LEFT AND RIGHT WALLS
        if (xfr) lhs(vLevel)(-1, all, all) = lhs(vLevel)(0, all, all);

        if (xlr) lhs(vLevel)(stagCore(vLevel).ubound(0) + 1, all, all) = lhs(vLevel)(stagCore(vLevel).ubound(0), all, all);
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updatePads()


    if (not inputParams.yPer) {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT FRONT AND BACK WALLS
        if (zeroBC) {
            if (yfr) lhs(vLevel)(all, -1, all) = -lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = -lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);

            //if (yfr) lhs(vLevel)(all, -1, all) = 0.0;
            //if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = 0.0;

        } else {
            if (yfr) lhs(vLevel)(all, -1, all) = 2.0*yWall(all, all) - lhs(vLevel)(all, 0, all);
            if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = 2.0*yWall(all, all) - lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);

            //if (yfr) lhs(vLevel)(all, -1, all) = yWall(all, all);
            //if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = yWall(all, all);
        }
#else
        // NEUMANN BOUNDARY CONDITION AT FRONT AND BACK WALLS
        if (yfr) lhs(vLevel)(all, -1, all) = lhs(vLevel)(all, 0, all);

        if (ylr) lhs(vLevel)(all, stagCore(vLevel).ubound(1) + 1, all) = lhs(vLevel)(all, stagCore(vLevel).ubound(1), all);
#endif
    } // PERIODIC BOUNDARY CONDITIONS ARE AUTOMATICALLY IMPOSED BY PERIODIC DATA TRANSFER ACROSS PROCESSORS THROUGH updatePads()


    if (inputParams.zPer) {
        // PERIODIC BOUNDARY CONDITION AT BOTTOM WALL
        lhs(vLevel)(all, all, -1) = lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));

        // PERIODIC BOUNDARY CONDITION AT TOP WALL
        lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = lhs(vLevel)(all, all, 0);

    } else {
#ifdef TEST_POISSON
        // DIRICHLET BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
        if (zeroBC) {
            lhs(vLevel)(all, all, -1) = -lhs(vLevel)(all, all, 0);
            lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = -lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));

            //lhs(vLevel)(all, all, -1) = 0.0;
            //lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = 0.0;
        } else {
            lhs(vLevel)(all, all, -1) = 2.0*zWall(all, all) - lhs(vLevel)(all, all, 0);
            lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = 2.0*zWall(all, all) - lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));

            //lhs(vLevel)(all, all, -1) = zWall(all, all);
            //lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = zWall(all, all);
        }
#else
        // NEUMANN BOUNDARY CONDITION AT BOTTOM AND TOP WALLS
        lhs(vLevel)(all, all, -1) = lhs(vLevel)(all, all, 0);

        lhs(vLevel)(all, all, stagCore(vLevel).ubound(2) + 1) = lhs(vLevel)(all, all, stagCore(vLevel).ubound(2));
#endif
    }
}


void multigrid_d3::updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data) {
    recvRequest = MPI_REQUEST_NULL;

    // TRANSFER DATA ACROSS FACES FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(mgRecvLft(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(mgRecvRgt(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));
    MPI_Irecv(&(data(vLevel)(mgRecvFrn(vLevel))), 1, yMGArray(vLevel), mesh.rankData.faceRanks(2), 3, MPI_COMM_WORLD, &recvRequest(2));
    MPI_Irecv(&(data(vLevel)(mgRecvBak(vLevel))), 1, yMGArray(vLevel), mesh.rankData.faceRanks(3), 4, MPI_COMM_WORLD, &recvRequest(3));

    MPI_Send(&(data(vLevel)(mgSendLft(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(0), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendRgt(vLevel))), 1, xMGArray(vLevel), mesh.rankData.faceRanks(1), 1, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendFrn(vLevel))), 1, yMGArray(vLevel), mesh.rankData.faceRanks(2), 4, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendBak(vLevel))), 1, yMGArray(vLevel), mesh.rankData.faceRanks(3), 3, MPI_COMM_WORLD);

    MPI_Waitall(4, recvRequest.dataFirst(), recvStatus.dataFirst());

    // TRANSFER DATA ACROSS EDGES FROM NEIGHBOURING CELLS TO IMPOSE SUB-DOMAIN BOUNDARY CONDITIONS
    MPI_Irecv(&(data(vLevel)(mgRecvLftFrn(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(0), 1, MPI_COMM_WORLD, &recvRequest(0));
    MPI_Irecv(&(data(vLevel)(mgRecvLftBak(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(1), 2, MPI_COMM_WORLD, &recvRequest(1));
    MPI_Irecv(&(data(vLevel)(mgRecvRgtFrn(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(2), 3, MPI_COMM_WORLD, &recvRequest(2));
    MPI_Irecv(&(data(vLevel)(mgRecvRgtBak(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(3), 4, MPI_COMM_WORLD, &recvRequest(3));

    MPI_Send(&(data(vLevel)(mgSendLftFrn(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(0), 4, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendLftBak(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(1), 3, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendRgtFrn(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(2), 2, MPI_COMM_WORLD);
    MPI_Send(&(data(vLevel)(mgSendRgtBak(vLevel))), 1, zMGArray(vLevel), mesh.rankData.edgeRanks(3), 1, MPI_COMM_WORLD);

    MPI_Waitall(4, recvRequest.dataFirst(), recvStatus.dataFirst());
}

