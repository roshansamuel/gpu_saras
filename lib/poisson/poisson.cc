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
/*! \file poisson.cc
 *
 *  \brief Definitions for functions of class poisson
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
 * \brief   Constructor of the base poisson class
 *
 *          The base constructor of the poisson class assigns the const references to the grid and parser
 *          class instances being used in the solver.
 *          Moreover, it resizes and populates a local array of multi-grid sizes as used in the grid class.
 *          An array of strides to be used at different V-cycle levels is also generated and stored.
 *
 *          It then calls a series of functions in sequence to initialize all the necessary parameters and data structures to
 *          store and manipulate the multi-grid data.
 *          Since the multi-grid solver operates on the staggered grid, it first computes the limits of the full and core
 *          staggered grid, as the grid class does the same for the collocated grid.
 *
 *          It then copies the staggered grid derivatives to local arrays with wide pads, and finally initializes all the
 *          blitz arrays used in the multigrid calculations.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   solParam is a const reference to the user-set parameters contained in the parser class
 ********************************************************************************************************************************************
 */
poisson::poisson(const grid &mesh, const parser &solParam): mesh(mesh), inputParams(solParam) {
    int maxIndex = 15;

    all = blitz::Range::all();

    mgSizeArray.resize(maxIndex);
    for (int i=0; i < maxIndex; i++) {
        mgSizeArray(i) = int(pow(2, i));
    }

    mgSizeArray(0) = 1;

    // SET FLAGS FOR FIRST AND LAST RANKS ALONG X AND Y DIRECTIONS
    xfr = (mesh.rankData.xRank == 0)? true: false;
    yfr = (mesh.rankData.yRank == 0)? true: false;
    xlr = (mesh.rankData.xRank == mesh.rankData.npX - 1)? true: false;
    ylr = (mesh.rankData.yRank == mesh.rankData.npY - 1)? true: false;

    // SET THE ARRAY LIMITS OF FULL AND CORE DOMAINS
    setStagBounds();

    // SET VALUES OF COEFFICIENTS USED FOR COMPUTING LAPLACIAN
    setCoefficients();

    // COPY THE STAGGERED GRID DERIVATIVES TO LOCAL ARRAYS
    copyDerivs();

    // RESIZE AND INITIALIZE ARRAYS
    initializeArrays();

    // THE allNeumann FLAG IMPOSES AN ADDITIONAL COMPATIBILITY CONDITION
    // SINCE ALL THE BOUNDARIES ARE ASSUMED TO HAVE NEUMANN BC WHEN 
    // SOLVING THE PRESSURE CORRECTION EQUATION.
    allNeumann = true;
}


/**
 ********************************************************************************************************************************************
 * \brief   The core, publicly accessible function of poisson to compute the solution for the Poisson equation
 *
 *          The function calls the V-cycle as many times as set by the user.
 *          Before doing so, the input data is transferred into the data-structures used by the poisson class to
 *          perform restrictions and prolongations without copying.
 *          Finally, the computed solution is transferred back from the internal data-structures back into the
 *          scalar field supplied by the calling function.
 *
 * \param   outLHS is a pointer to the plain scalar field (cell-centered) into which the computed soltuion must be transferred
 * \param   inpRHS is a const reference to the plain scalar field (cell-centered) which contains the RHS for the Poisson equation to solve
 ********************************************************************************************************************************************
 */
void poisson::mgSolve(plainsf &outLHS, const plainsf &inpRHS) {
    vLevel = 0;

    for (int i=0; i <= inputParams.vcDepth; i++) {
        lhs(i) = 0.0;
        rhs(i) = 0.0;
        smd(i) = 0.0;
    }

    // TRANSFER DATA FROM THE INPUT SCALAR FIELDS INTO THE DATA-STRUCTURES USED BY poisson
    rhs(0)(stagCore(0)) = inpRHS.F(stagCore(0));
    lhs(0)(stagCore(0)) = outLHS.F(stagCore(0));

    updatePads(rhs);
    updatePads(lhs);

    // PERFORM V-CYCLES AS MANY TIMES AS REQUIRED
    for (int i=0; i<inputParams.vcCount; i++) {
        vCycle();

        real mgResidual = computeError(inputParams.resType);

#ifdef TEST_POISSON
        blitz::Array<real, 3> pAnalytic, tempArray;

        pAnalytic.resize(blitz::TinyVector<int, 3>(stagCore(0).ubound() + 1));
        pAnalytic = 0.0;

#ifdef PLANAR
        real xDist, zDist;

        for (int i=0; i<=stagCore(0).ubound(0); ++i) {
            xDist = mesh.x(i) - 0.5;
            for (int k=0; k<=stagCore(0).ubound(2); ++k) {
                zDist = mesh.z(k) - 0.5;

                pAnalytic(i, 0, k) = (xDist*xDist + zDist*zDist)/4.0;
            }
        }
#else
        real xDist, yDist, zDist;

        for (int i=0; i<=stagCore(0).ubound(0); ++i) {
            xDist = mesh.x(i) - 0.5;
            for (int j=0; j<=stagCore(0).ubound(1); ++j) {
                yDist = mesh.y(j) - 0.5;
                for (int k=0; k<=stagCore(0).ubound(2); ++k) {
                    zDist = mesh.z(k) - 0.5;

                    pAnalytic(i, j, k) = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0;
                }
            }
        }
#endif

        tempArray.resize(pAnalytic.shape());
        tempArray = pAnalytic - lhs(0)(stagCore(0));

        real gloMax = 0.0;
        real locMax = blitz::max(fabs(tempArray));
        MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

        if (mesh.rankData.rank == 0) {
            std::cout << std::endl;
            std::cout << "Maximum absolute deviation from analytic solution is: " << std::scientific << std::setprecision(3) << gloMax << std::endl;
        }
#endif

        if (inputParams.printResidual)
            if (mesh.rankData.rank == 0)
                std::cout << std::endl << "Residual after V Cycle " << i << " is " << std::scientific << std::setprecision(3) << mgResidual << std::endl;

#ifndef TEST_POISSON
        if (mgResidual < inputParams.mgTolerance) break;
#endif
    }

#ifndef TEST_POISSON
    if (allNeumann) {
        // WHEN USING NEUMANN BC ON ALL SIDES, THERE ARE INFINTELY MANY SOLUTIONS.
        // SUBTRACT THE MEAN OF THE SOLUTION SO THAT THE PRESSURE CORRECTION HAS ZERO MEAN.
        // THIS CAN ENSURE THAT THE PRESSURE DOESN'T DRIFT AS THE SIMULATION EVOLVES.
        // THIS DOES NOT AFFECT THE PRESSURE GRADIENT, WHICH IS THE ACTUAL METRIC THAT
        // SATISFIES DIVERGENCE IN THIS FORMULATION OF NSE.
        real localMean = blitz::sum(lhs(0)(stagCore(0)))/mesh.totalPoints;
        real globalAvg = 0.0;

        MPI_Allreduce(&localMean, &globalAvg, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

        lhs(0) -= globalAvg;
    }
#endif

    // RETURN CALCULATED PRESSURE DATA
    outLHS.F(stagFull(0)) = lhs(0)(stagFull(0));
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to perform one loop of V-cycle
 *
 *          The V-cycle of restrictions, prolongations and smoothings are performed within this function.
 *          First the input data contained in \ref lhs is smoothed, after which the residual is computed and stored
 *          in the \ref rhs array.
 *          The restrictions, smoothing, and prolongations are performed on these two arrays subsequently.
 *
 ********************************************************************************************************************************************
 */
void poisson::vCycle() {
    /*
     * OUTLINE OF THE MULTI-GRID V-CYCLE
     * 1)  Starting at finest grid, perform N Gauss-Siedel pre-smoothing iterations to solve for the solution, Ax=b.
     * 2)  Compute the residual r=b-Ax, and restrict it to a coarser level.
     * 3)  Perform N Gauss-Siedel pre-smoothing iterations to solve for the error, Ae=r.
     * 4)  Repeat steps 2-3 until you reach the coarsest grid level.
     * 5)  Perform 2N (pre + post) Gauss-Siedel smoothing iterations to solve for the error 'e'.
     * 6)  Prolong the error 'e' to the next finer level.
     * 7)  Perform N post-smoothing iterations.
     * 8)  Repeat steps 6-7 until the finest grid is reached.
     * 9)  Add error 'e' to the solution 'x' and perform N post-smoothing iterations.
     * 10) End of one V-cycle - check for convergence by computing the normalized residual: r_normalized = ||b-Ax||/||b||. 
     */

    vLevel = 0;

    // When using Dirichlet BC, the residue, r, has homogeneous BC (r=0 at boundary) and only the original solution, x, has non-homogeneous BC.
    // Since pre-smoothing is performed on x, non-homogeneous (non-zero) Dirichlet BC is imposed
    zeroBC = false;

    // Step 1) Pre-smoothing iterations of Ax = b
    smooth(inputParams.preSmooth);

    // From now on, homogeneous Dirichlet BCs are used till end of V-Cycle
    zeroBC = true;

    // RESTRICTION OPERATIONS DOWN TO COARSEST MESH
    for (int i=0; i<inputParams.vcDepth; i++) {
        // Step 2) Compute the residual r = b - Ax
        computeResidual();

        // Copy lhs into smd
        smd(vLevel) = lhs(vLevel);

        // Restrict the residual to a coarser level
        coarsen();

        // Initialize lhs to 0, or the convergence will be drastically slow
        lhs(vLevel) = 0.0;

        // Step 3) Perform pre-smoothing iterations to solve for the error: Ae = r
        (vLevel == inputParams.vcDepth)?
            inputParams.solveFlag?
                solve():
                smooth(inputParams.preSmooth):
            smooth(inputParams.preSmooth);
    }
    // Step 4) Repeat steps 2-3 until you reach the coarsest grid level,

    // PROLONGATION OPERATIONS UP TO FINEST MESH
    for (int i=0; i<inputParams.vcDepth; i++) {
        // Step 6) Prolong the error 'e' to the next finer level.
        prolong();

        // Step 9) Add correction 'e' to the solution 'x' and perform post-smoothing iterations.
        lhs(vLevel) += smd(vLevel);

        // Once the error/residual has been added to the solution at finest level, the Dirichlet BC to be applied is again non-zero
        (vLevel == 0)? zeroBC = false: zeroBC = true;

        // Step 7) Perform post-smoothing iterations
        smooth(inputParams.postSmooth);
    }
    // Step 8) Repeat steps 6-7 until you reach the finest grid level,
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to initialize the arrays used in multi-grid
 *
 *          The memory required for various arrays in multi-grid solver are pre-allocated through this function.
 *          The function is called from within the constructor to perform this allocation once and for all.
 *          The arrays are initialized to 0.
 *
 ********************************************************************************************************************************************
 */
void poisson::initializeArrays() {
    lhs.resize(inputParams.vcDepth + 1);
    rhs.resize(inputParams.vcDepth + 1);
    tmp.resize(inputParams.vcDepth + 1);
    smd.resize(inputParams.vcDepth + 1);

    for (int i=0; i <= inputParams.vcDepth; i++) {
        lhs(i).resize(blitz::TinyVector<int, 3>(stagFull(i).ubound() - stagFull(i).lbound() + 1));
        lhs(i).reindexSelf(stagFull(i).lbound());
        lhs(i) = 0.0;

        tmp(i).resize(blitz::TinyVector<int, 3>(stagFull(i).ubound() - stagFull(i).lbound() + 1));
        tmp(i).reindexSelf(stagFull(i).lbound());
        tmp(i) = 0.0;

        rhs(i).resize(blitz::TinyVector<int, 3>(stagFull(i).ubound() - stagFull(i).lbound() + 1));
        rhs(i).reindexSelf(stagFull(i).lbound());
        rhs(i) = 0.0;

        smd(i).resize(blitz::TinyVector<int, 3>(stagFull(i).ubound() - stagFull(i).lbound() + 1));
        smd(i).reindexSelf(stagFull(i).lbound());
        smd(i) = 0.0;
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the RectDomain variables for all future references throughout the poisson solver
 *
 *          The function sets the core and full domain staggered grid sizes for all the sub-domains.
 *          The maximum allowable number of iterations for the Jacobi iterative solver used at the
 *          coarsest mesh is set as \f$ N_{max} = N_x \times N_y \times N_z \f$, where \f$N_x\f$, \f$N_y\f$ and \f$N_z\f$
 *          are the number of grid points in the collocated grid at the local sub-domains along x, y and z directions
 *          respectively.
 *
 ********************************************************************************************************************************************
 */
void poisson::setStagBounds() {
    blitz::TinyVector<int, 3> localSizeIndex;
    blitz::TinyVector<int, 3> loBound, upBound;

    stagFull.resize(inputParams.vcDepth + 1);
    stagCore.resize(inputParams.vcDepth + 1);

    xEnd.resize(inputParams.vcDepth + 1);
    yEnd.resize(inputParams.vcDepth + 1);
    zEnd.resize(inputParams.vcDepth + 1);

    /**
     ********************************************************************************************************************************************
     *  For the multi-grid solver, the number of grid points (cell-centers) must be \f$ 2^N \f$ to perform V-Cycles.
     *  The domain decomposition is also done such that each sub-domain will also have \f$ 2^M \f$ cells.
     *  As a result, the number of processors in each direction must be a power of 2.
     *  Then \f$ M = N - log_2(np) \f$ where \f$ np \f$ is the number of processors along the direction.
     ********************************************************************************************************************************************
     */
#ifdef PLANAR
    localSizeIndex = blitz::TinyVector<int, 3>(mesh.sizeIndex(0) - int(log2(inputParams.npX)),
                                               mesh.sizeIndex(1),
                                               mesh.sizeIndex(2));
#else
    localSizeIndex = blitz::TinyVector<int, 3>(mesh.sizeIndex(0) - int(log2(inputParams.npX)),
                                               mesh.sizeIndex(1) - int(log2(inputParams.npY)),
                                               mesh.sizeIndex(2));
#endif

    for (int i=0; i<=inputParams.vcDepth; i++) {
        // LOWER BOUND AND UPPER BOUND OF STAGGERED CORE - USED TO CONSTRUCT THE CORE SLICE
        loBound = 0, 0, 0;
#ifdef PLANAR
        upBound = mgSizeArray(localSizeIndex(0) - i) - 1, 0, mgSizeArray(localSizeIndex(2) - i) - 1;
#else
        upBound = mgSizeArray(localSizeIndex(0) - i) - 1, mgSizeArray(localSizeIndex(1) - i) - 1, mgSizeArray(localSizeIndex(2) - i) - 1;
#endif
        stagCore(i) = blitz::RectDomain<3>(loBound, upBound);

        // LOWER BOUND AND UPPER BOUND OF STAGGERED FULL SUB-DOMAIN - USED TO CONSTRUCT THE FULL SUB-DOMAIN SLICE
        // NOTE THAT THE PAD WIDTH USED IN POISSON SOLVER IS 1 BY DEFAULT, AND NOT THE SAME AS THAT IN GRID CLASS
        loBound = -1, -1, -1;
        upBound = stagCore(i).ubound() - loBound;
        stagFull(i) = blitz::RectDomain<3>(loBound, upBound);

        // SET THE LIMTS FOR ARRAY LOOPS IN smooth FUNCTION, AND A FEW OTHER PLACES
        xEnd(i) = stagCore(i).ubound(0);
#ifndef PLANAR
        yEnd(i) = stagCore(i).ubound(1);
#endif
        zEnd(i) = stagCore(i).ubound(2);
    }

    // SET MAXIMUM NUMBER OF ITERATIONS FOR THE GAUSS-SEIDEL SOLVER AT COARSEST LEVEL OF MULTIGRID SOLVER
    blitz::TinyVector<int, 3> cgSize = stagFull(inputParams.vcDepth).ubound() - stagFull(inputParams.vcDepth).lbound();
    maxCount = cgSize(0)*cgSize(1)*cgSize(2);
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to set the coefficients used for calculating laplacian and in smoothing
 *
 *          The function assigns values to the variables \ref hx, \ref hy etc.
 *          These coefficients are repeatedly used at many places in the Poisson solver.
 *
 ********************************************************************************************************************************************
 */
void poisson::setCoefficients() {
    ihx2.resize(inputParams.vcDepth + 1);
    i2hx.resize(inputParams.vcDepth + 1);
#ifndef PLANAR
    ihy2.resize(inputParams.vcDepth + 1);
    i2hy.resize(inputParams.vcDepth + 1);
#endif
    ihz2.resize(inputParams.vcDepth + 1);
    i2hz.resize(inputParams.vcDepth + 1);

    for(int i=0; i<=inputParams.vcDepth; i++) {
        int hInc = (1 << i);

        real hx = hInc*mesh.dXi;
        real hx2 = pow(hx, 2);
        i2hx(i) = 0.5/hx;
        ihx2(i) = 1.0/hx2;

#ifndef PLANAR
        real hy = hInc*mesh.dEt;
        real hy2 = pow(hy, 2);
        i2hy(i) = 0.5/hy;
        ihy2(i) = 1.0/hy2;
#endif

        real hz = hInc*mesh.dZt;
        real hz2 = pow(hz, 2);
        i2hz(i) = 0.5/hz;
        ihz2(i) = 1.0/hz2;
    }
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to copy the staggered grid derivatives from the grid class to local arrays
 *
 *          Though the grid derivatives in the grid class can be read and accessed, they cannot be used directly
 *          along with the arrays defined in the poisson class.
 *          Depending on the differentiation scheme selected by user, the arrays in grid class may have different
 *          pad widths, but within poisson class, only a single pad point is used everywhere, since the schemes
 *          used here are second order accurate by default.
 *          Therefore, corresponding arrays with single pad for grid derivatives are used, into which the staggered
 *          grid derivatives from the grid class are written and stored.
 *          This function serves this purpose of copying the grid derivatives.
 *
 ********************************************************************************************************************************************
 */
void poisson::copyDerivs() {
    // Sub-array start and end indices (in global indexing) at different levels of multigrid
    int ss, se, ls;

    xixx.resize(inputParams.vcDepth + 1);
    xix2.resize(inputParams.vcDepth + 1);
#ifndef PLANAR
    etyy.resize(inputParams.vcDepth + 1);
    ety2.resize(inputParams.vcDepth + 1);
#endif
    ztzz.resize(inputParams.vcDepth + 1);
    ztz2.resize(inputParams.vcDepth + 1);

    for(int n=0; n<=inputParams.vcDepth; ++n) {
        ss = mesh.subarrayStarts(0)/int(pow(2, n)) - 1;
        se = (mesh.subarrayEnds(0) - 1)/int(pow(2, n)) + 1;
        ls = 15*n + 3;

        xixx(n).resize(stagFull(n).ubound(0) - stagFull(n).lbound(0) + 1);
        xixx(n).reindexSelf(stagFull(n).lbound(0));
        xixx(n) = 0.0;
        xixx(n)(blitz::Range(-1, stagFull(n).ubound(0))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));

        ls = 15*n + 4;

        xix2(n).resize(stagFull(n).ubound(0) - stagFull(n).lbound(0) + 1);
        xix2(n).reindexSelf(stagFull(n).lbound(0));
        xix2(n) = 0.0;
        xix2(n)(blitz::Range(-1, stagFull(n).ubound(0))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));

#ifndef PLANAR
        ss = mesh.subarrayStarts(1)/int(pow(2, n)) - 1;
        se = (mesh.subarrayEnds(1) - 1)/int(pow(2, n)) + 1;
        ls = 15*n + 8;

        etyy(n).resize(stagFull(n).ubound(1) - stagFull(n).lbound(1) + 1);
        etyy(n).reindexSelf(stagFull(n).lbound(1));
        etyy(n) = 0.0;
        etyy(n)(blitz::Range(-1, stagFull(n).ubound(1))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));

        ls = 15*n + 9;

        ety2(n).resize(stagFull(n).ubound(1) - stagFull(n).lbound(1) + 1);
        ety2(n).reindexSelf(stagFull(n).lbound(1));
        ety2(n) = 0.0;
        ety2(n)(blitz::Range(-1, stagFull(n).ubound(1))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));
#endif

        ss = mesh.subarrayStarts(2)/int(pow(2, n)) - 1;
        se = (mesh.subarrayEnds(2) - 1)/int(pow(2, n)) + 1;
        ls = 15*n + 13;

        ztzz(n).resize(stagFull(n).ubound(2) - stagFull(n).lbound(2) + 1);
        ztzz(n).reindexSelf(stagFull(n).lbound(2));
        ztzz(n) = 0.0;
        ztzz(n)(blitz::Range(-1, stagFull(n).ubound(2))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));

        ls = 15*n + 14;

        ztz2(n).resize(stagFull(n).ubound(2) - stagFull(n).lbound(2) + 1);
        ztz2(n).reindexSelf(stagFull(n).lbound(2));
        ztz2(n) = 0.0;
        ztz2(n)(blitz::Range(-1, stagFull(n).ubound(2))) = mesh.globalMetrics(ls)(blitz::Range(ss, se));
    }
};


/**
 ********************************************************************************************************************************************
 * \brief   Function to coarsen the grid down the levels of the V-Cycle
 *
 *          Coarsening reduces the number of points in the grid by averaging values at two adjacent nodes onto an intermediate point between them
 *          As a result, the number of points in the domain decreases from \f$ 2^{N+1} + 1 \f$ at the input level to \f$ 2^N + 1 \f$.
 *          The vLevel variable is accordingly incremented by 1 to reflect this descent by one step down the V-Cycle.
 *
 ********************************************************************************************************************************************
 */
void poisson::coarsen() { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to perform prolongation on the array being solved
 *
 *          Prolongation makes the grid finer by averaging values at two adjacent nodes onto an intermediate point between them
 *          As a result, the number of points in the domain increases from \f$ 2^N + 1 \f$ at the input level to \f$ 2^{N+1} + 1 \f$.
 *          The vLevel variable is accordingly reduced by 1 to reflect this ascent by one step up the V-Cycle.
 *
 ********************************************************************************************************************************************
 */
void poisson::prolong() { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the residual at the start of each V-Cycle
 *
 *          The Poisson solver solves for the residual r = b - Ax
 *          This function computes this residual by calculating the Laplacian of the pressure field and
 *          subtracting it from the RHS of Poisson equation.
 *
 ********************************************************************************************************************************************
 */
void poisson::computeResidual() { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to perform smoothing operation on the input array
 *
 *          The smoothing operation is always performed on the data contained in the array \ref lhs.
 *          The array \ref tmp is used to store the temporary data and it is continuously swapped with the
 *          \ref lhs array at every iteration.
 *          This operation can be performed at any level of the V-cycle.
 *
 * \param   smoothCount is the integer value of the number of smoothing iterations to be performed
 ********************************************************************************************************************************************
 */
void poisson::smooth(const int smoothCount) { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the error at the end of each V-Cycle
 *
 *          To check for convergence, the residual must be computed throughout the domain.
 *          This function offers multiple ways to compute the residual (global maximum, rms, mean, etc.)
 *
 * \param   normOrder is the integer value of the order of norm used to calculate the residual.
 ********************************************************************************************************************************************
 */
real poisson::computeError(const int normOrder) {  return 0.0; };


/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions of Poisson solver at different levels of the V-cycle
 *
 *          This function is called mainly during smoothing operations to impose the boundary conditions for the
 *          Poisson equation.
 *          The sub-domains close to the wall will have the Neumann boundary condition on pressure imposeed at the walls.
 *          Meanwhile at the interior boundaries at the inter-processor sub-domains, data is transferred from the neighbouring cells
 *          by calling the \ref updatePads function.
 *
 ********************************************************************************************************************************************
 */
void poisson::imposeBC() { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to update the pad points of the local sub-domains at different levels of the V-cycle
 *
 *          This function is called mainly during smoothing operations by the \ref imposeBC function.
 *          At the interior boundaries at the inter-processor sub-domains, data is transferred from the neighbouring cells
 *          using a combination of MPI_Irecv and MPI_Send functions.
 *
 ********************************************************************************************************************************************
 */
void poisson::updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data) { };


/**
 ********************************************************************************************************************************************
 * \brief   Function to create the MPI sub-array data types necessary to transfer data across sub-domains
 *
 *          The inter-domain boundaries of all the sub-domains at different V-cycle levels need data to be transfered at
 *          with different mesh strides.
 *          The number of sub-arrays along each edge/face of the sub-domains are equal to the number of V-cycle levels.
 *          Since this data transfer has to take place at all the mesh levels including the finest mesh, there will be
 *          vcDepth + 1 elements.
 *
 ********************************************************************************************************************************************
 */
void poisson::createMGSubArrays() { };


poisson::~poisson() { };
