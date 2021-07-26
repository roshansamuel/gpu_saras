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
/*! \file poisson.h
 *
 *  \brief Class declaration of poisson
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef POISSON_H
#define POISSON_H

#include <sys/time.h>
#include <algorithm>

#include "plainsf.h"
#include "grid.h"

class poisson {
    protected:
        int vLevel, maxCount;

        bool zeroBC;

        /** Flags for first rank (fr) and last rank (lr) along X and Y directions */
        bool xfr, xlr, yfr, ylr;

        // THIS FLAG IS true WHEN USING NEUMANN BC ON ALL WALLS.
        // USING THIS, THE SOLVER IMPOSES COMPATIBILITY CONDITION. 
        bool allNeumann;

        const grid &mesh;
        const parser &inputParams;

        blitz::Range all;

        blitz::Array<blitz::Array<real, 3>, 1> lhs;
        blitz::Array<blitz::Array<real, 3>, 1> rhs;
        blitz::Array<blitz::Array<real, 3>, 1> tmp;
        blitz::Array<blitz::Array<real, 3>, 1> smd;

        blitz::Array<blitz::RectDomain<3>, 1> stagFull;
        blitz::Array<blitz::RectDomain<3>, 1> stagCore;
        blitz::Array<int, 1> xEnd, yEnd, zEnd;
        blitz::RectDomain<3> meanCore;

        blitz::Array<int, 1> mgSizeArray;

        blitz::Array<MPI_Request, 1> recvRequest;
        blitz::Array<MPI_Status, 1> recvStatus;

        blitz::Array<real, 1> i2hx, i2hy, i2hz;
        blitz::Array<real, 1> ihx2, ihy2, ihz2;

        blitz::Array<blitz::Array<real, 1>, 1> xixx, xix2;
        blitz::Array<blitz::Array<real, 1>, 1> etyy, ety2;
        blitz::Array<blitz::Array<real, 1>, 1> ztzz, ztz2;

        blitz::Array<MPI_Datatype, 1> xMGArray;
        blitz::Array<MPI_Datatype, 1> yMGArray;
        blitz::Array<MPI_Datatype, 1> zMGArray;

        blitz::Array<blitz::TinyVector<int, 3>, 1> mgSendLft, mgSendRgt;
        blitz::Array<blitz::TinyVector<int, 3>, 1> mgRecvLft, mgRecvRgt;

        blitz::Array<blitz::TinyVector<int, 3>, 1> mgSendFrn, mgSendBak;
        blitz::Array<blitz::TinyVector<int, 3>, 1> mgRecvFrn, mgRecvBak;

        blitz::Array<blitz::TinyVector<int, 3>, 1> mgSendLftFrn, mgSendRgtBak;
        blitz::Array<blitz::TinyVector<int, 3>, 1> mgRecvLftFrn, mgRecvRgtBak;

        blitz::Array<blitz::TinyVector<int, 3>, 1> mgSendRgtFrn, mgSendLftBak;
        blitz::Array<blitz::TinyVector<int, 3>, 1> mgRecvRgtFrn, mgRecvLftBak;

        static inline bool isOdd(int x) { return x % 2; };

        void initializeArrays();
        void copyDerivs();
        void setCoefficients();
        void setStagBounds();

        virtual void coarsen();
        virtual void prolong();
        virtual void computeResidual();
        virtual void smooth(const int smoothCount);
        virtual real computeError(const int normOrder);

        virtual void solve() {};
        virtual void imposeBC();
        virtual void createMGSubArrays();
        virtual void updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data);

        void vCycle();

    public:
        poisson(const grid &mesh, const parser &solParam);

        void mgSolve(plainsf &outLHS, const plainsf &inpRHS);

        virtual ~poisson();
};

/**
 ********************************************************************************************************************************************
 *  \class poisson poisson.h "lib/poisson.h"
 *  \brief The base class poisson and its derived classes multigrid_d2 and multigrid_d3
 *
 *  The class implements the geometric multi-grid method for solving the Poisson equation on a non-uniform grid across MPI decomposed
 *  domains for parallel computations.
 *  The data structure used by the class for computing multi-grid V-cycles across sub-domains is an array of blitz arrays
 *  with each blitz array corresponding to the data at a given V-cycle depth.
 *
 *  All the necessary functions to perform the V-cycle - prolongation, solving at coarsest mesh, smoothening, etc. are implemented
 *  within the \ref poisson class.
 ********************************************************************************************************************************************
 */

class multigrid_d2: public poisson {
    private:
        void coarsen();
        void prolong();
        void computeResidual();
        void smooth(const int smoothCount);
        real computeError(const int normOrder);

        void solve();

        void imposeBC();
        void initDirichlet();

        void createMGSubArrays();
        void updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data);

        blitz::Array<real, 1> xWall, zWall;

    public:
        multigrid_d2(const grid &mesh, const parser &solParam);

        ~multigrid_d2() {};
};

/**
 ********************************************************************************************************************************************
 *  \class multigrid_d2 poisson.h "lib/poisson.h"
 *  \brief The derived class from poisson to perform multi-grid operations on a 2D grid
 *
 *  The 2D implementation ignores the y-direction component of the computational domain.
 ********************************************************************************************************************************************
 */

class multigrid_d3: public poisson {
    private:
        void coarsen();
        void prolong();
        void computeResidual();
        void smooth(const int smoothCount);
        real computeError(const int normOrder);

        void solve();

        void imposeBC();
        void initDirichlet();

        void createMGSubArrays();
        void updatePads(blitz::Array<blitz::Array<real, 3>, 1> &data);

        blitz::Array<real, 2> xWall, yWall, zWall;

    public:
        multigrid_d3(const grid &mesh, const parser &solParam);

        ~multigrid_d3() {};
};

/**
 ********************************************************************************************************************************************
 *  \class multigrid_d3 poisson.h "lib/poisson.h"
 *  \brief The derived class from poisson to perform multi-grid operations on a 3D grid
 *
 *  The 3D implementation of the multi-grid method differs from the 2D version in that the \ref coarsen, \ref smooth etc use a different
 *  equation with extra terms, and the \ref prolong operation needs to perform extra interpolation steps in the y-direction.
 ********************************************************************************************************************************************
 */

#endif
