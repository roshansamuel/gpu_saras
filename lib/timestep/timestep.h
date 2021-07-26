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
/*! \file timestep.h
 *
 *  \brief Class declaration of timestep
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "poisson.h"
#include "tseries.h"
#include "force.h"
#include "les.h"

class timestep {
    public:
        real nu, kappa;

        timestep(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P);

        virtual void timeAdvance(vfield &V, sfield &P);
        virtual void timeAdvance(vfield &V, sfield &P, sfield &T);

    protected:
        // Const references to the time and time-step variables in the main solver.
        // These values can only be read by this class and not modified
        const real &solTime, &dt;

        const grid &mesh;

        blitz::RectDomain<3> core;

        int xSt, xEn;
        int ySt, yEn;
        int zSt, zEn;

        /** Plain scalar field into which the pressure correction is calculated and written by the Poisson solver */
        plainsf Pp;
        /** Plain scalar field into which the RHS for pressure Poisson equation is written and passed to the Poisson solver */
        plainsf mgRHS;

        /** A reference to the time-series I/O object to which relevant data to be written to I/O can be sent */
        tseries &tsWriter;

        /** Plain vector field which stores the pressure gradient term. */
        plainvf pressureGradient;
};

/**
 ********************************************************************************************************************************************
 *  \class timestep timestep.h "lib/timestep/timestep.h"
 *  \brief Contains all the global variables related to time-advancing the solution by one time-step
 *
 ********************************************************************************************************************************************
 */

class eulerCN_d2: public timestep {
    public:
        eulerCN_d2(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P);

        void timeAdvance(vfield &V, sfield &P);
        void timeAdvance(vfield &V, sfield &P, sfield &T);

    private:
        /** Maximum number of iterations for the iterative solvers solveVx, solveVy and solveVz */
        int maxIterations;

        real ihx2, ihz2;
        real i2hx, i2hz;

        multigrid_d2 mgSolver;

        void solveVx(vfield &V, plainvf &nseRHS);
        void solveVz(vfield &V, plainvf &nseRHS);

        void solveT(sfield &T, plainsf &tmpRHS);

        void setCoefficients();
};

/**
 ********************************************************************************************************************************************
 *  \class eulerCN_d2 timestep.h "lib/timestep/timestep.h"
 *  \brief The derived class from timestep to advance the 2D solution using the semi-implicit Crank-Nicholson-Euler method
 *
 ********************************************************************************************************************************************
 */

class eulerCN_d3: public timestep {
    public:
        eulerCN_d3(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P);

        void timeAdvance(vfield &V, sfield &P);
        void timeAdvance(vfield &V, sfield &P, sfield &T);

    private:
        /** Maximum number of iterations for the iterative solvers solveVx, solveVy and solveVz */
        int maxIterations;

        real ihx2, ihy2, ihz2;
        real i2hx, i2hy, i2hz;

        multigrid_d3 mgSolver;

        les *sgsLES;

        void solveVx(vfield &V, plainvf &nseRHS);
        void solveVy(vfield &V, plainvf &nseRHS);
        void solveVz(vfield &V, plainvf &nseRHS);

        void solveT(sfield &T, plainsf &tmpRHS);

        void setCoefficients();
};

/**
 ********************************************************************************************************************************************
 *  \class eulerCN_d3 timestep.h "lib/timestep/timestep.h"
 *  \brief The derived class from timestep to advance the 3D solution using the semi-implicit Crank-Nicholson-Euler method
 *
 ********************************************************************************************************************************************
 */

class lsRK3_d2: public timestep {
    public:
        lsRK3_d2(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P);

        void timeAdvance(vfield &V, sfield &P);
        void timeAdvance(vfield &V, sfield &P, sfield &T);

    private:
        /** Maximum number of iterations for the iterative solvers solveVx, solveVy and solveVz */
        int maxIterations;

        real ihx2, ihz2;
        real i2hx, i2hz;

        blitz::TinyVector<real, 3> alphRK3, betaRK3, zetaRK3, gammRK3;

        multigrid_d2 mgSolver;

        void solveVx(vfield &V, plainvf &nseRHS, real beta);
        void solveVz(vfield &V, plainvf &nseRHS, real beta);

        void solveT(sfield &T, plainsf &tmpRHS, real beta);

        void setCoefficients();
};

/**
 ********************************************************************************************************************************************
 *  \class lsRK3_d2 timestep.h "lib/timestep/timestep.h"
 *  \brief The derived class from timestep to advance the 2D solution using low-storage third-order Runge-Kutta scheme
 *
 ********************************************************************************************************************************************
 */

class lsRK3_d3: public timestep {
    public:
        lsRK3_d3(const grid &mesh, const real &sTime, const real &dt, tseries &tsIO, vfield &V, sfield &P);

        void timeAdvance(vfield &V, sfield &P);
        void timeAdvance(vfield &V, sfield &P, sfield &T);

    private:
        /** Maximum number of iterations for the iterative solvers solveVx, solveVy and solveVz */
        int maxIterations;

        real ihx2, ihy2, ihz2;
        real i2hx, i2hy, i2hz;

        blitz::TinyVector<real, 3> alphRK3, betaRK3, zetaRK3, gammRK3;

        multigrid_d3 mgSolver;

        les *sgsLES;

        void solveVx(vfield &V, plainvf &nseRHS, real beta);
        void solveVy(vfield &V, plainvf &nseRHS, real beta);
        void solveVz(vfield &V, plainvf &nseRHS, real beta);

        void solveT(sfield &T, plainsf &tmpRHS, real beta);

        void setCoefficients();
};

/**
 ********************************************************************************************************************************************
 *  \class lsRK3_d3 timestep.h "lib/timestep/timestep.h"
 *  \brief The derived class from timestep to advance the 3D solution using low-storage third-order Runge-Kutta scheme
 *
 ********************************************************************************************************************************************
 */

#endif
