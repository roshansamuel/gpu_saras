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
/*! \file hydro.h
 *
 *  \brief Class declaration of the hydro solver for both 2D and 3D cases.
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef HYDRO_H
#define HYDRO_H

#include "timestep.h"
#include "probes.h"
#include "sfield.h"
#include "vfield.h"

class hydro {
    public:
        /** The vector field that stores the velocity field */
        vfield V;

        /** The scalar field that stores the pressure field */
        sfield P;

        hydro(const grid &mesh, const parser &solParam, parallel &mpiParam);

        virtual void solvePDE();

        virtual ~hydro() { };

    protected:
        /** Integer value for the number of time-steps elapsed - it is incremented by 1 in each time-step. */
        int timeStepCount;

        real time, dt;

        const grid &mesh;
        const parser &inputParams;

        /** Instance of the \ref timestep class to perform time-integration. */
        timestep *ivpSolver;

        /** Instance of the \ref probe class to collect data from probes in the domain. */
        probes *dataProbe;

        /** Instance of the \ref parallel class that holds the MPI-related data like rank, xRank, etc. */
        parallel &mpiData;

        void checkPeriodic();

        void initVBCs();
        void initPBCs();
        void initVForcing();

        inline int roundNum(int numToRound, int multiple) {
            int remainder = numToRound % multiple;
            int halfNum = int(multiple/2);
            if (remainder == 0) return numToRound;
            if (remainder < halfNum) return numToRound - remainder;
            return numToRound + multiple - remainder;
        };
};

/**
 ********************************************************************************************************************************************
 *  \class hydro hydro.h "lib/hydro.h"
 *  \brief The base class hydro to solve the incompressible Navier-Stokes equations
 *
 *  The class initializes and stores the velocity vector field and the pressure scalar field along with a few auxilliary
 *  fields to solve the PDE.
 *  It solves the NSE using the \ref solvePDE function from within which the implicit Crank-Nicholson method is used
 *  to solve the PDE.
 ********************************************************************************************************************************************
 */

class hydro_d2: public hydro {
    public:
        hydro_d2(const grid &mesh, const parser &solParam, parallel &mpiParam);

        void solvePDE();

        ~hydro_d2();
};

/**
 ********************************************************************************************************************************************
 *  \class hydro_d2 hydro.h "lib/hydro.h"
 *  \brief The derived class from the hydro base class to solve the incompressible NSE in 2D
 *
 *  Since the class is instantiated when solving the NSE in 2D, the y-direction component of the grid is supressed.
 *  Consequently, the boundary conditions are imposed only on 4 sides of the domain.
 ********************************************************************************************************************************************
 */

class hydro_d3: public hydro {
    public:
        hydro_d3(const grid &mesh, const parser &solParam, parallel &mpiParam);

        void solvePDE();

        ~hydro_d3();
};

/**
 ********************************************************************************************************************************************
 *  \class hydro_d3 hydro.h "lib/hydro.h"
 *  \brief The derived class from the hydro base class to solve the incompressible NSE in 3D
 *
 *  Certain paramters to be used in the implicit calculation of velocity are defined separately from within the class.
 *  Moreover, it imposes boundary conditions on all the three faces of the computational domain.
 ********************************************************************************************************************************************
 */

#endif
