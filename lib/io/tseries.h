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
/*! \file tseries.h
 *
 *  \brief Class declaration of tseries
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef TSERIES_H
#define TSERIES_H

#include "plainsf.h"
#include "sfield.h"
#include "vfield.h"

class tseries {
    public:
        /** The real value for sub-grid energy computed by LES model is used only when LES switch is on */
        real subgridEnergy;

        /** Values momentum and thermal diffusion constants - these are set externally */
        real mDiff, tDiff;

        tseries(const grid &mesh, vfield &solverV, const real &solverTime, const real &timeStep);

        void writeTSHeader();
        void writeTSData();
        void writeTSData(const sfield &T);

        ~tseries();

    private:
        bool maxSwitch;

        int xLow, xTop;
        int yLow, yTop;
        int zLow, zTop;

        real totalVol;
        real divValue;
        real totalKineticEnergy, localKineticEnergy;
        real totalThermalEnergy, localThermalEnergy;
        real totalUzT, localUzT, NusseltNo, ReynoldsNo;

        const real &time, &tStp;

        const grid &mesh;

        vfield &V;

        plainsf divV;

        std::ofstream ofFile;
};

/**
 ********************************************************************************************************************************************
 *  \class tseries tseries.h "lib/io/tseries.h"
 *  \brief Handles the writing of time-series data for various global quantities
 *
 *  The class writes the output into a dat file as well as to the standard I/O.
 *  The class computes various global quantities use library functions and MPI reduce calls.
 ********************************************************************************************************************************************
 */

#endif
