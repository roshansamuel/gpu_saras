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
/*! \file uniformRandom.cc
 *
 *  \brief Definitions for functions of class initial
 *  \sa initial.h
 *  \author Roshan Samuel
 *  \date Sep 2020
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include <ctime>
#include <cstdlib>
#include "initial.h"

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose random initial condition for channel flow on the given input velocity field.
 *
 *          The function generates random values for the entire domain (with independent seeds for individual ranks).
 *          The random values chosen between -1.0 to 1.0 are first scaled by a factor.
 *          Then they are added to the computed uniform mean velocity field.
 *
 ********************************************************************************************************************************************
 */
void uniformRandom::initializeField(vfield &uField) {
    if (mesh.rankData.rank == 0) std::cout << "Imposing uniform random initial condition for channel flow" << std::endl << std::endl;

    // Seed the random number generator with both time and rank to get different random numbers in different MPI sub-domains
    int randSeed = std::time(0) + mesh.rankData.rank;
    std::srand(randSeed);

    // Scaling factor for random velocity fluctuations
    real vScale = mesh.inputParams.meanVelocity*mesh.inputParams.rfIntensity/100.0;

    // Random velocity fluctuation to be computed point-wise
    real vRandom;

#ifdef PLANAR
    // X-VELOCITY
    for (int i=uField.Vx.F.lbound(0); i <= uField.Vx.F.ubound(0); i++) {
        for (int k=uField.Vx.F.lbound(2); k <= uField.Vx.F.ubound(2); k++) {
            vRandom = vScale*(real(std::rand())/RAND_MAX - 0.5);

            uField.Vx.F(i, 0, k) = mesh.inputParams.meanVelocity + vRandom;
        }
    }

    // Z-VELOCITY
    for (int i=uField.Vz.F.lbound(0); i <= uField.Vz.F.ubound(0); i++) {
        for (int k=uField.Vz.F.lbound(2); k <= uField.Vz.F.ubound(2); k++) {
            vRandom = vScale*(real(std::rand())/RAND_MAX - 0.5);

            uField.Vz.F(i, 0, k) = vRandom;
        }
    }
#else
    // X-VELOCITY
    for (int i=uField.Vx.F.lbound(0); i <= uField.Vx.F.ubound(0); i++) {
        for (int j=uField.Vx.F.lbound(1); j <= uField.Vx.F.ubound(1); j++) {
            for (int k=uField.Vx.F.lbound(2); k <= uField.Vx.F.ubound(2); k++) {
                vRandom = vScale*(real(std::rand())/RAND_MAX - 0.5);

                uField.Vx.F(i, j, k) = mesh.inputParams.meanVelocity + vRandom;
            }
        }
    }

    // Y-VELOCITY
    for (int i=uField.Vy.F.lbound(0); i <= uField.Vy.F.ubound(0); i++) {
        for (int j=uField.Vy.F.lbound(1); j <= uField.Vy.F.ubound(1); j++) {
            for (int k=uField.Vy.F.lbound(2); k <= uField.Vy.F.ubound(2); k++) {
                vRandom = vScale*(real(std::rand())/RAND_MAX - 0.5);

                uField.Vy.F(i, j, k) = vRandom;
            }
        }
    }

    // Z-VELOCITY
    for (int i=uField.Vz.F.lbound(0); i <= uField.Vz.F.ubound(0); i++) {
        for (int j=uField.Vz.F.lbound(1); j <= uField.Vz.F.ubound(1); j++) {
            for (int k=uField.Vz.F.lbound(2); k <= uField.Vz.F.ubound(2); k++) {
                vRandom = vScale*(real(std::rand())/RAND_MAX - 0.5);

                uField.Vz.F(i, j, k) = vRandom;
            }
        }
    }
#endif
}
