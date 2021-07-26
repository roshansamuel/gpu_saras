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
/*! \file temperatureProfiles.cc
 *
 *  \brief Definitions of different initial temperature profiles
 *  \sa initial.h
 *  \author Roshan Samuel
 *  \date Feb 2021
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "initial.h"

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose linear profile on temperature and zero velocity condition
 *
 *          Depending on the preprocessor flag PLANAR, the function applies the
 *          initial conditions for both 2D and 3D cases.
 *
 ********************************************************************************************************************************************
 */
void linearProfile::initializeField(vfield &uField, sfield &tField) {
    if (mesh.rankData.rank == 0) std::cout << "Imposing linear temperature profile" << std::endl << std::endl;

    uField.Vx = 0.0;
#ifdef PLANAR
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
            tField.F.F(i, 0, k) = 1.0 - mesh.z(k)/mesh.zLen;
        }
    }
#else
    uField.Vy = 0.0;
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int j=tField.F.F.lbound(1); j <= tField.F.F.ubound(1); j++) {
            for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
                tField.F.F(i, j, k) = 1.0 - mesh.z(k)/mesh.zLen;
            }
        }
    }
#endif
    uField.Vz = 0.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to impose cosine profile on temperature and zero velocity condition
 *
 *          Depending on the preprocessor flag PLANAR, the function applies the
 *          initial conditions for both 2D and 3D cases.
 *
 ********************************************************************************************************************************************
 */
void cosineProfile::initializeField(vfield &uField, sfield &tField) {
    if (mesh.rankData.rank == 0) std::cout << "Imposing cosine temperature profile" << std::endl << std::endl;

    uField.Vx = 0.0;
#ifdef PLANAR
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
            tField.F.F(i, 0, k) = (1.0 + cos(M_PI*mesh.z(k)/mesh.zLen))/2.0;
        }
    }
#else
    uField.Vy = 0.0;
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int j=tField.F.F.lbound(1); j <= tField.F.F.ubound(1); j++) {
            for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
                tField.F.F(i, j, k) = (1.0 + cos(M_PI*mesh.z(k)/mesh.zLen))/2.0;
            }
        }
    }
#endif
    uField.Vz = 0.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to impose sine profile on temperature and zero velocity condition
 *
 *          Depending on the preprocessor flag PLANAR, the function applies the
 *          initial conditions for both 2D and 3D cases.
 *
 ********************************************************************************************************************************************
 */
void sineProfile::initializeField(vfield &uField, sfield &tField) {
    if (mesh.rankData.rank == 0) std::cout << "Imposing sine temperature profile" << std::endl << std::endl;

    uField.Vx = 0.0;
#ifdef PLANAR
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
            tField.F.F(i, 0, k) = (1.0 - 0.2*sin(2.0*M_PI*mesh.z(k)/mesh.zLen) - mesh.z(k)/mesh.zLen);
        }
    }
#else
    uField.Vy = 0.0;
    for (int i=tField.F.F.lbound(0); i <= tField.F.F.ubound(0); i++) {
        for (int j=tField.F.F.lbound(1); j <= tField.F.F.ubound(1); j++) {
            for (int k=tField.F.F.lbound(2); k <= tField.F.F.ubound(2); k++) {
                tField.F.F(i, j, k) = (1.0 - 0.2*sin(2.0*M_PI*mesh.z(k)/mesh.zLen) - mesh.z(k)/mesh.zLen);
            }
        }
    }
#endif
    uField.Vz = 0.0;
}
