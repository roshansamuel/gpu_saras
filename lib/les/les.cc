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
/*! \file les.cc
 *
 *  \brief Definitions for functions of class les
 *  \sa les.h
 *  \author Roshan Samuel
 *  \date Sep 2020
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "les.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the les class
 *
 *          The empty constructor merely initializes the local reference to the global mesh variable.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 ********************************************************************************************************************************************
 */
les::les(const grid &mesh): mesh(mesh) {
    core = mesh.coreDomain;
}

/**
 ********************************************************************************************************************************************
 * \brief   Prototype overloaded function to compute sub-grid contribution from LES model
 *
 *          The function uses the chosen LES model to compute the sub-grid stress terms, and add their contribution
 *          to the momentum equations of the Navier-Stokes equations.
 *
 * \param   nseRHS is a reference to the plain vector field which holds the RHS terms of the NSE
 ********************************************************************************************************************************************
 */
real les::computeSG(plainvf &nseRHS, vfield &V) { return 0; };

/**
 ********************************************************************************************************************************************
 * \brief   Prototype overloaded function to compute sub-grid contribution from LES model
 *
 *          The function uses the chosen LES model to compute the sub-grid stress terms, and add their contribution
 *          to both the momentum equation, as well as the temperature equation.
 *
 * \param   nseRHS is a reference to the plain vector field which holds the RHS terms of the NSE
 * \param   tmpRHS is a reference to the plain scalar field which holds the RHS terms of the temperature equation
 * \param   T is a const reference to the temperature scalar field
 ********************************************************************************************************************************************
 */
real les::computeSG(plainvf &nseRHS, plainsf &tmpRHS, vfield &V, sfield &T) { return 0; };
