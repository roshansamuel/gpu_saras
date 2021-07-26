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
/*! \file hotPlate.cc
 *
 *  \brief Definitions for functions of class boundary
 *  \sa boundary.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "boundary.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the hotPlate class
 *
 *          The constructor initializes the base boundary class using part of the arguments supplied to it.
 *          The radius of the heating plate, denoted by patchRadius, is also set in the initialization list.
 *
 * \param   mesh is a const reference to the global data contained in the grid class
 * \param   inField is a reference to scalar field to which the boundary conditions must be applied.
 * \param   bcWall is a const integer which specifies the wall to which the BC must be applied.
 * \param   plateRad is the const real value of the radius of the heating plate.
 ********************************************************************************************************************************************
 */
hotPlate::hotPlate(const grid &mesh, field &inField, const int bcWall, const real plateRad):
                            boundary(mesh, inField, bcWall), patchRadius(plateRad) {
    createPatch(patchRadius);

    dataSlice = mesh.shift(shiftDim, dField.fWalls(wallNum), 1);
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to create a heating patch for non-homogeneous boundary conditions
 *
 *          The current implementation is for a conducting circular heating plate surrounded by adiabatic insulated walls.
 *
 ********************************************************************************************************************************************
 */
void hotPlate::createPatch(const real patchRadius) {
    real patchCentX, patchCentY, patchCentZ;

    patchCentX = mesh.xLen/2.0;
    patchCentY = mesh.yLen/2.0;
    patchCentZ = 0.0;

    wallData.resize(blitz::TinyVector<int, 3>(dField.fWalls(wallNum).ubound() - dField.fWalls(wallNum).lbound() + 1));
    wallData.reindexSelf(dField.fWalls(wallNum).lbound());
    wallData = 0.0;

    wallMask.resize(blitz::TinyVector<int, 3>(dField.fWalls(wallNum).ubound() - dField.fWalls(wallNum).lbound() + 1));
    wallMask.reindexSelf(dField.fWalls(wallNum).lbound());
    wallMask = true;

    for (int iX = dField.fWalls(wallNum).lbound(0); iX <= dField.fWalls(wallNum).ubound(0); iX++) {
        for (int iY = dField.fWalls(wallNum).lbound(1); iY <= dField.fWalls(wallNum).ubound(1); iY++) {
            for (int iZ = dField.fWalls(wallNum).lbound(2); iZ <= dField.fWalls(wallNum).ubound(2); iZ++) {
                real ptRadius = sqrt(pow((mesh.x(iX) - patchCentX), 2) + pow((mesh.y(iY) - patchCentY), 2) + pow((mesh.z(iZ) - patchCentZ), 2));
                if (ptRadius <= patchRadius) {
                    wallData(iX, iY, iZ) = 1.0;
                    wallMask(iX, iY, iZ) = false;
                }
            }
        }
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to impose Mixed BC of a heating plate on a cell centered variable
 *
 *          For Saras solver, the wall passes through the cell centers of the variables.
 *          Hence the variable is lying on the wall for this case.
 *          Accordingly the value of the variable is directly set to 1.0 on the conducting sections of the wall.
 *          At the rest of the wall, Neumann BC is imposed using a boolean wallMask variable.
 *
 ********************************************************************************************************************************************
 */
void hotPlate::imposeBC() {
    // First impose Neumann BC everywhere for adiabatic wall
    dField.F(wallSlice) = dField.F(dataSlice);

    // Now in the area of the circular patch, make all values 0 in order to apply conducting BC
    dField.F(wallSlice) = dField.F(wallSlice)*wallMask;

    // Finally apply conducting BC in the circular patch alone
    dField.F(wallSlice) = dField.F(wallSlice) + wallData*(2.0*wallData - dField.F(dataSlice));
}
