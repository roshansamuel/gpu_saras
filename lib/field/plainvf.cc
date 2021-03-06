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
/*! \file plainvf.cc
 *
 *  \brief Definitions for functions of class plainvf - plain vector field
 *  \sa plainvf.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainvf.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the plainvf class
 *
 *          Three blitz arrays to store the data of the three components are initialized.
 *          The name for the plain vector field as given by the user is also assigned.
 *
 * \param   gridData is a const reference to the global data in the grid class
 ********************************************************************************************************************************************
 */
plainvf::plainvf(const grid &gridData): gridData(gridData) {
    blitz::TinyVector<int, 3> dSize = gridData.fullDomain.ubound() - gridData.fullDomain.lbound() + 1;
    blitz::TinyVector<int, 3> dlBnd = gridData.fullDomain.lbound();
    blitz::RectDomain<3> core = gridData.coreDomain;

    Vx.resize(dSize);
    Vx.reindexSelf(dlBnd);
    Vx = 0.0;

    mpiVxData = new mpidata(Vx, gridData.rankData);
    mpiVxData->createSubarrays(dSize, core.ubound() + 1, gridData.padWidths);

    Vy.resize(dSize);
    Vy.reindexSelf(dlBnd);
    Vy = 0.0;

    mpiVyData = new mpidata(Vy, gridData.rankData);
    mpiVyData->createSubarrays(dSize, core.ubound() + 1, gridData.padWidths);

    Vz.resize(dSize);
    Vz.reindexSelf(dlBnd);
    Vz = 0.0;

    mpiVzData = new mpidata(Vz, gridData.rankData);
    mpiVzData->createSubarrays(dSize, core.ubound() + 1, gridData.padWidths);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to multiply a given plainvf by a constant and add it to the result
 *
 *          The function serves to simplify the operation a = a + k*b.
 *          It combines the unary += operator with multiplication by scalar.
 *
 * \param   a is a const reference to the plainvf to be added to the member fields
 * \param   k is the real value to be multiplied to a before adding it to the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::multAdd(const plainvf &a, real k) {
    Vx += k*a.Vx;
    Vy += k*a.Vy;
    Vz += k*a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain vector field
 *
 *          The unary operator += adds a given plain vector field to the plainvf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be added to the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::operator += (plainvf &a) {
    Vx += a.Vx;
    Vy += a.Vy;
    Vz += a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain vector field
 *
 *          The unary operator -= subtracts a given plain vector field from the plainvf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::operator -= (plainvf &a) {
    Vx -= a.Vx;
    Vy -= a.Vy;
    Vz -= a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given vector field
 *
 *          The unary operator += adds a given vector field to the plainvf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be added to the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::operator += (vfield &a) {
    Vx += a.Vx.F;
    Vy += a.Vy.F;
    Vz += a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given vector field
 *
 *          The unary operator -= subtracts a given vector field from the plainvf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::operator -= (vfield &a) {
    Vx -= a.Vx.F;
    Vy -= a.Vy.F;
    Vz -= a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the plain vector field
 *
 *          The unary operator *= multiplies a real value to all the fields (Vx, Vy and Vz) stored in plainvf and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the plain vector field
 *
 * \return  A pointer to itself is returned by the plain vector field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainvf& plainvf::operator *= (real a) {
    Vx *= a;
    Vy *= a;
    Vz *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another plain vector field to the plain vector field
 *
 *          The operator = assigns all the three blitz arrays of a plain vector field (plainvf)
 *          to the corresponding three arrays of the plainvf.
 *
 * \param   a is a plainvf to be assigned to the plain vector field
 ********************************************************************************************************************************************
 */
void plainvf::operator = (plainvf &a) {
    Vx = a.Vx;
    Vy = a.Vy;
    Vz = a.Vz;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another vector field to the plain vector field
 *
 *          The operator = assigns all the three fields of a given vector field (vfield)
 *          to the corresponding three arrays of the plainvf.
 *
 * \param   a is a vfield to be assigned to the plain vector field
 ********************************************************************************************************************************************
 */
void plainvf::operator = (vfield &a) {
    Vx = a.Vx.F;
    Vy = a.Vy.F;
    Vz = a.Vz.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the plain vector field
 *
 *          The operator = assigns a real value to all the fields (Vx, Vy and Vz) stored in plainvf.
 *
 * \param   a is a real number to be assigned to the plain vector field
 ********************************************************************************************************************************************
 */
void plainvf::operator = (real a) {
    Vx = a;
    Vy = a;
    Vz = a;
}
