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
/*! \file plainsf.cc
 *
 *  \brief Definitions for functions of class plainsf - plain scalar field
 *  \sa plainsf.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainsf.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the plainsf class
 *
 *          The instance of the field class to store the data of the scalar field is initialized, and the necessary grid
 *          transformation derivatives along each direction are chosen according to the grid staggering.
 *          The arrays to store the output from various operators like derivatives, convective derivatives, etc. are also
 *          allocated.
 *          Finally, an instance of the <B>mpidata</B> class is initialized to store the sub-arrays to be sent/received
 *          across the processors during MPI communication.
 *
 * \param   gridData is a const reference to the global data contained in the grid class
 ********************************************************************************************************************************************
 */
plainsf::plainsf(const grid &gridData): gridData(gridData) {
    blitz::TinyVector<int, 3> dSize = gridData.fullDomain.ubound() - gridData.fullDomain.lbound() + 1;
    blitz::TinyVector<int, 3> dlBnd = gridData.fullDomain.lbound();

    F.resize(dSize);
    F.reindexSelf(dlBnd);
    F = 0.0;

    derS = new derivative(gridData, F);

    derivTemp.resize(dSize);
    derivTemp.reindexSelf(dlBnd);

    core = gridData.coreDomain;

    mpiHandle = new mpidata(F, gridData.rankData);
    mpiHandle->createSubarrays(dSize, core.ubound() + 1, gridData.padWidths);
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator to compute the gradient of the plain scalar field
 *
 *          The gradient operator computes the gradient of the scalar field, and stores it into a 
 *          plain vector field as defined by the tensor operation:
 *          \f$ \nabla f = \frac{\partial f}{\partial x}i + \frac{\partial f}{\partial y}j + \frac{\partial f}{\partial z}k \f$.
 *
 * \param   gradF is a reference to a plain vector field (plainvf) into which the computed gradient must be written.
 ********************************************************************************************************************************************
 */
void plainsf::gradient(plainvf &gradF) {
    derivTemp = 0.0;
    derS->calcDerivative1_x(derivTemp);
    gradF.Vx(core) = derivTemp(core);
#ifndef PLANAR
    derivTemp = 0.0;
    derS->calcDerivative1_y(derivTemp);
    gradF.Vy(core) = derivTemp(core);
#endif
    derivTemp = 0.0;
    derS->calcDerivative1_z(derivTemp);
    gradF.Vz(core) = derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to multiply a given plainsf by a constant and add it to the result
 *
 *          The function serves to simplify the operation a = a + k*b.
 *          It combines the unary += operator with multiplication by scalar.
 *
 * \param   a is a const reference to the plainsf to be added to the member fields
 * \param   k is the real value to be multiplied to a before adding it to the member fields
 *
 * \return  A pointer to itself is returned by the plain vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::multAdd(const plainsf &a, real k) {
    F += k*a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain scalar field
 *
 *          The unary operator += adds a given plain scalar field to the plainsf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another plainsf to be added to the member field
 *
 * \return  A pointer to itself is returned by the plain scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::operator += (plainsf &a) {
    F += a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain scalar field
 *
 *          The unary operator -= subtracts a given plain scalar field from the plainsf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another plainsf to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the plain scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::operator -= (plainsf &a) {
    F -= a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given scalar field
 *
 *          The unary operator += adds a given scalar field to the plainsf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be added to the member field
 *
 * \return  A pointer to itself is returned by the plain scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::operator += (sfield &a) {
    F += a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given scalar field
 *
 *          The unary operator -= subtracts a given scalar field from the plainsf and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the plain scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::operator -= (sfield &a) {
    F -= a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the scalar field
 *
 *          The unary operator *= multiplies a real value to the plainsf and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the scalar field
 *
 * \return  A pointer to itself is returned by the plain scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
plainsf& plainsf::operator *= (real a) {
    F *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another plain scalar field to the plain scalar field
 *
 *          The operator = copies the contents of the input plain scalar field to itself.
 *
 * \param   a is a plainsf to be assigned to the plain scalar field
 ********************************************************************************************************************************************
 */
void plainsf::operator = (plainsf &a) {
    F = a.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another scalar field to the plain scalar field
 *
 *          The operator = copies the contents of the input scalar field to itself.
 *
 * \param   a is a sfield to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void plainsf::operator = (sfield &a) {
    F = a.F.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the plain scalar field
 *
 *          The operator = assigns a real value to all the scalar field.
 *
 * \param   a is a real number to be assigned to the plain scalar field
 ********************************************************************************************************************************************
 */
void plainsf::operator = (real a) {
    F = a;
}
