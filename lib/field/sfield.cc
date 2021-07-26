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
/*! \file sfield.cc
 *
 *  \brief Definitions for functions of class sfield - scalar field
 *  \sa sfield.h
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainsf.h"
#include "plainvf.h"
#include "sfield.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the sfield class
 *
 *          One instance of the field class to store the data of the scalar field is initialized.
 *          The field is initialized with appropriate grid staggering to place the scalar on the cell centres.
 *          The name for the vector field as given by the user is also assigned.
 *
 * \param   gridData is a const reference to the global data contained in the grid class
 * \param   fieldName is a string value set by the user to name and identify the scalar field
 ********************************************************************************************************************************************
 */
sfield::sfield(const grid &gridData, std::string fieldName):
               gridData(gridData),
               F(gridData, fieldName),
               derS(gridData, F.F)
{
    this->fieldName = fieldName;

    derivTemp.resize(F.fSize);
    derivTemp.reindexSelf(F.flBound);

    core = gridData.coreDomain;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the diffusion term
 *
 *          It is assumed that the velocity is specified at face-centers, as required by the \ref sfield#computeNLin
 *          "computeNLin" function of sfield.
 *
 * \param   H is a reference to the plain scalar field (plainsf) to which the output will be written
 ********************************************************************************************************************************************
 */
void sfield::computeDiff(plainsf &H) {
    derivTemp = 0.0;
    derS.calcDerivative2xx(derivTemp);
    H.F(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derS.calcDerivative2yy(derivTemp);
    H.F(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derS.calcDerivative2zz(derivTemp);
    H.F(core) += derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the scalar field
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)f \f$ at the grid nodes of the scalar field f.
 *          To do so, the function needs the vector field (vfield) of velocity. It is assumed that the velocity is always
 *          specified at face-centers, and is interpolated accordingly to the scalar field grid points.
 *
 * \param   V is a const reference to a vector field (vfield) that specifies the convection velocity
 ********************************************************************************************************************************************
 */
void sfield::computeNLin(const vfield &V, plainsf &H) {
    derivTemp = 0.0;
    derS.calcDerivative1_x(derivTemp);
    H.F(core) -= V.Vx.F(core)*derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derS.calcDerivative1_y(derivTemp);
    H.F(core) -= V.Vy.F(core)*derivTemp(core);
#endif

    derivTemp = 0.0;
    derS.calcDerivative1_z(derivTemp);
    H.F(core) -= V.Vz.F(core)*derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator to compute the gradient of the scalar field
 *
 *          The function computes the gradient of the cell centered scalar field, and stores it into a face-centered plainvf:
 *          \f$ \nabla f = \frac{\partial f}{\partial x}i + \frac{\partial f}{\partial y}j + \frac{\partial f}{\partial z}k \f$.
 *
 * \param   gradF is a reference to a plain vector field (plainvf) into which the computed gradient must be written.
 ********************************************************************************************************************************************
 */
void sfield::gradient(plainvf &gradF) {
    derivTemp = 0.0;
    derS.calcDerivative1_x(derivTemp);
    gradF.Vx(core) = derivTemp(core);
#ifndef PLANAR
    derivTemp = 0.0;
    derS.calcDerivative1_y(derivTemp);
    gradF.Vy(core) = derivTemp(core);
#endif
    derivTemp = 0.0;
    derS.calcDerivative1_z(derivTemp);
    gradF.Vz(core) = derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across all processors when performing parallel computations
 *
 *          This function calls the \ref mpidata#syncData "syncData" function of mpidata class to
 *          perform data-transfer and thus update the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
void sfield::syncData() {
    F.syncData();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the scalar field
 *
 *          The function first calls the syncData() function of the field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *
 ********************************************************************************************************************************************
 */
void sfield::imposeBCs() {
    F.syncData();

    if (not gridData.inputParams.xPer) {
        tLft->imposeBC();
        tRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        tFrn->imposeBC();
        tBak->imposeBC();
    }
#endif
    tTop->imposeBC();
    tBot->imposeBC();
};

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain scalar field
 *
 *          The unary operator += adds a given plain scalar field to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to a plainsf to be added to the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator += (plainsf &a) {
    F.F += a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain scalar field
 *
 *          The unary operator -= subtracts a given plain scalar field from the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to a plainsf to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator -= (plainsf &a) {
    F.F -= a.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given scalar field
 *
 *          The unary operator += adds a given scalar field to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be added to the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator += (sfield &a) {
    F.F += a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given scalar field
 *
 *          The unary operator -= subtracts a given scalar field from the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another sfield to be subtracted from the member field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator -= (sfield &a) {
    F.F -= a.F.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the scalar field
 *
 *          The unary operator *= multiplies a real value to the sfield and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the scalar field
 *
 * \return  A pointer to itself is returned by the scalar field class to which the operator belongs
 ********************************************************************************************************************************************
 */
sfield& sfield::operator *= (real a) {
    F.F *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a plain scalar field to the scalar field
 *
 *          The operator = copies the contents of the input plain scalar field to itself.
 *
 * \param   a is the plainsf to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (plainsf &a) {
    F.F = a.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar field to the scalar field
 *
 *          The operator = copies the contents of the input scalar field to itself.
 *
 * \param   a is the scalar field to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (sfield &a) {
    F.F = a.F.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the scalar field
 *
 *          The operator = assigns a real value to all the scalar field.
 *
 * \param   a is a real number to be assigned to the scalar field
 ********************************************************************************************************************************************
 */
void sfield::operator = (real a) {
    F.F = a;
}
