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
/*! \file vfield.cc
 *
 *  \brief Definitions for functions of class vfield - vector field
 *  \sa vfield.h
 *  \author Roshan Samuel, Ali Asad
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "plainsf.h"
#include "plainvf.h"
#include "vfield.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the vfield class
 *
 *          Three instances of field class are initialized.
 *          Each instance corresponds to a component of the vector field.
 *          The fields are appropriately staggered to place them on the cell faces.
 *          The vector field is also assigned a name.
 *
 * \param   gridData is a const reference to the global data in the grid class
 * \param   fieldName is a string value used to name and identify the vector field
 ********************************************************************************************************************************************
 */
vfield::vfield(const grid &gridData, std::string fieldName):
               gridData(gridData),
               Vx(gridData, "Vx"), Vy(gridData, "Vy"), Vz(gridData, "Vz"),
               derVx(gridData, Vx.F), derVy(gridData, Vy.F), derVz(gridData, Vz.F)
{
    this->fieldName = fieldName;

    derivTemp.resize(Vx.fSize);
    derivTemp.reindexSelf(Vx.flBound);

    core = gridData.coreDomain;
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the diffusion term
 *
 *          The diffusion term (grad-squared) is caulculated here.
 *          The second derivatives of each component field are calculated along x, y and z.
 *          These terms are added to the corresponding components of the given plain
 *          vector field (plainvf), which is usually the RHS of the PDE being solved.
 *
 * \param   H is a reference to the plainvf into which the output is written
 ********************************************************************************************************************************************
 */
void vfield::computeDiff(plainvf &H) {
    derivTemp = 0.0;
    derVx.calcDerivative2xx(derivTemp);
    H.Vx(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVx.calcDerivative2yy(derivTemp);
    H.Vx(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVx.calcDerivative2zz(derivTemp);
    H.Vx(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVy.calcDerivative2xx(derivTemp);
    H.Vy(core) += derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative2yy(derivTemp);
    H.Vy(core) += derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative2zz(derivTemp);
    H.Vy(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative2xx(derivTemp);
    H.Vz(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVz.calcDerivative2yy(derivTemp);
    H.Vz(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative2zz(derivTemp);
    H.Vz(core) += derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the convective derivative of the vector field
 *
 *          The function calculates \f$ (\mathbf{u}.\nabla)\mathbf{v} \f$ on the vector field, \f$\mathbf{v}\f$.
 *          To do so, the function needs the vector field (vfield) of velocity, \f$\mathbf{u}\f$.
 *          For each term, there is first an interpolation operation to get the velocity at the
 *          location of data, and a differentiation operation to get the derivatives of the components of vfield.
 *
 * \param   V is a const reference to the vfield denoting convection velocity
 * \param   H is a reference to the plainvf into which the output is written
 ********************************************************************************************************************************************
 */
void vfield::computeNLin(const vfield &V, plainvf &H) {
    // Compute non-linear term for the Vx component
    derivTemp = 0.0;
    derVx.calcDerivative1_x(derivTemp);
    H.Vx(core) -= V.Vx.F(core)*derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVx.calcDerivative1_y(derivTemp);
    H.Vx(core) -= V.Vy.F(core)*derivTemp(core);
#endif

    derivTemp = 0.0;    
    derVx.calcDerivative1_z(derivTemp);
    H.Vx(core) -= V.Vz.F(core)*derivTemp(core);

    // Compute non-linear term for the Vy component
#ifndef PLANAR
    derivTemp = 0.0;
    derVy.calcDerivative1_x(derivTemp);
    H.Vy(core) -= V.Vx.F(core)*derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative1_y(derivTemp);
    H.Vy(core) -= V.Vy.F(core)*derivTemp(core);

    derivTemp = 0.0;
    derVy.calcDerivative1_z(derivTemp);
    H.Vy(core) -= V.Vz.F(core)*derivTemp(core);
#endif

    // Compute non-linear term for the Vz component
    derivTemp = 0.0;
    derVz.calcDerivative1_x(derivTemp);
    H.Vz(core) -= V.Vx.F(core)*derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVz.calcDerivative1_y(derivTemp);
    H.Vz(core) -= V.Vy.F(core)*derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative1_z(derivTemp);
    H.Vz(core) -= V.Vz.F(core)*derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator is used to calculate time step #dt using CFL Condition
 *           
 *          When the parameters specify that time-step must be adaptively
 *          calculated using the Courant Number given in the parameters file,
 *          this function will provide the \f$ dt \f$ using the maximum values
 *          of the components of the vfield.
 *
 * \param   dt is a reference to the real value into which the calculated value of time-step is written
 *********************************************************************************************************************************************
 */
void vfield::computeTStp(real &dt) {
    real locMax, gloMax;

#ifdef PLANAR
    locMax = blitz::max((blitz::abs(Vx.F)/gridData.dXi) +
                        (blitz::abs(Vz.F)/gridData.dZt));
#else
    locMax = blitz::max((blitz::abs(Vx.F)/gridData.dXi) +
                        (blitz::abs(Vy.F)/gridData.dEt) +
                        (blitz::abs(Vz.F)/gridData.dZt));
#endif

    MPI_Allreduce(&locMax, &gloMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

    dt = gridData.inputParams.courantNumber/gloMax;
}

/**
 ********************************************************************************************************************************************
 * \brief   Operator to compute the divergence of the vector field
 *
 *          The operator computes the divergence of a face-centered staggered vector field, and stores it into a cell centered
 *          scalar field as defined by the tensor operation:
 *          \f$ \nabla . \mathbf{v} = \frac{\partial \mathbf{v}}{\partial x} +
 *                                    \frac{\partial \mathbf{v}}{\partial y} +
 *                                    \frac{\partial \mathbf{v}}{\partial z} \f$.
 *
 * \param   divV is a reference to the plain scalar field (plainsf) into which the computed divergence is written.
 ********************************************************************************************************************************************
 */
void vfield::divergence(plainsf &divV) {
    divV = 0.0;

    derivTemp = 0.0;
    derVx.calcDerivative1_x(derivTemp);
    divV.F(core) += derivTemp(core);

#ifndef PLANAR
    derivTemp = 0.0;
    derVy.calcDerivative1_y(derivTemp);
    divV.F(core) += derivTemp(core);
#endif

    derivTemp = 0.0;
    derVz.calcDerivative1_z(derivTemp);
    divV.F(core) += derivTemp(core);
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across all processors when performing parallel computations
 *
 *          Each of the individual field components have their own subroutine, \ref field#syncData "syncData" to send and
 *          receive data across its MPI decomposed sub-domains.
 *          This function calls the \ref field#syncData "syncData" function of its components to update the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
void vfield::syncData() {
    Vx.syncData();
    Vy.syncData();
    Vz.syncData();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the X-component of the vector field
 *
 *          The function first calls the \ref field#syncData "syncData" function of the Vx field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVxBC() {
    Vx.syncData();

    if (not gridData.inputParams.xPer) {
        uLft->imposeBC();
        uRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        uFrn->imposeBC();
        uBak->imposeBC();
    }
#endif
    uTop->imposeBC();
    uBot->imposeBC();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the Y-component of the vector field
 *
 *          The function first calls the \ref field#syncData "syncData" function of the Vy field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVyBC() {
    Vy.syncData();

    if (not gridData.inputParams.xPer) {
        vLft->imposeBC();
        vRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        vFrn->imposeBC();
        vBak->imposeBC();
    }
#endif
    vTop->imposeBC();
    vBot->imposeBC();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for the Z-component of the vector field
 *
 *          The function first calls the \ref field#syncData "syncData" function of the Vz field to update the sub-domain pads.
 *          Then the boundary conditions are applied at the full domain boundaries by calling the imposeBC()
 *          of each boundary class object assigned to each wall.
 *          The order of imposing boundary conditions is - left, right, front, back, bottom and top boundaries.
 *          The corner values are not being imposed specifically and is thus dependent on the above order.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeVzBC() {
    Vz.syncData();

    if (not gridData.inputParams.xPer) {
        wLft->imposeBC();
        wRgt->imposeBC();
    }
#ifndef PLANAR
    if (not gridData.inputParams.yPer) {
        wFrn->imposeBC();
        wBak->imposeBC();
    }
#endif
    wTop->imposeBC();
    wBot->imposeBC();
}

/**
 ********************************************************************************************************************************************
 * \brief   Function to impose the boundary conditions for all the components of the vfield
 *
 *          The function merely calls \ref vfield#imposeVxBC "imposeVxBC", \ref vfield#imposeVyBC "imposeVyBC"
 *          and \ref vfield#imposeVzBC "imposeVzBC" functions together.
 *
 ********************************************************************************************************************************************
 */
void vfield::imposeBCs() {
    imposeVxBC();
#ifndef PLANAR
    imposeVyBC();
#endif
    imposeVzBC();
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given plain vector field
 *
 *          The unary operator += adds a given plain vector field to the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be added to the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator += (plainvf &a) {
    Vx.F += a.Vx;
    Vy.F += a.Vy;
    Vz.F += a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given plain vector field
 *
 *          The unary operator -= subtracts a given plain vector field from the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the plainvf to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator -= (plainvf &a) {
    Vx.F -= a.Vx;
    Vy.F -= a.Vy;
    Vz.F -= a.Vz;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given vector field
 *
 *          The unary operator += adds a given vector field to the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be added to the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator += (vfield &a) {
    Vx.F += a.Vx.F;
    Vy.F += a.Vy.F;
    Vz.F += a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given vector field
 *
 *          The unary operator -= subtracts a given vector field from the vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to the vfield to be subtracted from the member fields
 *
 * \return  A pointer to itself is returned by the vector field object to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator -= (vfield &a) {
    Vx.F -= a.Vx.F;
    Vy.F -= a.Vy.F;
    Vz.F -= a.Vz.F;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to multiply a scalar value to the vector field
 *
 *          The unary operator *= multiplies a real value to all the fields (Vx, Vy and Vz) stored in vfield and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be multiplied to the vector field
 *
 * \return  A pointer to itself is returned by the vector field class to which the operator belongs
 ********************************************************************************************************************************************
 */
vfield& vfield::operator *= (real a) {
    Vx.F *= a;
    Vy.F *= a;
    Vz.F *= a;

    return *this;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a plain vector field to the vector field
 *
 *          The operator = assigns all the three blitz arrays of a plain vector field (plainvf)
 *          to the corresponding arrays in the three fields of the vfield.
 *
 * \param   a is a plainvf to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (plainvf &a) {
    Vx.F = a.Vx;
    Vy.F = a.Vy;
    Vz.F = a.Vz;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign another vector field to the vector field
 *
 *          The operator = assigns all the three fields of a given vector field (vfield)
 *          to the corresponding fields of the vfield.
 *
 * \param   a is a vfield to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (vfield &a) {
    Vx.F = a.Vx.F;
    Vy.F = a.Vy.F;
    Vz.F = a.Vz.F;
}

/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the vector field
 *
 *          The operator = assigns a real value to all the fields (Vx, Vy and Vz) stored in vfield.
 *
 * \param   a is a real number to be assigned to the vector field
 ********************************************************************************************************************************************
 */
void vfield::operator = (real a) {
    Vx.F = a;
    Vy.F = a;
    Vz.F = a;
}
