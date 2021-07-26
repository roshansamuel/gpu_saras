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
/*! \file field.cc
 *
 *  \brief Definitions for functions of class field
 *  \sa field.h
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#include "field.h"

/**
 ********************************************************************************************************************************************
 * \brief   Constructor of the field class
 *
 *          The field class decides the limits necessary for a 3D array to store the
 *          data as per the specified grid staggering details.
 *          It initializes and stores necessary RectDomain objects for getting the
 *          core slice.
 *          Moreover, various offset slices of the core slice, used for performing
 *          finite difference operations, are also defined in this class.
 *          The upper and lower bounds of the array are calculated based on the directions
 *          along which the variable is staggered (or half-indexed).
 *          Finally, a blitz array to store the data of the field is resized according
 *          to the limits and initialized to 0.
 *
 * \param   gridData is a const reference to the global data in the grid class
 ********************************************************************************************************************************************
 */
field::field(const grid &gridData, std::string fieldName): gridData(gridData)
{
    blitz::TinyVector<int, 3> cuBound;

    this->fieldName = fieldName;

    fSize = gridData.fullSize;
    flBound = gridData.fullDomain.lbound();

    F.resize(fSize);
    F.reindexSelf(flBound);

    mpiHandle = new mpidata(F, gridData.rankData);

    core = gridData.coreDomain;
    cuBound = core.ubound();

    setWallSlices();

    mpiHandle->createSubarrays(fSize, cuBound + 1, gridData.padWidths);

    F = 0.0;
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to create the wall slices for the sub-domains
 *
 *          The wall slices of the sub-domain are used for imposing the boundary conditions.
 *          Hence they are important only for the near-boundary sub-domains in parallel computations.
 *          Moreover, only the staggered (half-indexed) grid has points on the boundary.
 *          In this regard, SARAS uses a slightly unconventional approach where the boundary of the
 *          full-domain passes through the cell-centers of the boundary cells.
 *          This has two advantages: i) no-slip BC is strictly enforced for two face centered velocity
 *          components, while only no-penetration BC is enforced through averaging,
 *          ii) there will be equal number of staggered (half-indexed) and collocated (full-indexed)
 *          points in all the MPI sub-domains, such that geometric multigrid operations, as well as
 *          computations in the PDE solver are load-balanced.
 *
 *          The collocated (full-indexed) grid points lie on either side of the domain boundaries.
 *          As a result the wall slices are defined only for those fields for which at least one
 *          of \ref field#xStag "xStag", \ref field#yStag "yStag" or \ref field#zStag "zStag" is false
 ********************************************************************************************************************************************
 */
void field::setWallSlices() {
    blitz::Array<blitz::TinyVector<int, 3>, 1> wlBound;
    blitz::Array<blitz::TinyVector<int, 3>, 1> wuBound;

    // 6 slices are stored in fWalls corresponding to the 6 faces of the 3D box
    fWalls.resize(6);

    wlBound.resize(6);
    wuBound.resize(6);

    // Wall slices are the locations where the BC (both Neumann and Dirichlet) is imposed.
    // In the places where these slices are being used, they should be on the LHS of equation.
    for (int i=0; i<6; i++) {
        wlBound(i) = F.lbound();
        wuBound(i) = F.ubound();
    }

    // The core slice corresponds to the part of the fluid within which all variables are computed at each time step.
    // Correspondingly, the boundary conditions are imposed on the layer just outside the core

    // UPPER BOUNDS OF LEFT WALL
    wlBound(0)(0) = wuBound(0)(0) = core.lbound(0) - 1;

    // LOWER BOUNDS OF RIGHT WALL
    wuBound(1)(0) = wlBound(1)(0) = core.ubound(0) + 1;

    // UPPER BOUNDS OF FRONT WALL
    wlBound(2)(1) = wuBound(2)(1) = core.lbound(1) - 1;

    // LOWER BOUNDS OF BACK WALL
    wuBound(3)(1) = wlBound(3)(1) = core.ubound(1) + 1;

    // UPPER BOUNDS OF BOTTOM WALL
    wlBound(4)(2) = wuBound(4)(2) = core.lbound(2) - 1;

    // LOWER BOUNDS OF TOP WALL
    wuBound(5)(2) = wlBound(5)(2) = core.ubound(2) + 1;

    for (int i=0; i<6; i++) {
        fWalls(i) = blitz::RectDomain<3>(wlBound(i), wuBound(i));
    }
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across all processors when performing parallel computations
 *
 *          This function calls the \ref mpidata#syncData "syncData" function of mpidata class to
 *          perform data-transfer and thus update the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
void field::syncData() {
    mpiHandle->syncData();
}


/**
 ********************************************************************************************************************************************
 * \brief   Function to extract the maximum value from the field
 *
 *          The function uses the in-built blitz function to obtain the maximum value in an array.
 *          Note that this function *takes the maximum of the absolute value* of the field.
 *          While performing parallel computation, the function performs an <B>MPI_Allreduce()</B> to get
 *          the global maximum from the entire computational domain.
 *
 * \return  The real value of the maximum is returned (it is implicitly assumed that only real values are used)
 ********************************************************************************************************************************************
 */
real field::fieldMax() {
    real localMax, globalMax;

    localMax = blitz::max(blitz::abs(F));

    /***************************************************************************************************************
     * DID YOU KNOW?                                                                                               *
     * In the line above, most compilers will not complain even if you omitted the namespace specification blitz:: *
     * This behaviour wasted an hour of my development time (including the effort of making this nice box).        *
     * Check Ref. [2] in README for explanation.                                                                   *
     ***************************************************************************************************************/

    MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

    return globalMax;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given field
 *
 *          The unary operator += adds a given field to the field stored by the class and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another to be added to the member field
 *
 * \return  A pointer to itself is returned by the field class to which the operator belongs
 ********************************************************************************************************************************************
 */
field& field::operator += (field &a) {
    F += a.F;

    return *this;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given field
 *
 *          The unary operator -= subtracts a given field from the field stored by the class and returns
 *          a pointer to itself.
 *
 * \param   a is a reference to another field to be deducted from the member field
 *
 * \return  A pointer to itself is returned by the field class to which the operator belongs
 ********************************************************************************************************************************************
 */
field& field::operator -= (field &a) {
    F -= a.F;

    return *this;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to add a given scalar value
 *
 *          The unary operator += adds a given constant scalar value to the field stored by the class and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be added to the field
 *
 * \return  A pointer to itself is returned by the field class to which the operator belongs
 ********************************************************************************************************************************************
 */
field& field::operator += (real a) {
    F += a;

    return *this;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to subtract a given scalar value
 *
 *          The unary operator -= subtracts a given constant scalar value from the field stored by the class and returns
 *          a pointer to itself.
 *
 * \param   a is a real number to be subtracted from the field
 *
 * \return  A pointer to itself is returned by the field class to which the operator belongs
 ********************************************************************************************************************************************
 */
field& field::operator -= (real a) {
    F -= a;

    return *this;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a scalar value to the field
 *
 *          The operator = assigns a real value to the entire field.
 *
 * \param   a is a real number to be assigned to the field
 ********************************************************************************************************************************************
 */
void field::operator = (real a) {
    F = a;
}


/**
 ********************************************************************************************************************************************
 * \brief   Overloaded operator to assign a field to the field
 *
 *          The operator = copies the contents of the input field to itself.
 *
 * \param   a is the field to be assigned to the field
 ********************************************************************************************************************************************
 */
void field::operator = (field &a) {
    F = a.F;
}


field::~field() { }
