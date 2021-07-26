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
/*! \file grid.h
 *
 *  \brief Class declaration of grid
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef GRID_H
#define GRID_H

#include <vector>

#include "parallel.h"

class grid {
    private:
        /** Grid stretching parameter for tangent-hyperbolic function along x, y and z directions */
        blitz::TinyVector<real, 3> thBeta;

        /** Arrays of the local values of \f$ \xi \f$, \f$ \eta \f$ and \f$ \zeta \f$ within sub-domains in the transformed plane */
        //@{
        blitz::Array<real, 1> xi, et, zt;
        //@}

        /** Arrays of the global values of \f$ \xi \f$, \f$ \eta \f$ and \f$ \zeta \f$ within the full domain in the transformed plane */
        //@{
        blitz::Array<real, 1> xiGlo, etGlo, ztGlo;
        //@}

        // Print flag - basically flag to ease printing to I/O. It is true for root rank (0)
        bool pf;

        void resizeGrid();
        void makeSizeArray();
        void setDomainSizes();
        void createXiEtaZeta();

        void createUniformGrid();
        void createTanHypGrid(int dim, blitz::Array<real, 1> xGlo, blitz::Array<real, 1> xiGl);

        void mgGridMetrics();
        void mgGridMetrics(int dim);

        void checkAnisotropy();
        void computeGlobalLimits();

    public:
        /** A const reference to the global variables stored in the parser class to access user set parameters */
        const parser &inputParams;

        /** A const reference to the global variables stored in the parallel class to access MPI related parameters */
        const parallel &rankData;

        /** Total number of points (cell-centers) in the full global domain */
        int totalPoints;

        /** The sizes of the core of MPI decomposed sub-domains without the pads (staggered points) */
        blitz::TinyVector<int, 3> coreSize;

        /** The sizes of the MPI decomposed sub-domains including the pads on all sides (staggered points) - staggrCoreSize + 2*padWidths */
        blitz::TinyVector<int, 3> fullSize;

        /** The sizes of the pad widths along the three directions - padX, padY, padZ */
        blitz::TinyVector<int, 3> padWidths;

        /** The size of the entire computational domain excluding the pads at the boundary of full domain - globalNx, globalNy, globalNz */
        blitz::TinyVector<int, 3> globalSize;

        /** The end indices of the MPI decomposed sub-domains within the global indexing of the full staggered grid - ztEn, etEn, xiEn */
        blitz::TinyVector<int, 3> subarrayEnds;

        /** The start indices of the MPI decomposed sub-domains within the global indexing of the full staggered grid - ztSt, etSt, xiSt */
        blitz::TinyVector<int, 3> subarrayStarts;

        /** Grid spacings in the transformed plane along the \f$ \xi \f$, \f$ \eta \f$ and \f$ \zeta \f$ directions */
        //@{
        real dXi, dEt, dZt;
        //@}

        /** Lengths of the physical computational domain along the x, y and z directions */
        //@{
        const real xLen, yLen, zLen;
        //@}

        /** Array of collocated grid sizes such that the corresponding staggered grid will be multi-grid compatible */
        blitz::Array<int, 1> sizeArray;

        /** Vector of indices pointing to the <B>sizeArray</B> that determines the global full domain size along the 3 directions */
        blitz::TinyVector<int, 3> sizeIndex;

        /** RectDomain object that defines the slice for the core of the local MPI decomposed sub-domain (staggered points) */
        blitz::RectDomain<3> coreDomain;

        /** RectDomain object that defines the slice for the full extent of the local MPI decomposed sub-domain (staggered points) */
        blitz::RectDomain<3> fullDomain;

        /*****************************************************************************************************************************************************/

        /** Grids along the x, y and z directions defined locally within MPI decomposed sub-domains */
        //@{
        blitz::Array<real, 1> x, y, z;
        //@}

        /*****************************************************************************************************************************************************/

        /** Grids points along the x, y and z directions for the global domain in physical plane */
        //@{
        blitz::Array<real, 1> xGlobal, yGlobal, zGlobal;
        //@}

        /*****************************************************************************************************************************************************/

        /** Arrays of grid derivative terms along \f$ \xi \f$ direction, local to each sub-domain. */
        //@{
        blitz::Array<real, 1> xi_x, xixx, xix2;
        //@}

        /** Arrays of grid derivative terms along \f$ \eta \f$ direction, local to each sub-domain. */
        //@{
        blitz::Array<real, 1> et_y, etyy, ety2;
        //@}

        /** Arrays of grid derivative terms along \f$ \zeta \f$ direction, local to each sub-domain. */
        //@{
        blitz::Array<real, 1> zt_z, ztzz, ztz2;
        //@}

        /** Global metrics along all 3 directions for multi-grid solver, stored in a single array of arrays */
        blitz::Array<blitz::Array<real, 1>, 1> globalMetrics;

        /*****************************************************************************************************************************************************/

        grid(const parser &solParam, parallel &parallelData);

        /**
        ********************************************************************************************************************************************
        * \brief   Function to shift a blitz RectDomain object by a given number of steps along a specified dimension.
        *
        *          The RectDomain objects offer a view of the blitz arrays on which the shift function operates.
        *          These objects are shifted along the dimension specified in the argument, by <B>dim</B>, through a number of steps,
        *          to offer offset views.
        *
        * \param   dim is the integer input to specify the dimension (direction) of the shift. (x -> 0, y -> 1, z -> 2)
        * \param   core is the input RectDomain object which is to be shifted to get the new view
        * \param   steps is the integer value by which the input view must be offset along the dimension specified by <B>dim</B>
        *
        * \return  A RectDomain object that specifies the new offset view of the data
        ********************************************************************************************************************************************
        */
        inline blitz::RectDomain<3> shift(int dim, blitz::RectDomain<3> core, int steps) const {
            core.lbound()(dim) += steps;
            core.ubound()(dim) += steps;

            return core;
        }

        /**
        ********************************************************************************************************************************************
        * \brief   Function to check if a given set of global indices lie within a rank
        *
        *          Based on the data from rankData, the function checks if the given point lies within the subdomain of a processor
        *          The processor in whose sub-domain the point lies returns true.
        *
        * \param   gloIndex is a blitz TinyVector that contains the global indices to check if it lies within an MPI subdomain
        *
        * \return  A boolean value that evaluates to true if the point lies within the sub-domain
        ********************************************************************************************************************************************
        */
        inline bool pointInDomain(blitz::TinyVector<int, 3> gloIndex) const {
            if ((gloIndex(0) < subarrayEnds(0)) and (gloIndex(0) >= subarrayStarts(0)) and (gloIndex(1) < subarrayEnds(1)) and (gloIndex(1) >= subarrayStarts(1))) return true;

            return false;
        };

        /**
        ********************************************************************************************************************************************
        * \brief   Function to obtain local indices from global indices
        *
        *          Based on the data from rankData, the function computes the local indices
        *          in a manner that is consistent for both staggered and collocated indices.
        *
        * \param   gloIndex is a blitz TinyVector that contains the global indices for which local indices have to be found
        *
        * \return  A blitz TinyVector that contains the local indices computed from the given global indices
        ********************************************************************************************************************************************
        */
        inline blitz::TinyVector<int, 3> glo2loc(blitz::TinyVector<int, 3> gloIndex) const {
            blitz::TinyVector<int, 3> locIndex;

            if (pointInDomain(gloIndex)) {
                locIndex(0) = gloIndex(0) % coreSize(0);
                locIndex(1) = gloIndex(1) % coreSize(1);
                locIndex(2) = gloIndex(2);
            } else {
                locIndex = 0, 0, 0;
            }

            return locIndex;
        };

        /**
        ********************************************************************************************************************************************
        * \brief   Function to obtain global indices from local indices
        *
        *          Based on the data from rankData, the function computes the global indices
        *          in a manner that is consistent for both staggered and collocated indices.
        *
        * \param   locIndex is a blitz TinyVector that contains the local indices for which global indices have to be found
        *
        * \return  A blitz TinyVector that contains the global indices computed from the given local indices
        ********************************************************************************************************************************************
        */
        inline blitz::TinyVector<int, 3> loc2glo(blitz::TinyVector<int, 3> locIndex) const {
            blitz::TinyVector<int, 3> gloIndex;

            gloIndex(0) = rankData.xRank*coreSize(0) + locIndex(0);
            gloIndex(1) = rankData.yRank*coreSize(1) + locIndex(1);
            gloIndex(2) = locIndex(2);

            return gloIndex;
        };
};

/**
 ********************************************************************************************************************************************
 *  \class grid grid.h "lib/grid.h"
 *  \brief  Contains all the global variables related to the grid, its slices, limits, and grid derivatives used
 *          throughout the solver
 ********************************************************************************************************************************************
 */

#endif
