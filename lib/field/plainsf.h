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
/*! \file plainsf.h
 *
 *  \brief Class declaration of plainsf - plain scalar field
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef PLAINSF_H
#define PLAINSF_H

#include "sfield.h"
#include "plainvf.h"

class plainsf {
    private:
        const grid &gridData;

        blitz::Array<real, 3> derivTemp;

        blitz::RectDomain<3> core;

        /** derS is an instance of the derivative class used to compute derivatives */
        derivative *derS;

    public:
        blitz::Array<real, 3> F;

        mpidata *mpiHandle;

        plainsf(const grid &gridData);

        void gradient(plainvf &gradF);

        plainsf& multAdd(const plainsf &a, real k);

        plainsf& operator += (plainsf &a);
        plainsf& operator -= (plainsf &a);

        plainsf& operator += (sfield &a);
        plainsf& operator -= (sfield &a);

        plainsf& operator *= (real a);

        void operator = (plainsf &a);
        void operator = (sfield &a);
        void operator = (real a);

/**
 ********************************************************************************************************************************************
 * \brief   Function to synchronise data across all processors when performing parallel computations
 *
 *          This function calls the \ref mpidata#syncData "syncData" function of mpidata class to perform perform data-transfer and thus update
 *          the sub-domain boundary pads.
 ********************************************************************************************************************************************
 */
        inline void syncData() {
            mpiHandle->syncData();
        }

/**
 ********************************************************************************************************************************************
 * \brief   Function to extract the maximum value from the plain scalar field
 *
 *          The function uses the in-built blitz function to obtain the maximum value in an array.
 *          While performing parallel computation, the function performs an <B>MPI_Allreduce()</B> to get
 *          the global maximum from the entire computational domain.
 *
 * \return  The real value of the maximum is returned (it is implicitly assumed that only real values are used)
 ********************************************************************************************************************************************
 */
        inline real fxMax() {
            real localMax, globalMax;

            localMax = blitz::max(F(gridData.coreDomain));

            MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

            return globalMax;
        }

/**
 ********************************************************************************************************************************************
 * \brief   Function to extract the maximum from absolute values of the plain scalar field
 *
 *          Like fxMax, this function uses the in-built blitz function to obtain the maximum value in an array.
 *          However, in doing so, it takes the absolute values of array.
 *          While performing parallel computation, the function performs an <B>MPI_Allreduce()</B> to get
 *          the global maximum from the entire computational domain.
 *
 * \return  The real value of the maximum is returned (it is implicitly assumed that only real values are used)
 ********************************************************************************************************************************************
 */
        inline real fxMaxAbs() {
            real localMax, globalMax;

            localMax = blitz::max(blitz::abs(F(gridData.coreDomain)));

            MPI_Allreduce(&localMax, &globalMax, 1, MPI_FP_REAL, MPI_MAX, MPI_COMM_WORLD);

            return globalMax;
        }

/**
 ********************************************************************************************************************************************
 * \brief   Function to compute the mean value from the plain scalar field
 *
 *          The function uses the in-built blitz function to obtain the mean value in an array.
 *          While performing parallel computation, the function performs an <B>MPI_Allreduce()</B> to get
 *          the global mean from the entire computational domain.
 *
 * \return  The real value of the mean is returned (it is implicitly assumed that only real values are used)
 ********************************************************************************************************************************************
 */
        inline real fxMean() {
            real localMean, globalSum;

            localMean = blitz::mean(F(gridData.coreDomain));

            MPI_Allreduce(&localMean, &globalSum, 1, MPI_FP_REAL, MPI_SUM, MPI_COMM_WORLD);

            return globalSum/gridData.rankData.nProc;
        }

        ~plainsf() { };
};

/**
 ********************************************************************************************************************************************
 *  \class plainsf plainsf.h "lib/plainsf.h"
 *  \brief Plain scalar field class to store simple scalar fields with no differentiation or interpolation
 *
 *  The class stores scalar fields in the form of a Blitz array
 ********************************************************************************************************************************************
 */

#endif
