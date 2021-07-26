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
/*! \file writer.h
 *
 *  \brief Class declaration of writer
 *
 *  \author Roshan Samuel
 *  \date Nov 2019
 *  \copyright New BSD License
 *
 ********************************************************************************************************************************************
 */

#ifndef WRITER_H
#define WRITER_H

#include <sys/stat.h>
#include <iomanip>
#include <vector>

#include "field.h"
#include "grid.h"
#include "hdf5.h"

class writer {
    public:
        writer(const grid &mesh, std::vector<field> &wFields);

        void writeTarang(real time);
        void writeSolution(real time);
        void writeRestart(real time);

        ~writer();

    private:
        const grid &mesh;

        // Print flag - basically flag to ease printing to I/O. It is true for root rank (0)
        bool pf;

        std::vector<field> &wFields;

#ifdef PLANAR
        blitz::Array<real, 2> fieldData;
#else
        blitz::Array<real, 3> fieldData;
#endif

        hid_t timeDSpace;
        hid_t xDSpace, yDSpace, zDSpace;
        hid_t sourceDSpace, targetDSpace;

        blitz::TinyVector<int, 3> locSize;

        void outputCheck();

        void initLimits();

        void copyData(field &outField);
};

/**
 ********************************************************************************************************************************************
 *  \class writer writer.h "lib/io/writer.h"
 *  \brief Class for all the global variables and functions related to writing output data of the solver.
 *
 *  The computational data from the solver is written in HDF5 format in a .h5 file.
 ********************************************************************************************************************************************
 */

#endif
