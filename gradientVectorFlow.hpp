#ifndef GVF_H
#define GVF_H
#include "SIPL/Types.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"

cl::Image3D createVectorField(OpenCL &ocl, cl::Image3D volume, SIPL::int3 &size);

cl::Image3D runFMGGVF(
        OpenCL &ocl,
        cl::Image3D *vectorField,
        SIPL::int3 &size,
        const int GVFIterations,
        const float MU,
        const bool no3Dwrite,
        const bool use16bit
);

#endif
