#ifndef GVF_H
#define GVF_H
#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"

cl::Image3D createVectorField(OpenCL &ocl, cl::Image3D volume, SIPL::int3 &size, const bool no3Dwrite, const bool use16bit);

cl::Image3D runFMGGVF(
        OpenCL &ocl,
        cl::Image3D *vectorField,
        SIPL::int3 &size,
        const int GVFIterations,
        const float MU,
        const bool no3Dwrite,
        const bool use16bit
);

class ErrorMeasurements {
    public:
        float averageError;
        float maxError;
};

ErrorMeasurements calculateMaxResidual(SIPL::Volume<SIPL::float3>* vectorField,
        SIPL::Volume<float>* volume,
        float mu);
#endif
