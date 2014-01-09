#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"
#include "gradientVectorFlow.hpp"

#ifndef KERNELS_DIR
#define KERNELS_DIR ""
#endif
#include <chrono>

#define TIMING

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now()-start \
    ).count() << " ms " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

int main(int argc, char ** argv) {
    const float mu = 0.2f;
    INIT_TIMER

    // Load MHD volume specified in arguments using SIPL
    SIPL::Volume<float> * volume = new SIPL::Volume<float>(argv[1], SIPL::IntensityTransformation(SIPL::NORMALIZED));
    SIPL::int3 size = volume->getSize();

    // Set up OpenCL
    OpenCL ocl;
    ocl.context = createCLContextFromArguments(argc, argv);
    VECTOR_CLASS<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.device = devices[0];
    ocl.queue = cl::CommandQueue(ocl.context, ocl.device);
    std::string filename = std::string(KERNELS_DIR) + std::string("3Dkernels.cl");
    ocl.program = buildProgramFromSource(ocl.context, filename);

    // Create texture on GPU and transfer
    cl::Image3D volumeGPU = cl::Image3D(
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_R, CL_FLOAT),
            size.x, size.y, size.z,
            0, 0,
            (float *)volume->getData()
    );

    // Create vector field on the GPU
    cl::Image3D vectorFieldGPU = createVectorField(ocl, volumeGPU, size);

    // Call the runFMGGVF method
    START_TIMER
    cl::Image3D resultGPU = runFMGGVF(
            ocl,
            &vectorFieldGPU,
            size,
            10, // iterations
            mu, // mu
            false, // no 3D write
            false // 16bit
    );
    STOP_TIMER("FMG GVF")
    
    // Transfer GVF vector field back to host
    const unsigned int totalSize = size.x*size.y*size.z;
    float * temp = new float[totalSize*4];

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;
    ocl.queue.enqueueReadImage(
        resultGPU,
        CL_TRUE,
        origin,
        region,
        0,0,
        temp
    );

    SIPL::float3 * data = new SIPL::float3[totalSize];
    for(int i = 0; i < totalSize; i++) {
        data[i].x = temp[i*4];
        data[i].y = temp[i*4+1];
        data[i].z = temp[i*4+2];
    }
    delete[] temp;
    SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size);
    result->setData(data);

    std::cout << "Maximum magnitude of residuals was: " << calculateMaxResidual(result, volume, mu) << std::endl;
    delete volume;

    // Display using OpenCL
    result->display(0.0, 0.1);
    
    // TODO: calculate the max magnitude of residuals
    // Create magnitude image and display it
    SIPL::Volume<float> * magnitude = new SIPL::Volume<float>(size);
    for(int i = 0; i < magnitude->getTotalSize(); i++)
        magnitude->set(i, result->get(i).length());
    magnitude->display();

    // TODO: create 2D version?
    // TODO: 16 bit support
    // TODO: work-group sizes
    // TODO: no 3d write
}

