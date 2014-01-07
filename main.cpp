#include "SIPL/Core.hpp"
#include "OpenCLUtilities/openCLUtilities.hpp"
#include "gradientVectorFlow.hpp"

int main(int argc, char ** argv) {

    // Load MHD volume specified in arguments using SIPL
    SIPL::Volume<float> * volume = new SIPL::Volume<float>(argv[1]);
    SIPL::int3 size = volume->getSize();

    // Set up OpenCL
    OpenCL ocl;
    ocl.context = createCLContextFromArguments(argc, argv);
    VECTOR_CLASS<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.device = devices[0];
    ocl.queue = cl::CommandQueue(ocl.context, ocl.device);

    // Create texture on GPU and transfer
    cl::Image3D volumeGPU = cl::Image3D(
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_R, CL_FLOAT),
            size.x, size.y, size.z,
            0, 0,
            volume->getData()
    );

    // Create vector field on the GPU
    cl::Image3D vectorFieldGPU = createVectorField(ocl, volumeGPU, size);

    // Call the runFMGGVF method
    cl::Image3D resultGPU = runFMGGVF(
            ocl,
            &vectorFieldGPU,
            size,
            10, // iterations
            0.05, // mu
            false, // no 3D write
            true // 16bit
    );
    
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
        data[i].x = temp[i*3];
        data[i].y = temp[i*3+1];
        data[i].z = temp[i*3+2];
    }
    delete[] temp;
    SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size);
    result->setData(data);

    // Display using OpenCL
    result->display();
    
    // Further: Time the application and calculate the max magnitude of residuals
}

