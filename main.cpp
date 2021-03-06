#include "SIPL/Core.hpp"
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

#define MAX(a,b) a > b ? a : b


int main(int argc, char ** argv) {
    if(argc < 4) {
        std::cout << "Usage: " << argv[0] << " filename.mhd mu iterations" << std::endl;
        return 0;
    }

    const bool printOutForPlotting = false;
    const float mu = atof(argv[2]);
    const int iterations = atoi(argv[3]);
    const bool use16bit = argc > 4 && std::string(argv[4]) == "--16bit";
    if(!printOutForPlotting) {
        if(use16bit) {
            std::cout << "Using 16 bit storage format for the vector fields" << std::endl;
        } else {
            std::cout << "Using 32 bit storage format for the vector fields" << std::endl;
        }
    }
    INIT_TIMER

    // Load MHD volume specified in arguments using SIPL
    SIPL::Volume<float> * volume = new SIPL::Volume<float>(argv[1], SIPL::IntensityTransformation(SIPL::NORMALIZED));
    SIPL::int3 size = volume->getSize();
    if(!printOutForPlotting)
        std::cout << "Loaded dataset of size " << size.x << ", " << size.y << ", " << size.z << std::endl;

    // Set up OpenCL
    oul::OpenCLManager * manager = oul::OpenCLManager::getInstance();
    manager->setDebugMode(true);
    //try {
    oul::DeviceCriteria defaultCriteria;
    defaultCriteria.setTypeCriteria(oul::DEVICE_TYPE_GPU);
    defaultCriteria.setDeviceCountCriteria(1);
    oul::Context context = manager->createContext(defaultCriteria);
    if(!printOutForPlotting)
        std::cout << "Using device: " << context.getDevice(0).getInfo<CL_DEVICE_NAME>() << std::endl;
    std::string filename;
    bool no3Dwrite;
    if((int)context.getDevice(0).getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
        filename = std::string(KERNELS_DIR) + std::string("3Dkernels.cl");
        no3Dwrite = false;
    } else {
        if(!printOutForPlotting)
            std::cout << "This device doesn't support writing to 3D textures. Using buffers instead. 16bit storage format is also disabled." << std::endl;
        filename = std::string(KERNELS_DIR) + std::string("3Dkernels_no_3d_write.cl");
        no3Dwrite = true;
    }
    context.createProgramFromSource(filename);

    // Create texture on GPU and transfer
    cl::Image3D volumeGPU = cl::Image3D(
            context.getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_R, CL_FLOAT),
            size.x, size.y, size.z,
            0, 0,
            (float *)volume->getData()
    );

    OpenCL ocl;
    ocl.context = context.getContext();
    ocl.program = context.getProgram(0);
    ocl.queue = context.getQueue(0);

    // Create vector field on the GPU
    cl::Image3D vectorFieldGPU = createVectorField(ocl, volumeGPU, size, no3Dwrite, use16bit);

    // Call the runFMGGVF method
    START_TIMER
    cl::Image3D resultGPU = runFMGGVF(
            ocl,
            &vectorFieldGPU,
            size,
            iterations, // iterations
            mu, // mu
            no3Dwrite, // no 3D write
            use16bit // 16bit
    );
    if(printOutForPlotting) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count() << ";";
    } else {
        STOP_TIMER("FMG GVF")
    }

    // Transfer GVF vector field back to host
    const unsigned int totalSize = size.x*size.y*size.z;

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    SIPL::float3 * data = new SIPL::float3[totalSize];
    if(use16bit) {
        short * temp = new short[totalSize*4];
        context.getQueue(0).enqueueReadImage(
                resultGPU,
                CL_TRUE,
                origin,
                region,
                0,0,
                temp
            );

            for(int i = 0; i < totalSize; i++) {
                data[i].x = MAX(-1.0f, (float)temp[i*4]/32767.0f);
                data[i].y = MAX(-1.0f, (float)temp[i*4+1]/32767.0f);
                data[i].z = MAX(-1.0f, (float)temp[i*4+2]/32767.0f);
            }
            delete[] temp;
    } else {
        float * temp = new float[totalSize*4];
        context.getQueue(0).enqueueReadImage(
            resultGPU,
            CL_TRUE,
            origin,
            region,
            0,0,
            temp
        );

        for(int i = 0; i < totalSize; i++) {
            data[i].x = temp[i*4];
            data[i].y = temp[i*4+1];
            data[i].z = temp[i*4+2];
        }
        delete[] temp;
    }
    SIPL::Volume<SIPL::float3> * result = new SIPL::Volume<SIPL::float3>(size);
    result->setData(data);

    ErrorMeasurements e = calculateMaxResidual(result, volume, mu);
    if(printOutForPlotting) {
        std::cout << e.averageError << std::endl;
    } else {
        std::cout << "Maximum magnitude of residuals was: " << e.maxError << std::endl;
        std::cout << "Average error was: " << e.averageError << std::endl;
    }
    delete volume;

    // Display using OpenCL
    if(!printOutForPlotting)
        result->display(0.0, 0.1);

    // Create magnitude image and display it
    if(!printOutForPlotting) {
        SIPL::Volume<float> * magnitude = new SIPL::Volume<float>(size);
        for(int i = 0; i < magnitude->getTotalSize(); i++)
            magnitude->set(i, result->get(i).length());
        magnitude->display();
    }
    /*
    } catch(cl::Error &error) {
        std::cout << "An OpenCL error occured!" << std::endl;
        std::cout << "Error message: " << getCLErrorString(error.err()) << std::endl;
        std::cout << "What caused the error: " << error.what() << std::endl;
        std::cout << "Error code: " << (int)error.err() << std::endl;
        std::cout << "Aborting due to error." << std::endl;
    }
    */

    // TODO: create 2D version?
    // TODO: work-group sizes
    // TODO: no 3d write
}

