#include "gradientVectorFlow.hpp"
using namespace cl;

Image3D initSolutionToZero(OpenCL &ocl, SIPL::int3 size, int imageType, int bufferSize, bool no3Dwrite) {
    Image3D v = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    if(no3Dwrite) {
        Kernel initToZeroKernel(ocl.program, "initFloatBuffer");
        Buffer vBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);
        initToZeroKernel.setArg(0,vBuffer);
        ocl.queue.enqueueNDRangeKernel(
                initToZeroKernel,
                NullRange,
                NDRange(size.x*size.y*size.z),
                NullRange
        );
		cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        ocl.queue.enqueueCopyBufferToImage(vBuffer,v,0,offset,region);
    } else {
        Kernel initToZeroKernel(ocl.program, "init3DFloat");
        initToZeroKernel.setArg(0,v);
        ocl.queue.enqueueNDRangeKernel(
                initToZeroKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

    }

    return v;
}
void gaussSeidelSmoothing(
        OpenCL &ocl,
        Image3D &v,
        Image3D &r,
        Image3D &sqrMag,
        int iterations,
        SIPL::int3 size,
        float mu,
        float spacing,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {

    if(iterations <= 0)
        return;

    Kernel gaussSeidelKernel = Kernel(ocl.program, "GVFgaussSeidel");
    Kernel gaussSeidelKernel2 = Kernel(ocl.program, "GVFgaussSeidel2");

    Image3D v_2 = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
     );

    gaussSeidelKernel.setArg(0, r);
    gaussSeidelKernel.setArg(1, sqrMag);
    gaussSeidelKernel.setArg(2, mu);
    gaussSeidelKernel.setArg(3, spacing);
    gaussSeidelKernel2.setArg(0, r);
    gaussSeidelKernel2.setArg(1, sqrMag);
    gaussSeidelKernel2.setArg(2, mu);
    gaussSeidelKernel2.setArg(3, spacing);

    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        Buffer v_2_buffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);

        for(int i = 0; i < iterations*2; i++) {
             if(i % 2 == 0) {
                 gaussSeidelKernel.setArg(4, v);
                 gaussSeidelKernel.setArg(5, v_2_buffer);
                 ocl.queue.enqueueNDRangeKernel(
                    gaussSeidelKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
                );
                ocl.queue.enqueueCopyBufferToImage(v_2_buffer, v_2,0,offset,region);
             } else {
                 gaussSeidelKernel2.setArg(4, v_2);
                 gaussSeidelKernel2.setArg(5, v_2_buffer);
                 ocl.queue.enqueueNDRangeKernel(
                    gaussSeidelKernel2,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
                );
                ocl.queue.enqueueCopyBufferToImage(v_2_buffer, v,0,offset,region);
             }
        }
    } else {
         for(int i = 0; i < iterations*2; i++) {
             if(i % 2 == 0) {
                 gaussSeidelKernel.setArg(4, v);
                 gaussSeidelKernel.setArg(5, v_2);
                 ocl.queue.enqueueNDRangeKernel(
                    gaussSeidelKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
                );
             } else {
                 gaussSeidelKernel2.setArg(4, v_2);
                 gaussSeidelKernel2.setArg(5, v);
                 ocl.queue.enqueueNDRangeKernel(
                    gaussSeidelKernel2,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
                );
             }
        }
    }
}

Image3D restrictVolume(
        OpenCL &ocl,
        Image3D &v,
        SIPL::int3 newSize,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {

    // Check to see if size is a power of 2 and equal in all dimensions

    Image3D v_2 = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            newSize.x,
            newSize.y,
            newSize.z
    );

    Kernel restrictKernel = Kernel(ocl.program, "restrictVolume");
    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = newSize.x;
		region[1] = newSize.y;
		region[2] = newSize.z;
        Buffer v_2_buffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*newSize.x*newSize.y*newSize.z);
        restrictKernel.setArg(0, v);
        restrictKernel.setArg(1, v_2_buffer);
        ocl.queue.enqueueNDRangeKernel(
                restrictKernel,
                NullRange,
                NDRange(newSize.x,newSize.y,newSize.z),
                NDRange(4,4,4)
        );
        ocl.queue.enqueueCopyBufferToImage(v_2_buffer, v_2,0,offset,region);
    } else {
        restrictKernel.setArg(0, v);
        restrictKernel.setArg(1, v_2);
        ocl.queue.enqueueNDRangeKernel(
                restrictKernel,
                NullRange,
                NDRange(newSize.x,newSize.y,newSize.z),
                NDRange(4,4,4)
        );
    }

    return v_2;
}

Image3D prolongateVolume(
        OpenCL &ocl,
        Image3D &v_l,
        Image3D &v_l_p1,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {
    Image3D v_2 = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    Kernel prolongateKernel = Kernel(ocl.program, "prolongate");
    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        Buffer v_2_buffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);
        prolongateKernel.setArg(0, v_l);
        prolongateKernel.setArg(1, v_l_p1);
        prolongateKernel.setArg(2, v_2_buffer);
        ocl.queue.enqueueNDRangeKernel(
                prolongateKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

        ocl.queue.enqueueCopyBufferToImage(v_2_buffer, v_2,0,offset,region);
    } else {
        prolongateKernel.setArg(0, v_l);
        prolongateKernel.setArg(1, v_l_p1);
        prolongateKernel.setArg(2, v_2);
        ocl.queue.enqueueNDRangeKernel(
                prolongateKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }

    return v_2;
}

Image3D prolongateVolume2(
        OpenCL &ocl,
        Image3D &v_l_p1,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {
    Image3D v_2 = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    Kernel prolongateKernel = Kernel(ocl.program, "prolongate2");
    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        Buffer v_2_buffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);
        prolongateKernel.setArg(0, v_l_p1);
        prolongateKernel.setArg(1, v_2_buffer);
        ocl.queue.enqueueNDRangeKernel(
                prolongateKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

        ocl.queue.enqueueCopyBufferToImage(v_2_buffer, v_2,0,offset,region);
    } else {
        prolongateKernel.setArg(0, v_l_p1);
        prolongateKernel.setArg(1, v_2);
        ocl.queue.enqueueNDRangeKernel(
                prolongateKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }

    return v_2;
}


Image3D residual(
        OpenCL &ocl,
        Image3D &r,
        Image3D &v,
        Image3D &sqrMag,
        float mu,
        float spacing,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {
    Image3D newResidual = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    Kernel residualKernel(ocl.program, "residual");
    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        Buffer newResidualBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);
        residualKernel.setArg(0, r);
        residualKernel.setArg(1, v);
        residualKernel.setArg(2, sqrMag);
        residualKernel.setArg(3, mu);
        residualKernel.setArg(4, spacing);
        residualKernel.setArg(5, newResidualBuffer);
        ocl.queue.enqueueNDRangeKernel(
                residualKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

        ocl.queue.enqueueCopyBufferToImage(newResidualBuffer, newResidual,0,offset,region);
    } else {
        residualKernel.setArg(0, r);
        residualKernel.setArg(1, v);
        residualKernel.setArg(2, sqrMag);
        residualKernel.setArg(3, mu);
        residualKernel.setArg(4, spacing);
        residualKernel.setArg(5, newResidual);
        ocl.queue.enqueueNDRangeKernel(
                residualKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }
    return newResidual;
}


SIPL::int3 calculateNewSize(SIPL::int3 size) {
    bool sizeIsOkay = false;
    if(size.x == size.y && size.x == size.z) {
        float p = (float)log((float)size.x) / log(2.0f);
        if(floor(p) == p)
            sizeIsOkay = true;
    }
    int newSize;
    if(!sizeIsOkay) {
        int maxSize = std::max(size.x, std::max(size.y, size.z));
        int i = 1;
        while(true) {
            if(pow(2.0f, (float)i) >= maxSize) {
                newSize = (int)pow(2.0f, (float)(i-1));
                break;
            }
            i++;
        }
    } else {
        newSize = size.x / 2;
    }

    return SIPL::int3(newSize,newSize,newSize);

}

void multigridVcycle(
        OpenCL &ocl,
        Image3D &r_l,
        Image3D &v_l,
        Image3D &sqrMag,
        int l,
        int v1,
        int v2,
        int l_max,
        float mu,
        float spacing,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {

    // Pre-smoothing
    gaussSeidelSmoothing(ocl,v_l,r_l,sqrMag,v1,size,mu,spacing,imageType,bufferSize,no3Dwrite);

    if(l < l_max) {
        SIPL::int3 newSize = calculateNewSize(size);

        // Compute new residual
        Image3D p_l = residual(ocl, r_l, v_l, sqrMag, mu, spacing, size,imageType,bufferSize,no3Dwrite);

        // Restrict residual
        Image3D r_l_p1 = restrictVolume(ocl, p_l, newSize,imageType,bufferSize,no3Dwrite);

        // Restrict sqrMag
        Image3D sqrMag_l_p1 = restrictVolume(ocl, sqrMag, newSize,imageType,bufferSize,no3Dwrite);

        // Initialize v_l_p1
        Image3D v_l_p1 = initSolutionToZero(ocl,newSize,imageType,bufferSize,no3Dwrite);

        // Solve recursively
        multigridVcycle(ocl, r_l_p1, v_l_p1, sqrMag_l_p1, l+1,v1,v2,l_max,mu,spacing*2,newSize,imageType,bufferSize,no3Dwrite);

        // Prolongate
        v_l = prolongateVolume(ocl, v_l, v_l_p1, size,imageType,bufferSize,no3Dwrite);
    }

    // Post-smoothing
    gaussSeidelSmoothing(ocl,v_l,r_l,sqrMag,v2,size,mu,spacing,imageType,bufferSize,no3Dwrite);
}

/*
Image3D runMGGVF(OpenCL &ocl, Image3D *vectorField, paramList &parameters, SIPL::int3 &size) {

    const int GVFIterations = getParam(parameters, "gvf-iterations");
    const bool no3Dwrite = !getParamBool(parameters, "3d_write");
    const float MU = getParam(parameters, "gvf-mu");
    const int totalSize = size.x*size.y*size.z;
    const bool use16bit = getParamBool(parameters, "16bit-vectors");
    int imageType;
    if(use16bit) {
        imageType = CL_SNORM_INT16;
    } else {
        imageType = CL_FLOAT;
    }

    Kernel initKernel = Kernel(ocl.program, "MGGVFInit");

    int v1 = 2;
    int v2 = 2;
    int l_max = 6; // TODO this should be calculated
    float spacing = 1.0f;

    // create sqrMag
    Kernel createSqrMagKernel(ocl.program, "createSqrMag");
    Image3D sqrMag = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    createSqrMagKernel.setArg(0, *vectorField);
    createSqrMagKernel.setArg(1, sqrMag);
    ocl.queue.enqueueNDRangeKernel(
            createSqrMagKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
    );
    std::cout << "sqrMag created" << std::endl;

    // create fx and rx
    Image3D fx = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    Image3D *rx = new Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    ocl.GC.addMemObject(rx);
    initKernel.setArg(0, *vectorField);
    initKernel.setArg(1, fx);
    initKernel.setArg(2, *rx);
    initKernel.setArg(3, 1);
    ocl.queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
    );
    std::cout << "fx initialized" << std::endl;

    // X component
    for(int i = 0; i < GVFIterations; i++) {
        multigridVcycle(ocl,*rx,fx,sqrMag,0,v1,v2,l_max,MU,spacing,size,imageType);
        ocl.queue.finish();
    }
    std::cout << "fx finished" << std::endl;

    // delete rx
    ocl.GC.deleteMemObject(rx);

    // create fy and ry
    Image3D fy = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    Image3D *ry = new Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    ocl.GC.addMemObject(ry);
    initKernel.setArg(0, *vectorField);
    initKernel.setArg(1, fy);
    initKernel.setArg(2, *ry);
    initKernel.setArg(3, 2);
    ocl.queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
    );
    std::cout << "fy initialized" << std::endl;
    // Y component
    for(int i = 0; i < GVFIterations; i++) {
        multigridVcycle(ocl,*ry,fy,sqrMag,0,v1,v2,l_max,MU,spacing,size,imageType);
        ocl.queue.finish();
    }
    std::cout << "fy finished" << std::endl;

    // delete ry
    ocl.GC.deleteMemObject(ry);
    // create fz and rz
    Image3D fz = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    Image3D *rz = new Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    ocl.GC.addMemObject(rz);
    initKernel.setArg(0, *vectorField);
    initKernel.setArg(1, fz);
    initKernel.setArg(2, *rz);
    initKernel.setArg(3, 3);
    ocl.queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
    );
    std::cout << "fz initialized" << std::endl;
    ocl.GC.deleteMemObject(vectorField);
    // Z component
    for(int i = 0; i < GVFIterations; i++) {
        multigridVcycle(ocl,*rz,fz,sqrMag,0,v1,v2,l_max,MU,spacing,size,imageType);
        ocl.queue.finish();
    }
    std::cout << "fz finished" << std::endl;

    // delete rz
    ocl.GC.deleteMemObject(rz);


    Image3D finalVectorField = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_RGBA, imageType),
            size.x,
            size.y,
            size.z
    );
    Kernel finalizeKernel = Kernel(ocl.program, "MGGVFFinish");
    finalizeKernel.setArg(0, fx);
    finalizeKernel.setArg(1, fy);
    finalizeKernel.setArg(2, fz);
    finalizeKernel.setArg(3, finalVectorField);
    ocl.queue.enqueueNDRangeKernel(
            finalizeKernel,
            NullRange,
            NDRange(size.x,size.y,size.z),
            NullRange
    );
    std::cout << "MG GVF finished" << std::endl;


    return finalVectorField;
}
*/

Image3D computeNewResidual(
        OpenCL &ocl,
        Image3D &f,
        Image3D &vectorField,
        float mu,
        float spacing,
        int component,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {
    Image3D newResidual = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );

    Kernel residualKernel(ocl.program, "fmgResidual");
    if(no3Dwrite) {
        cl::size_t<3> offset;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = 0;
		cl::size_t<3> region;
		region[0] = size.x;
		region[1] = size.y;
		region[2] = size.z;
        Buffer newResidualBuffer = Buffer(ocl.context, CL_MEM_WRITE_ONLY, bufferSize*size.x*size.y*size.z);
        residualKernel.setArg(0,vectorField);
        residualKernel.setArg(1, f);
        residualKernel.setArg(2, mu);
        residualKernel.setArg(3, spacing);
        residualKernel.setArg(4, component);
        residualKernel.setArg(5, newResidualBuffer);
        ocl.queue.enqueueNDRangeKernel(
                residualKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
        ocl.queue.enqueueCopyBufferToImage(newResidualBuffer, newResidual,0,offset,region);
    } else {
        residualKernel.setArg(0,vectorField);
        residualKernel.setArg(1, f);
        residualKernel.setArg(2, mu);
        residualKernel.setArg(3, spacing);
        residualKernel.setArg(4, component);
        residualKernel.setArg(5, newResidual);
        ocl.queue.enqueueNDRangeKernel(
                residualKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }

    return newResidual;
}

Image3D fullMultigrid(
        OpenCL &ocl,
        Image3D &r_l,
        Image3D &sqrMag,
        int l,
        int v0,
        int v1,
        int v2,
        int l_max,
        float mu,
        float spacing,
        SIPL::int3 size,
        int imageType,
        int bufferSize,
        bool no3Dwrite
        ) {
    Image3D v_l;
    if(l < l_max) {
        SIPL::int3 newSize = calculateNewSize(size);
        Image3D r_l_p1 = restrictVolume(ocl, r_l, newSize, imageType,bufferSize,no3Dwrite);
        Image3D sqrMag_l = restrictVolume(ocl,sqrMag,newSize,imageType,bufferSize,no3Dwrite);
        Image3D v_l_p1 = fullMultigrid(ocl,r_l_p1,sqrMag_l,l+1,v0,v1,v2,l_max,mu,spacing*2,newSize, imageType,bufferSize,no3Dwrite);
        v_l = prolongateVolume2(ocl,v_l_p1, size,imageType,bufferSize,no3Dwrite);
    } else {
        v_l = initSolutionToZero(ocl,size,imageType,bufferSize,no3Dwrite);
    }

    for(int i = 0; i < v0; i++) {
        multigridVcycle(ocl,r_l,v_l,sqrMag,l,v1,v2,l_max,mu,spacing,size,imageType,bufferSize,no3Dwrite);
    }

    return v_l;

}

Image3D runFMGGVF(
        OpenCL &ocl,
        Image3D *vectorField,
        SIPL::int3 &size,
        const int GVFIterations,
        const float MU,
        const bool no3Dwrite,
        const bool use16bit
        ) {

    const int totalSize = size.x*size.y*size.z;
    int imageType, bufferTypeSize;
    if(use16bit) {
        imageType = CL_SNORM_INT16;
        bufferTypeSize = sizeof(short);
    } else {
        imageType = CL_FLOAT;
        bufferTypeSize = sizeof(float);
    }

    Kernel initKernel = Kernel(ocl.program, "MGGVFInit");

    int v0 = 1;
    int v1 = 2;
    int v2 = 2;
    int l_max = 2; // TODO this should be calculated
    float spacing = 1.0f;

    // create sqrMag
    Kernel createSqrMagKernel(ocl.program, "createSqrMag");
    Image3D sqrMag = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_R, imageType),
            size.x,
            size.y,
            size.z
    );
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    if(no3Dwrite) {
        Buffer sqrMagBuffer = Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                totalSize*bufferTypeSize
        );
        createSqrMagKernel.setArg(0, *vectorField);
        createSqrMagKernel.setArg(1, sqrMagBuffer);
        ocl.queue.enqueueNDRangeKernel(
                createSqrMagKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
        ocl.queue.enqueueCopyBufferToImage(sqrMagBuffer,sqrMag,0,offset,region);
    } else {
        createSqrMagKernel.setArg(0, *vectorField);
        createSqrMagKernel.setArg(1, sqrMag);
        ocl.queue.enqueueNDRangeKernel(
                createSqrMagKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
    }
    std::cout << "sqrMag created" << std::endl;

    Kernel addKernel(ocl.program, "addTwoImages");
    Image3D fx = initSolutionToZero(ocl,size,imageType,bufferTypeSize,no3Dwrite);

    // X component
    for(int i = 0; i < GVFIterations; i++) {
        Image3D rx = computeNewResidual(ocl,fx,*vectorField,MU,spacing,1,size,imageType,bufferTypeSize,no3Dwrite);
        Image3D fx2 = fullMultigrid(ocl,rx,sqrMag,0,v0,v1,v2,l_max,MU,spacing,size,imageType,bufferTypeSize,no3Dwrite);
        ocl.queue.finish();
        if(no3Dwrite) {
            Buffer fx3 = Buffer(
                    ocl.context,
                    CL_MEM_WRITE_ONLY,
                    totalSize*bufferTypeSize
            );

            addKernel.setArg(0,fx);
            addKernel.setArg(1,fx2);
            addKernel.setArg(2,fx3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.enqueueCopyBufferToImage(fx3,fx,0,offset,region);
            ocl.queue.finish();
        } else {
            Image3D fx3 = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, imageType),
                size.x,
                size.y,
                size.z
            );

            addKernel.setArg(0,fx);
            addKernel.setArg(1,fx2);
            addKernel.setArg(2,fx3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.finish();

            fx = fx3;
        }

    }
    std::cout << "fx finished" << std::endl;

    // create fy and ry
    // Y component
    Image3D fy = initSolutionToZero(ocl,size,imageType,bufferTypeSize,no3Dwrite);
    for(int i = 0; i < GVFIterations; i++) {
        Image3D ry = computeNewResidual(ocl,fy,*vectorField,MU,spacing,2,size,imageType,bufferTypeSize,no3Dwrite);
        Image3D fy2 = fullMultigrid(ocl,ry,sqrMag,0,v0,v1,v2,l_max,MU,spacing,size,imageType,bufferTypeSize,no3Dwrite);
        ocl.queue.finish();
        if(no3Dwrite) {
            Buffer fy3 = Buffer(
                    ocl.context,
                    CL_MEM_WRITE_ONLY,
                    totalSize*bufferTypeSize
            );

            addKernel.setArg(0,fy);
            addKernel.setArg(1,fy2);
            addKernel.setArg(2,fy3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.enqueueCopyBufferToImage(fy3,fy,0,offset,region);
            ocl.queue.finish();
        } else {
            Image3D fy3 = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, imageType),
                size.x,
                size.y,
                size.z
            );

            addKernel.setArg(0,fy);
            addKernel.setArg(1,fy2);
            addKernel.setArg(2,fy3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.finish();

            fy = fy3;
        }

    }

    std::cout << "fy finished" << std::endl;

    // create fz and rz
    // Z component
    Image3D fz = initSolutionToZero(ocl,size,imageType,bufferTypeSize,no3Dwrite);
    for(int i = 0; i < GVFIterations; i++) {
        Image3D rz = computeNewResidual(ocl,fz,*vectorField,MU,spacing,3,size,imageType,bufferTypeSize,no3Dwrite);
        Image3D fz2 = fullMultigrid(ocl,rz,sqrMag,0,v0,v1,v2,l_max,MU,spacing,size,imageType,bufferTypeSize,no3Dwrite);
        ocl.queue.finish();
        if(no3Dwrite) {
            Buffer fz3 = Buffer(
                    ocl.context,
                    CL_MEM_WRITE_ONLY,
                    totalSize*bufferTypeSize
            );

            addKernel.setArg(0,fz);
            addKernel.setArg(1,fz2);
            addKernel.setArg(2,fz3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.enqueueCopyBufferToImage(fz3,fz,0,offset,region);
            ocl.queue.finish();
        } else {
            Image3D fz3 = Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                ImageFormat(CL_R, imageType),
                size.x,
                size.y,
                size.z
            );

            addKernel.setArg(0,fz);
            addKernel.setArg(1,fz2);
            addKernel.setArg(2,fz3);
            ocl.queue.enqueueNDRangeKernel(
                    addKernel,
                    NullRange,
                    NDRange(size.x,size.y,size.z),
                    NDRange(4,4,4)
            );
            ocl.queue.finish();

            fz = fz3;
        }

    }

    ocl.GC.deleteMemObject(vectorField);

    std::cout << "fz finished" << std::endl;


    Image3D finalVectorField = Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            ImageFormat(CL_RGBA, imageType),
            size.x,
            size.y,
            size.z
    );
    Kernel finalizeKernel = Kernel(ocl.program, "MGGVFFinish");
    if(no3Dwrite) {
        Buffer finalVectorFieldBuffer = Buffer(
                ocl.context,
                CL_MEM_WRITE_ONLY,
                4*totalSize*bufferTypeSize
        );

        finalizeKernel.setArg(0, fx);
        finalizeKernel.setArg(1, fy);
        finalizeKernel.setArg(2, fz);
        finalizeKernel.setArg(3, finalVectorFieldBuffer);
        ocl.queue.enqueueNDRangeKernel(
                finalizeKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );
        ocl.queue.enqueueCopyBufferToImage(finalVectorFieldBuffer,finalVectorField,0,offset,region);
    } else {
        finalizeKernel.setArg(0, fx);
        finalizeKernel.setArg(1, fy);
        finalizeKernel.setArg(2, fz);
        finalizeKernel.setArg(3, finalVectorField);
        ocl.queue.enqueueNDRangeKernel(
                finalizeKernel,
                NullRange,
                NDRange(size.x,size.y,size.z),
                NDRange(4,4,4)
        );

    }
    std::cout << "MG GVF finished" << std::endl;


    return finalVectorField;
}

