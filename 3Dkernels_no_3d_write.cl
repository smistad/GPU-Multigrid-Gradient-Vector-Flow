__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant sampler_t interpolationSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t hpSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define LPOS(pos) pos.x+pos.y*get_global_size(0)+pos.z*get_global_size(0)*get_global_size(1)
#define NLPOS(pos) ((pos).x) + ((pos).y)*size.x + ((pos).z)*size.x*size.y

#ifdef VECTORS_16BIT
#define FLOAT_TO_SNORM16_4(vector) convert_short4_sat_rte(vector * 32767.0f)
#define SNORM16_TO_FLOAT_4(vector) max(-1.0f, convert_float4(vector) / 32767.0f)
#define FLOAT_TO_SNORM16_3(vector) convert_short3_sat_rte(vector * 32767.0f)
#define SNORM16_TO_FLOAT_3(vector) max(-1.0f, convert_float3(vector) / 32767.0f)
#define FLOAT_TO_SNORM16_2(vector) convert_short2_sat_rte(vector * 32767.0f)
#define SNORM16_TO_FLOAT_2(vector) max(-1.0f, convert_float2(vector) / 32767.0f)
#define FLOAT_TO_SNORM16(vector) convert_short_sat_rte(vector * 32767.0f)
#define SNORM16_TO_FLOAT(vector) max(-1.0f, convert_float(vector) / 32767.0f)
#define VECTOR_FIELD_TYPE short
#define UNORM16_TO_FLOAT(v) (float)v / 65535.0f
#define FLOAT_TO_UNORM16(v) convert_ushort_sat_rte(v * 65535.0f)
#define TDF_TYPE ushort
#else
#define FLOAT_TO_SNORM16_4(vector) vector
#define SNORM16_TO_FLOAT_4(vector) vector
#define FLOAT_TO_SNORM16_3(vector) vector
#define SNORM16_TO_FLOAT_3(vector) vector
#define FLOAT_TO_SNORM16_2(vector) vector
#define SNORM16_TO_FLOAT_2(vector) vector
#define FLOAT_TO_SNORM16(vector) vector
#define SNORM16_TO_FLOAT(vector) vector
#define VECTOR_FIELD_TYPE float
#define UNORM16_TO_FLOAT(v) v
#define FLOAT_TO_UNORM16(v) v
#define TDF_TYPE float
#endif

__kernel void GVFgaussSeidel(
        __read_only image3d_t r,
        __read_only image3d_t sqrMag,
        __private float mu,
        __private float spacing,
        __read_only image3d_t v_read,
        __global VECTOR_FIELD_TYPE * v_write
        ) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    // Calculate manhatten address
    int i = pos.x+pos.y+pos.z;

        // Compute red and put into v_write
        if(i % 2 == 0) {
            float value = native_divide(2.0f*mu*(
                    read_imagef(v_read, sampler, pos + (int4)(1,0,0,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(1,0,0,0)).x+
                    read_imagef(v_read, sampler, pos + (int4)(0,1,0,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(0,1,0,0)).x+
                    read_imagef(v_read, sampler, pos + (int4)(0,0,1,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(0,0,1,0)).x
                    ) - 2.0f*spacing*spacing*read_imagef(r, sampler, pos).x,
                    12.0f*mu+spacing*spacing*read_imagef(sqrMag, sampler, pos).x);
            v_write[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
        }
}

__kernel void GVFgaussSeidel2(
        __read_only image3d_t r,
        __read_only image3d_t sqrMag,
        __private float mu,
        __private float spacing,
        __read_only image3d_t v_read,
        __global VECTOR_FIELD_TYPE * v_write
        ) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    // Calculate manhatten address
    int i = pos.x+pos.y+pos.z;

        if(i % 2 == 0) {
            // Copy red
            float value = read_imagef(v_read, sampler, pos).x;
            v_write[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
        } else {
            // Compute black
                float value = native_divide(2.0f*mu*(
                    read_imagef(v_read, sampler, pos + (int4)(1,0,0,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(1,0,0,0)).x+
                    read_imagef(v_read, sampler, pos + (int4)(0,1,0,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(0,1,0,0)).x+
                    read_imagef(v_read, sampler, pos + (int4)(0,0,1,0)).x+
                    read_imagef(v_read, sampler, pos - (int4)(0,0,1,0)).x
                    ) - 2.0f*spacing*spacing*read_imagef(r, sampler, pos).x,
                    12.0f*mu+spacing*spacing*read_imagef(sqrMag, sampler, pos).x);
            v_write[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
        }
}


__kernel void addTwoImages(
        __read_only image3d_t i1,
        __read_only image3d_t i2,
        __global VECTOR_FIELD_TYPE * i3
        ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float v = read_imagef(i1,sampler,pos).x+read_imagef(i2,sampler,pos).x;
    i3[LPOS(pos)] = FLOAT_TO_SNORM16(v);
}


__kernel void createSqrMag(
        __read_only image3d_t vectorField,
        __global VECTOR_FIELD_TYPE * sqrMag
        ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    const float4 v = read_imagef(vectorField, sampler, pos);

    float mag = v.x*v.x+v.y*v.y+v.z*v.z;
    sqrMag[LPOS(pos)] = FLOAT_TO_SNORM16(mag);
}

__kernel void MGGVFFinish(
        __read_only image3d_t fx,
        __read_only image3d_t fy,
        __read_only image3d_t fz,
        __global VECTOR_FIELD_TYPE * vectorField
        ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    float4 value;
    value.x = read_imagef(fx,sampler,pos).x;
    value.y = read_imagef(fy,sampler,pos).x;
    value.z = read_imagef(fz,sampler,pos).x;
    value.w = length(value.xyz);
    vstore4(FLOAT_TO_SNORM16_4(value), LPOS(pos), vectorField);
}

__kernel void restrictVolume(
        __read_only image3d_t v_read,
        __global VECTOR_FIELD_TYPE * v_write
        ) {
        int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0)*2, get_global_size(1)*2, get_global_size(2)*2, 0};
    int4 pos = writePos*2;
    pos = select(pos, size-3, pos >= size-1);

    const int4 readPos = pos;
    const float value = 0.125*(
            read_imagef(v_read, hpSampler, readPos+(int4)(0,0,0,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(1,0,0,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(0,1,0,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(0,0,1,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(1,1,0,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(0,1,1,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(1,1,1,0)).x +
            read_imagef(v_read, hpSampler, readPos+(int4)(1,0,1,0)).x
            );

    v_write[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
}

__kernel void prolongate(
        __read_only image3d_t v_l_read,
        __read_only image3d_t v_l_p1,
        __global VECTOR_FIELD_TYPE * v_l_write
        ) {
    const int4 writePos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    const int4 readPos = convert_int4(floor(convert_float4(writePos)/2.0f));
    const float value = read_imagef(v_l_read, hpSampler, writePos).x + read_imagef(v_l_p1, hpSampler, readPos).x;
    v_l_write[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
}

__kernel void prolongate2(
        __read_only image3d_t v_l_p1,
        __global VECTOR_FIELD_TYPE * v_l_write
        ) {
    const int4 writePos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    const int4 readPos = convert_int4(floor(convert_float4(writePos)/2.0f));
    v_l_write[LPOS(writePos)] = FLOAT_TO_SNORM16(read_imagef(v_l_p1, hpSampler, readPos).x);
}

__kernel void residual(
        __read_only image3d_t r,
        __read_only image3d_t v,
        __read_only image3d_t sqrMag,
        __private float mu,
        __private float spacing,
        __global VECTOR_FIELD_TYPE * newResidual
        ) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    const float value = read_imagef(r, hpSampler, pos).x -
            ((mu*(
                    read_imagef(v, hpSampler, pos+(int4)(1,0,0,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(1,0,0,0)).x+
                    read_imagef(v, hpSampler, pos+(int4)(0,1,0,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(0,1,0,0)).x+
                    read_imagef(v, hpSampler, pos+(int4)(0,0,1,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(0,0,1,0)).x-
                    6*read_imagef(v, hpSampler, pos).x
                ) / (spacing*spacing))
            - read_imagef(sqrMag, hpSampler, pos).x*read_imagef(v, hpSampler, pos).x);

    newResidual[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
}

__kernel void fmgResidual(
        __read_only image3d_t vectorField,
        __read_only image3d_t v,
        __private float mu,
        __private float spacing,
        __private int component,
        __global VECTOR_FIELD_TYPE * newResidual
        ) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    float4 vector = read_imagef(vectorField, sampler, pos);
    float v0;
    if(component == 1) {
        v0 = vector.x;
    } else if(component == 2) {
       v0 = vector.y;
    } else {
       v0 = vector.z;
    }
    const float sqrMag = vector.x*vector.x+vector.y*vector.y+vector.z*vector.z;

    float residue = (mu*(
                    read_imagef(v, hpSampler, pos+(int4)(1,0,0,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(1,0,0,0)).x+
                    read_imagef(v, hpSampler, pos+(int4)(0,1,0,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(0,1,0,0)).x+
                    read_imagef(v, hpSampler, pos+(int4)(0,0,1,0)).x+
                    read_imagef(v, hpSampler, pos-(int4)(0,0,1,0)).x-
                    6*read_imagef(v, hpSampler, pos).x)
                ) / (spacing*spacing);

    const float value = -sqrMag*v0-(residue - sqrMag*read_imagef(v, hpSampler, pos).x);

    newResidual[LPOS(writePos)] = FLOAT_TO_SNORM16(value);
}


__kernel void initFloatBuffer(
        __global VECTOR_FIELD_TYPE * buffer
        ) {
        buffer[get_global_id(0)] = FLOAT_TO_SNORM16(0.0f);
}

__kernel void createVectorField(
        __read_only image3d_t volume,
        __global VECTOR_FIELD_TYPE * vectorField
        ) {
    const int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 gradient;
    gradient.x = 0.5f*(read_imagef(volume, sampler, pos+(int4)(1,0,0,0)).x - read_imagef(volume, sampler, pos-(int4)(1,0,0,0)).x);
    gradient.y = 0.5f*(read_imagef(volume, sampler, pos+(int4)(0,1,0,0)).x - read_imagef(volume, sampler, pos-(int4)(0,1,0,0)).x);
    gradient.z = 0.5f*(read_imagef(volume, sampler, pos+(int4)(0,0,1,0)).x - read_imagef(volume, sampler, pos-(int4)(0,0,1,0)).x);
    gradient.w = 0;

    vstore4(FLOAT_TO_SNORM16_4(gradient), LPOS(pos), vectorField);
}
