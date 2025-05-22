//
//  ocl.h
//  mg1
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef ocl_h
#define ocl_h



#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


//object
struct ocl_obj
{
    //environment
    cl_int              err;
    cl_platform_id      platform_id;
    cl_device_id        device_id;
    cl_uint             num_devices;
    cl_uint             num_platforms;
    cl_context          context;
    cl_command_queue    command_queue;
    cl_program          program;
    char                device_str[50];
    size_t              device_num[3];
    cl_event            event;
};



void ocl_ini(struct ocl_obj *ocl);
void ocl_fin(struct ocl_obj *ocl);



#endif /* ocl_h */
