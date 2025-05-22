//
//  ocl.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#include <stdio.h>
#include "ocl.h"


//init
void ocl_ini(struct ocl_obj *ocl)
{
    /*
     =============================
     environment
     =============================
     */
    
    ocl->err            = clGetPlatformIDs(1, &ocl->platform_id, &ocl->num_platforms);                                              //platform
    ocl->err            = clGetDeviceIDs(ocl->platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl->device_id, &ocl->num_devices);              //devices
    ocl->context        = clCreateContext(NULL, ocl->num_devices, &ocl->device_id, NULL, NULL, &ocl->err);                          //context
//    ocl->command_queue  = clCreateCommandQueue(ocl->context, ocl->device_id, 0, &ocl->err);                                         //command queue
    ocl->command_queue  = clCreateCommandQueue(ocl->context, ocl->device_id, CL_QUEUE_PROFILING_ENABLE, &ocl->err);                 //command queue, with profiling
    ocl->err            = clGetDeviceInfo(ocl->device_id, CL_DEVICE_NAME, 50, &ocl->device_str[0], NULL);                           //device info
    ocl->err            = clGetDeviceInfo(ocl->device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 24, &ocl->device_num, NULL);                 //device info CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_ITEM_SIZES

    printf("%s\n", ocl->device_str);
    printf("%zu,%zu,%zu\n", ocl->device_num[0], ocl->device_num[1], ocl->device_num[2]);

    /*
     =============================
     program
     =============================
     */
    
    //src
    FILE* src_file = fopen("prg.cl", "r");
    if(!src_file)
    {
        fprintf(stderr, "prg.cl not found\n");
        exit(1);
    }

    //length
    fseek(src_file, 0, SEEK_END);
    size_t src_len =  ftell(src_file);
    fseek(src_file, 0, SEEK_SET);

    //source
    char *src_mem = (char*)malloc(src_len);
    fread(src_mem, sizeof(char), src_len, src_file);
    fclose(src_file);

    //create
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&src_mem, (const size_t*)&src_len, &ocl->err);
    printf("prg %d\n",ocl->err);
    
    //build
    ocl->err = clBuildProgram(ocl->program, 1, &ocl->device_id, NULL, NULL, NULL);  //-cl-fast-relaxed-math
    printf("bld %d\n",ocl->err);
    
    //clean
    ocl->err = clUnloadPlatformCompiler(ocl->platform_id);
    free(src_mem);
    
    /*
     =============================
     log
     =============================
     */

    //log
    size_t log_size = 0;
    
    //log size
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    //allocate
    char *log = (char*)malloc(log_size);

    //log text
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    //print
    printf("%s\n", log);

    //clear
    free(log);
}


//final
void ocl_fin(struct ocl_obj *ocl)
{
    //queue
    ocl->err = clFlush(ocl->command_queue);
    ocl->err = clFinish(ocl->command_queue);
    
    //context
    ocl->err = clReleaseProgram(ocl->program);
    ocl->err = clReleaseCommandQueue(ocl->command_queue);
    ocl->err = clReleaseContext(ocl->context);
    
    return;
}
