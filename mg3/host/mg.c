//
//  mg.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//
#include <time.h>
#include <math.h>
#include "mg.h"


//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh)
{
    printf("mg %d %d %d\n",mg->nl, mg->nj, mg->nc);
    
    //levels
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));
    
    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //instance
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //dims
        lvl->msh.le = (cl_int3){msh->le.x-l, msh->le.y-l, msh->le.z-l};
        
        //dx
        lvl->msh.dx = msh->dx*powf(2e0f,l);
        lvl->msh.dt = msh->dt;
        
        //mesh
        msh_ini(&lvl->msh);
        
        //device
        lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->bb = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->rr = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->aa = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
    }
    
    //ini
    mg->ele_ini = clCreateKernel(ocl->program, "ele_ini", &ocl->err);
    
    //trans
    mg->ele_prj = clCreateKernel(ocl->program, "ele_prj", &ocl->err);
    mg->ele_itp = clCreateKernel(ocl->program, "ele_itp", &ocl->err);
    
    //err
    mg->ele_rsq = clCreateKernel(ocl->program, "ele_rsq", &ocl->err);
    mg->ele_esq = clCreateKernel(ocl->program, "ele_esq", &ocl->err);
    mg->vec_sum = clCreateKernel(ocl->program, "vec_sum", &ocl->err);
    
    //poisson
    mg->ops[0].ele_fwd = clCreateKernel(ocl->program, "ele_fwd", &ocl->err);
    mg->ops[0].ele_res = clCreateKernel(ocl->program, "ele_res", &ocl->err);
    mg->ops[0].ele_jac = clCreateKernel(ocl->program, "ele_jac", &ocl->err);
    
    return;
}


//forward b = Au, with timings
void mg_fwd(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl)
{
    //wall time ms/ns (longs)
    struct timespec cpu_t0;
    struct timespec cpu_t1;
    
    //ocl time ns
    cl_ulong gpu_t0;
    cl_ulong gpu_t1;
    
    //args
    ocl->err = clSetKernelArg(op->ele_fwd,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->ele_fwd,  1, sizeof(cl_mem),            (void*)&lvl->aa);
    ocl->err = clSetKernelArg(op->ele_fwd,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    
    //clock
    clock_gettime(CLOCK_REALTIME, &cpu_t0);

    //fwd
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_fwd, 3, NULL, lvl->msh.ie_sz, NULL, 0, NULL, &ocl->event);
    
    //complete
    clWaitForEvents(1, &ocl->event);
    clock_gettime(CLOCK_REALTIME, &cpu_t1);

    ocl->err = clGetEventProfilingInfo(ocl->event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gpu_t0, NULL);
    ocl->err = clGetEventProfilingInfo(ocl->event, CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &gpu_t1, NULL);
    
 
    printf("fwd [%4lu,%4lu,%4lu]%12lu %e %e\n",
           lvl->msh.iv_sz[0],lvl->msh.iv_sz[1],lvl->msh.iv_sz[2],
           lvl->msh.iv_sz[0]*lvl->msh.iv_sz[1]*lvl->msh.iv_sz[2],
           (1e9*cpu_t1.tv_sec + cpu_t1.tv_nsec) - (1e9*cpu_t0.tv_sec + cpu_t0.tv_nsec),
           (double)(gpu_t1 - gpu_t0));
    
    return;
}


//jacobi
void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl)
{
    //args
    ocl->err = clSetKernelArg(op->ele_res,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->ele_res,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->ele_res,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->ele_res,  3, sizeof(cl_mem),            (void*)&lvl->rr);

    ocl->err = clSetKernelArg(op->ele_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->ele_jac,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->ele_jac,  2, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //smooth
    for(int j=0; j<mg->nj; j++)
    {
        //solve
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_res, 3, NULL, lvl->msh.ie_sz, NULL, 0, NULL, NULL);
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_jac, 3, NULL, lvl->msh.ie_sz, NULL, 0, NULL, NULL);
    }
    
    //residual
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_res, 3, NULL, lvl->msh.ie_sz, NULL, 0, NULL, NULL);
    
    return;
}


//interp
void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->ele_itp,  0, sizeof(struct msh_obj),    (void*)&lf->msh);     //fine
    ocl->err = clSetKernelArg(mg->ele_itp,  1, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->ele_itp,  2, sizeof(cl_mem),            (void*)&lf->uu);      //fine
    
    //interp
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_itp, 3, NULL, lf->msh.ne_sz, NULL, 0, NULL, NULL);
    
    return;
}


//project
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->ele_prj,  0, sizeof(struct msh_obj),    (void*)&lc->msh);     //coarse
    ocl->err = clSetKernelArg(mg->ele_prj,  1, sizeof(cl_mem),            (void*)&lf->rr);      //fine
    ocl->err = clSetKernelArg(mg->ele_prj,  2, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->ele_prj,  3, sizeof(cl_mem),            (void*)&lc->bb);      //coarse
    
    //project
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_prj, 3, NULL, lc->msh.ne_sz, NULL, 0, NULL, NULL);
    
    return;
}

//v-cycles
void mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op)
{
    //cycle
    for(int c=0; c<mg->nc; c++)
    {
        //descend
        for(int l=0; l<(mg->nl-1); l++)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //pre
            mg_jac(ocl, mg, op, lf);
            
            //prj
            mg_prj(ocl, mg, lf, lc);
            
        } //dsc
        
        //coarse
        mg_jac(ocl, mg, op, &mg->lvls[mg->nl-1]);
        
        //ascend
        for(int l=(mg->nl-2); l>=0; l--)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //itp
            mg_itp(ocl, mg, lf, lc);
           
            //post
            mg_jac(ocl, mg, op, lf);
            
        } //dsc
        
        //norms
        mg_nrm(ocl, mg,  &mg->lvls[0]);
        
    }   //cycle
    
    return;
}




//norms
void mg_nrm(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl)
{
    //resid
    float r = mg_red(ocl, mg, lvl->rr, lvl->msh.ne_tot);
    
    //err
    ocl->err = clSetKernelArg(mg->ele_esq,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(mg->ele_esq,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(mg->ele_esq,  2, sizeof(cl_mem),            (void*)&lvl->aa);
    ocl->err = clSetKernelArg(mg->ele_esq,  3, sizeof(cl_mem),            (void*)&lvl->rr);
    
    //err
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_esq, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, &ocl->event);
    float e = mg_red(ocl, mg, lvl->rr, lvl->msh.ne_tot);
    
    float dx3 = lvl->msh.dx*lvl->msh.dx2;
    
    //norms
//    printf("nrm [%2u,%2u,%2u] %+e %+e  %+e %+e  %+e %+e\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, r, e, dx3*r, dx3*e, sqrt(dx3*r), sqrt(dx3*e));
    printf("nrm [%2u,%2u,%2u] %+e %+e\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, sqrt(dx3*r), sqrt(dx3*e));
//    printf("nrm [%2u,%2u,%2u] %+e %+e\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, r, e);
    
    return;
}


//fold (max 1024ˆ3)
float mg_red(struct ocl_obj *ocl, struct mg_obj *mg, cl_mem uu, cl_int n)
{
    //args
    ocl->err = clSetKernelArg(mg->vec_sum, 0, sizeof(cl_mem), (void*)&uu);
    ocl->err = clSetKernelArg(mg->vec_sum, 1, sizeof(cl_int), (void*)&n);

    uint l = ceil(log2(n));
    
    //loop
    for(int i=0; i<l; i++)
    {
        size_t p = pow(2,l-i-1);
        
//        printf("%2d %2d %u %zu\n", i, l, n, p);
    
        //calc
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vec_sum, 1, NULL, &p, NULL, 0, NULL, NULL);
    }
    
    //result
    float r;
    
    //read
    ocl->err = clEnqueueReadBuffer(ocl->command_queue, uu, CL_TRUE, 0, sizeof(float), &r, 0, NULL, NULL);

    return r;
}


//final
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    ocl->err = clReleaseKernel(mg->ele_ini);
    ocl->err = clReleaseKernel(mg->ele_prj);
    ocl->err = clReleaseKernel(mg->ele_itp);
    
    ocl->err = clReleaseKernel(mg->ele_esq);
    ocl->err = clReleaseKernel(mg->vec_sum);
    
    ocl->err = clReleaseKernel(mg->ops[0].ele_res);
    ocl->err = clReleaseKernel(mg->ops[0].ele_jac);
    ocl->err = clReleaseKernel(mg->ops[0].ele_fwd);

    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //device
        ocl->err = clReleaseMemObject(mg->lvls[l].uu);
        ocl->err = clReleaseMemObject(mg->lvls[l].bb);
        ocl->err = clReleaseMemObject(mg->lvls[l].rr);
        ocl->err = clReleaseMemObject(mg->lvls[l].aa);
    }
    
    //mem
    free(mg->lvls);

    return;
}
