//
//  msh.h
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#ifndef msh_h
#define msh_h

#include <stdio.h>
#include <math.h>
#include "ocl.h"

//object
struct msh_obj
{
    cl_int3    le;
    cl_int3    ne;
    cl_int3    nv;
    
    cl_int     ne_tot;
    cl_int     nv_tot;
    
    cl_float    dt;
    cl_float    dx;
    cl_float    dx2;
    cl_float    rdx;
    cl_float    rdx2;
    
    size_t      nv_sz[3];
    size_t      ne_sz[3];
    size_t      iv_sz[3];
    size_t      ie_sz[3];
};


void msh_ini(struct msh_obj *msh);

#endif /* msh_h */
