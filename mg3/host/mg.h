//
//  mg.h
//  mg1
//
//  Created by toby on 10.06.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef mg_h
#define mg_h

#include <math.h>
#include "msh.h"

//object
struct lvl_obj
{
    struct msh_obj  msh;
    
    //memory
    cl_mem  uu;
    cl_mem  bb;
    cl_mem  rr;
    cl_mem  aa;
};


//object
struct op_obj
{
    //operator
    cl_kernel       ele_fwd;
    cl_kernel       ele_res;
    cl_kernel       ele_jac;
};


//object
struct mg_obj
{
    //levels
    cl_int             nl;     //levels
    cl_int             nj;     //jac iter
    cl_int             nc;     //v-cycles
    
    //array
    struct lvl_obj *lvls;
    
    //ini
    cl_kernel       ele_ini;
    
    //trans
    cl_kernel       ele_prj;
    cl_kernel       ele_itp;
    
    //err
    cl_kernel       ele_rsq;
    cl_kernel       ele_esq;
    cl_kernel       vec_sum;
    
    //ops
    struct op_obj ops[1];

};


//init
void    mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh);
void    mg_fin(struct ocl_obj *ocl, struct mg_obj *mg);

void    mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);
void    mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc);

void    mg_fwd(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl);
void    mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl);
void    mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl);

void    mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op);

float   mg_red(struct ocl_obj *ocl, struct mg_obj *mg, cl_mem uu, const cl_int n);
void    mg_nrm(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl);

#endif /* mg_h */
