//
//  io.h
//  mg1
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef io_h
#define io_h


#include <stdio.h>
#include "ocl.h"
#include "msh.h"


#define ROOT_WRITE  "/Users/toby/Downloads/"



void wrt_xmf(struct ocl_obj *ocl, struct msh_obj *msh, int idx);
void wrt_flt1(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, cl_int n_tot);
void wrt_flt3(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, cl_int n_tot);

#endif /* io_h */
