//
//  prg.cl
//  fsi2
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


#include "utl.h"


/*
 ===================================
 mesh
 ===================================
 */

//object
struct msh_obj
{
    int3    le;
    int3    ne;
    int3    nv;
    
    int     ne_tot;
    int     nv_tot;
    
    float   dt;
    float   dx;
    float   dx2;
    float   rdx;
    float   rdx2;
    
    ulong   nv_sz[3];
    ulong   ne_sz[3];
    ulong   iv_sz[3];
    ulong   ie_sz[3];
};


/*
 ===================================
 ini
 ===================================
 */


kernel void ele_ini(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *bb,
                    global float           *rr,
                    global float           *aa)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = msh.dx*(convert_float3(ele_pos) + 0.5f);
    
    float u = sin(M_PI*x.x);
    
    //write
    uu[ele_idx] = u*utl_bnd2(ele_pos, msh.ne);
    bb[ele_idx] = 0e0f;
    rr[ele_idx] = 0e0f;
    aa[ele_idx] = u;

    
    return;
}


/*
 ============================
 operator
 ============================
 */



//forward
kernel void ele_fwd(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)} + 1; //interior
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float s = 0.0f;
    
    //stencil
    for(int i=0; i<6; i++)
    {
        int3    adj_pos = ele_pos + off_fac[i];
        int     adj_idx = utl_idx1(adj_pos, msh.ne);
        
        s += uu[adj_idx];
    }
    
    //store
    bb[ele_idx] = msh.rdx2*(6.0f*uu[ele_idx] - s);
    
    return;
}



//residual
kernel void ele_res(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)} + 1; //interior
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float s = 0.0f;
    
    //stencil
    for(int i=0; i<6; i++)
    {
        int3    adj_pos = ele_pos + off_fac[i];
        int     adj_idx = utl_idx1(adj_pos, msh.ne);
        
        s += uu[adj_idx];
    }
    
    //scale
    float Au = msh.rdx2*(6.0f*uu[ele_idx] - s);
    
    //store
    rr[ele_idx] = bb[ele_idx] - Au;
    
    return;
}


//jacobi
kernel void ele_jac(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *rr)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)} + 1; //interior
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //du = D^-1(r)
    uu[ele_idx] += 0.99f*msh.dx2*rr[ele_idx]/6.0f;
    
    return;
}


/*
 ============================
 multigrid
 ============================
 */


//projection
kernel void ele_prj(const  struct msh_obj   mshc,    //coarse    (out)
                    global float            *rrf,    //fine      (in)
                    global float            *uuc,    //coarse    (out)
                    global float            *bbc)    //coarse    (out)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, mshc.ne);
    
    
    //fine
    int3 pos = 2*ele_pos;
    int3 dim = 2*mshc.ne;
    
    //sum
    float s = 0e0f;
    
    //sum fine
    for(int i=0; i<8; i++)
    {
        int3 adj_pos = pos + off_vtx[i];
        int  adj_idx = utl_idx1(adj_pos, dim);
        s += rrf[adj_idx];
    }
    
    //store/reset
    uuc[ele_idx] = 0e0f;
    bbc[ele_idx] = s;
    
    return;
}


//interp
kernel void ele_itp(const  struct msh_obj   mshf,    //fine      (out)
                    global float            *uuc,    //coarse    (in)
                    global float            *uuf)    //fine      (out)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, mshf.ne);   //fine
    
    //    printf("%2d %v3hlu\n", ele_idx, ele_pos/2);
    
    //coarse
    int3 pos = ele_pos/2;
    int3 dim = mshf.ne/2;
    
    //write - scale
    uuf[ele_idx] += 0.125f*uuc[utl_idx1(pos, dim)];
    
    return;
}

/*
 ============================
 error
 ============================
 */


////residual squared
//kernel void ele_rsq(const  struct msh_obj   msh,
//                    global float            *rr)
//{
//    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
//    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
//    
//    //square/write
//    rr[ele_idx] = pown(rr[ele_idx],2.0f);
//    
//    return;
//}


//error squared
kernel void ele_err(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *aa,
                    global float            *rr)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //square/write
    rr[ele_idx] = aa[ele_idx] - uu[ele_idx];
    
    return;
}

/*
 ============================
 reduction
 ============================
 */


////fold
//kernel void vec_sum(global float *uu,
//                    const  int   n)
//{
//    int i = get_global_id(0);
//    int m = get_global_size(0);
//
////    printf("%d %d %d %f %f\n",i, n, m, uu[i], uu[m+i]);
//    
//    if((m+i)<n)
//    {
//        uu[i] += uu[m+i];
//    }
//      
//    return;
//}



//fold - inf
kernel void vec_sum(global float *uu,
                    const  int   n)
{
    int i = get_global_id(0);
    int m = get_global_size(0);

//    printf("%d %d %d %f %f\n",i, n, m, uu[i], uu[m+i]);
    
    float u1 = uu[i];
    float u2 = uu[m+i];
    
    if((m+i)<n)
    {
        uu[i] = (fabs(u1)>fabs(u2))?u1:u2;
    }
      
    return;
}
