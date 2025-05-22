//
//  utl.cl
//  fsi2
//
//  Created by toby on 05.05.25.
//  Copyright © 2025 Toby Simpson. All rights reserved.
//

#ifndef utl_h
#define utl_h



/*
 ===================================
 constants
 ===================================
 */


//util
constant int3       off_fac[6]  = {{-1,0,0},{+1,0,0},{0,-1,0},{0,+1,0},{0,0,-1},{0,0,+1}};
constant int3       off_vtx[8]  = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};


/*
 ===================================
 util
 ===================================
 */

//global index
int utl_idx1(int3 pos, int3 dim)
{
    return pos.x + dim.x*pos.y + dim.x*dim.y*pos.z;
}

//local index 3x3x3
int utl_idx3(int3 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

//in-bounds
int utl_bnd1(int3 pos, int3 dim)
{
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int utl_bnd2(int3 pos, int3 dim)
{
    return any(pos==0)||any(pos==(dim-1));
}


#endif
