//
//  io.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//

#include "io.h"



//write xdmf
void wrt_xmf(struct ocl_obj *ocl, struct msh_obj *msh, int idx)
{
    FILE* file1;
    char file1_name[250];
//    float* ptr1;
    
    //file name
    sprintf(file1_name, "%s/xmf/%s.%02u%02u%02u.%03u.xmf", ROOT_WRITE, "grid", msh->le.x, msh->le.y, msh->le.z, idx);
    
    //open
    file1 = fopen(file1_name,"w");
    
    fprintf(file1,"<Xdmf>\n");
    fprintf(file1,"  <Domain>\n");
    fprintf(file1,"    <Topology name=\"topo\" TopologyType=\"3DCoRectMesh\" Dimensions=\"%u %u %u\"></Topology>\n", msh->nv.z, msh->nv.y, msh->nv.x);
    fprintf(file1,"      <Geometry name=\"geo\" Type=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file1,"        <!-- Origin -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", 0e0f, 0e0f, 0e0f);
    fprintf(file1,"        <!-- DxDyDz -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", msh->dx, msh->dx, msh->dx);
    fprintf(file1,"      </Geometry>\n");
    fprintf(file1,"      <Grid Name=\"T1\" GridType=\"Uniform\">\n");
    fprintf(file1,"        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n");
    fprintf(file1,"        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n");
    fprintf(file1,"         <Attribute Name=\"uu\" Center=\"Cell\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
    fprintf(file1,"             /Users/toby/Downloads/raw/uu.%02u%02u%02u.%03u.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"         <Attribute Name=\"bb\" Center=\"Cell\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
    fprintf(file1,"             /Users/toby/Downloads/raw/bb.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"         <Attribute Name=\"rr\" Center=\"Cell\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
    fprintf(file1,"             /Users/toby/Downloads/raw/rr.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"         <Attribute Name=\"aa\" Center=\"Cell\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%u %u %u 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", msh->ne.z, msh->ne.y, msh->ne.x);
    fprintf(file1,"             /Users/toby/Downloads/raw/aa.%02u%02u%02u.%03d.raw\n", msh->le.x, msh->le.y, msh->le.z, idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    fprintf(file1,"    </Grid>\n");
    fprintf(file1," </Domain>\n");
    fprintf(file1,"</Xdmf>\n");
    
    //clean up
    fclose(file1);
}



//float
void wrt_flt1(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, int n_tot)
{
    FILE* file1;
    char file1_name[250];
    cl_float* ptr1;
    
    //buffer
    sprintf(file1_name, "%s/raw/%s.%02u%02u%02u.%03d.raw", ROOT_WRITE, dsc, msh->le.x, msh->le.y, msh->le.z, idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, *buf, CL_TRUE, CL_MAP_READ, 0, n_tot*sizeof(cl_float), 0, NULL, NULL, &ocl->err);
    fwrite(ptr1, sizeof(float), n_tot, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, *buf, ptr1, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}


//float4
void wrt_flt3(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx, int n_tot)
{
    FILE* file1;
    char file1_name[250];
    cl_float4* ptr1;
    
    //buffer
    sprintf(file1_name, "%s/raw/%s.%02u%02u%02u.%03d.raw", ROOT_WRITE, dsc, msh->le.x, msh->le.y, msh->le.z, idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, *buf, CL_TRUE, CL_MAP_READ, 0, n_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    //skip .w
    for(int i=0; i<n_tot; i++)
    {
        fwrite(&ptr1[i], sizeof(float), 3, file1);
    }
    clEnqueueUnmapMemObject(ocl->command_queue, *buf, ptr1, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}
