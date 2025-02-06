__kernel void gray(__read_only image2d_t input, __write_only image2d_t output){
   // sampler for control how does 2D object are read with read_imagef function (addressing mode, filter mode, normalized coordinates)
   const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

   // get dimensions of input
   const int2 size = get_image_dim(input);
   // get coordinates of the image
   int2 coord = (int2)(get_global_id(0),get_global_id(1));

   float4 color = (float4)(0,0,0,1.0f);
   float4 srcColor = read_imagef(input,sampler,(int2)(coord.x,coord.y));
   // convert actual values to B&W
   color.xyz = srcColor.x*0.21 + srcColor.y*0.72 + srcColor.z*0.07;
    // write output image
    write_imagef(output,coord,color);
}