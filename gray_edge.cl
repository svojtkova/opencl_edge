__kernel void detectedge(__global int *buff,__global int *out){
       // work group and work item ids
      int x = get_global_id(1);
      int y = get_global_id(0);
      // get the width first input value in kernel
      int width = %d;

      // convert to B&W
      buff[width*y+x] = (int)buff[width*y+x]*0.21 + buff[width*y+x]*0.72 + buff[width*y+x]*0.07;

      // wait for work items with local and global memory
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      // horizontal sobel operator
      float hx = buff[width*(y-1) + (x-1)] + 2*buff[width*(y)+(x-1)]
			 +buff[width*(y+1)+(x-1)] - buff[width*(y-1)+(x+1)]
			 -2*buff[width*(y)+(x+1)] - buff[width*(y+1)+(x+1)];

      // vertical sobel operator
      float vx = buff[width*(y-1)+(x-1)] +2*buff[width*(y-1)+(x)] +buff[width*(y-1)+(x+1)]
			 -buff[width*(y+1)+(x-1)] - 2*buff[width*(y+1)+(x)] - buff[width*(y+1)+(x+1)];

      // gradient
      int value = (int)sqrt(hx*hx+vx*vx);
      // detect edge
      int val = (value > 255 ? 255 : value);
      // return value
      out[y*width + x] = val;
  }
