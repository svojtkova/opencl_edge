
import numpy as np
import pyopencl as cl
from PIL import Image
from time import time
import os


# adjust work size of work groups
def RoundUp(groupSize, globalSize):
    r = globalSize % groupSize;
    if r == 0:
        return globalSize
    else:
        return globalSize + groupSize - r


# open file with written kernel program
def getKernel(krnl):
    kernel = open(krnl).read()
    return kernel


# function with openCL conversion to B&W and edge detection
def gray_edge(filename, localWorkSize):
    start_time = time()
    # image loading
    img = Image.open(filename)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_width = img.size[0]
    img_height = img.size[1]

    # prepare host memory for OpenCL
    img_bytes = img.tobytes()

    #  ctx = cl.create_some_context()
    # creating context and queue connected to this context
    platforms = cl.get_platforms()
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Compile OpenCl Program by character string
    gray = cl.Program(ctx, """
        __kernel void gray(__read_only image2d_t input,
                               __write_only image2d_t output){
           // sampler for control how does 2D object are read with read_imagef function (addressing mode, filter mode, normalized coordinates)
           const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                                     CLK_ADDRESS_CLAMP_TO_EDGE |
                                     CLK_FILTER_NEAREST;
            // get dimensions of input
           const int2 size = get_image_dim(input);
            // get coordinates of the image
           int2 coord = (int2)(get_global_id(0),get_global_id(1));
            float4 color = (float4)(0,0,0,1.0f);
            //read input image with sampler flags
           float4 srcColor = read_imagef(input,sampler,(int2)(coord.x,coord.y));
           // convert actual values to B&W
           color.xyz = srcColor.x*0.21 + srcColor.y*0.72 + srcColor.z*0.07;
            // write output image
            write_imagef(output,coord,color);
        }

        """).build()

    # Image format with channel_order(color), channel_type(size)
    dev_image_format = cl.ImageFormat(cl.channel_order.RGBA,
                                      cl.channel_type.UNSIGNED_INT8)

    # Image(context, flags, format, shape=None, pitches=None, hostbuf=None, is_array=False, buffer=None)
    # If hostbuf is given and shape is None, then hostbuf.shape is used as the shape parameter
    input_image = cl.Image(ctx,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           dev_image_format,
                           img.size,
                           None,
                           img_bytes)
    gray_image = cl.Image(ctx,
                          cl.mem_flags.WRITE_ONLY,
                          dev_image_format,
                          img.size)

    # adjusting work size
    globalWorkSize = (RoundUp(localWorkSize[0], img.size[0]),
                      RoundUp(localWorkSize[1], img.size[1]))

    # run kernel function gray
    gray.gray(queue, globalWorkSize, localWorkSize, input_image, gray_image)

    # prepare for reading output grey image
    buffer = np.zeros(img_width * img_height * 4, np.uint8)
    origin = (0, 0, 0)
    region = (img_width, img_height, 1)
    # copy of image from kernel
    cl.enqueue_copy(queue, dest=buffer, src=gray_image,
                    origin=origin, region=region).wait()

    # convert output image from previous kernel to 8 bit image
    out_im = Image.frombytes("RGBA", img.size, buffer.tobytes())
    img = out_im.convert('RGB').convert('P', palette=Image.ADAPTIVE)

    # prepare image for next kerneô
    data = np.asarray(img).astype(np.uint32)

    # buffer preparation of input/output data for kernel function
    im = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, data.nbytes)
    # init of kernel function
    prgm = cl.Program(ctx, getKernel('edge.cl') % (data.shape[1])).build()

    # get the size of work items in work group
    globalWorkSize = (RoundUp(localWorkSize[0], data.shape[0]),
                      RoundUp(localWorkSize[1], data.shape[1]))

    # run kernel program detectedge
    prgm.detectedge(queue, globalWorkSize, localWorkSize, im, out)

    # copy output from program to readable form from out to result
    result = np.empty_like(data)
    cl.enqueue_copy(queue, result, out)
    result = result.astype(np.uint8)

    img = Image.fromarray(result)

    end_time = time()
    print("{}".format(end_time - start_time))
    # save image
    img.save('parallel.png')


if __name__ == '__main__':
    first_file = '500x500.png'
    second_file = '1000x1000.png'
    third_file = '1500x1500.png'
    print("Picture " + first_file + ' 500x500')
    print('1')
    gray_edge(first_file, (1, 1))
    print('2')
    gray_edge(first_file, (2,2))
    print('8')
    gray_edge(first_file, (8,8))
    print('16')
    gray_edge(first_file, (16,16))

    print("Picture " + second_file + ' 1000x1000')
    print('1')
    gray_edge(second_file, (1, 1))
    print('2')
    gray_edge(second_file, (2, 2))
    print('8')
    gray_edge(second_file, (8, 8))
    print('16')
    gray_edge(second_file, (16, 16))

    print("Picture " + third_file + ' 1500x1500')
    print('1')
    gray_edge(third_file, (1, 1))
    print('2')
    gray_edge(third_file, (2, 2))
    print('8')
    gray_edge(third_file, (8, 8))
    print('16')
    gray_edge(third_file, (16, 16))

