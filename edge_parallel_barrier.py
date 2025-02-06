import numpy as np
import pyopencl as cl
from PIL import Image
from time import time


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
    img_size = img_width * img_height

    # creating context and queue connected to this context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # convert image to 8bit
    img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE)

    # The image is one uint8 per pixel, buffer is according to data.nbytes.
    # The kernel is reading and writing 4 byte pixels.
    data = np.asarray(img).astype(np.uint32)

    # create buffers input and output image
    im = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, data.nbytes)

    # assigning program to context
    prgm = cl.Program(ctx, getKernel('gray_edge.cl') % (data.shape[1])).build()

    # adjusting the size of work groups
    globalWorkSize = (RoundUp(localWorkSize[0], data.shape[0]),
                      RoundUp(localWorkSize[1], data.shape[1]))

    # run the program
    prgm.detectedge(queue, globalWorkSize, localWorkSize, im, out)

    result = np.empty_like(data)

    # copy output from program to readable form from out to result
    cl.enqueue_copy(queue, result, out)
    result = result.astype(np.uint8)
    img = Image.fromarray(result)

    end_time = time()
    print("{}".format(end_time - start_time))
    # save image
    img.save('parallel_barrier.png')


if __name__ == '__main__':
    first_file = '500x500.png'
    second_file = '1000x1000.png'
    third_file = '1500x1500.png'
    print("Picture " + first_file + ' 500x500')
    print('1')
    gray_edge(first_file, (1, 1))    # with 1 item in each work group
    print('2')
    gray_edge(first_file, (2,2))     # with 2 items in each work group
    print('8')
    gray_edge(first_file, (8,8))     # with 8 items in each work group
    print('16')
    gray_edge(first_file, (16,16))   # with 16 items in each work group

    print("Picture " + second_file + ' 1000x1000')
    print('1')
    gray_edge(second_file, (1, 1))     # with 1 item in each work group
    print('2')
    gray_edge(second_file, (2, 2))     # with 2 items in each work group
    print('8')
    gray_edge(second_file, (8, 8))     # with 8 items in each work group
    print('16')
    gray_edge(second_file, (16, 16))   # with 16 items in each work group

    print("Picture " + third_file + ' 1500x1500')
    print('1')
    gray_edge(third_file, (1, 1))     # with 1 item in each work group
    print('2')
    gray_edge(third_file, (2, 2))     # with 2 items in each work group
    print('8')
    gray_edge(third_file, (8, 8))     # with 8 items in each work group
    print('16')
    gray_edge(third_file, (16, 16))   # with 16 items in each work group




