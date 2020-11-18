import numpy as np
from time import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mendelbrot_graph, int max_iters, float upper_bound",
    """
    mendalbrot_graph[i] = 1;

pycuda::complex<float> c = lattice[i];
pycuda::complex<float> z(0,0)

for (int j=0; j<max_iter; j++){
   z = z*z + c
   if (abs(z) > upper_bound){
     mandalbrot_graph[i] = 0;
     break;
   }
}
""",
    "mandel_ker")


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    real_values = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_values = np.matrix(np.linspace(imag_low, imag_high, height), dtype=np.complex64)
    mandelbrot_lattice = np.array(real_values + imag_values.transpose(), dtype=np.complex64)

    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)
    # Allocating Empty GPU Array:
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)
    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))

    mandelbrot_graph = mandelbrot_graph_gpu.get()
    return mandelbrot_graph


if __name__ == "__main__":
    t = time()
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)
    t_ = time()

    mandel_elpsed = t_ - t

    t__ = time()
    fig = plt.figure(1)
    plt.imshow('mandelbrot.png', dpi=fig.dpi)

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_elpsed))
