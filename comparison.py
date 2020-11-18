import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel

host_data = np.float32(np.random.random(50000000))

gpu_kernel = ElementwiseKernel(
    "float * in, float * out",
    "out[i] = 2*in[i]",
    "gpu_kernel"
)

#Allocating a blank GPU array:

def compare():
    t1 = time()
    host_data_2x = host_data * np.float32(2)

    t2 = time()
    print(f'Time to compute on CPU: {t2-t1}')

    device_data = gpuarray.to_gpu(host_data)
    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_kernel(device_data, device_data_2x)
    t2 = time()

    print(f'Time to compute on MX250: {t2-t1}')

    from_gpu = device_data_2x.get()

    assert np.allclose(from_gpu, host_data_2x)


if __name__ == "__main__":
    compare()