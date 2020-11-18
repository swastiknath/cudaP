import pycuda.driver as drv
import pycuda

drv.init()

print(drv.Device.count())
device = drv.Device(0)
print(device.name())
print(device.compute_capability())
print(device.total_memory()/(1024**2))


print(device.get_attributes())