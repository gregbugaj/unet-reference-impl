import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time
import os

import mxnet as mx
from mxnet import profiler

batch_shape = (1, 3, 224, 224)
resnet18 = vision.resnet18_v2(pretrained=True)
resnet18.hybridize()
resnet18.forward(mx.nd.zeros(batch_shape))
resnet18.export('resnet18_v2')
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet18_v2', 0)


# Create sample input
input = mx.nd.zeros(batch_shape)

ctx = mx.cpu()
# ctx =mx.gpu(0)

# Execute with MXNet
executor = sym.simple_bind(ctx = ctx, data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)


os.environ["MXNet_USE_TENSORRT"] = "1"
os.environ["MXNET_SUBGRAPH_BACKEND"] = "MKLDNN"

# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()

# Timing
print('Starting MXNet timed run')
# Run Performance Runs
profiler.set_config(profile_all=True, aggregate_stats=True)
# Start Profiler
profiler.set_state('run')

start = time.process_time()
for i in range(0, 100):#10000
    print(i)
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)

# Stop Profiler
profiler.set_state('stop')
# Fetch Results from Profiler
# We will add a new API in Profiler - profiler.get_summary(reset=True)
# profiler.get_summary() => Return a JSON string representing the output as shown below.
#                        => Resets all the counter in the current profiler.
 
print("Done Running performance runs....")
print(profiler.dumps(reset=True))


# Execute with TensorRT
print('Building TensorRT engine')
trt_sym = sym.get_backend_symbol('TensorRT')
arg_params, aux_params = mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
mx.contrib.tensorrt.set_use_fp16(True)
executor = trt_sym.simple_bind(ctx = ctx, data=batch_shape,
                               grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)

#Warmup
print('Warming up TensorRT')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()

# Timing
print('Starting TensorRT timed run')
start = time.process_time()
for i in range(0, 10000):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)