import mxnet as mx
from mxnet import profiler

# Configurations
warmup = 25
runs = 100
run_backward = True
 
# Operator to benchmark
F = mx.nd.add
 
# Prepare data for the operator
lhs = mx.nd.ones(shape=(1024, 1024))
rhs = mx.nd.ones(shape=(1024, 1024))
lhs.attach_grad()
rhs.attach_grad()
mx.nd.waitall()
 
# Warmup
print("Warming up....")
for _ in range(warmup):
    with mx.autograd.record():
        res = mx.nd.add(lhs, rhs)
    res.backward()
    mx.nd.waitall()
print("Done warming up....")
 
# Run Performance Runs
print("Running performance runs....")
profiler.set_config(profile_all=True, aggregate_stats=True)
# Start Profiler
profiler.set_state('run')
for _ in range(runs):
    with mx.autograd.record():
        res = mx.nd.add(lhs, rhs)
    res.backward()
    mx.nd.waitall()
 
# Stop Profiler
profiler.set_state('stop')
 
# Fetch Results from Profiler
# We will add a new API in Profiler - profiler.get_summary(reset=True)
# profiler.get_summary() => Return a JSON string representing the output as shown below.
#                        => Resets all the counter in the current profiler.
 
print("Done Running performance runs....")
print(profiler.dumps(reset=True))