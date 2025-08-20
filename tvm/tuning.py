import os
import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_executor as runtime
from tvm import rpc
from tvm.contrib import utils
import torch
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite

def get_network(model_path, batch_size=1, input_size=300):
    """Load and convert the model to TVM Relay format."""
    input_shape = (batch_size, 3, input_size, input_size)
    
    model = create_mobilenetv2_ssd_lite(2, is_test=True, device="cpu")
    model.load(model_path)
    model.eval()
    
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data)
    mod, params = relay.frontend.from_pytorch(scripted_model, [("data", input_shape)])
    
    return mod, params, input_shape

def create_tuning_option():
    """Create tuning options for the Raspberry Pi."""
    tuning_option = {
        'log_filename': 'rasp_mobilenet_ssd_tuning.log',
        'tuner': 'random',
        'n_trial': 1000,
        'early_stopping': None,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                '',  # key
                '',  # host
                9190,  # port
                number=10,
                repeat=1,
                min_repeat_ms=1000,
                timeout=10
            ),
        ),
    }
    return tuning_option

def tune_tasks(tasks, measure_option, tuner='random', n_trial=1000):
    """Tune the extracted tasks with the specified options."""
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        
        # Create tuner
        if tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(task)
        elif tuner == 'xgb':
            tuner_obj = autotvm.tuner.XGBTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        
        # Tune
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file('rasp_mobilenet_ssd_tuning.log')
            ]
        )

def compile_model(mod, params, target, target_host):
    """Compile the model with tuned parameters."""
    with autotvm.apply_history_best('rasp_mobilenet_ssd_tuning.log'):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)
    return lib

def save_compiled_model(lib):
    """Save the compiled model."""
    # Export the library as .so file
    lib_fname = 'ssd_tuned.so'
    lib.export_library(lib_fname)
    print(f"Tuned and compiled model saved as {lib_fname}")

def main():
    model_path = '/computer_vision/models_parameters/model.pth'
    target = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon'
    target_host = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon'
    
    try:
        # Load and convert model
        print("Loading and converting model...")
        mod, params, input_shape = get_network(model_path)
        
        # Extract tuning tasks
        tasks = autotvm.task.extract_from_program(mod['main'], target=target, params=params)
        
        print("Connecting to Raspberry Pi")
        remote = rpc.connect('192.168.3.60', 9190)
        
        if not tasks:
            print("No tuning tasks extracted. Proceeding with default compilation...")
        else:
            # Create tuning options and tune
            print("Starting tuning...")
            tuning_option = create_tuning_option()
            tune_tasks(tasks, tuning_option['measure_option'], 
                      tuning_option['tuner'], tuning_option['n_trial'])
        
        # Compile the model with tuned parameters
        print("Compiling model with tuned parameters...")
        lib = compile_model(mod, params, target, target_host)
        
        # Save the compiled model
        print("Saving compiled model...")
        save_compiled_model(lib)

        
        return lib, remote
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    main()