import tvm
from tvm import relay
import torch
import os
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite

if __name__ == "__main__":
    MODEL_PATH = "models_parameters/front_model.pth"
    SAVE_DIR = "tvm_model/front"
    SAVE_NAME = "front_compiled_model.so"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = create_mobilenetv2_ssd_lite(2, is_test=True, device="cpu")
    model.load(MODEL_PATH)
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, 300, 300)
    
    # Trace the model
    scripted_model = torch.jit.trace(model, dummy_input).eval()
    
    # Convert to Relay
    input_name = "input0"
    shape_list = [(input_name, dummy_input.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    
    # Set target to Raspberry Pi 
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    
    lib_path = os.path.join(SAVE_DIR, SAVE_NAME)
    lib.export_library(lib_path)
    
    print(f"Model successfully compiled and saved to: {lib_path}")