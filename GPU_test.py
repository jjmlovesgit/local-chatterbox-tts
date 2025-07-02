import torch
import os

print("--- PyTorch CUDA Diagnostic ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("\n!!! CUDA is NOT available. Forcing initialization to see the error...")
    try:
        # This is the command that will force the error message
        torch.ones(1).cuda()
        print("This is unexpected. Forced initialization worked, but is_available() is False.")
    except Exception as e:
        print("\n>>> CAUGHT THE DETAILED ERROR:")
        print(e)

print("\n--- System Environment Variables ---")
print(f"CUDA_PATH variable set: {'CUDA_PATH' in os.environ}")
if 'CUDA_PATH' in os.environ:
    print(f"CUDA_PATH value: {os.environ['CUDA_PATH']}")

print("\n--- End of Diagnostic ---")