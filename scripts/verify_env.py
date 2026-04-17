import torch
import sys

def verify():
    print("Verifying AttenX Environment...")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  Running on CPU. Training will be slow.")

    # Check dependencies
    try:
        import numpy
        import pandas
        import matplotlib
        print("All core dependencies (numpy, pandas, matplotlib) are installed.")
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")

    print("\nEnvironment verification complete.")

if __name__ == "__main__":
    verify()
