import torch.multiprocessing as mp
import runpy
import sys

if __name__ == "__main__":
    # Set 'spawn' before any CUDA code
    mp.set_start_method("spawn", force=True)

    # Keep original CLI args
    sys.argv = sys.argv[1:]  # drop the script name
    runpy.run_module("oe_eval.launch", run_name="__main__")
