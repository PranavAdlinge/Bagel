import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer

# Assuming Bagel's custom inferencer is imported here
# from inferencer import InterleaveInferencer

def run_bagel_tp_inference(rank, world_size):
    """
    Worker function to run distributed Tensor Parallelism on a specific GPU.
    """
    # 1. Initialize PyTorch Distributed Process Group (Required for HF Tensor Parallelism)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Bind this process to the specific GPU
    torch.cuda.set_device(rank)

    model_path = "ByteDance-Seed/BAGEL-7B-MoT"
    
    # 2. Load model with ACTUAL Tensor Parallelism
    # Replacing `device_map="auto"` with `tp_plan="auto"` instructs Transformers 
    # to slice the linear/attention weights across the device mesh.
    print(f"[GPU {rank}] Loading model with Tensor Parallelism...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        tp_plan="auto"  # <-- This enables native Tensor Parallelism
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 3. Setup your Inference logic
    # Make sure to map inputs specifically to the current rank's device
    # inferencer = InterleaveInferencer(model, ...)
    
    # -----------------------------------------------------
    # ADD YOUR DATA / PROMPT HERE
    # -----------------------------------------------------
    prompt = "A female cosplayer portraying an ethereal fairy..."
    # inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{rank}")
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=100)
    
    # Only print/save the output on the master rank to prevent duplicate outputs
    if rank == 0:
        print("[GPU 0] Inference complete.")
        # print(tokenizer.decode(outputs[0]))
        
    # Cleanup the distributed group
    dist.destroy_process_group()

# -----------------------------------------------------
# Notebook Execution Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    # Detect available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Tensor Parallelism requires at least 2 GPUs. Running standard inference.")
        run_bagel_tp_inference(0, 1)
    else:
        print(f"Starting Tensor Parallel inference across {world_size} GPUs...")
        # Spawn a separate process for each GPU
        mp.spawn(
            run_bagel_tp_inference, 
            args=(world_size,), 
            nprocs=world_size, 
            join=True
        )