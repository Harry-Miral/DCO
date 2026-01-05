import subprocess
import os
import concurrent.futures
import time
from datetime import datetime

# ================= 🚀 GLOBAL CONFIGURATION =================
# Define available GPU IDs (e.g., [0, 1, 2, 3] for a 4-GPU machine)
GPU_IDS = [0, 1] 
# Maximum experiments to run at once (usually same as number of GPUs for 8B)
MAX_PARALLEL_JOBS = len(GPU_IDS)
LOG_DIR = "logs_8b_full_eval"
os.makedirs(LOG_DIR, exist_ok=True)

COMMON_ENV = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "PYTHONPATH": "."
}

# ================= 🧪 EXPERIMENT DEFINITIONS =================
# All configurations follow your provided tested parameters exactly.
EXPERIMENTS = [
    # --- Group 1: Faithfulness ---
    {
        "name": "1_XSum",
        "args": "experiment=xsum/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=0.5 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=25 debug=True"
    },
    {
        "name": "2_NQ_Swap",
        "args": "experiment=nq_swap/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=3.5 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=10 decoder.configs.intervention_end_layer=30 debug=True"
    },
    {
        "name": "3_MemoTrap",
        "args": "experiment=memo_trap/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=25 debug=True"
    },
    {
        "name": "4_IFEval",
        "args": "experiment=ifeval/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=1.5 decoder.configs.tau=0.75 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=25 debug=True"
    },
    {
        "name": "5_NQ_Open_Oracle",
        "args": "experiment=nq/baseline/llama3_8b_instruct decoder=dco data.variation=oracle decoder.configs.beta=3.0 decoder.configs.tau=0.75 decoder.configs.intervention_start_layer=10 decoder.configs.intervention_end_layer=25 debug=True"
    },
    # --- Group 2: MuSiQue (Reasoning) ---
    {
        "name": "6_MuSiQue_Open_CoT",
        "args": "experiment=musique/baseline/llama3_8b_instruct decoder=dco data.variation=cot_open_book decoder.configs.beta=2.0 decoder.configs.tau=2.0 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=30 debug=True"
    },
    {
        "name": "7_MuSiQue_Open_Direct",
        "args": "experiment=musique/baseline/llama3_8b_instruct decoder=dco data.variation=direct_open_book decoder.configs.beta=2.0 decoder.configs.tau=0.75 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=25 debug=True"
    },
    {
        "name": "8_MuSiQue_Closed_CoT",
        "args": "experiment=musique/baseline/llama3_8b_instruct decoder=dco data.variation=cot_closed_book decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=30 debug=True"
    },
    {
        "name": "9_MuSiQue_Closed_Direct",
        "args": "experiment=musique/baseline/llama3_8b_instruct decoder=dco data.variation=direct_closed_book decoder.configs.beta=2.0 decoder.configs.tau=1.75 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=25 debug=True"
    },
    # --- Group 3: Factuality ---
    {
        "name": "10_TruthfulQA",
        "args": "experiment=truthfulqa/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=2.0 decoder.configs.tau=0.8 decoder.configs.intervention_start_layer=10 decoder.configs.intervention_end_layer=28 debug=True"
    },
    {
        "name": "11_TriviaQA",
        "args": "experiment=triviaqa/baseline/llama3_8b_instruct decoder=dco decoder.configs.beta=5.0 decoder.configs.tau=1.2 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=30 debug=True"
    },
    {
        "name": "12_NQ_Closed",
        "args": "experiment=nq/baseline/llama3_8b_instruct decoder=dco data.variation=closed_book decoder.configs.beta=5.0 decoder.configs.tau=1.1 decoder.configs.intervention_start_layer=15 decoder.configs.intervention_end_layer=30 debug=True"
    }
]

# ================= 🛠️ EXECUTION ENGINE =================

gpu_queue = GPU_IDS.copy()

def run_experiment(exp, gpu_id):
    name = exp["name"]
    args = exp["args"]
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    
    # Construct the final command
    cmd = f"python scripts/main.py {args}"
    
    # Set environment for this specific process (assign GPU)
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🚀 [STARTING] {name} on GPU {gpu_id}")
    start_time = datetime.now()
    
    with open(log_path, "w") as log_file:
        log_file.write(f"Task: {name}\nStart Time: {start_time}\nGPU: {gpu_id}\nCMD: {cmd}\n{'-'*50}\n")
        log_file.flush()
        
        process = subprocess.Popen(
            cmd, shell=True, env=env, 
            stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
        process.wait()

    end_time = datetime.now()
    duration = end_time - start_time
    status = "SUCCESS" if process.returncode == 0 else f"FAILED (Code {process.returncode})"
    
    print(f"🏁 [FINISHED] {name} | Status: {status} | Duration: {duration}")
    return name, status

def main():
    print(f"🔥 Llama-3-8B DCO Full Evaluation Orchestrator")
    print(f"Parallel Jobs: {MAX_PARALLEL_JOBS} | Total Tasks: {len(EXPERIMENTS)}")
    print("="*60)

    # Use ThreadPoolExecutor to manage parallel subprocesses
    # Note: Llama-3-8B usually fits in ~16GB-20GB VRAM during inference (fp16/bf16)
    # If you have 24GB+ cards, running 1 per card is optimal.
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_exp = {}
        
        # Dispatch logic to handle GPU assignment
        for i, exp in enumerate(EXPERIMENTS):
            # Assign GPU in a round-robin or simple available-id fashion
            gpu_id = GPU_IDS[i % len(GPU_IDS)]
            future = executor.submit(run_experiment, exp, gpu_id)
            future_to_exp[future] = exp

        for future in concurrent.futures.as_completed(future_to_exp):
            results.append(future.result())

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    for name, status in results:
        print(f" - {name.ljust(25)}: {status}")
    print("="*60)

if __name__ == "__main__":
    main()