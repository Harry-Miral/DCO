import subprocess
import os
import time
import datetime
import sys

# ================= 🚀 Global Configuration =================
LOG_FILE = "final_70b_full_results.txt"

# General arguments
COMMON_ARGS = [
    "HF_ENDPOINT=https://hf-mirror.com",
    "python", "scripts/main.py",
    "data_loader.batch_size=1",   # ⚠️ Critical for 70B VRAM limits
    "debug=True"                  # Set to True to avoid mandatory WandB config
]

# ⚠️ Full Run Configuration
NUM_SAMPLES = -1 

# ================= 🧪 Experiment Queue (12 Groups) =================
EXPERIMENTS = [
    # ==========================================
    # Group 1: Faithfulness (Table 1)
    # ==========================================
    
    # 1. XSum (Summarization)
    # Strategy: 70B performs better with stronger intervention (Beta=3.5)
    {
        "name": "1_XSum",
        "cmd": "experiment=xsum/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=3.5 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=38 decoder.configs.intervention_end_layer=62"
    },

    # 2. NQ-Swap (Resistance to Interference)
    # Strategy: Truncate length to prevent OOM. Parameters follow "Center" config (Beta=3.5)
    {
        "name": "2_NQ_Swap",
        "cmd": "experiment=nq_swap/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=3.5 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=25 decoder.configs.intervention_end_layer=75 model.configs.max_seq_len=2560"
    },

    # 3. MemoTrap (Instruction Traps)
    # Strategy: Similar to instruction following; use robust "Center" parameters
    {
        "name": "3_MemoTrap",
        "cmd": "experiment=memo_trap/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=38 decoder.configs.intervention_end_layer=62"
    },

    # 4. IFEval (Instruction Following)
    # Strategy: Previously verified that Beta=3.5 yields best performance (0.87)
    {
        "name": "4_IFEval",
        "cmd": "experiment=ifeval/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=3.5 decoder.configs.tau=0.75 decoder.configs.intervention_start_layer=38 decoder.configs.intervention_end_layer=62"
    },

    # 5. NQ-Open Oracle (Open Book)
    # Strategy: Document-supported; parameters similar to NQ-Swap
    {
        "name": "5_NQ_Open_Oracle",
        "cmd": "experiment=nq/baseline/llama3_70b_instruct decoder=dco data.variation=oracle decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=25 decoder.configs.intervention_end_layer=75"
    },

    # ==========================================
    # Group 2: Reasoning / MuSiQue (Table 3)
    # ==========================================

    # 6. MuSiQue Open Book + CoT
    # Strategy: High-scoring configuration (0.86) using "Center" parameters
    {
        "name": "6_MuSiQue_Open_CoT",
        "cmd": "experiment=musique/baseline/llama3_70b_instruct decoder=dco data.variation=cot_open_book decoder.configs.beta=2.0 decoder.configs.tau=2.0 decoder.configs.intervention_start_layer=38 decoder.configs.intervention_end_layer=75 model.configs.max_seq_len=2560"
    },

    # 7. MuSiQue Open Book + Direct (No CoT)
    # Strategy: No CoT protection; use stricter Tau (1.0) to prevent hallucinations
    {
        "name": "7_MuSiQue_Open_Direct",
        "cmd": "experiment=musique/baseline/llama3_70b_instruct decoder=dco data.variation=direct_open_book decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=38 decoder.configs.intervention_end_layer=75 model.configs.max_seq_len=2560"
    },

    # 8. MuSiQue Closed Book + CoT
    # Strategy: Pure memorization; ablation shows earlier layers (Start=33) perform better
    {
        "name": "8_MuSiQue_Closed_CoT",
        "cmd": "experiment=musique/baseline/llama3_70b_instruct decoder=dco data.variation=cot_closed_book decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=33 decoder.configs.intervention_end_layer=70"
    },

    # 9. MuSiQue Closed Book + Direct (No CoT)
    # Strategy: Same as above, targeting earlier layers
    {
        "name": "9_MuSiQue_Closed_Direct",
        "cmd": "experiment=musique/baseline/llama3_70b_instruct decoder=dco data.variation=direct_closed_book decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=33 decoder.configs.intervention_end_layer=70"
    },

    # ==========================================
    # Group 3: Factuality (Table 2)
    # ==========================================

    # 10. TruthfulQA (Safety/Truthfulness)
    # Strategy: "Center" configuration previously verified as most stable (0.42)
    {
        "name": "10_TruthfulQA",
        "cmd": "experiment=truthfulqa/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=2.0 decoder.configs.tau=0.8 decoder.configs.intervention_start_layer=25 decoder.configs.intervention_end_layer=70"
    },

    # 11. TriviaQA (Knowledge)
    # Strategy: Closed-book knowledge recall; use standard "Center" config
    {
        "name": "11_TriviaQA",
        "cmd": "experiment=triviaqa/baseline/llama3_70b_instruct decoder=dco decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=30 decoder.configs.intervention_end_layer=70"
    },

    # 12. NQ-Open Closed Book (Pure Memorization)
    # Strategy: Similar to TriviaQA
    {
        "name": "12_NQ_Closed",
        "cmd": "experiment=nq/baseline/llama3_70b_instruct decoder=dco data.variation=closed_book decoder.configs.beta=2.0 decoder.configs.tau=1.0 decoder.configs.intervention_start_layer=30 decoder.configs.intervention_end_layer=70"
    }
]

# ================= 🛠️ Execution Engine =================

def run_command_live(cmd_str, task_name):
    print(f"\n🚀 [{task_name}] Executing: {cmd_str}\n" + "-"*60)
    
    # Log start time
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\n{'='*60}\nTASK: {task_name}\nSTART: {datetime.datetime.now()}\nCMD: {cmd_str}\n{'='*60}\n")

    process = subprocess.Popen(
        cmd_str, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        env=os.environ.copy()
    )
    
    full_log = ""
    # Real-time console output + incremental logging
    for line in process.stdout:
        print(line, end="") # Display in terminal
        sys.stdout.flush()
        full_log += line
        
        # Simple real-time persistence for key metrics (prevents data loss on crash)
        # Note: I/O overhead is negligible for LLM inference with batch=1
        if any(keyword in line for keyword in ["RESULT", "Acc", "EM", "Score"]):
             with open(LOG_FILE, "a") as f:
                f.write(line)

    process.wait()
    
    # Finalize log block
    with open(LOG_FILE, "a") as f:
        f.write(f"\nFINISHED: {datetime.datetime.now()}\n")
        f.write("-" * 60 + "\n")

def main():
    print(f"🔥 Starting FINAL FULL 70B Run (12 Experiments)")
    print(f"📄 Global Log: {LOG_FILE}")
    print(f"🔢 Samples: {NUM_SAMPLES} (Full Dataset)")
    print("="*60)

    for i, exp in enumerate(EXPERIMENTS):
        task_name = exp["name"]
        print(f"\n🔶 Progress: [{i+1}/{len(EXPERIMENTS)}] -> {task_name}")
        
        # Assemble command
        full_cmd = COMMON_ARGS + exp["cmd"].split() + [f"data.num_samples={NUM_SAMPLES}"]
        cmd_str = " ".join(full_cmd)
        
        try:
            run_command_live(cmd_str, task_name)
            print(f"✅ {task_name} Completed.")
        except Exception as e:
            print(f"❌ {task_name} FAILED: {e}")
            with open(LOG_FILE, "a") as f:
                f.write(f"\n❌ ERROR: {e}\n")

    print(f"\n🎉🎉🎉 All Experiments Finished! Data collected in {LOG_FILE}")

if __name__ == "__main__":
    main()