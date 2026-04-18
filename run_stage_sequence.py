"""Sequenced stage0 → stage2 training with checkpointing and eval."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s0-steps", type=int, default=150_000)
    p.add_argument("--s2-steps", type=int, default=400_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    print(f"=== Stage 0 pretrain: {a.s0_steps} steps ===", flush=True)
    subprocess.run([sys.executable, "-m", "simulator.train.train_ppo",
        "--stage", "0", "--total-timesteps", str(a.s0_steps),
        "--n-envs", str(a.n_envs), "--seed", str(a.seed),
        "--out", "ppo_seq_s0"], check=True)

    print(f"=== Eval stage 0 ===", flush=True)
    subprocess.run([sys.executable, "-m", "simulator.train.eval",
        "--model", "ppo_seq_s0", "--episodes", "3",
        "--map-size", "30", "30", "--round-turns", "100"], check=True)

    print(f"=== Stage 2 finetune: {a.s2_steps} steps ===", flush=True)
    subprocess.run([sys.executable, "-m", "simulator.train.train_ppo",
        "--stage", "2", "--total-timesteps", str(a.s2_steps),
        "--n-envs", str(a.n_envs), "--seed", str(a.seed),
        "--out", "ppo_seq_s2"], check=True)

    print(f"=== Eval stage 2 (target 600K+) ===", flush=True)
    subprocess.run([sys.executable, "-m", "simulator.train.eval",
        "--model", "ppo_seq_s2", "--episodes", "5",
        "--map-size", "100", "100", "--round-turns", "600"], check=True)


if __name__ == "__main__":
    main()
