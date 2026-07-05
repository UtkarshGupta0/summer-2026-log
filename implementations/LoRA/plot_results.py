import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

# Rank vs accuracy
lora_df = df[df["mode"] == "lora"].sort_values("r")
plt.figure(figsize=(6,4))
plt.plot(lora_df["r"], lora_df["val_accuracy"], marker="o")
plt.axhline(df[df["mode"]=="full"]["val_accuracy"].iloc[0], color="gray", linestyle="--", label="Full fine-tune")
plt.xscale("log", base=2)
plt.xlabel("LoRA rank (r)"); plt.ylabel("Validation accuracy")
plt.title("SST-2 accuracy vs LoRA rank")
plt.legend(); plt.tight_layout()
plt.savefig("results/plots/rank_vs_accuracy.png", dpi=150)

# Trainable params comparison (bar, log scale)
plt.figure(figsize=(6,4))
modes = df.groupby("mode")["trainable_params"].first()
plt.bar(modes.index, modes.values)
plt.yscale("log")
plt.ylabel("Trainable parameters (log scale)")
plt.title("Trainable parameters by method")
plt.tight_layout()
plt.savefig("results/plots/param_comparison.png", dpi=150)

# isolate the clean rank sweep: only q_lin+v_lin rows
rank_sweep_df = df[(df["mode"] == "lora") & (df["target_modules"] == "q_lin+v_lin")].sort_values("r")

plt.figure(figsize=(6,4))
plt.plot(rank_sweep_df["r"], rank_sweep_df["val_accuracy"], marker="o")
plt.axhline(df[df["mode"]=="full"]["val_accuracy"].iloc[0], color="gray", linestyle="--", label="Full fine-tune")
plt.xscale("log", base=2)
plt.xlabel("LoRA rank (r)"); plt.ylabel("Validation accuracy")
plt.title("SST-2 accuracy vs LoRA rank (q_lin + v_lin only)")
plt.legend(); plt.tight_layout()
plt.savefig("results/plots/rank_vs_accuracy_isolated.png", dpi=150)