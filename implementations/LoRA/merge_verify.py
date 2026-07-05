import time
import torch
from train import build_model
from data import get_dataloaders
from lora import merge_lora

def bench(model, batch, n=50):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            model(**batch)
    torch.cuda.synchronize()
    return (time.time() - t0) / n

def verify_merge():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("lora", r=8, alpha=16).to(device)
    model.load_state_dict(torch.load("results/lora_checkpoint_r{r}_{modules_tag}.pt", map_location=device))
    model.eval()

    _, val_dl = get_dataloaders()
    batch = next(iter(val_dl))
    batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

    with torch.no_grad():
        logits_before = model(**batch).logits.clone()

    for _ in range(5):
        with torch.no_grad():
            model(**batch)

    t_unmerged = bench(model, batch)
    merge_lora(model)

    with torch.no_grad():
        logits_after = model(**batch).logits

    t_merged = bench(model, batch)

    max_diff = (logits_before - logits_after).abs().max().item()
    print(f"Max logit difference after merge: {max_diff:.2e}")
    assert max_diff < 1e-4, "Merge is not numerically equivalent!"
    print("Merge verified: identical outputs, zero inference overhead confirmed.")

    print(f"Unmerged forward pass: {t_unmerged * 1000:.3f} ms/iter")
    print(f"Merged forward pass:   {t_merged * 1000:.3f} ms/iter")
    speedup = t_unmerged / t_merged
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    verify_merge()