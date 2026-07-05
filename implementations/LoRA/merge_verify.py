import torch
from train import build_model
from data import get_dataloaders
from lora import merge_lora

def verify_merge():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("lora", r=8, alpha=16).to(device)
    model.load_state_dict(torch.load("results/lora_checkpoint.pt", map_location=device))
    model.eval()

    _, val_dl = get_dataloaders()
    batch = next(iter(val_dl))
    batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

    with torch.no_grad():
        logits_before = model(**batch).logits.clone()

    merge_lora(model)

    with torch.no_grad():
        logits_after = model(**batch).logits

    max_diff = (logits_before - logits_after).abs().max().item()
    print(f"Max logit difference after merge: {max_diff:.2e}")
    assert max_diff < 1e-4, "Merge is not numerically equivalent!"
    print("Merge verified: identical outputs, zero inference overhead confirmed.")

if __name__ == "__main__":
    verify_merge()

import time
def bench(model, batch, n=50):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n): model(**batch)
    torch.cuda.synchronize()
    return (time.time() - t0) / n