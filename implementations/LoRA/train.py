import torch, time
from transformers import AutoModelForSequenceClassification
from lora import inject_lora, freeze_all_but_lora_and_head, count_trainable_params
from data import get_dataloaders

def build_model(mode, r=8, alpha=16):
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)

    if mode == "full":
        pass
    elif mode == "frozen_head":
        for n, p in model.named_parameters():
            p.requires_grad = "classifier" in n or "pre_classifier" in n
    elif mode == "lora":
        model = inject_lora(model, target_modules=("q_lin", "v_lin"), r=r, alpha=alpha)
        freeze_all_but_lora_and_head(model)
    return model


def train(mode, epochs=3, lr=None, subset_size=None, r=8, alpha=16, log_path="results/metrics.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_dl, val_dl = get_dataloaders(subset_size=subset_size)
    model = build_model(mode, r=r, alpha=alpha).to(device)

    trainable, total, pct = count_trainable_params(model)
    lr = lr or (2e-5 if mode == "full" else 1e-3)  # LoRA/head-only tolerate higher LR
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    opt_state_mb = trainable * 2 * 4 / 1e6

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()

    train_time = time.time() - start
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    acc = evaluate(model, val_dl, device)

    result = dict(mode=mode, r=r if mode=="lora" else None, trainable_params=trainable,
                   total_params=total, pct_trainable=pct, opt_state_mb=opt_state_mb,
                   peak_mem_mb=peak_mem_mb, train_time_s=train_time, val_accuracy=acc)
    log_result(result, log_path)
    if mode == "lora":
        torch.save(model.state_dict(), "results/lora_checkpoint.pt")
    return model, result


def evaluate(model, val_dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(preds)
    return correct / total


def log_result(result, path):
    import csv, os
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if write_header: w.writeheader()
        w.writerow(result)

if __name__ == "__main__":
    for mode in ["full", "frozen_head", "lora"]:
        train(mode, epochs=3, subset_size=20000)