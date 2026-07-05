from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloaders(batch_size=32, subset_size=None):
    ds = load_dataset("nyu-mll/glue", "sst2")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(ex):
        return tok(ex["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    if subset_size:
        ds["train"] = ds["train"].select(range(subset_size))

    from torch.utils.data import DataLoader
    train_dl = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(ds["validation"], batch_size=batch_size)
    return train_dl, val_dl