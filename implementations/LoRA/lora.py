import torch, torch.nn as nn, math

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f) * (1 / math.sqrt(in_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T
        return base_out + self.scale * lora_out

    @torch.no_grad()
    def merge(self):
        self.base.weight += self.scale * (self.B @ self.A)
        return self.base


def inject_lora(model, target_modules=("q_lin", "v_lin"), r=8, alpha=16):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total


def freeze_all_but_lora_and_head(model):
    for n, p in model.named_parameters():
        p.requires_grad = ("A" in n or "B" in n or "classifier" in n or "pre_classifier" in n)