from train import train

configs = {
    "q_only":        ("q_lin",),
    "q_v":           ("q_lin", "v_lin"),
    "q_k_v_o":       ("q_lin", "k_lin", "v_lin", "out_lin"),
    "attn_plus_ffn": ("q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"),
}

# roughly fix trainable-param budget by lowering r as target_modules grows
rank_by_config = {
    "q_only": 32,
    "q_v": 16,
    "q_k_v_o": 8,
    "attn_plus_ffn": 4,
}

for name, modules in configs.items():
    r = rank_by_config[name]
    train("lora", epochs=3, subset_size=20000, r=r, alpha=2*r, target_modules=modules)