"""CLI 推理配置。"""

from dataclasses import dataclass


@dataclass
class CLIConfig:
    load_from: str = "model"
    save_dir: str = "out"
    weight: str = "full_sft"
    lora_weight: str = "None"
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    max_new_tokens: int = 8192
    temperature: float = 0.85
    top_p: float = 0.95
    open_thinking: bool = False
    historys: int = 0
    show_speed: bool = True
    device: str = "cuda"
