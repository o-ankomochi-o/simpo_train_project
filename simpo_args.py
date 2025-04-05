from dataclasses import dataclass, field

from trl import CPOConfig


@dataclass
class SimPOTrainingArguments(CPOConfig):
    diversity_weight: float = field(
        default=0.05, metadata={"help": "Weight for the diversity loss term"}
    )
    diversity_alpha: float = field(
        default=1.0,
        metadata={"help": "Temperature for diversity softmax (entropy sharpness)"},
    )
