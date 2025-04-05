from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
    # trl==0.12.1 の DPOTrainer は ref_model_init_kwargs という引数（初期値：None）を内部的に参照します。
    # これは CPOConfig や TrainingArguments には 元々定義されていない ので、エラーになります。
    ref_model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Keyword arguments to initialize reference model."},
    )
    model_adapter_name: Optional[str] = field(default=None)
