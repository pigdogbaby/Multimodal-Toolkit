from .tabular_combiner import TabularFeatCombiner
from .tabular_config import TabularConfig
from .tabular_modeling_auto import AutoModelWithTabular
from .tabular_transformers import (
    BertWithTabular,
    RobertaWithTabular,
    DistilBertWithTabular,
    LongformerWithTabular,
    PtWithTabular,
)


__all__ = [
    "TabularFeatCombiner",
    "TabularConfig",
    "AutoModelWithTabular",
    "BertWithTabular",
    "RobertaWithTabular",
    "DistilBertWithTabular",
    "LongformerWithTabular",
    "PtWithTabular",
]

from .configuration_pt import PtConfig
from .modeling_pt import PtModel, PtForMaskedLM
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
AutoConfig.register("pt", PtConfig)
AutoModel.register(PtConfig, PtModel)
# AutoModelForMaskedLM.register(PtConfig, PtForMaskedLM)