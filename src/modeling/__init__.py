from src.modeling.interface import ModelInterface, load_model, counting_parameters

# trigger registration of subclasses
from src.modeling.huggingface import evaluation as _
from src.modeling.my_bert import evaluation as _
