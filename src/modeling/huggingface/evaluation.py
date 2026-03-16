from src.modeling import _head
from src.modeling.huggingface.upstream import (
    _HuggingfaceModelForDownstreamConfigBase,
    _HuggingfaceModelForDownstreamBase,
)


# sequence classification
# fmt:off
class HuggingfaceModelForSequenceClassificationConfig(
    _head._SequenceClassificationConfig, _HuggingfaceModelForDownstreamConfigBase
):  pass
class HuggingfaceModelForSequenceClassification(
    _head._SequenceClassificationHead, _HuggingfaceModelForDownstreamBase
):  pass # fmt:on


# token classification
# fmt:off
class HuggingfaceModelForTokenClassificationConfig(
    _head._TokenClassificationConfig, _HuggingfaceModelForDownstreamConfigBase
):  pass
class HuggingfaceModelForTokenClassification(
    _head._TokenClassificationHead, _HuggingfaceModelForDownstreamBase
):  pass # fmt:on


# question answering
# fmt:off
class HuggingfaceModelForQuestionAnsweringConfig(
    _head._QuestionAnsweringConfig, _HuggingfaceModelForDownstreamConfigBase
):  pass
class HuggingfaceModelForQuestionAnswering(
    _head._QuestionAnsweringHead, _HuggingfaceModelForDownstreamBase
):  pass # fmt:on
