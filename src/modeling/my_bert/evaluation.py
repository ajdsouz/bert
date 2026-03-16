from src.modeling import _head
from src.modeling.my_bert.upstream import (
    _MyBertModelForDownstreamConfigBase,
    _MyBertModelForDownstreamBase,
)


# sequence classification
# fmt:off
class MyBertModelForSequenceClassificationConfig(
    _head._SequenceClassificationConfig, _MyBertModelForDownstreamConfigBase
):  pass
class MyBertModelForSequenceClassification(
    _head._SequenceClassificationHead, _MyBertModelForDownstreamBase
):  pass # fmt:on


# token classification
# fmt:off
class MyBertModelForTokenClassificationConfig(
    _head._TokenClassificationConfig, _MyBertModelForDownstreamConfigBase
):  pass
class MyBertModelForTokenClassification(
    _head._TokenClassificationHead, _MyBertModelForDownstreamBase
):  pass # fmt:on


# question answering
# fmt:off
class MyBertModelForQuestionAnsweringConfig(
    _head._QuestionAnsweringConfig, _MyBertModelForDownstreamConfigBase
):  pass
class MyBertModelForQuestionAnswering(
    _head._QuestionAnsweringHead, _MyBertModelForDownstreamBase
):  pass # fmt:on
