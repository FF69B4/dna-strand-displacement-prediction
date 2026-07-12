"""Dissertation-local DNABERT-2 backend.

This package vendors the model classes needed by ``zhihan1996/DNABERT-2-117M``
so the embedding script can load the checkpoint without Hugging Face
``trust_remote_code``.
"""

from .configuration_bert import BertConfig
from .bert_layers import BertModel

__all__ = ["BertConfig", "BertModel"]
