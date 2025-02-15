# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

# HF-HAT
# For subtransformers, we slice the parameters of the supertransformer
# Here, each parameter is an object and not a scalar, and hence the parameters
# are shallow copied and hence the changes in parameters(during backprop) will
# be reflected in the supertransformer


import math
import os
from re import L
from typing_extensions import final
import warnings
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from random import randrange

# https://discuss.pytorch.org/t/attributeerror-builtin-function-or-method-object-has-no-attribute-fftn/109744
# import torch.fft
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
import torch.nn.functional as F


from custom_layers.custom_embedding import CustomEmbedding
from custom_layers.custom_linear import CustomLinear
from custom_layers.custom_layernorm import CustomLayerNorm, CustomNoNorm
from custom_layers.DynamicSeparableConv2d import DynamicSeparableConv1d
from copy import deepcopy
from loss import CrossEntropyLossSoft
from loss import *
from custom_layers.HyperNetDynamicLinear import HyperNetDynamicLinear

from utils import get_overlap_order
from utils import dropout_layers

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


NORM2FN = {"layer_norm": CustomLayerNorm, "no_norm": CustomNoNorm}

def min_max_normalization(arr, min_val, max_val):
    '''
    min_val, max_val = 10000, -10000 # todo: set it based on search space
    for item in arr:
        item = float(item)
        if item < min_val:
            min_val = item
        if item > max_val:
            max_val = item
    '''
    new_arr = []
    for i in range(len(arr)):
        new_arr.append(0.01 + ((arr[i] - min_val)/(max_val - min_val) if (max_val-min_val) !=0 else 0))
    return new_arr

def standard2onehot(standard_input):
    possible_hid_dims = [120, 240, 360, 480, 540, 600, 768]
    onehot = []
    for sin in standard_input:
        for hdim in possible_hid_dims:
            if sin == hdim:
                onehot.append(1)
            else:
                onehot.append(0)
    return onehot

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


# TODO: use config instead of sample_hidden_size, super_hidden_size
# ^ rethinking this. For all lowest level set_sample_function, we directly use
# the required variable instead of config. Looks fine for now.
def calc_dropout(dropout, sample_hidden_size, super_hidden_size):
    return dropout * 1.0 * sample_hidden_size / super_hidden_size


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = CustomEmbedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = CustomEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = CustomEmbedding(
            config.type_vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.use_bottleneck = config.mixing == "bert-bottleneck"
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def set_sample_config(self, config):
        # we name all param inside sampling ocnfig as sample_*
        # hidden_size -> sample_emb_hidden_size
        sample_hidden_size = config.sample_hidden_size
        if self.use_bottleneck:
            # dont slice embeddings if you are using bottleneck
            # we will slice it with bottleneck layers
            sample_hidden_size = config.hidden_size

        self.word_embeddings.set_sample_config(sample_hidden_size, part="encoder")
        self.position_embeddings.set_sample_config(sample_hidden_size, part="encoder")
        self.token_type_embeddings.set_sample_config(sample_hidden_size, part="encoder")

        self.LayerNorm.set_sample_config(sample_hidden_size)

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=sample_hidden_size,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    def get_active_subnet(self, config):
        ## TODO: Handle the effect of bottleneck for get_active_subnet!!
        sublayer = BertEmbeddings(config)
        sublayer.word_embeddings = self.word_embeddings.get_active_subnet(
            part="encoder"
        )
        sublayer.position_embeddings = self.position_embeddings.get_active_subnet(
            part="encoder"
        )
        sublayer.token_type_embeddings = self.token_type_embeddings.get_active_subnet(
            part="encoder"
        )

        sublayer.LayerNorm = self.LayerNorm.get_active_subnet()
        return sublayer

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # TODO: make all these hidden_sizes different for supertransformer # might be error
        self.query = CustomLinear(config.hidden_size, self.all_head_size)
        self.key = CustomLinear(config.hidden_size, self.all_head_size)
        self.value = CustomLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            # not elastic but it uses the customembedding wrapper
            self.distance_embedding = CustomEmbedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

        self.sample_num_attention_heads = self.num_attention_heads
        self.sample_num_attention_heads = self.attention_head_size ## ?? doesn;t matter as set_sample_config is called
        self.sample_all_head_size = self.all_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.sample_num_attention_heads,
            self.sample_attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def set_sample_config(self, config, tiny_attn=False):

        if tiny_attn:
            self.sample_num_attention_heads = 1
            # no sampling
            sample_hidden_size = config.hidden_size
            self.sample_attention_head_size = int(
                sample_hidden_size / self.sample_num_attention_heads
            )
        else:
            self.sample_num_attention_heads = config.sample_num_attention_heads
            sample_hidden_size = config.sample_hidden_size
            self.sample_attention_head_size = int(
                sample_hidden_size / self.sample_num_attention_heads
            )

        self.sample_all_head_size = (
            self.sample_num_attention_heads * self.sample_attention_head_size
        )

        self.query.set_sample_config(sample_hidden_size, self.sample_all_head_size)
        self.key.set_sample_config(sample_hidden_size, self.sample_all_head_size)
        self.value.set_sample_config(sample_hidden_size, self.sample_all_head_size)
        sample_hidden_dropout_prob = calc_dropout(
            config.attention_probs_dropout_prob,
            super_hidden_size=config.num_attention_heads,
            sample_hidden_size=self.sample_num_attention_heads,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    def get_active_subnet(self, config):
        sublayer = BertSelfAttention(config)
        sublayer.set_sample_config(config)  ## Necessary evil
        sublayer.query = self.query.get_active_subnet()
        sublayer.key = self.key.get_active_subnet()
        sublayer.value = self.value.get_active_subnet()

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.sample_attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.sample_all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, config, prev_layer_importance_order=None):
        sample_hidden_size = config.sample_hidden_size
        self.dense.set_sample_config(sample_hidden_size, sample_hidden_size)
        self.LayerNorm.set_sample_config(sample_hidden_size)
        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=sample_hidden_size,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

        if self.config.rewire:
            if hasattr(self.dense, "importance_order"):
                # slicing importance order
                self.dense.sample_importance_order = self.dense.importance_order

                # for sliced training
                if prev_layer_importance_order is not None:
                    # slice the prev importance order p1'
                    prev_layer_importance_order = prev_layer_importance_order[
                        :sample_hidden_size
                    ]
                    # final_importance_indices = get_overlap_order(
                    #     self.dense.sample_importance_order, prev_layer_importance_order
                    # )
                    final_importance_indices = torch.arange(
                        sample_hidden_size, requires_grad=False
                    )
                    self.dense.sample_importance_order = final_importance_indices

    def get_active_subnet(self, config):
        sublayer = BertSelfOutput(config)

        sublayer.dense = self.dense.get_active_subnet()
        sublayer.LayerNorm = self.LayerNorm.get_active_subnet()

        return sublayer

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.config.rewire:
            if hasattr(self.dense, "sample_importance_order"):
                # reorder the input_tensor according to the new importance order
                # input_tensor = self.dense.importance_order(hidden_states)
                input_tensor = input_tensor[:, :, self.dense.sample_importance_order]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


"""
Implementation for Spatial Unit and Dense layers inspired from https://github.com/lucidrains/g-mlp-pytorch
"""


class SpatialUnit(nn.Module):
    def __init__(self, intermediate_size, seq_len, act, causal=False, init_eps=1e-3):
        super().__init__()
        out = intermediate_size // 2
        self.causal = causal

        self.norm = CustomLayerNorm(out)
        self.proj = nn.Conv1d(seq_len, seq_len, 1)

        self.act = act

        init_eps /= seq_len
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.0)

    def set_sample_config(self, sample_intermediate_size):
        self.norm.set_sample_config(sample_intermediate_size)

    def get_active_subnet(self, config):
        sublayer = SpatialUnit(config)
        sublayer.norm = self.norm.get_active_subnet(config.sample_intermediate_size)
        sublayer.proj.weight.data.copy_(self.proj.weight)
        sublayer.proj.bias.data.copy_(self.proj.bias)

    def forward(self, hidden_states, gate_res=None):
        device, n = hidden_states.device, hidden_states.shape[1]

        res, gate = hidden_states.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device=device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.0)

        gate = F.conv1d(gate, weight, bias)

        if gate_res is not None:
            gate = gate + gate_res

        return self.act(gate) * res


class BertDense(nn.Module):
    def __init__(self, config, act=nn.Identity(), attn=None):
        super().__init__()

        ### Can we have this as an elasticization parameter ###
        self.attention = BertSelfAttention(config) if attn is not None else None

        self.channel_projection_in = CustomLinear(
            config.hidden_size, config.intermediate_size
        )
        self.proj_act = nn.GELU()
        self.spatial_projection = SpatialUnit(
            config.intermediate_size, config.sequence_length, act
        )
        self.channel_projection_out = CustomLinear(
            config.intermediate_size // 2, config.hidden_size
        )

    def set_sample_config(self, config):
        sample_hidden_size = config.sample_hidden_size
        sample_intermediate_size = config.sample_intermediate_size

        self.channel_projection_in.set_sample_config(
            sample_hidden_size, sample_intermediate_size
        )
        self.spatial_projection.set_sample_config(sample_intermediate_size)
        self.channel_projection_out.set_sample_config(
            sample_hidden_size, sample_intermediate_size
        )

        if self.attention is not None:
            self.attention.set_sample_config(config)

    def get_active_subnet(self, config):
        sublayer = BertDense(config)
        sublayer.channel_projection_in = self.channel_projection_in.get_active_subnet()
        sublayer.spatial_projection = self.spatial_projection.get_active_subnet()
        sublayer.channel_projection_out = (
            self.channel_projection_out.get_active_subnet()
        )

        if self.attention is not None:
            sublayer.attention.get_active_subnet(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        gate_res = self.attention(hidden_states) if self.attention is not None else None

        x = self.channel_projection_in(hidden_states)
        x = self.spatial_projection(x, gate_res=gate_res)
        x = self.channel_projection_out(x)

        return x


class BertMobile(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.is_identity_layer = False

    def set_sample_config(self, config, is_identity_layer=False):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.attention.set_sample_config(config)
        if hasattr(self, "crossattention"):
            self.crossattention.set_sample_config(config)
        self.intermediate.set_sample_config(config)
        self.output.set_sample_config(config)

    def get_active_subnet(self, config):
        sublayer = BertLayer(config)

        sublayer.attention.self.set_sample_config(
            config
        )  ## Just to access those variables

        sublayer.attention = self.attention.get_active_subnet(config)

        #### Building the intermediate layer
        sublayer.intermediate.dense = self.intermediate.dense.get_active_subnet()

        #### Building the output layer
        sublayer.output.dense = self.output.dense.get_active_subnet()
        sublayer.output.LayerNorm = self.output.LayerNorm.get_active_subnet()

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        if self.is_identity_layer:
            return (hidden_states,)

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
Implementation for Spatial Unit and Dense layers inspired from https://github.com/lucidrains/g-mlp-pytorch
"""


class SpatialUnit(nn.Module):
    def __init__(
        self,
        intermediate_size,
        seq_len,
        act,
        layer_norm_eps,
        causal=False,
        init_eps=1e-3,
    ):
        super().__init__()
        out = intermediate_size // 2
        self.causal = causal

        self.norm = CustomLayerNorm(out, eps=layer_norm_eps)
        self.proj = nn.Conv1d(seq_len, seq_len, 1)

        self.act = act

        init_eps /= seq_len
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.0)

    def set_sample_config(self, sample_intermediate_size):
        self.norm.set_sample_config(sample_intermediate_size // 2)

    def get_active_subnet(self, config, act):
        sublayer = SpatialUnit(
            config.sample_intermediate_size,
            config.max_seq_length,
            act,
            config.layer_norm_eps,
        )
        sublayer.norm = self.norm.get_active_subnet(
            config.sample_intermediate_size // 2
        )
        sublayer.proj.weight.data.copy_(self.proj.weight)
        sublayer.proj.bias.data.copy_(self.proj.bias)

        return sublayer

    def forward(self, hidden_states, gate_res=None):
        device, n = hidden_states.device, hidden_states.shape[1]

        res, gate = hidden_states.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device=device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.0)

        gate = F.conv1d(gate, weight, bias)
        if gate_res is not None:
            # gate_res is output of bertselfattention which is a tuple of
            # (outputs, attention_scores), where attention scores can be None
            gate = gate + gate_res[0]
        # make gate a tuple
        gate = (self.act(gate) * res,)

        if gate_res is not None:
            # add attention scores if we output them
            gate = gate + gate_res[1:]

        return gate


class BertFNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.linear_inter = CustomLinear(config.hidden_size, config.intermediate_size)
        self.linear_final = CustomLinear(config.intermediate_size, config.hidden_size)
        self.dropout_inter = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_final = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()
        self.is_identity_layer = False

    def set_sample_config(self, config, is_identity_layer=False):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.norm.set_sample_config(config.sample_hidden_size)
        self.linear_inter.set_sample_config(
            config.sample_hidden_size, config.sample_intermediate_size
        )
        self.linear_final.set_sample_config(
            config.sample_intermediate_size, config.sample_hidden_size
        )

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.intermediate_size,
            sample_hidden_size=config.sample_intermediate_size,
        )
        self.dropout_inter = nn.Dropout(sample_hidden_dropout_prob)

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=config.sample_hidden_size,
        )
        self.dropout_final = nn.Dropout(sample_hidden_dropout_prob)

    def get_active_subnet(self, config):
        subnet = BertFNet(config)
        subnet.norm = self.norm.get_active_subnet(config)

        subnet.linear_inter = self.linear_inter.get_active_subnet()
        subnet.linear_final = self.linear_final.get_active_subnet()

        return subnet

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        residual = hidden_states

        if self.is_identity_layer:
            return (hidden_states,)

        x = torch.fft.fft(hidden_states, dim=-1)  # Applying FFT along the embedding dim
        x = torch.fft.fft(x, dim=-2).real  # Applying FFT along the token or seq_len dim
        x += residual

        x = self.norm(x)
        x_res = x

        x = self.linear_inter(x)
        x = self.act(x)
        x = self.dropout_inter(x)
        x = self.linear_final(x)
        x = self.dropout_final(x)

        x += x_res

        x = self.norm(x)

        return (x,)


class BertDense(nn.Module):
    def __init__(self, config, act=nn.Identity()):
        super().__init__()
        ### Can we have this as an elasticization parameter ##
        ### TODO: Figure out if we need to add this as a elasticization/sampling choice in SuperNet ###
        if config.tiny_attn:
            # hard setting this to 1
            config.num_attention_heads = 1
            self.attention = BertSelfAttention(config)
        else:
            self.attention = None

        self.act = act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.norm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.channel_projection_in = CustomLinear(
            config.hidden_size, config.intermediate_size
        )
        self.proj_act = nn.GELU()
        self.spatial_projection = SpatialUnit(
            config.intermediate_size, config.max_seq_length, act, config.layer_norm_eps
        )
        self.channel_projection_out = CustomLinear(
            config.intermediate_size // 2, config.hidden_size
        )
        self.is_identity_layer = False

    def set_sample_config(self, config, is_identity_layer=False):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        sample_hidden_size = config.sample_hidden_size
        sample_intermediate_size = config.sample_intermediate_size

        self.norm.set_sample_config(sample_hidden_size)

        self.channel_projection_in.set_sample_config(
            sample_hidden_size, sample_intermediate_size
        )
        self.spatial_projection.set_sample_config(sample_intermediate_size)
        self.channel_projection_out.set_sample_config(
            sample_intermediate_size // 2, sample_hidden_size
        )

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.intermediate_size // 2,
            sample_hidden_size=sample_intermediate_size // 2,
        )
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)
        if self.attention is not None:
            self.attention.set_sample_config(config, tiny_attn=True)

    def get_active_subnet(self, config):

        sublayer = BertDense(config)
        sublayer.norm = self.norm.get_active_subnet(config.sample_hidden_size)
        sublayer.channel_projection_in = self.channel_projection_in.get_active_subnet()
        sublayer.spatial_projection = self.spatial_projection.get_active_subnet(
            config, self.act
        )
        sublayer.channel_projection_out = (
            self.channel_projection_out.get_active_subnet()
        )

        if self.attention is not None:
            sublayer.attention.get_active_subnet(config)

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        if self.is_identity_layer:
            return (hidden_states,)

        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        gate_res = (
            self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            if self.attention is not None
            else None
        )

        x = self.channel_projection_in(hidden_states)
        x = self.proj_act(x)
        outputs = self.spatial_projection(x, gate_res=gate_res)
        # only pass the hidden states to fully connected
        x = self.channel_projection_out(outputs[0])
        # add attention scores if we output them
        x += residual

        x = self.dropout(x)
        outputs = (x,) + outputs[1:]

        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def set_sample_config(self, config):
        prev_layer_importance_order = None
        # prev_layer_inv_importance_order = None
        sample_hidden_size = config.sample_hidden_size
        if self.config.rewire:
            self.invert_importance_order = self.config.hidden_size == sample_hidden_size

            if self.invert_importance_order is False and hasattr(
                self.self.query, "inv_importance_order"
            ):
                prev_layer_importance_order = self.self.query.importance_order
                # prev_layer_inv_importance_order = self.self.query.inv_importance_order

        sample_hidden_size = config.sample_hidden_size
        self.self.set_sample_config(config)
        self.output.set_sample_config(
            config, prev_layer_importance_order=prev_layer_importance_order
        )

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def get_active_subnet(self, config):
        sublayer = BertAttention(config)
        sublayer.self = self.self.get_active_subnet(config)
        sublayer.output = self.output.get_active_subnet(config)

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if self.config.rewire:
            if self.invert_importance_order and hasattr(
                self.self.query, "inv_importance_order"
            ):
                inv_importance_order = self.self.query.inv_importance_order

                # inverse the permutation before applying it in residual
                hidden_states = hidden_states[:, :, inv_importance_order]

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        if hasattr(config, "search_space_id") and (config.search_space_id is not None and config.search_space_id in ["v4.5"]):
            # normal: Linear -> relu -> Linear
            # 4.5: Depthwise Convnet -> Linear -> gelu -> Linear
            self.conv1d = BertConv1d(config)

    def set_sample_config(self, config):
        sample_intermediate_size = config.sample_intermediate_size if config.sample_intermediate_size > 10 else int(config.sample_intermediate_size * config.sample_hidden_size) # todo: make it dynamic
        sample_hidden_size = config.sample_hidden_size
        self.dense.set_sample_config(sample_hidden_size, sample_intermediate_size)

    def forward(self, hidden_states):
        if hasattr(self, "conv1d"):
            hidden_states = self.conv1d(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertConv1d(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_input_channels = config.hidden_size
        if config.search_space_id == "v4.2" or config.search_space_id == "v4.3":
            self.num_input_channels = config.intermediate_size
        self.normval_conv = False
        if config.search_space_id == "v4.1":
            self.normval_conv = True
        self.is_actvn = False
        if config.search_space_id == "v4.1" or config.search_space_id == "v4.2":
            self.is_actvn = True

        self.conv1d = DynamicSeparableConv1d(self.num_input_channels, [7], normal_conv=self.normval_conv) # todo: make 7 command line
        if self.is_actvn:
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.conv1d(hidden_states)
        if self.is_actvn:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = CustomLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.search_space_id = config.search_space_id if hasattr(config, "search_space_id") else None
        if hasattr(config, "search_space_id") and (config.search_space_id is not None and config.search_space_id.startswith("v4") and config.search_space_id!="v4.5"):
            # normal: Linear -> relu -> Linear
            # 4.1: Linear -> relu -> Linear -> Normal Convnet -> relu  (current) 
            # 4.2: Linear -> relu -> depthwise Convnet -> relu -> Linear 
            # 4.3: Linear -> relu -> Linear -> Depthwise Convnet  
            # 4.5: Depthwise Convnet -> Linear -> gelu -> Linear
            self.conv1d = BertConv1d(config)

    def set_sample_config(self, config, prev_layer_importance_order=None):
        sample_intermediate_size = config.sample_intermediate_size if config.sample_intermediate_size > 10 else int(config.sample_intermediate_size * config.sample_hidden_size) # todo: make it dynamic
        sample_hidden_size = config.sample_hidden_size
        self.LayerNorm.set_sample_config(sample_hidden_size)
        self.dense.set_sample_config(sample_intermediate_size, sample_hidden_size)
        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.intermediate_size,
            sample_hidden_size=sample_intermediate_size,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

        if self.config.rewire:
            if hasattr(self.dense, "importance_order"):
                self.dense.sample_importance_order = self.dense.importance_order

                # sliced training
                if prev_layer_importance_order is not None:
                    # slice prev_layer_importance_order p1'
                    prev_layer_importance_order = prev_layer_importance_order[
                        :sample_hidden_size
                    ]

                    # final_importance_indices = get_overlap_order(
                    #     self.dense.sample_importance_order, prev_layer_importance_order
                    # )
                    final_importance_indices = torch.arange(
                        sample_hidden_size, requires_grad=False
                    )
                    self.dense.sample_importance_order = final_importance_indices

    def forward(self, hidden_states, input_tensor):
        # normal: Linear -> relu -> Linear
        # 4.1: Linear -> relu -> Linear -> Normal Convnet -> relu  (current) 
        # 4.2: Linear -> relu -> depthwise Convnet -> relu -> Linear 
        # 4.3: Linear -> relu -> Linear -> Depthwise Convnet  
        # 4.5: Depthwise Convnet -> Linear -> gelu -> Linear
        if self.search_space_id is None or self.search_space_id == "v4.5" or not self.search_space_id.startswith("v4"):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if self.config.rewire:
                if hasattr(self.dense, "sample_importance_order"):
                    importance_order = self.dense.sample_importance_order
                    input_tensor = input_tensor[:, :, importance_order]
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.search_space_id == "v4.1":
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if hasattr(self, "conv1d"):
                hidden_states = self.conv1d(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.search_space_id == "v4.2":
            if hasattr(self, "conv1d"):
                hidden_states = self.conv1d(hidden_states)
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.search_space_id == "v4.3":
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if hasattr(self, "conv1d"):
                hidden_states = self.conv1d(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.use_bottleneck = config.mixing == "bert-bottleneck"
        if self.use_bottleneck:
            # TODO: add initializer to the linear layer
            if config.use_hypernet_w_low_rank == 0:
                self.input_bottleneck = CustomLinear(config.hidden_size, config.hidden_size)
                self.output_bottleneck = CustomLinear(
                    config.hidden_size, config.hidden_size
                )
                if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_2L", "archrouting_1L", "archrouting_jack_2L", "archrouting_jack_1L"]:
                    self.arch_embeds = config.sample_hidden_size
                    if hasattr(config, "hypernet_input_format") and config.hypernet_input_format == "onehot":
                        self.arch_embeds = standard2onehot(self.arch_embeds)

                    self.arch_expert = None
                    self.expert_routing_type = config.expert_routing_type
                    if config.expert_routing_type == "archrouting_2L" or config.expert_routing_type == "archrouting_jack_2L":
                        if hasattr(config, "hypernet_input_format") and config.hypernet_input_format == "onehot":
                            self.arch_expert = torch.nn.Sequential(
                                torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                                torch.nn.Linear(config.hypernet_hidden_size, config.hypernet_hidden_size),
                                torch.nn.ReLU(),
                                torch.nn.Linear(config.hypernet_hidden_size, config.max_experts),
                                torch.nn.Softmax(dim=-1)
                            )
                        else:
                            self.arch_expert = torch.nn.Sequential(
                                torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                                torch.nn.ReLU(),
                                torch.nn.Linear(config.hypernet_hidden_size, config.max_experts),
                                torch.nn.Softmax(dim=-1)
                            )
                    elif config.expert_routing_type == "archrouting_1L" or config.expert_routing_type == "archrouting_jack_1L":
                        self.arch_expert = torch.nn.Sequential(
                            torch.nn.Linear(len(self.arch_embeds), config.max_experts),
                            torch.nn.Softmax(dim=-1)
                        )
                    # todo
                    self.max_hidden_size = float(768) 
                    self.min_hidden_size = float(120)
                    self.register_buffer("active_arch_embed", torch.zeros(len(self.arch_embeds)))
                elif hasattr(config, "expert_routing_type") and config.expert_routing_type in ["neuronrouting_jack_2L", "neuronrouting_jack_drop_2L"]:
                    self.arch_embeds = config.sample_hidden_size
                    if hasattr(config, "hypernet_input_format") and config.hypernet_input_format == "onehot":
                        self.arch_embeds = standard2onehot(self.arch_embeds)
                    self.expert_routing_type = config.expert_routing_type
                    self.arch_expert = None
                    if config.expert_routing_type == "neuronrouting_jack_2L":
                        self.arch_expert_fc1 = torch.nn.Sequential(
                            torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                            torch.nn.ReLU(),
                            # torch.nn.Linear(config.hypernet_hidden_size, config.max_experts*config.intermediate_size)
                            CustomLinear(config.hypernet_hidden_size, config.max_experts*config.intermediate_size)
                        )
                        self.arch_expert_fc2 = torch.nn.Sequential(
                            torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                            torch.nn.ReLU(),
                            CustomLinear(config.hypernet_hidden_size, config.max_experts*config.hidden_size)
                            # torch.nn.Linear(config.hypernet_hidden_size, config.max_experts*config.hidden_size)
                        )
                    elif config.expert_routing_type == "neuronrouting_jack_drop_2L":
                        self.arch_expert_fc1 = torch.nn.Sequential(
                            torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                            torch.nn.ReLU(),
                            # torch.nn.Dropout(0.1),
                            torch.nn.Linear(config.hypernet_hidden_size, config.max_experts*config.intermediate_size)
                        )
                        self.arch_expert_fc2 = torch.nn.Sequential(
                            torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                            torch.nn.ReLU(),
                            # torch.nn.Dropout(0.1),
                            CustomLinear(config.hypernet_hidden_size, config.max_experts*config.hidden_size)
                        )
                    self.max_hidden_size = float(768) 
                    self.min_hidden_size = float(120)
                    self.register_buffer("active_arch_embed", torch.zeros(len(self.arch_embeds)))
            else:
                self.input_bottleneck = HyperNetDynamicLinear(config.hidden_size, config.hidden_size, config.bottleneck_rank, config.sample_hidden_size, config.hypernet_hidden_size)
                self.output_bottleneck = HyperNetDynamicLinear(config.hidden_size, config.hidden_size, config.bottleneck_rank, config.sample_hidden_size, config.hypernet_hidden_size)
                self.max_hidden_size = float(768)  # max(config.sample_hidden_size))
                self.min_hidden_size = float(120)  # min(config.sample_hidden_size))
        elif config.search_space_id.startswith("v5."):
            self.arch_embeds = [float(config.hidden_size)/768.0, float(config.num_attention_heads)/12.0, float(config.intermediate_size)/2560.0]
            if config.search_space_id == "v5.2":
                self.arch_embeds.append(float(config.num_hidden_layers)/12.0)
            if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_2L"]:
                self.arch_expert = None
                self.expert_routing_type = config.expert_routing_type
                if config.expert_routing_type == "archrouting_jack_2L":
                    self.arch_expert = torch.nn.Sequential(
                        torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Linear(config.hypernet_hidden_size, config.max_experts),
                        torch.nn.Softmax(dim=-1)
                    )
                # todo
                self.max_hidden_size = float(768) 
                self.min_hidden_size = float(120)
                self.register_buffer("active_arch_embed", torch.zeros(len(self.arch_embeds)))
            elif hasattr(config, "expert_routing_type") and config.expert_routing_type in ["neuronrouting_jack_2L", "neuronrouting_jack_drop_2L"]:
                self.expert_routing_type = config.expert_routing_type
                self.arch_expert = None
                if config.expert_routing_type == "neuronrouting_jack_2L":
                    self.arch_expert_fc1 = torch.nn.Sequential(
                        torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                        torch.nn.ReLU(),
                        # torch.nn.Linear(config.hypernet_hidden_size, config.max_experts*config.intermediate_size)
                        CustomLinear(config.hypernet_hidden_size, config.max_experts*config.intermediate_size)
                    )
                    self.arch_expert_fc2 = torch.nn.Sequential(
                        torch.nn.Linear(len(self.arch_embeds), config.hypernet_hidden_size),
                        torch.nn.ReLU(),
                        CustomLinear(config.hypernet_hidden_size, config.max_experts*config.hidden_size)
                        # torch.nn.Linear(config.hypernet_hidden_size, config.max_experts*config.hidden_size)
                    )
                self.max_hidden_size = float(768) 
                self.min_hidden_size = float(120)
                self.register_buffer("active_arch_embed", torch.zeros(len(self.arch_embeds)))

        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.is_identity_layer = False

        # add expert layers
        if hasattr(config, "max_experts"):
            self.max_experts = getattr(config, "max_experts")
            assert(self.max_experts>1)
            # self.intermediate is first expert and other experts are:
            self.other_intermediate_experts = nn.ModuleList([BertIntermediate(config) for _ in range(self.max_experts-1)] )
            self.other_output_experts = nn.ModuleList([BertOutput(config) for _ in range(self.max_experts-1)] )

    def set_sample_config(self, config, is_identity_layer=False):
        sample_hidden_size = config.sample_hidden_size
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False

        if self.use_bottleneck:
            if config.use_hypernet_w_low_rank == 0:
                self.input_bottleneck.set_sample_config(
                    config.hidden_size, config.sample_hidden_size
                )
                self.output_bottleneck.set_sample_config(
                    config.sample_hidden_size,
                    config.hidden_size,
                )
                if hasattr(self, "arch_expert"):
                    norm_active_arch_embed = min_max_normalization(config.master_sample_hidden_size, self.min_hidden_size, self.max_hidden_size)
                    if hasattr(config, "hypernet_input_format") and config.hypernet_input_format == "onehot":
                        norm_active_arch_embed = standard2onehot(config.master_sample_hidden_size)
                    for i in range(len(norm_active_arch_embed)):
                        if hasattr(config, "fixed_hypernet_input") and config.fixed_hypernet_input == "yes":
                            self.active_arch_embed[i] = 0.5
                        else:
                            self.active_arch_embed[i] = norm_active_arch_embed[i]
            else:
                arch_embed = config.sample_hidden_size
                self.input_bottleneck.set_sample_config(
                    config.hidden_size, config.sample_hidden_size, min_max_normalization(config.master_sample_hidden_size, self.min_hidden_size, self.max_hidden_size) 
                )
                self.output_bottleneck.set_sample_config(
                    config.sample_hidden_size, config.hidden_size, min_max_normalization(config.master_sample_hidden_size, self.min_hidden_size, self.max_hidden_size)
                )
            # todo: set arch_encoding
        elif hasattr(self, "active_arch_embed"):
            self.active_arch_embed[0] = config.sample_hidden_size / 768.0
            self.active_arch_embed[1] = config.sample_num_attention_heads / 12.0
            self.active_arch_embed[2] = config.sample_intermediate_size / 3072.0
            if config.search_space_id == "v5.2":
                self.active_arch_embed[3] = config.num_hidden_layers / 12.0

        self.attention.set_sample_config(config)
        if hasattr(self, "crossattention"):
            self.crossattention.set_sample_config(config)

        prev_layer_importance_order = None
        # prev_layer_inv_importance_order = None

        if self.config.rewire:
            self.invert_importance_order = self.config.hidden_size == sample_hidden_size
            # for sliced training
            if self.invert_importance_order is False and hasattr(
                self.intermediate.dense, "inv_importance_order"
            ):
                prev_layer_importance_order = self.intermediate.dense.importance_order
                # prev_layer_inv_importance_order = (
                #    self.intermediate.dense.inv_importance_order
                # )

        self.intermediate.set_sample_config(config)

        self.output.set_sample_config(
            config, prev_layer_importance_order=prev_layer_importance_order
        )

        self.active_expert_id = 0
        if hasattr(self, "other_intermediate_experts"):
            additional_experts = self.max_experts-1 if config.last_expert_averaging_expert == "no" else self.max_experts-2
            for i in range(additional_experts):
                self.other_intermediate_experts[i].set_sample_config(config)
                self.other_output_experts[i].set_sample_config(config, prev_layer_importance_order=prev_layer_importance_order)
            self.active_expert_id = config.master_sample_expert_id
            self.last_expert_averaging_expert = config.last_expert_averaging_expert
            
            if self.expert_routing_type in ["neuronrouting_jack_2L", "neuronrouting_jack_drop_2L"]:
                self.arch_expert_fc2[-1].set_sample_config(config.hypernet_hidden_size, self.max_experts*config.sample_hidden_size)
                self.arch_expert_fc1[-1].set_sample_config(config.hypernet_hidden_size, self.max_experts*config.sample_intermediate_size)

    def get_active_subnet(self, config):
        sublayer = BertLayer(config)

        sublayer.attention.self.set_sample_config(
            config
        )  ## Just to access those variables

        if self.use_bottleneck:
            sublayer.input_bottleneck = self.input_bottleneck.get_active_subnet()
            sublayer.output_bottleneck = self.output_bottleneck.get_active_subnet()

        sublayer.attention = self.attention.get_active_subnet(config)

        #### Building the intermediate layer
        sublayer.intermediate.dense = self.intermediate.dense.get_active_subnet()

        #### Building the output layer
        sublayer.output.dense = self.output.dense.get_active_subnet()
        sublayer.output.LayerNorm = self.output.LayerNorm.get_active_subnet()

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        if self.is_identity_layer:
            return (hidden_states, None, None)

        if self.use_bottleneck:
            hidden_states = self.input_bottleneck(hidden_states)
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        if self.use_bottleneck:
            layer_output = self.output_bottleneck(layer_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        if hasattr(self, "last_expert_averaging_expert") and self.last_expert_averaging_expert == "yes" and self.active_expert_id == self.max_experts-1:
            # simulate averaging expert
            intermediate_weights = self.intermediate.dense.samples["weight"]
            intermediate_bias = self.intermediate.dense.samples["bias"]
            layer_output_weights = self.output.dense.samples["weight"]
            layer_output_bias = self.output.dense.samples["bias"]
            for expert_id in range(1, self.max_experts-2):
                intermediate_weights = intermediate_weights + self.other_intermediate_experts[expert_id].dense.samples["weight"]
                intermediate_bias = intermediate_bias + self.other_intermediate_experts[expert_id].dense.samples["bias"]
                layer_output_weights = layer_output_weights + self.other_output_experts[expert_id].dense.samples["weight"]
                layer_output_bias = layer_output_bias + self.other_output_experts[expert_id].dense.samples["bias"]
            intermediate_weights = intermediate_weights / (self.max_experts-1)
            intermediate_bias = intermediate_bias / (self.max_experts-1)
            layer_output_weights = layer_output_weights / (self.max_experts-1)
            layer_output_bias = layer_output_bias / (self.max_experts-1)
            intermediate_output = F.linear(attention_output, intermediate_weights, intermediate_bias)
            layer_output = F.linear(intermediate_output, layer_output_weights, layer_output_bias)
            layer_output = self.output.dropout(layer_output)
            layer_output = self.output.LayerNorm(layer_output + attention_output)
        elif hasattr(self, "arch_expert"):
            if self.expert_routing_type == "archrouting_jack_1L" or self.expert_routing_type == "archrouting_jack_2L":
                route_prob = self.arch_expert(self.active_arch_embed)
                route_prob_max, routes = torch.max(route_prob, dim=-1)
                intermediate_weights = route_prob[0] * self.intermediate.dense.samples["weight"]
                intermediate_bias = route_prob[0] * self.intermediate.dense.samples["bias"]
                layer_output_weights = route_prob[0] * self.output.dense.samples["weight"]
                layer_output_bias = route_prob[0] * self.output.dense.samples["bias"]
                for expert_id in range(self.max_experts-1):
                    intermediate_weights = intermediate_weights + (route_prob[expert_id+1] * self.other_intermediate_experts[expert_id].dense.samples["weight"])
                    intermediate_bias = intermediate_bias + (route_prob[expert_id+1] * self.other_intermediate_experts[expert_id].dense.samples["bias"])
                    layer_output_weights = layer_output_weights + (route_prob[expert_id+1] * self.other_output_experts[expert_id].dense.samples["weight"])
                    layer_output_bias = layer_output_bias + (route_prob[expert_id+1] * self.other_output_experts[expert_id].dense.samples["bias"])
                intermediate_weights = intermediate_weights 
                intermediate_bias = intermediate_bias 
                layer_output_weights = layer_output_weights 
                layer_output_bias = layer_output_bias 
                intermediate_output = F.linear(attention_output, intermediate_weights, intermediate_bias)
                intermediate_output = self.intermediate.intermediate_act_fn(intermediate_output)
                layer_output = F.linear(intermediate_output, layer_output_weights, layer_output_bias)
                layer_output = self.output.dropout(layer_output)
                layer_output = self.output.LayerNorm(layer_output + attention_output)
            elif self.expert_routing_type in ["neuronrouting_jack_2L", "neuronrouting_jack_drop_2L"]:
                fc1_expert_out = self.arch_expert_fc1(self.active_arch_embed)
                fc1_expert_out = fc1_expert_out.view(-1, self.max_experts)
                fc1_expert_out = torch.nn.Softmax(dim=-1)(fc1_expert_out)
                fc2_expert_out = self.arch_expert_fc2(self.active_arch_embed)
                fc2_expert_out = fc2_expert_out.view(-1, self.max_experts)
                fc2_expert_out = torch.nn.Softmax(dim=-1)(fc2_expert_out)
                intermediate_weights = self.intermediate.dense.samples["weight"] * fc1_expert_out[:, 0].view(-1,1)
                intermediate_bias = self.intermediate.dense.samples["bias"] * fc1_expert_out[:, 0]
                layer_output_weights = self.output.dense.samples["weight"] * fc2_expert_out[:, 0].view(-1,1)
                layer_output_bias = self.output.dense.samples["bias"] * fc2_expert_out[:, 0]
                for expert_id in range(self.max_experts-1):
                    intermediate_weights = intermediate_weights + (self.other_intermediate_experts[expert_id].dense.samples["weight"] * fc1_expert_out[:, expert_id+1].view(-1,1))
                    intermediate_bias = intermediate_bias + (self.other_intermediate_experts[expert_id].dense.samples["bias"] * fc1_expert_out[:, expert_id+1] )
                    layer_output_weights = layer_output_weights + (self.other_output_experts[expert_id].dense.samples["weight"] * fc2_expert_out[:, expert_id+1].view(-1,1))
                    layer_output_bias = layer_output_bias + (self.other_output_experts[expert_id].dense.samples["bias"] * fc2_expert_out[:, expert_id+1] )
                intermediate_output = F.linear(attention_output, intermediate_weights, intermediate_bias)
                intermediate_output = self.intermediate.intermediate_act_fn(intermediate_output)
                layer_output = F.linear(intermediate_output, layer_output_weights, layer_output_bias)
                layer_output = self.output.dropout(layer_output)
                layer_output = self.output.LayerNorm(layer_output + attention_output)
            else:
                intermediate_output = self.intermediate(attention_output) if routes == 0 else self.other_intermediate_experts[routes - 1](attention_output)
                layer_output = self.output(intermediate_output, attention_output) if routes == 0 else self.other_output_experts[routes - 1](intermediate_output, attention_output)
                layer_output = route_prob_max * layer_output
        else:
            intermediate_output = self.intermediate(attention_output) if self.active_expert_id == 0 else self.other_intermediate_experts[self.active_expert_id - 1](attention_output)
            layer_output = self.output(intermediate_output, attention_output) if self.active_expert_id == 0 else self.other_output_experts[self.active_expert_id - 1](intermediate_output, attention_output)
            
        '''
        if self.config.rewire:
            if self.invert_importance_order and hasattr(
                self.intermediate.dense, "inv_importance_order"
            ):
                inv_importance_order = self.intermediate.dense.inv_importance_order
                # undo permutation before adding in residual
                attention_output = attention_output[:, :, inv_importance_order]
        '''
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.mixing == "attention" or config.mixing == "bert-bottleneck":
            layer_function = BertLayer
        elif config.mixing == "gmlp":
            layer_function = BertDense
        elif config.mixing == "fnet":
            layer_function = BertFNet
        else:
            raise NotImplementedError(f"{config.mixing} is currently not implemented")

        self.use_bottleneck = config.mixing == "bert-bottleneck"

        self.layer = nn.ModuleList(
            [layer_function(config) for _ in range(config.sample_num_hidden_layers)] # changed to sample_num_hidden_layers (finetuning bug)
        )
        self.sample_num_hidden_layers = config.num_hidden_layers
        # to count the amount of times a layer is dropped
        self.layer_drop_counts = [0] * self.sample_num_hidden_layers

    def set_sample_config(self, config, drop_layers=True, drop_vector=None):

        self.sample_num_hidden_layers = config.sample_num_hidden_layers
        if config.layer_drop_prob > 0:
            assert self.sample_num_hidden_layers == self.config.num_hidden_layers

        if isinstance(config.sample_intermediate_size, list):
            sample_intermediate_sizes = config.sample_intermediate_size
        else:
            sample_intermediate_sizes = [config.sample_intermediate_size] * len(
                self.layer
            )
        if isinstance(config.sample_num_attention_heads, list):
            sample_num_attention_heads_list = config.sample_num_attention_heads
        else:
            sample_num_attention_heads_list = [config.sample_num_attention_heads] * len(
                self.layer
            )

        if self.use_bottleneck:
            sample_hidden_size = config.sample_hidden_size

        # TODO: We can either modify self.layer to just have layers that are not in layerdrop
        # or we could just iterate all layers and set identity layer to True (which we are currently doing)
        # Decide what is best and change this
        if drop_vector is not None:
            drop_layers = True
            layers_to_drop = drop_vector
        else:
            layers_to_drop = dropout_layers(
                config.sample_num_hidden_layers, config.layer_drop_prob
            )
        # for i, (drop, layer) in enumerate(zip(layers_to_drop, self.layer)): # to check
        for i, layer in enumerate(self.layer):
            layer_config = deepcopy(config)
            layer_config.master_sample_hidden_size = layer_config.sample_hidden_size
            if hasattr(config, "sample_expert_ids"):
                layer_config.master_sample_expert_id = config.sample_expert_ids[i]

            if i < self.sample_num_hidden_layers:
                layer_config.sample_intermediate_size = sample_intermediate_sizes[i]
                layer_config.sample_num_attention_heads = (
                    sample_num_attention_heads_list[i]
                )
                if self.use_bottleneck:
                    # for bert-bottleneck, use diff hidden size for each layer
                    layer_config.sample_hidden_size = sample_hidden_size[i]

                # if drop and drop_layers:
                #     layer.set_sample_config(layer_config, is_identity_layer=True)
                #    self.layer_drop_counts[i] += 1
                #else:
                layer.set_sample_config(layer_config, is_identity_layer=False)
            else:
                layer.set_sample_config(layer_config, is_identity_layer=True)

    def get_active_subnet(self, config):
        sublayer = BertEncoder(config)

        if isinstance(config.sample_intermediate_size, list):
            sample_intermediate_sizes = config.sample_intermediate_size
        else:
            sample_intermediate_sizes = [
                config.sample_intermediate_size
            ] * config.sample_num_hidden_layers
        if isinstance(config.sample_num_attention_heads, list):
            sample_num_attention_heads_list = config.sample_num_attention_heads
        else:
            sample_num_attention_heads_list = [
                config.sample_num_attention_heads
            ] * config.sample_num_hidden_layers

        if self.use_bottleneck:
            sample_hidden_size = config.sample_hidden_size

        ### Extracting the subnetworks
        for i in range(config.sample_num_hidden_layers):
            layer_config = deepcopy(config)

            layer_config.sample_intermediate_size = sample_intermediate_sizes[i]
            layer_config.sample_num_attention_heads = sample_num_attention_heads_list[i]

            if self.use_bottleneck:
                layer_config.sample_hidden_size = sample_hidden_size[i]

            sublayer.layer[i].set_sample_config(layer_config, is_identity_layer=False)

            sublayer.layer[i] = self.layer[i].get_active_subnet(layer_config)

        return sublayer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None and i < len(head_mask) else None # changed for variable num layers during finetuning
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, config):
        if isinstance(config.sample_hidden_size, list):
            # For bert bottleneck, sample_hidden_size is a list of bert bottleneck choices
            # this is not needed for pooler layer and hence we set it to full hidden size
            sample_hidden_size = config.hidden_size
        else:
            sample_hidden_size = config.sample_hidden_size
        self.dense.set_sample_config(sample_hidden_size, sample_hidden_size)

    def get_active_subnet(self, config):
        sublayer = BertPooler(config)
        self.dense.set_sample_config(config)  ## Should be unnecessary in principle
        sublayer.dense = self.dense.get_active_subnet()
        return sublayer

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN[config.normalization_type](
            config.hidden_size, eps=config.layer_norm_eps
        )

    def set_sample_config(self, config):
        sample_hidden_size = config.sample_hidden_size
        if config.mixing == "bert-bottleneck":
            sample_hidden_size = config.hidden_size
        self.dense.set_sample_config(sample_hidden_size, sample_hidden_size)
        self.LayerNorm.set_sample_config(sample_hidden_size)

    def get_active_subnet(self, config):
        subnet = BertPredictionHeadTransform(config)

        subnet.dense = self.dense.get_active_subnet()
        subnet.LayerNorm = self.LayerNorm.get_active_subnet(config)

        return subnet

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = CustomLinear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def set_sample_config(self, config):
        sample_hidden_size = config.sample_hidden_size
        if config.mixing == "bert-bottleneck":
            sample_hidden_size = config.hidden_size

        self.transform.set_sample_config(config)
        self.decoder.set_sample_config(sample_hidden_size, config.vocab_size)

    def get_active_subnet(self, config):
        subnet = BertLMPredictionHead(config)
        subnet.transform = self.transform.get_active_subnet(config)
        subnet.decoder = self.decoder.get_active_subnet()
        subnet.bias.data.copy_(self.bias)

        return subnet

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def set_sample_config(self, config):
        self.predictions.set_sample_config(config)

    def get_active_subnet(self, config):
        subnet = BertOnlyMLMHead(config)
        subnet.predictions = self.predictions.get_active_subnet(config)

        return subnet

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = CustomLinear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = CustomLinear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, CustomLinear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, CustomEmbedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, CustomLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def set_sample_config(self, config, drop_layers=True, drop_vector=None):
        self.embeddings.set_sample_config(config)
        # drop_layers is needed for layerdrop
        self.encoder.set_sample_config(
            config, drop_layers=drop_layers, drop_vector=drop_vector
        )
        if self.pooler is not None:
            self.pooler.set_sample_config(config)

        # if self.config.rewire:
        #    if hasattr(self, "inv_importance_order"):
        #        self.sample_inv_importance_order = self.inv_importance_order[
        #            :sample_hidden_size
        #        ]

    def get_active_subnet(self, config):
        subnet = BertModel(config)

        subnet.embeddings = self.embeddings.get_active_subnet(config)
        subnet.encoder = self.encoder.get_active_subnet(config)
        if self.pooler is not None:
            subnet.pooler = self.pooler.get_active_subnet(config)

        return subnet

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.sample_num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.config.rewire:
            if hasattr(self, "inv_importance_order"):
                sequence_output = sequence_output[:, :, self.inv_importance_order]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning. """,
    BERT_START_DOCSTRING,
)
class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning(
                "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING
)
class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        setattr(config, "normalization_type", "layer_norm")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.config = config
        if self.config.additional_random_softmaxing:
            self.random_softmaxing_idx = random.randint(1, 11)
        if hasattr(config, "add_distill_linear_layer") and config.add_distill_linear_layer:
            self.distill_input_features =  config.hidden_size[-1] if isinstance(config.hidden_size, list) else config.hidden_size
            self.fit_dense = CustomLinear(self.distill_input_features, self.distill_input_features)
        self.init_weights()

    def sample_next_layer(self):
        if random.random() <= self.config.random_layer_selection_probability:
            self.random_softmaxing_idx = random.choice(
                list(range(1, self.random_softmaxing_idx - 1))
                + list(range(self.random_softmaxing_idx + 2, 12))
            )
        else:
            if self.random_softmaxing_idx == 1:
                self.random_softmaxing_idx = 2
            elif self.random_softmaxing_idx == 11:
                self.random_softmaxing_idx = 10
            else:
                self.random_softmaxing_idx = random.choice(
                    [self.random_softmaxing_idx - 1, self.random_softmaxing_idx + 1]
                )
        return self.random_softmaxing_idx

    def set_sample_config(self, config, drop_layers=True, drop_vector=None):
        # pass drop_layers flag to bertmodel for layerdrop
        self.bert.set_sample_config(
            config, drop_layers=drop_layers, drop_vector=drop_vector
        )
        self.cls.set_sample_config(config)
        if hasattr(self, 'fit_dense'):
            # active_distill_input_features = config.sample_hidden_size[-1] if isinstance(config.sample_hidden_size, list) else config.sample_hidden_size
            self.fit_dense.set_sample_config(self.distill_input_features, self.distill_input_features)
            # self.active_distill_input_features = [768] + config.sample_hidden_size # embed: todo: adapt for other spaces beyond v1

    def get_active_subnet(self, config):
        subnet = BertForMaskedLM(config)
        # subnet.set_sample_config(config)
        subnet.bert = self.bert.get_active_subnet(config)
        subnet.cls = self.cls.get_active_subnet(config)
        # subnet.classifier = self.classifier.get_active_subnet()

        return subnet

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_soft_loss=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            if use_soft_loss:
                if self.config.alpha_divergence:
                    loss_fct = AdaptiveLossSoft(
                        self.config.alpha_min,
                        self.config.alpha_max,
                        self.config.beta_clip,
                        logits=True,  # both predictions and target are logits (non-softmaxed)
                    )
                else:
                    loss_fct = CrossEntropyLossSoft()

                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1, self.config.vocab_size),
                )
            else:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
                )

            if self.config.additional_random_softmaxing and self.training:
                prediction_scores = self.cls(
                    outputs.hidden_states[self.random_softmaxing_idx]
                )
                if use_soft_loss:
                    masked_lm_loss += loss_fct(
                        prediction_scores.view(-1, self.config.vocab_size),
                        labels.view(-1, self.config.vocab_size),
                    )
                else:
                    masked_lm_loss += loss_fct(
                        prediction_scores.view(-1, self.config.vocab_size),
                        labels.view(-1),
                    )
                    
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        hidden_states = outputs.hidden_states
        if hasattr(self, 'fit_dense'):
            tmp_hidden_states = []
            for s_id, sequence_layer in enumerate(hidden_states):
                # self.fit_dense.set_sample_config(self.active_distill_input_features[s_id], self.distill_input_features)
                tmp_hidden_states.append(self.fit_dense(sequence_layer))
            hidden_states = tmp_hidden_states

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert (
            self.config.pad_token_id is not None
        ), "The PAD token should be defined for generation"
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring). Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 2), labels.view(-1)
            )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (
                ((next_sentence_loss,) + output)
                if next_sentence_loss is not None
                else output
            )

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    def set_sample_config(self, config, drop_layers=True, drop_vector=None):
        if isinstance(config.sample_hidden_size, list):
            sample_hidden_size = config.hidden_size
        else:
            sample_hidden_size = config.sample_hidden_size
        self.bert.set_sample_config(
            config, drop_layers=drop_layers, drop_vector=drop_vector
        )
        self.classifier.set_sample_config(sample_hidden_size, config.num_labels)

        sample_hidden_dropout_prob = calc_dropout(
            config.hidden_dropout_prob,
            super_hidden_size=config.hidden_size,
            sample_hidden_size=sample_hidden_size,
        )
        # reinitialize the dropout module with new dropout rate
        # we can also directly use F.dropout as a function with the input
        # embedding on forward and the new dropout rate. But for now, we are just
        # reinitialing the module and using this in the forward function
        self.dropout = nn.Dropout(sample_hidden_dropout_prob)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def get_active_subnet(self, config):
        subnet = BertForSequenceClassification(config)
        # subnet.set_sample_config(config)
        subnet.bert = self.bert.get_active_subnet(config)
        subnet.classifier = self.classifier.get_active_subnet()

        return subnet

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        input_ids = (
            input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = CustomLinear(config.hidden_size, config.num_labels)

        self.init_weights()

    def set_sample_config(self, config, drop_layers=True, drop_vector=None):
        sample_hidden_size = config.sample_hidden_size
        if config.mixing == "bert-bottleneck":
            sample_hidden_size = config.hidden_size
        self.bert.set_sample_config(
            config, drop_layers=drop_layers, drop_vector=drop_vector
        )
        self.qa_outputs.set_sample_config(sample_hidden_size, config.num_labels)

    def get_active_subnet(self, config):
        subnet = BertForMaskedLM(config)
        # subnet.set_sample_config(config)
        subnet.bert = self.bert.get_active_subnet(config)
        subnet.cls = self.cls.get_active_subnet(config)
        # subnet.classifier = self.classifier.get_active_subnet()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
