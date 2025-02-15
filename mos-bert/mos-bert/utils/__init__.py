# taken from https://github.com/deepset-ai/FARM/blob/master/farm/utils.py
import os
from datetime import datetime
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from collections import OrderedDict
import json
from random import randrange

import math
import functools
import copy

millnames = ["", " Thousand", " Million", " Billion", " Trillion"]

# taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


# def unique_everseen(iterable, key=None):
#     "List unique elements, preserving order. Remember all elements ever seen."
#     # unique_everseen('AAAABBBCCDAABBB') --> A B C D
#     # unique_everseen('ABBCcAD', str.lower) --> A B C D
#     seen = set()
#     seen_add = seen.add
#     if key is None:
#         for element in filterfalse(seen.__contains__, iterable):
#             seen_add(element)
#             yield element
#     else:
#         for element in iterable:
#             k = key(element)
#             if k not in seen:
#                 seen_add(k)
#                 yield element


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# verify with this https://github.com/google-research/bert/issues/656
def calculate_params(
    emb_dims,
    num_attention_heads,
    d_ff_list,
    num_enc,  # num hidden layers
    vocab_size=28996,  # for bert base cased its 28,996
    add_output_emb_layer=True,
    add_embs_dim=True,
    bottleneck=True,
    max_emb_dim=768,
    merged_bottleneck=False,
    depth_features=None,  # for handling cases where layers are dropped
):

    if not bottleneck:
        # if there are no bottleneck layers, we cant have varying emb sizes
        assert not isinstance(emb_dims, list)
        emb_dims = [emb_dims] * num_enc
        max_emb_dim = emb_dims[-1]
    # (vocab_size + positional_embeddings + token_type_embeddings) * max_emb_dim +
    # layernorm weights and bias (2 * max_emb_dim)
    emb_params = (vocab_size + 512 + 2) * max_emb_dim + 2 * max_emb_dim

    # if add_embs_dim:
    #     # vocab_size + position_embeddings + token_type_embeddings
    #     emb_params = (vocab_size + 512 + 2) * emb_dim + 2 * emb_dim
    # else:
    #     # for scaling laws dont add emb_dim and output emb dimension
    #     emb_params = 0
    #     add_output_emb_layer = False

    assert len(d_ff_list) == num_enc
    if depth_features is not None:
        # assert sum(depth_features) != num_enc, "Cannot drop all layers"
        if sum(depth_features) == num_enc:
            # dropping all layers
            # this should technically be the emb size
            return 0.0
    per_layer_params = 0
    prev_bottleneck_dim = max_emb_dim
    for layer_idx, (d_ff, emb_dim) in enumerate(zip(d_ff_list, emb_dims)):

        if depth_features is not None:
            # if layer is dropped, we dont add the params
            if depth_features[layer_idx] == 1:
                continue
        # weight and bias in layernorm
        layer_norm_params = 2 * emb_dim

        per_layer_params += (
            4
            * (
                (emb_dim * emb_dim) + emb_dim
            )  # q, k,v and fc layer and their biases in bertattention
            + 2 * (emb_dim * d_ff)  # intermediate and bertouput
            + (d_ff + emb_dim)  # intermediate and bertoutput bias terms
            + 2 * layer_norm_params  # 2 layernorms in a transformer block
        )
        # if training with bottleneck, we can compose layers to reduce parameter size
        # for instance, output bottleneck of layer1 -> (128, 768) and input bottleneck of layer2 -> (768, 256)
        # we can then compose these two bottlenecks to one (128, 256)
        if merged_bottleneck:
            # Assuming we have 3 layers with these hidden sizes:
            # 120, 240, 360
            # Total bottleneck params: (768*120 + 120) + (120*240 + 240) + (240 * 360 + 360) + (360*768+768)
            per_layer_params += prev_bottleneck_dim * emb_dim + emb_dim
            prev_bottleneck_dim = emb_dim

        elif bottleneck:
            # for bottlenck params
            per_layer_params += (emb_dim * max_emb_dim + max_emb_dim) + (
                emb_dim * max_emb_dim + emb_dim
            )

    if merged_bottleneck:
        # to add final layer (360*768+768)
        per_layer_params += prev_bottleneck_dim * max_emb_dim + max_emb_dim

    # BertPredictionHeadTransform parameters
    output_emb_params = (max_emb_dim * max_emb_dim) + max_emb_dim + layer_norm_params

    if add_output_emb_layer:
        output_emb_layer = vocab_size * emb_dim + vocab_size
    else:
        output_emb_layer = 0

    return emb_params + per_layer_params + output_emb_params + output_emb_layer


def calculate_params_from_config(
    config,
    scaling_laws=False,
    add_output_emb_layer=False,
    merged_bottleneck=True,  # compose the bottlenecks and calculate the params count
    ffn_expansion = True,
    search_space_id = None
):
    add_embs_dim = scaling_laws != True

    # todo: do dynamically
    if max(getattr(config, "sample_intermediate_size")) < 10:
        new_config = copy.deepcopy(config)
        sample_intermediate_size = getattr(new_config, "sample_intermediate_size")
        sample_hidden_size = getattr(new_config, "sample_hidden_size")
        new_sample_intermediate_size = []
        for int_ratio, hid_size in zip(sample_intermediate_size, sample_hidden_size if isinstance(sample_hidden_size, list) else [new_config.sample_hidden_size]*new_config.sample_num_hidden_layers):
            new_sample_intermediate_size.append(int(int_ratio * hid_size))
        setattr(new_config, "sample_intermediate_size", new_sample_intermediate_size)
        config = new_config

    depth_features = None
    if hasattr(config, "depth_features"):
        if config.depth_features != []:
            depth_features = config.depth_features

    return calculate_params(
        config.sample_hidden_size,
        config.sample_num_attention_heads,
        config.sample_intermediate_size,
        config.sample_num_hidden_layers,
        config.vocab_size,
        add_output_emb_layer=add_output_emb_layer,
        add_embs_dim=add_embs_dim,
        bottleneck=(config.mixing == "bert-bottleneck"),
        merged_bottleneck=(config.mixing == "bert-bottleneck"), # earlier: merged_bottleneck
        depth_features=depth_features,
    )

# print(calculate_params(768, [12]*12, [3072]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=False, merged_bottleneck=False, depth_features=None))
# print(calculate_params([120]*12, [12]*12, [3072]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=True, merged_bottleneck=True, depth_features=None))
# print(calculate_params([120]*12, [12]*12, [1024]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=True, merged_bottleneck=True, depth_features=None))
# print(calculate_params(120, [12]*12, [1024]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=False, merged_bottleneck=False, depth_features=None))
# print(calculate_params([120]*12, [12]*12, [512]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=True, merged_bottleneck=True, depth_features=None))
# print(calculate_params(768, [12]*12, [3072]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=False, merged_bottleneck=False, depth_features=None))
# print(calculate_params(120, [12]*12, [240]*12, 12, 30522, add_output_emb_layer=False, add_embs_dim=True, bottleneck=False, merged_bottleneck=False, depth_features=None))

def check_path(path, error_message_template="Specified path - {} does not exist"):
    assert os.path.exists(path), error_message_template.format(path)


def get_current_datetime():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dropout_layers(num_layers, layer_drop_prob):
    if layer_drop_prob == 0:
        return torch.zeros(num_layers)

    to_drop = torch.zeros(num_layers).uniform_(0.0, 1.0) <= layer_drop_prob

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    # layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return to_drop


# taken from https://github.com/pytorch/pytorch/issues/3025#issuecomment-392601780
def get_overlap_order(ar1, ar2):
    # ar1 is the bigger matrix
    # ar2 is the sliced matrix
    # we get overlapping values' order in ar2

    assert ar1.shape[-1] >= ar2.shape[-1]

    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)
    mask[ar2.unique()] = True
    indexes = mask[ar1].nonzero().squeeze()
    overlapping_elements = ar1[indexes]
    return ar2.argsort()[overlapping_elements.argsort()]
