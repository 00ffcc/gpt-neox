# # Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model. with deep embed"""

import math
import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, recursive_setattr
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import (
    ParallelTransformerLayer,
    NormPipe,
    ParallelLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)

from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from typing import Union, List

from mole.dist_embed.a2a_embed import SparseEmbedding

class ParallelTransformerLayerDeepEmb(ParallelTransformerLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, attention_mask, emb=None, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()

        # x: [b, s, h] TODO: ? should be [s, b, h]

        # Enable delayedscaling if TransformerEngine's FP8 is used for MHA layer.
        if self.neox_args.te_fp8_mha:
            from megatron.model.transformer_engine import TEDelayedScaling

            fp8_recipe = TEDelayedScaling(neox_args=self.neox_args)
            fp8_context = fp8_recipe.get_context()
        else:
            from contextlib import nullcontext

            fp8_context = nullcontext()

        with fp8_context:
            if self.gpt_j_residual:
                # pseudocode:
                # x = x + attn(ln(x)) + mlp(ln(x))
                # this means we can avoid doing the allreduce in the attn / mlp outputs
                # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
                # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
                # we preserve the functionality for backwards compatibility

                residual = x
                # applies the correct normalization depending on if the norms are tied
                if self.gpt_j_tied and not self.neox_args.te_layernorm_mlp:
                    x = self.input_layernorm(x)
                    x1, x2 = x, x
                elif self.gpt_j_tied and self.neox_args.te_layernorm_mlp:
                    x2 = x
                    x = self.input_layernorm(x)
                    x1 = x
                elif self.neox_args.te_layernorm_mlp:
                    x1, x2 = self.input_layernorm(x), x
                else:
                    x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

                # attention operator
                attention_output, attention_bias = self.attention(
                    x1, attention_mask, layer_past=layer_past
                )
                if self.use_cache:
                    attention_output, presents = attention_output
                    self.layer_past = presents

                if attention_bias is not None:
                    with torch.enable_grad() if not self.eval else nullcontext():
                        attention_output = bias_dropout_fn(
                            attention_output,
                            bias=attention_bias.expand_as(attention_output),
                            residual=None,
                            prob=self.hidden_dropout,
                        )

                # mlp operator
                mlp_output, mlp_bias = self.mlp(x2)

                if emb is not None:
                    mlp_output = mlp_output * emb

                if mlp_bias is not None:
                    with torch.enable_grad() if not self.eval else nullcontext():
                        output = bias_dropout_fn(
                            mlp_output,
                            bias=mlp_bias.expand_as(mlp_output),
                            residual=attention_output,
                            prob=self.hidden_dropout,
                        )
                else:
                    output = mlp_output

                # output = (x + attn(ln(x)) + mlp(ln(x))
                output = residual + self.reduce(output)
            else:
                # pseudocode:
                # x = x + attn(ln1(x))
                # x = x + mlp(ln2(x))*emb

                residual = x

                # x = x + attn(ln1(x))
                attention_output, attention_bias = self.attention(
                    self.input_layernorm(x), attention_mask, layer_past=layer_past
                )

                if self.use_cache:
                    attention_output, presents = attention_output
                    self.layer_past = presents
                with torch.enable_grad() if not self.eval else nullcontext():
                    if attention_bias is not None:
                        # Use special bias_dropout_fn if we have a bias term from the above attention layer
                        attention_output = bias_dropout_fn(
                            attention_output,
                            bias=attention_bias.expand_as(residual),
                            residual=residual,
                            prob=self.hidden_dropout,
                        )
                    else:
                        # Otherwise just apply dropout + residual
                        attention_output = (
                            torch.nn.functional.dropout(
                                attention_output,
                                p=self.hidden_dropout,
                                training=self.training,
                            )
                            + residual
                        )

                # output = x + mlp(ln2(x))
                if self.neox_args.te_layernorm_mlp:
                    layernorm_output = attention_output
                else:
                    layernorm_output = self.post_attention_layernorm(attention_output)
                mlp_bias = torch.tensor(
                    0.0, device=layernorm_output.device, dtype=layernorm_output.dtype
                )

                # call signatures of both dense and MoE are the same
                mlp_output, mlp_bias = self.mlp(layernorm_output)

                if emb is not None:
                    mlp_output = mlp_output * emb

                with torch.enable_grad() if not self.eval else nullcontext():
                    if mlp_bias == None or (self.num_experts > 1):
                        # No dropout either
                        assert mlp_bias is None
                        output = mlp_output + attention_output
                    else:
                        output = bias_dropout_fn(
                            mlp_output,
                            bias=mlp_bias.expand_as(attention_output),
                            residual=attention_output,
                            prob=self.hidden_dropout,
                        )

            return output

class SequentialWrapperDeepEmb(torch.nn.Module):
    """
    Used to convert a deepspeed PipelineModule to an nn.Sequential like model whilst retaining
    activation checkpointing.
    """

    def __init__(
        self,
        neox_args,
        layers,
        emb,
        activation_checkpoint_interval,
        activation_checkpoint_func,
        parent_class_name=None,
    ):
        super().__init__()
        self.neox_args = neox_args
        self.sequential = torch.nn.Sequential(*layers)
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.parent_class_name = parent_class_name
        self.activation_checkpoint_func = activation_checkpoint_func
        self.batch_fn = None
        self.emb = emb

    def _is_checkpointable(self, funcs):
        if self.parent_class_name == "GPT2DeepEmbModelPipe":
            return all(
                "ParallelTransformerLayerDeepEmb" in f.__class__.__name__ for f in funcs
            )
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        _set_use_cache(self.sequential, use_cache)
        recursive_setattr(self.sequential, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching.
        """
        _set_use_cache(self.sequential, False)
        recursive_setattr(self.sequential, "training", True)

    def forward(
        self,
        forward_input,
        curriculum_seqlen=None,
        labels=None,
        neox_args=None,
        return_moe_losses=False,
    ):

        if self.batch_fn:
            forward_input = self.batch_fn(forward_input)

        if (
            curriculum_seqlen is not None
            and isinstance(forward_input, tuple)
            and len(forward_input) == 3
        ):
            neox_args.update_value("curriculum_seqlen", curriculum_seqlen)
            tokens = forward_input[0]
            input_ids = forward_input[1]
            attention_mask = forward_input[2]
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                # position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()
                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[
                    :, :, :curriculum_seqlen, :curriculum_seqlen
                ].contiguous()
            forward_input = (tokens, input_ids, attention_mask)

        moe_losses = []

        tokens, position_ids, attention_mask = forward_input
        if self.emb is not None:
            emb = self.emb[0](tokens.transpose(0, 1).contiguous()) # [b, s, h] -> [s, b, h]
        else:
            emb = None

        def exec_range_func(start, end):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.sequential[start:end]):
                    if not isinstance(layer, ParallelTransformerLayerDeepEmb):
                        inputs = layer(inputs)
                    else:
                        layer_idx = layer.layer_number
                        if layer_idx in self.neox_args.emb_layers_idx:
                            emb_idx = self.neox_args.emb_layers_idx.index(layer_idx)
                            hidden_size = self.neox_args.hidden_size
                            layer_emb = emb[:, :, hidden_size * emb_idx: hidden_size * (emb_idx + 1)]
                        else:
                            layer_emb = None
                        hidden_states, attention_mask = inputs
                        inputs = layer(hidden_states, attention_mask, emb=layer_emb), attention_mask
                    if hasattr(layer, "last_moe_loss"):
                        moe_losses.append(layer.last_moe_loss)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.sequential))
            x = func(forward_input)
        else:
            num_layers = len(self.sequential)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval): # TODO: 可优化?
                end_idx = min(
                    start_idx + self.activation_checkpoint_interval, num_layers
                )

                funcs = self.sequential[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x,)

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx, end_idx), *x
                    )
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        if return_moe_losses:
            return x, moe_losses
        else:
            return x

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.sequential, "layer_past", None)


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    mask_value = torch.finfo(attention_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(
        mask_value, dtype=attention_scores.dtype, device=attention_scores.device
    )
    attention_scores.masked_fill_(ltor_mask, mask_value)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


def _pre_transformer_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _post_transformer_block(args):
    # from (hidden_states, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)


class GPT2DeepEmbModelPipe(PipelineModule, torch.nn.Module):
    """GPT2Model adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.

    :param neox_args: NeoX arguments object (configuration)
    :param num_tokentypes: number of token types (TODO: deprecated, remove)
    :param parallel_output: if true, don't gather the output logits, and calculate loss in parallel. Set to true by default in training for efficiency, but set to false for inference.
    :param topology: deepspeed topology object specifying pipe / model parallelism topology.
    :param use_cache: if true, cache key/value pairs for each layer in inference.
    """

    def __init__(
        self,
        neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.neox_args = neox_args

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(
            self.neox_args
        )
        self.__topology__ = topology

        self.specs = []
        self.init_specs()  # initializes the layer specs (basically a fancy nn.Sequential)

        embedding_dim = len(neox_args.emb_layers_idx) * neox_args.hidden_size

        if embedding_dim > 0:
            self.emb = [SparseEmbedding(
                        num_embeddings         = neox_args.padded_vocab_size, 
                        embedding_dim          = embedding_dim,
                        optimizer_params       = neox_args.emb_optimizer_params,
                        std                    = neox_args.emb_init_std,
                        embedding_output_dtype = neox_args.emb_output_dtype,
                        params_dtype           = neox_args.emb_params_dtype,
                    )]
            nn.init.constant_(self.emb[0].weight, 1) # for DeepEmb
        else:
            self.emb = None

        super().__init__(
            layers=self.specs,
            loss_fn=partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy),
            topology=topology,
            activation_checkpoint_interval=self.neox_args.checkpoint_num_layers
            if self.neox_args.checkpoint_activations
            else 0,
            partition_method=neox_args.pipe_partition_method,
            checkpointable_layers=[
            ],
        )

    def insert_layers(
        self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx
    ):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any(
            [isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]
        ):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all(
                [hasattr(l, "__call__") for l in layers]
            ), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.neox_args.pipe_partition_method,
            checkpointable_layers=[
            ],
        )

    def init_specs(self):

        weight_tying = not self.neox_args.no_weight_tying
        self.specs = []

        # Embedding layer
        # input will be (input_ids, position_ids, attention_mask)

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        # NB: the attention mask always needs to be the *last* item in the args when being passed from
        # one stage to the next, because deepspeed is hacks on top of hacks.
        #
        # outputs are now (hidden_states,  attention_mask)

        self.specs.append(_pre_transformer_block)

        # T5 RPE positional embedding
        if self.neox_args.pos_emb == "rpe":
            hidden_size_per_attention_head = mpu.divide(
                self.neox_args.hidden_size, self.neox_args.num_attention_heads
            )
            rpe_scale = math.sqrt(hidden_size_per_attention_head)
            rpe_emb = ParallelRelativePositionBias(
                neox_args=self.neox_args,
                scale=rpe_scale,
                causal=True,
                num_buckets=self.neox_args.rpe_num_buckets,
                max_distance=self.neox_args.rpe_max_distance,
                heads=self.neox_args.num_attention_heads,
            )

        # Transformer layers
        for i in range(self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            assert layer_type not in ['gmlp', 'amlp', 'rwkv','mamba'], f"Layer {i} is of type {layer_type}, which is not supported by GPT2DeepEmb"
            self.specs.append(
                LayerSpec(
                    ParallelTransformerLayerDeepEmb,
                    neox_args=self.neox_args,
                    attention_mask_func=gpt2_attention_mask_func,
                    init_method=self.init_method,
                    output_layer_init_method=self.output_layer_init_method,
                    layer_number=i,
                    rpe=rpe_emb if self.neox_args.pos_emb == "rpe" else None,
                    rotary=self.neox_args.pos_emb == "rotary",
                    use_cache=self.use_cache,
                )
            )

        # used to drop attention mask + reshape hidden states
        self.specs.append(_post_transformer_block)

        # NormPipe is a (deprecated) helper class that used to be used to pass presents along the pipeline - since presents are now cached to the `TransformerLayer` class this is no longer needed
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        # outputs are now a single tensor: hidden_states

        def _logits_helper(embedding, lm_output):
            """Just a wrapper to massage inputs/outputs from pipeline."""
            if self.neox_args.use_mup:
                # Since we're using pipeline parallelism, we can't directly use MuReadout. Instead, use this workaround that does the same thing as MuReadout.
                # https://github.com/microsoft/mup/issues/6#issuecomment-1082156274
                lm_output = (
                    lm_output
                    / self.tied_modules.embed.word_embeddings.weight.infshape.width_mult()
                )

            logits = parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output,
                seq_parallel=self.neox_args.sequence_parallel,
            )
            return logits

        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    forward_fn=_logits_helper,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    ParallelLinearPipe,
                    neox_args=self.neox_args,
                    init_method=self.init_method,
                    parallel_output=self.parallel_output,
                    is_last_layer=True,
                )
            )

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.

        :param cache: (bool) True if you want to use caching during inference, False otherwise
        """
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)
        recursive_setattr(self.forward_funcs, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to True,
        so logits are not gathered across model parallel ranks, and loss is computed in parallel (more efficient).
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)
        recursive_setattr(self.forward_funcs, "training", True)

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(
                        Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x))
                    )
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, "__call__"):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f"Layer number {n} ({spec}) Not recognized")
        model = SequentialWrapperDeepEmb(
            self.neox_args,
            layers,
            self.emb,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model
