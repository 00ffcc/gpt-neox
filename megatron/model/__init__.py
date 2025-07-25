# Copyright (c) 2025, EleutherAI
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

from .gpt2_model import GPT2ModelPipe
from .emb_model import GPT2DeepEmbModelPipe
from .emb_model_fast import GPT2DeepEmbModelPipe as GPT2DeepEmbModelPipeFast
from .utils import (
    get_params_for_weight_decay_optimization,
    mark_norms_for_sequence_parallel_grad_sync,
)
from .word_embeddings import SoftEmbedding
