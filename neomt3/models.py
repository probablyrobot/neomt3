# Copyright 2024 The MT3 Authors.
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

"""Feature converter and model for continuous inputs."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from transformers import PreTrainedModel, PreTrainedTokenizer


class ContinuousInputsEncDecFeatureConverter:
    """Feature converter for continuous inputs with encoder-decoder architecture."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.feature_specs = {
            "inputs": {"dtype": tf.float32, "rank": 2},
            "targets": {"dtype": tf.int32},
            "targets_position": {"dtype": tf.int32},
            "targets_segmentation": {"dtype": tf.int32},
            "targets_type_ids": {"dtype": tf.int32},
            "targets_attention_mask": {"dtype": tf.int32},
        }

    def _convert_features(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Convert features to model inputs."""
        # Get input features
        inputs = features["inputs"]
        targets = features["targets"]

        # Create decoder input tokens
        decoder_input_tokens = self._create_decoder_input_tokens(targets)

        # Create attention masks
        encoder_attention_mask = tf.ones_like(inputs[:, :, 0], dtype=tf.int32)
        decoder_attention_mask = tf.ones_like(targets, dtype=tf.int32)

        # Create position IDs
        encoder_position_ids = tf.range(tf.shape(inputs)[1], dtype=tf.int32)
        decoder_position_ids = tf.range(tf.shape(targets)[1], dtype=tf.int32)

        # Create type IDs
        encoder_type_ids = tf.zeros_like(encoder_position_ids)
        decoder_type_ids = tf.ones_like(decoder_position_ids)

        return {
            "encoder_inputs": inputs,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_position_ids": encoder_position_ids,
            "encoder_type_ids": encoder_type_ids,
            "decoder_input_ids": decoder_input_tokens,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "decoder_type_ids": decoder_type_ids,
            "labels": targets,
        }

    def _create_decoder_input_tokens(self, targets: tf.Tensor) -> tf.Tensor:
        """Create decoder input tokens by shifting targets right."""
        # Add start token at beginning
        start_token = tf.constant([self.tokenizer.bos_token_id], dtype=tf.int32)
        decoder_input_tokens = tf.concat([start_token, targets[:, :-1]], axis=1)
        return decoder_input_tokens


class ContinuousInputsEncoderDecoderModel:
    """Encoder-decoder model with continuous inputs."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        input_depth: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = True,
    ):
        """Initialize the model.

        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for encoding/decoding
            input_depth: Input feature dimension
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.input_depth = input_depth
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def generate(self, inputs: tf.Tensor, max_length: int) -> tf.Tensor:
        """Generate sequence from inputs.

        Args:
            inputs: Input tensor
            max_length: Maximum sequence length

        Returns:
            Generated sequence
        """
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return outputs
