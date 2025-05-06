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

"""Network definitions for MT3."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


class MT3Model(tf.keras.Model):
    """MT3 model for music transcription."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dff: int = 2048,
        dropout_rate: float = 0.1,
        max_sequence_length: int = 1000,
    ):
        """Initialize the model.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate
            max_sequence_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create positional encoding
        self.pos_encoding = self._get_positional_encoding()

        # Create embedding layers
        self.embedding = tf.keras.layers.Dense(d_model)

        # Create encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Create decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Create final layer
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def _get_positional_encoding(self) -> tf.Tensor:
        """Get positional encoding matrix.

        Returns:
            Positional encoding tensor of shape [1, max_sequence_length, d_model]
        """
        position = tf.range(self.max_sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / self.d_model)
        )

        pos_encoding = tf.zeros([self.max_sequence_length, self.d_model])

        # Calculate sine and cosine terms
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack(
                [
                    tf.repeat(tf.range(self.max_sequence_length), len(div_term)),
                    tf.tile(tf.range(0, self.d_model, 2), [self.max_sequence_length]),
                ],
                axis=1,
            ),
            tf.reshape(tf.sin(position * div_term), [-1]),
        )

        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack(
                [
                    tf.repeat(tf.range(self.max_sequence_length), len(div_term)),
                    tf.tile(tf.range(1, self.d_model, 2), [self.max_sequence_length]),
                ],
                axis=1,
            ),
            tf.reshape(tf.cos(position * div_term), [-1]),
        )

        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output logits
        """
        # Add positional encoding to input embeddings
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, : tf.shape(inputs)[1], :]

        # Apply dropout during training
        if training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        # Pass through encoder layers
        enc_output = x
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, training=training)

        # Pass through decoder layers
        dec_output = enc_output
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, training=training)

        # Final layer
        return self.final_layer(dec_output)

    def generate(self, inputs, max_length, temperature=1.0, top_k=0, top_p=0.0):
        """Generate sequence from inputs.

        Args:
            inputs: Input tensor
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter

        Returns:
            Generated sequence
        """
        batch_size = tf.shape(inputs)[0]

        # Initialize output sequence
        output = tf.zeros([batch_size, 0], dtype=tf.int32)

        # Generate tokens one by one
        for _ in range(max_length):
            # Get model predictions
            logits = self(inputs, training=False)
            logits = logits[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = (
                    logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
                )
                logits = tf.where(
                    indices_to_remove, tf.ones_like(logits) * -float("inf"), logits
                )

            # Apply top-p sampling
            if top_p > 0:
                sorted_logits = tf.sort(logits, direction="DESCENDING")
                cumulative_probs = tf.cumsum(
                    tf.nn.softmax(sorted_logits, axis=-1), axis=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat(
                    [
                        tf.zeros_like(sorted_indices_to_remove[:, :1]),
                        sorted_indices_to_remove[:, :-1],
                    ],
                    axis=-1,
                )
                indices_to_remove = tf.gather(
                    sorted_indices_to_remove,
                    tf.argsort(
                        tf.argsort(logits, direction="DESCENDING"),
                        direction="ASCENDING",
                    ),
                    batch_dims=1,
                )
                logits = tf.where(
                    indices_to_remove, tf.ones_like(logits) * -float("inf"), logits
                )

            # Sample next token
            probs = tf.nn.softmax(logits, axis=-1)
            next_token = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)

            # Append to output
            output = tf.concat([output, next_token], axis=1)

            # Stop if end token is generated
            if tf.reduce_all(next_token == 2):  # EOS token
                break

        return output


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer encoder layer."""

    def __init__(self, d_model, num_heads, dff, dropout_rate):
        """Initialize the layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        """Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        # Multi-head attention
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Transformer decoder layer."""

    def __init__(self, d_model, num_heads, dff, dropout_rate):
        """Initialize the layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None):
        """Forward pass.

        Args:
            x: Input tensor
            enc_output: Encoder output tensor
            training: Whether in training mode
            look_ahead_mask: Look-ahead mask for decoder

        Returns:
            Output tensor
        """
        # Self attention
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Multi-head attention
        attn2 = self.mha2(out1, enc_output, enc_output, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(ffn_output + out2)
