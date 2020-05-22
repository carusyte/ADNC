# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf

from adnc.model.utils import layer_norm, get_activation
"""
A implementation of the LSTM unit, it performs a bit faster as the TF implementation and implements layer norm.
"""


class CustomLSTMCell(tf.keras.layers.Layer):
    def __init__(self,
                 num_units,
                 layer_norm=False,
                 activation='tanh',
                 seed=100,
                 **kwargs):

        self.num_units = num_units
        self.layer_norm = layer_norm

        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed

        self._forget_bias = -1.0

        self._activation = activation
        self.act = get_activation(activation)

        super(CustomLSTMCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def zero_state(self, batch_size, dtype=tf.float32):
        zero_state = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return zero_state

    def build(self, input_shapes):
        if self.layer_norm:
            self.lstm_layer = self._lnlstm_layer
        else:
            self.lstm_layer = self._lstm_layer

    def call(self, inputs, cell_state):
        with tf.name_scope("{}".format(self.name)):
            outputs, cell_states = self.lstm_layer(inputs,
                                                   cell_state,
                                                   name="{}".format(self.name))
        return outputs, cell_states

    def get_config(self):
        return {
            'num_units': self.num_units,
            'layer_norm': self.layer_norm,
            'rng': self.rng,
            'seed': self.seed,
            'name': self.name,
            'dtype': self.dtype,
            '_forget_bias': self._forget_bias,
            'activation':self._activation
        }

    def _lstm_cell(self, inputs, pre_cell_state, cell_size, w_ifco, b_ifco):

        ifco = tf.matmul(inputs, w_ifco) + b_ifco

        gates = tf.sigmoid(ifco[:, 0 * cell_size:3 * cell_size])
        cell_state = tf.add(
            tf.multiply(gates[:, 0:cell_size], pre_cell_state),
            tf.multiply(gates[:, cell_size:2 * cell_size],
                        self.act(ifco[:, 3 * cell_size:4 * cell_size])))
        output = gates[:, 2 * cell_size:3 * cell_size] * self.act(cell_state)

        return output, cell_state

    def _lstm_layer(self, inputs, pre_cell_state, name=0):

        inputs_shape = inputs.get_shape()
        if inputs_shape.__len__() != 2:
            raise UserWarning(
                "invalid shape: inputs at _lstm_layer {}".format(name))
        input_size = inputs_shape[1].value

        cell_shape = pre_cell_state.get_shape()
        if cell_shape.__len__() != 2:
            raise UserWarning(
                "invalid shape: cell_shape at _lstm_layer {}".format(name))
        cell_size = cell_shape[1].value

        with tf.name_scope("cell_{}".format(name)):
            w_ifco = tf.Variable(
                name="w_ifco_{}".format(name),
                initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                    seed=self.seed)(shape=(input_size, 4 * cell_size),
                                    dtype=self.dtype),
            )
            b_ifco = tf.Variable(name="b_ifco_{}".format(name),
                                 initializer=tf.zeros(shape=(4 * cell_size, )),
                                 dtype=self.dtype)

            output, cell_state = self._lstm_cell(inputs, pre_cell_state,
                                                 cell_size, w_ifco, b_ifco)

        return output, cell_state

    def _lnlstm_cell(self, inputs, pre_cell_state, cell_size, w_ifco, b_ifco):

        ifco = layer_norm(tf.matmul(inputs, w_ifco),
                          name="w_ifco",
                          dtype=self.dtype) + b_ifco
        gates = tf.sigmoid(ifco[:, 0 * cell_size:3 * cell_size])
        cell_state = tf.add(
            tf.multiply(gates[:, 0:cell_size], pre_cell_state),
            tf.multiply(gates[:, cell_size:2 * cell_size],
                        self.act(ifco[:, 3 * cell_size:4 * cell_size])))
        output = gates[:, 2 * cell_size:3 * cell_size] * self.act(
            layer_norm(cell_state, name="out_act", dtype=self.dtype))

        return output, cell_state

    def _lnlstm_layer(self, inputs, pre_cell_state, name):

        cell_state = pre_cell_state

        input_size = inputs.get_shape()[1].value

        print(pre_cell_state)

        cell_shape = cell_state.get_shape()
        cell_size = cell_shape[1].value

        with tf.name_scope("{}".format(name)):
            w_ifco = tf.Variable(
                name="w_ifco_{}".format(name),
                initial_value=tf.keras.initializers.VarianceScaling(
                    scale=1.0,
                    mode='fan_avg',
                    distribution='uniform',
                    seed=self.seed)(shape=(input_size, 4 * cell_size),
                                    dtype=self.dtype))

            b_ifco = tf.Variable(name="b_ifco_ln_{}".format(name),
                                 initial_value=tf.zeros(shape=(4 * cell_size),
                                                        dtype=self.dtype))

            output, cell_state = self._lnlstm_cell(inputs, cell_state,
                                                   cell_size, w_ifco, b_ifco)

        return output, cell_state
