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
import tensorflow as tf


def layer_norm(weights, name, dtype=tf.float32):
    _eps = 1e-6

    with tf.name_scope("ln_{}".format(name)):
        scale = tf.Variable(
            name='scale',
            initializer=tf.ones(shape=[weights.get_shape()[1]]),
            dtype=dtype)
        beta = tf.Variable(name='beta',
                           initializer=tf.zeros(shape=[weights.get_shape()[1]
                                                       ], ),
                           dtype=dtype)

    mean, var = tf.nn.moments(x=weights, axes=[1], keepdims=True)
    norm_weights = (weights - mean) / tf.sqrt(var + _eps)

    return norm_weights * scale + beta
