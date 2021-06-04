import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor
import lib.datautility as du
import warnings
import time
import csv
import sys
import os
import pandas as pd


class Normalization:
    NONE = 'None'
    Z_SCORE = 'z_score'
    MAX = 'max'


class Cost:
    NONE = 'None'
    MSE = 'MSE'
    L2_NORM = 'L2'
    CROSS_ENTROPY = 'cross_entropy'
    BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
    MULTICLASS_CROSS_ENTROPY = 'multiclass_cross_entropy'
    ROUNDED_CROSS_ENTROPY = 'rounded_cross_entropy'
    RMSE = 'rmse'
    ROUNDED_RMSE = 'rounded_rmse'
    CROSS_ENTROPY_RMSE = 'cross_entropy_rmse'
    HINGE_LOSS = 'hinge_loss'


class Optimizer:
    SGD = 'sgd'
    ADAM = 'adam'
    ADAGRAD = 'adagrad'


NETWORK_ID = 0


class Network:
    RAW_INPUT = 'raw_input'
    NORMALIZED_INPUT = 'normalized_input'

    @staticmethod
    def __get_next_id__(increment=True):
        global NETWORK_ID
        nid = NETWORK_ID
        if increment:
            NETWORK_ID += 1
        return nid

    def __init__(self, name=None):
        self.__id = Network.__get_next_id__()
        self.scope_name = 'network{}'.format(self.__id) if name is None else name
        self.layers = []
        self.__is_init = False
        self.step_size = None
        self.batch_size = None

        self.training_epochs = None

        self.args = dict()

        self.normalization = Normalization.NONE

        self.cost_method = Cost.MSE
        self.cost_function = None
        self.__cost = []
        self.__outputs = []
        self.__output_layers = []
        self.__output_weights = []
        self.__output_tf_wts = None
        self.__learn_wts = False

        self.__training_cost = None
        self.__validation_cost = None

        self.raw_input = None
        self.norm_input = None

        self.optimizer = Optimizer.SGD

        self.__tmp_multi_out = None

        self.deepest_hidden_layer = None

        self.recurrent = False
        # self.use_last = False
        self.deepest_recurrent_layer = None

        self.__deepest_hidden_layer_ind = None
        self.__deepest_non_output_layer_ind = None

        self.graph = tf.get_default_graph()

        self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU
        # self.session = tf.InteractiveSession()  # use GPU
        self.saver = None
        self.save_path = None

    def get_network_id(self):
        return self.__id

    def set_default_cost_method(self, cost_method=Cost.MSE):
        self.cost_method = cost_method

    def set_optimizer(self, optimizer=Optimizer.SGD):
        self.optimizer = optimizer

    def get_deepest_hidden_layer(self):
        return self.deepest_hidden_layer

    def get_deepest_hidden_layer_index(self):
        return self.__deepest_hidden_layer_ind

    def get_layer(self, index):
        if not (0 <= index < len(self.layers)):
            raise IndexError('Index is out of range for the network with {} layers'.format(len(self.layers)))
        return self.layers[index]

    def begin_multi_output(self, cost_methods=None, weights=None):
        self.__tmp_multi_out = dict()

        if cost_methods is None:
            cost_methods = ['default']
        cost_methods = np.array(cost_methods).reshape((-1))

        if weights is None:
            weights = [1]
        weights = np.array(weights).reshape((-1))

        self.__tmp_multi_out['methods'] = cost_methods
        self.__tmp_multi_out['weights'] = weights
        self.__tmp_multi_out['deepest_hidden'] = len(self.layers)-1

        return self

    def end_multi_output(self):
        if self.__tmp_multi_out is None:
            return self

        self.__deepest_hidden_layer_ind = self.__tmp_multi_out['deepest_hidden']
        self.deepest_hidden_layer = self.layers[self.__tmp_multi_out['deepest_hidden']]

        for i in range(self.__tmp_multi_out['deepest_hidden'] + 1, len(self.layers)):
            if len(self.__tmp_multi_out['methods']) == 1:
                m = self.__tmp_multi_out['methods'][0]
            elif i < len(self.__tmp_multi_out['methods']):
                m = self.__tmp_multi_out['methods'][i]
            else:
                m = 'default'

            if len(self.__tmp_multi_out['weights']) == 1:
                w = self.__tmp_multi_out['weights'][0]

            elif i < len(self.__tmp_multi_out['weights']):
                w = self.__tmp_multi_out['weights'][i]
            else:
                w = 1

            self.__learn_wts = True

            self.__outputs.append(self.layers[i]['h'])
            self.__output_layers.append(i)
            self.__cost.append(m)
            self.__output_weights.append(w)

        self.__tmp_multi_out = None
        return self

    def add_input_layer(self, n, normalization=Normalization.NONE):
        layer = dict()
        with tf.variable_scope(self.scope_name):
            layer['n'] = n
            layer['z'] = tf.placeholder(tf.float32, [None, None, n], name='x')
            layer['param'] = {'w': None, 'b': None, 'type': 'input',
                              'arg': {'stat1': tf.placeholder_with_default(tf.zeros([n]), shape=[n],
                                                                           name='input_stat1'),
                                      'stat2': tf.placeholder_with_default(tf.ones([n]), shape=[n],
                                                                           name='input_stat2')}}

            layer['a'] = tf.identity

            self.normalization = normalization
            if normalization == Normalization.Z_SCORE:
                layer['h'] = layer['a']((layer['z'] - layer['param']['arg']['stat1']) /
                                        tf.maximum(layer['param']['arg']['stat2'], tf.constant(1e-12, dtype=tf.float32)))
            elif normalization == Normalization.MAX:
                layer['h'] = layer['a']((layer['z'] - layer['param']['arg']['stat2']) /
                                        tf.maximum(layer['param']['arg']['stat1'] - layer['param']['arg']['stat2'],
                                                   tf.constant(1e-12, dtype=tf.float32)))
            else:
                layer['h'] = layer['a'](layer['z'])

        self.raw_input = layer['z']
        self.norm_input = layer['h']

        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_input_layer_from_network(self, network, layer_index):
        # network_layer = network.get_layer(layer_index)

        for i in range(layer_index+1):
            self.layers.insert(max(0, len(self.layers)), network.layers[i])

        # layer = dict()
        # layer['n'] = network_layer['n']
        # layer['z'] = network.layers[0]['z']
        # layer['param'] = network_layer['param']
        # layer['a'] = network_layer['a']
        # layer['h'] = network_layer['h']

        self.normalization = network.normalization
        self.args = network.args
        self.recurrent = network.recurrent

        self.raw_input = network.raw_input
        self.norm_input = network.norm_input

        # self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def copy_layer_from_network(self, network, layer_index):
        layer = network.layers[layer_index]
        network = Network()
        if layer['param']['type'] == 'input':
            self.add_input_layer(layer['n'], network.normalization)

        elif layer['param']['type'] == 'dense':
            pass
        elif layer['param']['type'] == 'inverse':
            pass
        elif layer['param']['type'] == 'lstm':
            self.add_lstm_layer(**layer['param']['ctor'])


            self.layers[-1]['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[-1]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))

            layer['param']['arg']['cell'] = tf.get_variable('Layer' + str(len(self.layers)) + '_C',
                                                            (layer['n']), dtype=tf.float32,
                                                            initializer=tf.constant_initializer(0.0))
        elif layer['param']['type'] == 'gru':
            pass
        elif layer['param']['type'] == 'rnn':
            pass
        elif layer['param']['type'] == 'merge':
            pass
        elif layer['param']['type'] == 'dropout':
            pass

    def add_dense_layer(self, n, activation=tf.identity):
        layer = dict()
        connecting_layer = self.get_deepest_hidden_layer_index()
        with tf.variable_scope(self.scope_name):
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'dense', 'arg': None}
            layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[connecting_layer]['n'], layer['n']),
                                                                  stddev=1. / np.sqrt(
                                                                      self.layers[connecting_layer]['n'])),
                                              dtype=tf.float32, name='Layer' + str(len(self.layers)) + '_W')
            layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]), name='Layer' + str(len(self.layers)) + '_B')

            bsize = tf.shape(self.layers[-1]['h'])[0]
            layer['z'] = tf.matmul(
                tf.reshape(self.layers[connecting_layer]['h'], [-1, self.layers[connecting_layer]['n']]),
                layer['param']['w']) + layer['param']['b']
            layer['a'] = activation
            layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, n]))
        self.layers.insert(max(0, len(self.layers)), layer)
        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_inverse_layer(self, layer_index, activation=tf.identity):
        if layer_index < 0:
            layer_index += len(self.layers)
        assert layer_index > 0 and layer_index < len(self.layers)
        inv = self.layers[layer_index]
        layer = dict()
        with tf.variable_scope(self.scope_name):
            layer['n'] = self.layers[layer_index - 1]['n']
            layer['param'] = {'w': None, 'b': None, 'type': 'inverse', 'arg': layer_index}
            layer['param']['w'] = None
            layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]))

            bsize = tf.shape(self.layers[-1]['h'])[0]
            layer['z'] = tf.matmul(tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]),
                                   tf.transpose(inv['param']['w'])) + \
                         layer['param']['b']
            layer['a'] = activation
            layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, self.layers[layer_index - 1]['n']]))
        self.layers.insert(max(0, len(self.layers)), layer)
        return self

    def __init_gate(self, n, feeding_n, activation=tf.identity, name='gate'):
        with tf.variable_scope(self.scope_name):
            gate = dict()
            gate['n'] = n
            gate['param'] = {'w': None, 'b': None, 'type': 'gate',
                             'arg': None}

            gate['param']['w'] = tf.get_variable(name + '_W', initializer=tf.truncated_normal(
                (feeding_n, gate['n']), stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                 dtype=tf.float32)
            gate['param']['b'] = tf.get_variable(name + '_B', (gate['n']), dtype=tf.float32,
                                                 initializer=tf.constant_initializer(0.0))

            gate['a'] = activation
        return gate

    def add_lstm_layer(self, n, peepholes=False, reverse=False, as_decoder=False, activation=tf.identity):
        self.recurrent = True
        # self.use_last = use_last
        connecting_layer = self.get_deepest_hidden_layer_index()

        with tf.variable_scope(self.scope_name):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'lstm',
                              'arg': {'timesteps': None, 'hsubt': None, 'cell': None},
                              'ctor':{'n':n,'peepholes':peepholes,'reverse':reverse,
                                      'as_decoder':as_decoder,'activation':activation}}

            if as_decoder:
                layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                      initializer=tf.truncated_normal(
                                                          (self.layers[connecting_layer-1]['n'] + layer['n'], layer['n']),
                                                          stddev=1. / np.sqrt(self.layers[-2]['n'])),
                                                      dtype=tf.float32)
            else:

                layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                      initializer=tf.truncated_normal(
                                                          (self.layers[connecting_layer]['n'] + layer['n'], layer['n']),
                                                          stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                      dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))

            layer['param']['arg']['cell'] = tf.get_variable('Layer' + str(len(self.layers)) + '_C',
                                                            (layer['n']), dtype=tf.float32,
                                                            initializer=tf.constant_initializer(0.0))

            layer['a'] = activation

        feeding_n = self.layers[-1]['n'] + n if not as_decoder else self.layers[-2]['n'] + n

        if peepholes:
            feeding_n += n

        forget_g = self.__init_gate(n, feeding_n, tf.sigmoid,
                                    name='Layer' + str(len(self.layers)) + '_forget')
        input_g = self.__init_gate(n, feeding_n, tf.sigmoid,
                                   name='Layer' + str(len(self.layers)) + '_input')
        output_g = self.__init_gate(n, feeding_n, tf.sigmoid,
                                    name='Layer' + str(len(self.layers)) + '_output')


        L = self.layers

        def __lstm_step(state, input):
            # with tf.variable_scope(self.scope_name, reuse=True):
            state = layer['a'](state)
            W = tf.get_variable('Layer' + str(len(L)) + '_W')
            W = tf.identity(W)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

            b = tf.get_variable('Layer' + str(len(L)) + '_B')
            b = tf.identity(b)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

            C = tf.get_variable('Layer' + str(len(L)) + '_C')
            C = tf.identity(C)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', C)

            # forget gate
            forgetW = tf.get_variable('Layer' + str(len(L)) + '_forget_W')
            forgetW = tf.identity(forgetW)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', forgetW)

            forgetb = tf.get_variable('Layer' + str(len(L)) + '_forget_B')
            forgetb = tf.identity(forgetb)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', forgetb)

            # input gate
            inputW = tf.get_variable('Layer' + str(len(L)) + '_input_W')
            inputW = tf.identity(inputW)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', inputW)

            inputb = tf.get_variable('Layer' + str(len(L)) + '_input_B')
            inputb = tf.identity(inputb)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', inputb)

            # output gate
            outputW = tf.get_variable('Layer' + str(len(L)) + '_output_W')
            outputW = tf.identity(outputW)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', outputW)

            outputb = tf.get_variable('Layer' + str(len(L)) + '_output_B')
            outputb = tf.identity(outputb)
            tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', outputb)

            concat = tf.concat([tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n'] if not as_decoder else L[connecting_layer-1]['n']]), tf.float32), state], 1)
            cell_prime = tf.tanh(tf.matmul(concat, W) + b)

            p_concat = state if not peepholes else tf.concat([state, tf.cast(tf.reshape(
                tf.tile(C, [tf.shape(state)[0]]), [-1, layer['n']]), tf.float32)], 1)

            concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n'] if not as_decoder else L[connecting_layer-1]['n']]), tf.float32), p_concat], 1)

            forget_h = forget_g['a'](tf.matmul(concat_g, forgetW) + forgetb)
            input_h = input_g['a'](tf.matmul(concat_g, inputW) + inputb)

            C = (C * forget_h) + (cell_prime * input_h)

            pr_concat = state if not peepholes else tf.concat([state, C], 1)
            concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n'] if not as_decoder else L[connecting_layer-1]['n']]), tf.float32), pr_concat], 1)
            output_h = output_g['a'](tf.matmul(concat_g, outputW) + outputb)

            layer['z'] = (output_h * tf.tanh(C))
            return layer['z']

        with tf.variable_scope(self.scope_name, reuse=True):
            shape = tf.shape(self.layers[connecting_layer]['h'])

            if not as_decoder:
                init = tf.Variable(tf.ones((1, layer['n'])) * 0.0)

                if reverse:
                    lstm_zs = tf.reverse(
                        tf.scan(__lstm_step, tf.reverse(tf.transpose(self.layers[connecting_layer]['h'], [1, 0, 2]), axis=[0]),
                                initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']])), axis=[0])
                else:
                    lstm_zs = tf.scan(__lstm_step, tf.transpose(self.layers[connecting_layer]['h'], [1, 0, 2]),
                                      initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']]))
            else:
                init = tf.transpose(self.layers[connecting_layer]['h'], [1, 0, 2])[-1]

                lstm_zs = tf.scan(__lstm_step, tf.transpose(tf.zeros_like(self.layers[connecting_layer-1]['h']), [1, 0, 2]),
                                  initializer=tf.reshape(init, [-1, layer['n']]))

            layer['h'] = layer['a'](tf.transpose(lstm_zs, [1, 0, 2]))
        self.layers.insert(max(0, len(self.layers)), layer)

        self.deepest_recurrent_layer = len(self.layers) - 1

        if self.__tmp_multi_out is None and not as_decoder:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_bidirectional_lstm_layer(self, n,  peepholes=False, as_decoder=False, activation=tf.identity):
        self.recurrent = True
        # self.use_last = use_last

        layer = dict()
        layer['n'] = n*2
        layer['param'] = {'w': None, 'b': None, 'type': 'merge', 'arg': None}

        deepest_rec = self.deepest_recurrent_layer
        deepest_hid = self.deepest_hidden_layer
        deepest_hid_ind = self.__deepest_hidden_layer_ind

        self.add_lstm_layer(n, peepholes=peepholes, as_decoder=as_decoder)
        forward = self.layers[-1]
        del self.layers[-1]
        self.layers.insert(0, dict())

        self.add_lstm_layer(n, peepholes=peepholes, reverse=True, as_decoder=as_decoder)
        reverse = self.layers[-1]

        self.layers.insert(-1,forward)
        del self.layers[0]

        self.deepest_recurrent_layer = deepest_rec
        self.deepest_hidden_layer = deepest_hid
        self.__deepest_hidden_layer_ind = deepest_hid_ind

        with tf.variable_scope(self.scope_name):
            bsize = tf.shape(self.layers[-1]['h'])[0]
            layer['z'] = tf.concat([forward['h'],reverse['h']], -1)
            layer['a'] = activation
            layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, layer['n']]))

        self.layers.insert(max(0, len(self.layers)), layer)

        self.deepest_recurrent_layer = len(self.layers) - 1

        if self.__tmp_multi_out is None and not as_decoder:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1

        return self

    def add_gru_layer(self, n,  activation=tf.identity):
        self.recurrent = True
        # self.use_last = use_last

        connecting_layer = self.get_deepest_hidden_layer_index()

        with tf.variable_scope(self.scope_name):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'gru',
                              'arg': {'timesteps': None, 'hsubt': None, 'cell': None}}

            layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[connecting_layer]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))

            layer['a'] = activation

        update_g = self.__init_gate(n, self.layers[connecting_layer]['n'] + n, activation=tf.sigmoid,
                                    name='Layer' + str(len(self.layers)) + '_update')
        reset_g = self.__init_gate(n, self.layers[connecting_layer]['n'] + n, activation=tf.sigmoid,
                                   name='Layer' + str(len(self.layers)) + '_reset')

        L = self.layers

        def __gru_step(state, input):
            with tf.variable_scope(self.scope_name, reuse=True):
                state = layer['a'](state)
                W = tf.get_variable('Layer' + str(len(L)) + '_W')
                W = tf.identity(W)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

                b = tf.get_variable('Layer' + str(len(L)) + '_B')
                b = tf.identity(b)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

                updateW = tf.get_variable('Layer' + str(len(L)) + '_update_W')
                updateW = tf.identity(updateW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', updateW)

                updateb = tf.get_variable('Layer' + str(len(L)) + '_update_B')
                updateb = tf.identity(updateb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', updateb)

                resetW = tf.get_variable('Layer' + str(len(L)) + '_reset_W')
                resetW = tf.identity(resetW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', resetW)

                resetb = tf.get_variable('Layer' + str(len(L)) + '_reset_B')
                resetb = tf.identity(resetb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', resetb)

                concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n']]), tf.float32), state], 1)
                update_h = update_g['a'](tf.matmul(concat_g, updateW) + updateb)
                reset_h = reset_g['a'](tf.matmul(concat_g, resetW) + resetb)

                concat = tf.concat([tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n']]), tf.float32), reset_h * state], 1)
                cell_prime = tf.tanh(tf.matmul(concat, W) + b)

                layer['z'] = (1 - update_h) * state + update_h * cell_prime

                return layer['z']

        with tf.variable_scope(self.scope_name):

            shape = tf.shape(self.layers[connecting_layer]['h'])
            init = tf.Variable(tf.zeros((1, layer['n'])))
            gru_zs = tf.scan(__gru_step, tf.transpose(self.layers[-1]['h'], [1, 0, 2]),
                             initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']]))

            layer['h'] = layer['a'](tf.transpose(gru_zs, [1, 0, 2]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_recurrent_layer = len(self.layers) - 1

        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_rnn_layer(self, n, activation=tf.identity):
        self.recurrent = True
        # self.use_last = use_last
        connecting_layer = self.get_deepest_hidden_layer_index()
        with tf.variable_scope(self.scope_name):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'rnn',
                              'arg': {'init': None, 'hsubt': None, 'cell': None}}

            layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[connecting_layer]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[connecting_layer]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))
            layer['param']['arg']['init'] = tf.Variable(tf.zeros((1, layer['n'])),
                                                        name='Layer' + str(len(self.layers)) + '_init')

            layer['a'] = activation
        L = self.layers

        def __rnn_step(state, input):
            with tf.variable_scope(self.scope_name, reuse=True):
                state = layer['a'](state)
                W = tf.get_variable('Layer' + str(len(L)) + '_W')
                W = tf.identity(W)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

                b = tf.get_variable('Layer' + str(len(L)) + '_B')
                b = tf.identity(b)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

                concat = tf.concat(
                    [tf.cast(tf.reshape(input, [-1, L[connecting_layer]['n']]), tf.float32), state], 1)

                layer['z'] = tf.tanh(tf.matmul(concat, W) + b)
                return layer['z']

        with tf.variable_scope(self.scope_name):
            shape = tf.shape(self.layers[connecting_layer]['h'])
            init = tf.Variable(tf.zeros((1, layer['n'])))
            rnn_zs = tf.scan(__rnn_step, tf.transpose(self.layers[connecting_layer]['h'], [1, 0, 2]),
                             initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']]))

            layer['h'] = layer['a'](tf.transpose(rnn_zs, [1, 0, 2]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_recurrent_layer = len(self.layers) - 1

        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_dropout_layer(self, n, keep=0.5, activation=tf.identity):
        layer = dict()
        connecting_layer = self.get_deepest_hidden_layer_index()

        with tf.variable_scope(self.scope_name):
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'dropout',
                              'arg': tf.placeholder(tf.float32, name='keep')}

            layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal((self.layers[connecting_layer]['n'],
                                                                                   layer['n']),
                                                                  stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))

            bsize = tf.shape(self.layers[connecting_layer]['h'])[0]
            layer['z'] = tf.matmul(tf.nn.dropout(tf.reshape(self.layers[connecting_layer]['h'],
                                                            [-1, self.layers[connecting_layer]['n']]),
                                                 layer['param']['arg']), layer['param']['w']) + layer['param']['b']
            layer['a'] = activation
            layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, n]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']] = keep
        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def __initialize(self):
        if not self.__is_init:


            self.y = []
            self.c = []
            # print(self.cost_method)

            if len(self.__outputs) == 0:
                self.__outputs.append(self.layers[-1]['h'])
                self.__output_layers.append(len(self.layers) - 1)
                self.__cost.append(self.cost_method)
                self.__output_weights.append(1)
                self.__deepest_hidden_layer_ind -= 1
                self.deepest_hidden_layer = self.layers[self.__deepest_hidden_layer_ind]

            for i in range(len(self.__cost)):
                if self.__cost[i] == 'default':
                    self.__cost[i] = self.cost_method

            # if self.__learn_wts:
            #     self.__output_tf_wts = None
            #
            #     self.__learn_wts = True
            #     self.__output_tf_wts = tf.nn.sigmoid(tf.Variable(tf.zeros([len(self.__outputs)]), name='output_wts'))
            #     # tf.Variable(1, name='output_wt' + str(i))
            # else:
            #     self.__output_tf_wts = tf.constant(self.__output_weights, dtype=tf.float32, name='output_wts')

            # cost_vec = []
            # if self.optimizer == Optimizer.ADAM:
            #     self.update = tf.train.AdamOptimizer(self.step_size, name='update')
            # elif self.optimizer == Optimizer.ADAGRAD:
            #     self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
            # else:
            #     self.update = tf.train.GradientDescentOptimizer(self.step_size, name='update')

            out_gradients = dict()
            out_grad_count = dict()

            for i in range(len(self.__outputs)):

                out_y = tf.placeholder(tf.float32, [None, None, self.layers[self.__output_layers[i]]['n']],
                                       name='y' + str(i))

                method = self.__cost[i]

                # print(method)

                if method == Cost.CROSS_ENTROPY:
                    # sum_cross_entropy = -tf.reduce_sum(
                    #     tf.where(tf.is_nan(out_y), self.__outputs[i], out_y) * tf.log(self.__outputs[i]),
                    #     reduction_indices=[-1])
                    # sce = tf.reduce_sum(tf.where(tf.is_nan(sum_cross_entropy), tf.zeros_like(sum_cross_entropy),
                    #                              sum_cross_entropy))
                    # cost_fn = sce/(tf.cast(tf.count_nonzero(sum_cross_entropy),
                    #                        tf.float32)+tf.constant(1e-4, dtype=tf.float32))
                    # cost_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.where(tf.is_nan(out_y[0]),
                    #                                                                       tf.zeros_like(out_y[0]),
                    #                                                                       out_y[0]),
                    #                                                       logits=tf.where(tf.is_nan(out_y[0]),
                    #                                                                       tf.zeros_like(out_y[0]),
                    #                                                                       self.__outputs[i]['z'])))

                    ##
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    z_flat = tf.reshape(self.layers[self.__output_layers[i]]['z'],
                                        (-1, self.layers[self.__output_layers[i]]['n']))

                    non_nan = tf.where(tf.logical_not(tf.is_nan(y_flat)))
                    if 'sigmoid' in str(self.layers[self.__output_layers[i]]['a']):
                        cost_fn = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.gather_nd(y_flat, non_nan),
                                                                    logits=tf.gather_nd(z_flat, non_nan)))
                    else:
                        cost_fn = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n'])),
                                logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n']))))
                        # self.debug1 = tf.nn.softmax_cross_entropy_with_logits(
                        #     labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                        #                       (-1, self.layers[self.__output_layers[i]]['n'])),
                        #     logits=tf.reshape(tf.gather_nd(y_flat, non_nan),
                        #                       (-1, self.layers[self.__output_layers[i]]['n'])))

                        # cost_fn = tf.where(tf.is_nan(cost_fn), tf.zeros_like(cost_fn), cost_fn)

                        #
                        # self.debug1 = tf.nn.softmax_cross_entropy_with_logits(
                        #             labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                        #                               (-1, self.layers[self.__output_layers[i]]['n'])),
                        #             logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                        #                               (-1, self.layers[self.__output_layers[i]]['n'])))
                        # self.debugn = tf.gather_nd(z_flat, tf.where(tf.logical_not(tf.is_nan(y_flat))))
                        # self.debugc = cost_fn
                        # self.debug2 = cost_fn

                        # cost_fn = self.debugc

                elif method == Cost.BINARY_CROSS_ENTROPY:
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    z_flat = tf.reshape(self.layers[self.__output_layers[i]]['z'],
                                        (-1, self.layers[self.__output_layers[i]]['n']))
                    non_nan = tf.where(tf.logical_not(tf.is_nan(y_flat)))
                    cost_fn = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.gather_nd(y_flat, non_nan),
                                                                logits=tf.gather_nd(z_flat, non_nan)))
                elif method == Cost.MULTICLASS_CROSS_ENTROPY:
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    z_flat = tf.reshape(self.layers[self.__output_layers[i]]['z'],
                                        (-1, self.layers[self.__output_layers[i]]['n']))

                    non_nan = tf.where(tf.logical_and(tf.logical_not(tf.is_nan(z_flat)),
                                                      tf.logical_and(tf.logical_not(tf.is_nan(y_flat)),
                                                                     tf.logical_not(
                                                                         tf.less(y_flat, tf.constant(-9e8))))))

                    cost_fn = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n'])),
                            logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n']))))

                    self.y_flat = tf.reshape(tf.gather_nd(y_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n']))
                    self.z_flat = tf.reshape(tf.gather_nd(z_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n']))
                    self.entropy = tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n'])),
                            logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n'])))


                elif method == Cost.ROUNDED_CROSS_ENTROPY:
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    z_flat = tf.reshape(tf.round(self.layers[self.__output_layers[i]]['z']),
                                        (-1, self.layers[self.__output_layers[i]]['n']))

                    non_nan = tf.where(tf.logical_not(tf.is_nan(y_flat)))
                    if 'sigmoid' in str(self.layers[self.__output_layers[i]]['a']):
                        cost_fn = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.gather_nd(y_flat, non_nan),
                                                                    logits=tf.gather_nd(z_flat, non_nan)))
                    else:
                        cost_fn = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n'])),
                                logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n']))))

                elif method == Cost.CROSS_ENTROPY_RMSE:
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    z_flat = tf.reshape(self.layers[self.__output_layers[i]]['z'],
                                        (-1, self.layers[self.__output_layers[i]]['n']))

                    non_nan = tf.where(tf.logical_not(tf.is_nan(y_flat)))

                    if 'sigmoid' in str(self.layers[self.__output_layers[i]]['a']):
                        cost_fn = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.gather_nd(y_flat, non_nan),
                                                                    logits=tf.gather_nd(z_flat, non_nan)))
                    else:
                        cost_fn = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n'])),
                                logits=tf.reshape(tf.gather_nd(z_flat, non_nan),
                                                  (-1, self.layers[self.__output_layers[i]]['n']))))

                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = cost_fn + tf.sqrt(sse / (tf.cast(tf.count_nonzero(sq_dif),
                                                               tf.float32) + tf.constant(1e-8, dtype=tf.float32)))

                elif method == Cost.L2_NORM:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = sse / 2

                elif method == Cost.RMSE:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                        reduction_indices=[-1])
                    # mean, var = tf.nn.moments(sse,axes=[0])
                    # sse = tf.where(tf.abs(sse-mean) < 6 * tf.sqrt(var),sse,tf.zeros_like(sse))

                    ssse = tf.reduce_sum(sse)

                    cost_fn = tf.sqrt(ssse / (tf.cast(tf.count_nonzero(sq_dif),
                                                     tf.float32) + tf.constant(1e-8, dtype=tf.float32)))

                elif method == Cost.ROUNDED_RMSE:
                    sq_dif = tf.squared_difference(tf.round(self.__outputs[i]), tf.where(tf.is_nan(out_y),
                                                                                         self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = tf.sqrt(sse / (tf.cast(tf.count_nonzero(sq_dif),
                                                     tf.float32) + tf.constant(1e-8, dtype=tf.float32)))

                elif method == Cost.HINGE_LOSS:
                    y_flat = tf.reshape(out_y, (-1, self.layers[self.__output_layers[i]]['n']))
                    h_flat = tf.reshape(self.layers[self.__output_layers[i]]['h'],
                                        (-1, self.layers[self.__output_layers[i]]['n']))

                    non_nan = tf.where(tf.logical_and(tf.logical_not(tf.is_nan(h_flat)),
                                                      tf.logical_and(tf.logical_not(tf.is_nan(y_flat)),
                                                                     tf.logical_not(
                                                                         tf.less(y_flat, tf.constant(-9e8))))))
                    cost_fn = tf.reduce_mean(
                        tf.losses.hinge_loss(
                            labels=tf.reshape(tf.gather_nd(y_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n'])),
                            logits=tf.reshape(tf.gather_nd(h_flat, non_nan),
                                              (-1, self.layers[self.__output_layers[i]]['n']))))

                    # tf.losses.hinge_loss(

                else:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = sse / (tf.cast(tf.count_nonzero(sq_dif),
                                             tf.float32) + tf.constant(1e-8, dtype=tf.float32))

                # self.var_grads = self.update.compute_gradients(self.cost_function, tf.trainable_variables())
                # self.clipped_var_grads = [(tf.clip_by_norm(
                #     tf.where(tf.is_nan(grad if grad is not None else tf.zeros_like(var)), tf.zeros_like(var),
                #              grad if grad is not None else tf.zeros_like(var)), 1.), var) for grad, var in
                #     self.var_grads]
                #####
                # for grad,var in self.update.compute_gradients(cost_fn, tf.trainable_variables()):
                #     # print(var)
                #     if var in out_gradients.keys():
                #         out_gradients[var] += grad if grad is not None else 0
                #         out_grad_count[var] += 1 if grad is not None else 0
                #     else:
                #         out_gradients[var] = grad if grad is not None else 0
                #         out_grad_count[var] = 1 if grad is not None else 0
                # print('----')
                #####

                w = self.__output_weights[i]
                # w = tf.gather(self.__output_tf_wts, tf.constant(i))
                # min_w = tf.gather(self.__output_tf_wts, tf.argmin(self.__output_tf_wts))
                self.c.append(w*cost_fn)

                # if self.cost_function is None:
                #     self.cost_function = w * cost_fn
                # else:
                #     self.cost_function += w * cost_fn
                self.y.append(out_y)

            # self.cost_function += tf.reduce_mean(1.-self.__output_tf_wts)
            self.cost_function = tf.reduce_sum(tf.where(tf.is_nan(self.c),
                                                        tf.zeros_like(self.c),
                                                        self.c)) if len(self.c) > 1 else self.c[0]

            if self.optimizer == Optimizer.ADAM:
                self.update = tf.train.AdamOptimizer(self.step_size, name='update')
            elif self.optimizer == Optimizer.ADAGRAD:
                self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
            else:
                self.update = tf.train.GradientDescentOptimizer(self.step_size, name='update')

            self.minimize_cost = self.update.minimize(self.cost_function)

            # self.var_grads = [(out_gradients[v]/out_grad_count[v], v) for v in out_gradients.keys()]
            #
            # self.clipped_var_grads = [(tf.clip_by_norm(
            #     tf.where(tf.is_nan(grad if grad is not None else tf.zeros_like(var)), tf.zeros_like(var),
            #              grad if grad is not None else tf.zeros_like(var)), 1.), var) for grad, var in self.var_grads]

            self.var_grads = self.update.compute_gradients(self.cost_function, tf.trainable_variables())
            self.clipped_var_grads = [(tf.clip_by_norm(
                tf.where(tf.is_nan(grad if grad is not None else tf.zeros_like(var)), tf.zeros_like(var),
                         grad if grad is not None else tf.zeros_like(var)), 1.), var) for grad, var in self.var_grads]
            self.update_weights = self.update.apply_gradients(self.clipped_var_grads)

            tf.global_variables_initializer().run()

            # tf.get_default_graph().finalize()

            self.__is_init = True

    def save_model_weights(self):
        if not os.path.exists('temp/'):
            os.makedirs('temp/')
        if not os.path.exists('temp/{}'.format(self.scope_name)):
            os.makedirs('temp/{}'.format(self.scope_name))
        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.session, 'temp/{}/mod'.format(self.scope_name))
        arg_arr = []
        for i in self.args:
            arg_arr.append([i.name,'temp/{}/{}.npy'.format(self.scope_name,i.name.replace('/','_').replace(':','-'))])
            np.save('temp/{}/{}.npy'.format(self.scope_name,i.name.replace('/','_').replace(':','-')),self.args[i])
        du.write_csv(np.array(arg_arr),'temp/{}/args.csv'.format(self.scope_name),['key','file'])

    def restore_model_weights(self):
        res_graph = tf.Graph()
        sess = tf.InteractiveSession(graph=res_graph)
        self.saver = tf.train.import_meta_graph('temp/{}/mod.meta'.format(self.scope_name))
        self.saver.restore(sess, tf.train.latest_checkpoint('temp/{}/'.format(self.scope_name)))

        res_var_name = [v.name for v in res_graph.get_collection('variables')]
        var_name = [v.name for v in self.graph.get_collection('variables')]
        for v in range(len(var_name)):
            if var_name[v] in res_var_name:
                a =tf.assign(self.graph.get_tensor_by_name(var_name[v]), sess.run(res_graph.get_tensor_by_name(res_var_name[v])))
                self.session.run(a)

        arg_arr,_ = du.read_csv('temp/{}/args.csv'.format(self.scope_name))
        for k,v in arg_arr:
            self.args[self.graph.get_tensor_by_name(k)] = np.load(v)
        sess.close()

    def rename_network(self, name):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sgv = graph_editor.make_view_from_scope(self.scope_name, self.graph)
            graph_editor.copy(sgv, self.graph, name, self.scope_name, True)

        # graph_editor.detach_(sgv)

        self.scope_name = name

    def __backprop_through_time(self, x, y, s):
        batch_cost = []

        valid = np.argwhere(np.array([len(k) for k in x[s]]) > 0).ravel()

        series_batch = x[s][valid]

        # print(len(y))
        # print(np.array(y, dtype=object))
        # exit(1)
        series_label = None
        # if not self.use_last:
        series_label = y[s][valid]
        n_timestep = max([len(k) for k in series_batch])

        series_batch_padded = np.array(
            [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

        series_label_padded = []

        # print(series_label)
        # print(np.array([a[i] for a in series_label]))
        # exit(1)

        for i in range(len(series_label[0])):
            series_label_padded.append(np.array(
                [np.pad(sl, ((0, n_timestep - len(sl)), (0, 0)), 'constant', constant_values=np.nan) for sl in
                 np.array([a[i] for a in series_label])]).reshape((len(series_batch), n_timestep, -1)))

        # print(series_label_padded)
        # exit(1)

        self.args[self.layers[0]['z']] = series_batch_padded.reshape((len(series_batch), n_timestep, -1))

        if not len(self.y) == len(series_label_padded):
            raise IndexError('The number of output layers does not match the labels supplied')

        nan_check = []
        for i in range(len(self.y)):
            self.args[self.y[i]] = series_label_padded[i]
            nan_check.append(np.all(np.isnan(series_label_padded[i].ravel())))

        if np.all(nan_check) or len(nan_check) == 0:
            return None

        # print(self.session.run(self.layers[-1]['h'], feed_dict=self.args))
        # print(self.session.run(self.layers[-2]['h'], feed_dict=self.args))
        # print(self.session.run(self.layers[-3]['h'], feed_dict=self.args))
        # print(self.session.run(self.layers[-4]['h'], feed_dict=self.args))
        # print(self.session.run(self.layers[-5]['h'], feed_dict=self.args))

        # exit(1)

        # self.session.run([self.update_weights], feed_dict=self.args)
        _, cost = self.session.run([self.update_weights, self.cost_function], feed_dict=self.args)

        # print(cost)
        # if cost == float('nan') or np.isnan(cost):
        # print('debug1', self.session.run(self.debug1, feed_dict=self.args))
        # print('debugn', self.session.run(self.debugn, feed_dict=self.args))
        # print('debugc', self.session.run(self.debugc, feed_dict=self.args))
        # print('debug2', self.session.run(self.debug2, feed_dict=self.args))

        # print('labels',series_label_padded)
        # print('maybe',self.session.run(self.maybe,feed_dict=self.args))
        # print('cost',self.session.run(self.cost_function, feed_dict=self.args))
        # print('cost n', self.session.run(self.cost_n,feed_dict=self.args))
        # exit(1)
        batch_cost.append(cost)

        return batch_cost

    def build(self):
        self.step_size = 0.1
        self.__initialize()
        return self

    def train(self, x, y, use_validation=True, validation_data=None, validation_labels=None,
              step=0.1, max_epochs=100, threshold=0.01, batch=1):

        # if not (du.ndims(x) == 3 and du.ndims(y) == 4):
        #     pass
        # TODO: if data is passed as wrong shape, reformat
        # TODO: ensure validation data also has correct format/shape

        self.__training_cost = []
        self.__validation_cost = []

        has_validation_set = False
        if validation_labels is not None and validation_data is not None:
            has_validation_set = True

        if use_validation and not has_validation_set:
            print('WARNING - Incomplete validation set information was provided. Using default 70/30% split.')

        self.step_size = step
        self.batch_size = batch
        self.training_epochs = max_epochs

        self.__initialize()

        if self.optimizer == Optimizer.ADAM:
            self.update = tf.train.AdamOptimizer(self.step_size, name='update')
        elif self.optimizer == Optimizer.ADAGRAD:
            self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
        else:
            self.update = tf.train.GradientDescentOptimizer(self.step_size, name='update')

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name,n.input)
        #
        # for n in range(50):
        #     print('=============================================================================================')
        # self.rename_network('asdf')
        # for n in range(50):
        #     print('=============================================================================================')
        #
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name,n.input)


        # exit(1)
        print("{:=<40}".format(''))
        print("{:^40}".format("Training Network"))
        print("{:=<40}".format(''))

        structure = describe_network_structure(self)
        print("-{} layers: {}".format(structure['n_layers'], structure['string']))
        print("-{} epochs".format(max_epochs))
        print("-step size = {}".format(step))
        print("-batch size = {}".format(batch))

        if not use_validation:
            print("{:=<40}".format(''))
            print("{:<10}{:^10}{:>10}".format("Epoch", "Cost", "Time"))
            print("{:=<40}".format(''))
        else:
            print("{:=<50}".format(''))
            print("{:<10}{:^10}{:^10}{:^10}{:>10}".format("Epoch", "Cost", "Val Cost", "Delta", "Time"))
            print("{:=<50}".format(''))

        # w = self.session.run(self.__output_tf_wts, feed_dict=self.args)*1000
        # print('wts: ', 1. + (w-(np.min(w))))

        # x = np.array(x, dtype=np.float32)
        # for i in range(len(x)):
        #     x[i] = np.array(x[i], dtype=np.float32)

        if self.normalization == Normalization.Z_SCORE:
            self.args[self.layers[0]['param']['arg']['stat1']] = np.nanmean(np.hstack([np.array(i,dtype=np.float32).ravel() for i in x])
                                                                            .reshape((-1, np.array(x[0]).shape[1])),
                                                                            axis=0).reshape((-1))
            self.args[self.layers[0]['param']['arg']['stat2']] = np.nanstd(np.hstack([np.array(i,dtype=np.float32).ravel() for i in x])
                                                                           .reshape((-1, np.array(x[0],dtype=np.float32).shape[1])),
                                                                           axis=0).reshape((-1))
        elif self.normalization == Normalization.MAX:
            self.args[self.layers[0]['param']['arg']['stat1']] = np.nanmax(np.hstack([np.array(i,dtype=np.float32).ravel() for i in x])
                                                                           .reshape((-1, np.array(x[0],dtype=np.float32).shape[1])),
                                                                           axis=0).reshape((-1))
            self.args[self.layers[0]['param']['arg']['stat2']] = np.nanmin(np.hstack([np.array(i,dtype=np.float32).ravel() for i in x])
                                                                           .reshape((-1, np.array(x[0],dtype=np.float32).shape[1])),
                                                                           axis=0).reshape((-1))

        desc = describe_multi_label(y)
        vdesc = None
        if has_validation_set:
            vdesc = describe_multi_label(validation_labels)

        seq_y = []
        warn = True

        for j in range(desc['n_label_sets']):
            if type(y[0][j][0][0]) is np.str_:
                print('Building label set {} from input...'.format(j))
                if y[0][j][0][0] == Network.RAW_INPUT:
                    target = self.raw_input
                elif y[0][j][0][0] == Network.NORMALIZED_INPUT:
                    target = self.norm_input
                else:
                    raise ValueError('Invalid label values for label set {}.'.format(j))

                for i in range(desc['n_samples']):
                    fd = dict()
                    fd[self.layers[0]['z']] = [x[i]]
                    y[i][j] = self.session.run(target, feed_dict=fd)[0]

                if has_validation_set:
                    for i in range(vdesc['n_samples']):
                        fd = dict()
                        fd[self.layers[0]['z']] = [validation_data[i]]
                        validation_labels[i][j] = self.session.run(target, feed_dict=fd)[0]
                print("{:=<50}".format(''))

        train_start = time.time()

        if use_validation and not has_validation_set:
            cut = int(np.floor(x.shape[0] * 0.7))

            tx = x[:cut]
            ty = y[:cut]
            vx = x[cut:]
            vy = y[cut:]
        else:
            tx = x
            ty = y
            vx = validation_data
            vy = validation_labels

        e = 1
        e_cost = []

        while e <= max_epochs:
            epoch_start = time.time()

            v = list(range(x.shape[0]))
            np.random.shuffle(v)
            # x = x[v]
            # y = y[v]
            cost = []
            val_cost = []

            for i in range(0, tx.shape[0], batch):
                s = np.array(range(i, min(tx.shape[0], i + batch)))
                if len(s) < batch:
                    continue

                if self.recurrent:
                    batch_cost = self.__backprop_through_time(tx, ty, s)

                    if batch_cost is None:
                        continue

                    for j in batch_cost:
                        cost.append(j)
                else:
                    ys = []

                    for j in range(len(ty[s][0])):
                        ys.append(np.array([a[j] for a in ty[s]]).reshape((batch, 1, -1)))

                    self.args[self.layers[0]['z']] = tx[s].reshape((batch, 1, -1))

                    if not len(self.y) == len(ys):
                        raise IndexError('The number of output layers does not match the labels supplied')

                    for j in range(len(self.y)):
                        self.args[self.y[j]] = ys[j]

                    self.minimize_cost.run(feed_dict=self.args)
                    cost.append(self.get_cost(tx[s], ty[s], False))
            # print(self.get_cost(tx, ty, True))

            if use_validation:
                # print(vy)
                # print(len(vy))
                val_batch_cost = []
                for ii in range(len(vy)):
                    val_batch_cost.append(self.get_cost(np.array([vx[ii]]), np.array([vy[ii]]), True))
                val_batch_cost = np.array(val_batch_cost)
                val_cost.append(np.mean(val_batch_cost))
                # val_cost.append(
                #     np.mean(val_batch_cost[abs(val_batch_cost - np.mean(val_batch_cost)) < 6 * np.std(val_batch_cost)]))
                # exit(1)
                # val_cost.append(self.get_cost(vx, vy, True))

                # print(val_cost[-1])
                if np.isnan(val_cost[-1]):
                    # vp = self.predict(vx)
                    print('Invalid sample found in validation set...')
                    rem = []
                    for q in range(len(vx)):
                        # print(q,'-',self.get_cost(np.array([vx[q]]),np.array([vy[q]]),True))
                        if np.isnan(self.get_cost(np.array([vx[q]]), np.array([vy[q]]), True)):
                            rem.append(q)
                            print('Sample {} marked for removal...'.format(q))
                            # print(np.array([vx[q]]))
                            # for v in np.array([vx[q]]):
                            #     for vi in v:
                            #         for vii in vi:
                            #             print(vii)
                            #     #print(v)
                            # print(np.array([vy[q]]))
                            # print(self.predict(np.array([vx[q]])))
                            #
                            #
                            # # print(self.session.run(self.cost_function,
                            # exit(1)

                    vx = np.delete(vx, rem)
                    vy = np.delete(vy, rem)

                    print('{} sample{} removed.'.format(len(rem), 's' if len(rem) > 1 else ''))
                    if len(vx) == 0:
                        raise ValueError('ERROR - All samples have been removed from validation')
                    # c = self.get_cost(vx, vy, True)
                    # print(c)
                    val_cost[-1] = self.get_cost(vx, vy, True)
                    # print(val_cost[-1])

            if use_validation:
                e_cost.append(np.nanmean(val_cost))
            else:
                e_cost.append(np.nanmean(cost))

            m_avg_dist = 5
            if e > m_avg_dist + 1:
                mean_last_ten = np.mean(e_cost[(-1 * (m_avg_dist + 1)):-1])
            else:
                mean_last_ten = 0

            self.__training_cost.append(np.nanmean(cost))

            if not use_validation:
                print("{:<10}{:^10.4f}{:>9.1f}s".format("Epoch " + str(e), e_cost[-1],
                                                        time.time() - epoch_start))
            else:
                self.__validation_cost.append(e_cost[-1])
                delta = np.mean(e_cost[(-1 * (min(m_avg_dist, e) + 1)):-1]) - \
                        np.mean(e_cost[-m_avg_dist:]) if e > 1 else 0
                print("{:<10}{:^10.4f}{:^10.4f}{:^10.4f}{:>9.1f}s".format("Epoch " + str(e), np.nanmean(cost),
                                                                          e_cost[-1],
                                                                          delta,
                                                                          time.time() - epoch_start))

            # w = self.session.run(self.__output_tf_wts, feed_dict=self.args)*1000
            # print('wts: ', 1. + (w - (np.min(w))))

            # flat_vx = flatten_sequence(self.predict(x=validation_data)[0])
            # flat_vy = flatten_sequence(extract_from_multi_label(validation_labels,0))
            # print(">> {:<.3f},{}".format(eu.auc(flat_vy,flat_vx),eu.auc(flat_vy,flat_vx,False)))
            # print(">> {:<.3f},{}".format(eu.cohen_kappa(flat_vy, flat_vx, 0.5),
            #                              eu.cohen_kappa(flat_vy, flat_vx, 0.5, False)))

            t = float(threshold)
            # print(np.abs(np.mean(e_cost[-m_avg_dist:]) - mean_last_ten))

            ##
            if (e > m_avg_dist + 1 and mean_last_ten - np.mean(e_cost[-m_avg_dist:]) < t) or e >= max_epochs:
                break
            ##
            # if e >= max_epochs:
            #     break

            e += 1

        if not use_validation:
            print("{:=<40}".format(''))
        else:
            print("{:=<50}".format(''))
        print("Total Time: {:<.1f}s".format(time.time() - train_start))

    def predict(self, x, layer_index=-1, batch=None):
        # TODO: add ability to predict from last hidden layer

        if len(x.shape) == 2:
            x = x.reshape((1,x.shape[0],x.shape[1]))
        if batch is None:
            batch = x.shape[0]

        if not (du.ndims(x) == 3):
            pass
            # TODO: if data is passed as wrong shape, reformat

        arg = dict(self.args)
        for i in arg:
            if 'keep' in i.name:
                arg[i] = 1
        for i in self.y:
            try:
                del arg[i]
            except KeyError:
                pass

        if self.recurrent:
            pred = None

            for b in range(0, x.shape[0], batch):
                s = np.array(range(b, min(x.shape[0], b + batch)))
                xb = x[s]

                valid = np.argwhere(np.array([len(k) for k in xb]) > 0).ravel()
                series_batch = xb[valid]

                samp_timestep = [len(k) for k in series_batch]
                n_timestep = max(samp_timestep)

                series_batch_padded = np.array(
                    [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in
                     series_batch])

                arg[self.layers[0]['z']] = series_batch_padded.reshape((len(xb), n_timestep, -1))

                if layer_index == -1:
                    p = self.session.run(self.__outputs, feed_dict=arg)
                else:
                    p = [self.session.run(self.get_layer(layer_index)['h'], feed_dict=arg)]

                n_out = 1
                if layer_index == -1:
                    n_out = len(self.__outputs)

                if pred is None:
                    pred = []
                    for i in range(n_out):
                        out_p = []
                        for j in range(len(samp_timestep)):
                            out_p.append(np.array(p[i][j])[:samp_timestep[j]])
                        pred.append(np.array(out_p))
                else:
                    for i in range(len(self.__outputs)):
                        out_p = []
                        for j in range(len(samp_timestep)):
                            out_p.append(np.array(p[i][j])[:samp_timestep[j]])
                        pred[i] = np.append(pred[i], np.array(out_p))
            return pred
        else:
            out_p = None
            for b in range(0, x.shape[0], batch):
                s = np.array(range(b, min(x.shape[0], b + batch)))
                arg[self.layers[0]['z']] = x[s]
                if layer_index == -1:
                    out = self.session.run(self.__outputs, feed_dict=arg)
                else:
                    out = self.session.run(self.get_layer(layer_index)['h'], feed_dict=arg)

                if out_p is None:
                    out_p = out
                else:
                    out_p = np.append(out_p, out)
            return [flatten_sequence(a) for a in out_p]

    def get_cost(self, x, y, test=True):

        if not (du.ndims(x) == 3 and du.ndims(y) == 4):
            pass
            # TODO: if data is passed as wrong shape, reformat

        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1

        if self.recurrent:
            pred = []

            valid = np.argwhere(np.array([len(k) for k in x]) > 0).ravel()

            try:
                series_batch = x[valid]
            except TypeError:
                print(valid)
                print(x)
                print(y)
                exit(1)

            series_label = None
            # if not self.use_last:
            series_label = y[valid]
            n_timestep = max([len(k) for k in series_batch])

            series_batch_padded = np.array(
                [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

            series_label_padded = []

            for i in range(len(series_label[0])):
                series_label_padded.append(np.array(
                    [np.pad(sl, ((0, n_timestep - len(sl)), (0, 0)), 'constant', constant_values=np.nan) for sl in
                     np.array([a[i] for a in series_label])]).reshape((len(series_batch), n_timestep, -1)))

            arg[self.layers[0]['z']] = series_batch_padded.reshape((len(series_batch), n_timestep, -1))

            if not len(self.y) == len(series_label_padded):
                raise IndexError('The number of output layers does not match the labels supplied ({} vs {})'.format(
                    len(self.y), len(series_label_padded)))

            for i in range(len(self.y)):
                arg[self.y[i]] = series_label_padded[i]
        else:
            ys = []

            for j in range(len(y[0])):
                ys.append(np.array([a[j] for a in y]).reshape((len(y), 1, -1)))

            arg[self.layers[0]['z']] = x.reshape((len(x), 1, -1))

            if not len(self.y) == len(ys):
                raise IndexError('The number of output layers does not match the labels supplied')

            for j in range(len(self.y)):
                # print(ys[j])
                arg[self.y[j]] = ys[j]

        # print(self.session.run([self.c[0], self.c[1], self.c[2], self.c[3], self.c[4]], feed_dict=arg))

        # cy, cz, ce = self.session.run([self.y_flat, self.z_flat, self.entropy], feed_dict=arg)
        #
        # for i in range(len(cy)):
        #     print(cz[i],cy[i],ce[i])
        # exit(1)

        return self.session.run(self.cost_function, feed_dict=arg)

    def get_training_series(self):

        series = dict()
        series['Training'] = self.__training_cost

        if self.__validation_cost is not None:
            series['Validation'] = self.__validation_cost

        return series


def describe_network_structure(net):

    structure = dict()
    nodes = []
    for i in range(net.get_deepest_hidden_layer_index() + 1):
        if net.layers[i]['param']['type'] == 'merge':
            continue
        if i < net.get_deepest_hidden_layer_index() and net.layers[i+1]['param']['type'] == 'merge':
                nodes[-1].append(net.layers[i]['n'])
        else:
            nodes.append([net.layers[i]['n']])

    if len(net.layers) > net.get_deepest_hidden_layer_index() + 1:
        nodes.append([])
        for i in range(net.get_deepest_hidden_layer_index() + 1, len(net.layers)):
            if net.layers[i]['param']['type'] == 'inverse':
                nodes[-1].append(net.layers[i]['n'])
                if i < len(net.layers)-1:
                    nodes.append([])
            else:
                nodes[-1].append(net.layers[i]['n'])

    str_structure = []
    for i in nodes:
        str_structure.append(', '.join(['{}n'.format(j) for j in i]))

    structure['n_layers'] = len(nodes)
    structure['list'] = len(nodes)
    structure['string'] = ' -> '.join(str_structure)

    return structure


def flatten_sequence(sequence, key=None, identifier=None):
    # TODO: move to datautility

    # the following handle the case when trying to flatten a single sample
    if type(sequence) is dict:
        sequence = np.array([sequence])

    if len(sequence.shape) == 2:
        sequence = sequence.reshape((1, sequence.shape[0], sequence.shape[1]))

    if key is not None and len(key.shape) == 1:
        key = key.reshape((1, -1))

    if identifier is not None and len(identifier.shape) == 2:
        identifier = identifier.reshape((1, identifier.shape[0], identifier.shape[1]))

    try:
        if key is not None:
            assert len(key) == len(sequence)
    except AssertionError:
        print(len(key))
        print(len(sequence))
        assert len(key) == len(sequence)

    if identifier is not None:
        assert len(identifier) == len(sequence) and du.ndims(identifier) == 3
        id = flatten_sequence(identifier)

    seq = list(sequence)
    dims = du.ndims(seq)

    if dims <= 2:
        return seq

    try:  # try the simple case
        if key is not None:
            # print(sum([len(np.array(seq[i])) for i in range(len(seq))]))
            ex_key = np.hstack([np.tile(key[i], len(np.array(seq[i]))) for i in range(len(seq))]).reshape(
                (-1, np.array(key).shape[1]))
            fl_seq = np.hstack([np.array(i).ravel() for i in seq]).reshape((-1, np.array(seq[0]).shape[1]))

            if identifier is None:
                return np.append(ex_key, fl_seq, axis=1)
            else:
                # print(ex_key.shape)
                # print(id.shape)
                # print(fl_seq.shape)
                return np.append(np.append(id, ex_key, axis=1), fl_seq, axis=1)

        if identifier is None:
            return np.hstack([np.array(i).ravel() for i in seq]).reshape((-1, np.array(seq[0]).shape[1]))
        else:
            return np.append(id, np.hstack([np.array(i).ravel() for i in seq]).reshape((-1, np.array(seq[0]).shape[1])),
                             axis=1)
    except (ValueError, IndexError, TypeError):
        # try:
        ar = None
        ind = 0
        # print('in flatten except')
        # print(len(seq))
        for i in range(len(seq)):
            row = seq[i][0]

            # print(row)

            for t in range(1, len(seq[i])):
                ntime = len(seq[i][t])
                row = np.append(row, np.hstack([np.array(j).ravel() for j in seq[i][t]]).reshape((ntime, -1)), 1)

            if key is not None:
                # print(seq[i][0])
                # print(seq[i])
                # print(key[i])
                # print(len(seq[i][0]))

                ex_key = np.array(np.tile(key[i], len(seq[i][0]))).reshape((len(seq[i][0]), -1))
                # print(ex_key)
                # print(row)
                # try:
                #     row = np.append(ex_key, row, axis=1)
                # except ValueError:
                #     print(ex_key)
                #     print(row)

                row = np.append(ex_key, row, axis=1)
                # print(row)

            if ar is None:
                ar = row
            else:
                ar = np.append(ar, row, 0)
        if identifier is None:
            return ar
        else:
            return np.append(id, ar, axis=1)
        # except ValueError:
        #     raise ValueError('sequence must be in the basic shape: (sample, time step, ... )')


def format_data(table, identifier=None, labels=None, columns=None, order=None, as_sequence=False, verbose=True):
    # TODO: move to datautility

    if len(np.array(table).shape) == 1:
        table = np.array([table])
    # print(np.array(table).shape)

    if as_sequence:
        if identifier is None:
            raise ValueError('identifier cannot be None when formatting as a sequence.')
        return reshape_sequence(table, identifier, labels, columns, order, verbose)

    table = np.array(table)
    table[np.where(table == '')] = 'nan'

    if order is None:
        ordering = list(range(len(table)))
    else:
        try:
            tbl_order = np.array(table[:, order], dtype=np.float32)
        except ValueError:
            tbl_order = np.array(table[:, order], dtype=str)
        ordering = np.argsort(tbl_order)

    table = table[ordering]

    if identifier is not None and not hasattr(identifier, '__iter__'):
        identifier = [identifier]

    if identifier is None:
        id_ind = table.shape[1]
    else:
        if hasattr(identifier, '__iter__'):
            id_ind = list(identifier)
            id_ind.append(table.shape[1])
        else:
            id_ind = [identifier, table.shape[1]]

    # id_ind = table.shape[1]
    table = np.append(table, np.array(range(len(table))).reshape((-1, 1)), 1)

    return reshape_sequence(table, id_ind, labels, columns, None, verbose)


def reshape_sequence(table, pivot, labels=None, columns=None, order=None, verbose=True):
    # TODO: move to datautility
    # TODO: redirect to format_sequence(...)

    if columns is None:
        columns = range(table.shape[-1])
    col = np.array(columns)

    table = np.array(table)
    table[np.where(table == '')] = 'nan'

    if hasattr(pivot, '__iter__'):
        pivot_ind = table.shape[1]
        # print('>>')
        # print(np.array([table[i, pivot] for i in range(len(table))], dtype=str).reshape((-1, 1)))
        table = np.append(table,
                          np.array(['~'.join(table[i, pivot]) for i in range(len(table))], dtype=str).reshape((-1, 1)),
                          1)
        pivot = pivot_ind
        # print('>>')

    try:
        tbl_order = np.array(table[:, pivot], dtype=np.float32)
    except ValueError:
        tbl_order = np.array(table[:, pivot], dtype=str)

    # print('>>')
    _, piv = np.unique(tbl_order, return_index=True)
    p = table[piv, pivot]
    # print('>>')

    seq = dict()
    key = []
    x = []
    y = []

    import sys

    if verbose:
        output_str = '-- formatting sequence...({}%)'.format(0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str

    n_rows = len(table)

    inc = 0
    for i in p:

        match = np.argwhere(np.array(table[:, pivot]) == i).ravel()

        ki = i
        if isinstance(ki, str):
            ki = ki.split('~')

        key.append(ki)

        if order is None:
            ordering = list(range(len(match)))
        else:
            if not hasattr(order,'__iter__'):
                order = [order]

            field = []
            for j in range(len(order)):
                try:
                    _ = np.array(table[match, :], dtype=np.float32)
                    field.append((str(j), float))
                except ValueError:
                    field.append((str(j), str))

            tbl_order = np.array(table[match], dtype=field)
            ordering = np.argsort(tbl_order, order=[str(j) for j in range(len(order))])

        x.append(np.array(table[match, col.reshape((-1, 1))],
                          dtype=np.float32).T[ordering].reshape((-1, len(col))))

        lab = None
        if labels is not None:
            labels = np.array(labels)

            # if not hasattr(labels[0], '__iter__'):
            #     lab = {0: np.array(table[match, np.array(labels).reshape((-1, 1))],
            #                    dtype=np.float32).T[ordering].reshape((-1, len(labels)))}
            # else:
            #     lab = dict()
            #     ind = 0
            #     for j in labels:
            #         mlabel = np.array(j).reshape((-1))
            #         # print(type(mlabel[0]))
            #         lab[ind] = np.array(table[match, np.array(mlabel).reshape((-1, 1))],
            #                             dtype=np.float32).T[ordering].reshape((-1, len(mlabel)))
            #         ind += 1

            if not hasattr(labels[0], '__iter__'):
                if type(labels[0]) == np.str_:
                    lab = {0: np.full((len(match), len(col)), labels[0])}
                else:
                    lab = {0: np.array(table[match, np.array(labels).reshape((-1, 1))],
                                       dtype=np.float32).T[ordering].reshape((-1, len(labels)))}
            else:
                lab = dict()
                ind = 0
                for j in labels:
                    mlabel = np.array(j).reshape((-1))
                    is_str = False
                    try:
                        np.array(mlabel[0], dtype=np.int)
                    except ValueError:
                        is_str=True
                    # print(mlabel[0])
                    # print(is_str)
                    if is_str:
                        lab[ind] = np.full((len(match), len(col)), mlabel[0])
                    else:
                        lab[ind] = np.array(table[match, np.array(mlabel, dtype=int).reshape((-1, 1))],
                                            dtype=np.float32).T[ordering].reshape((-1, len(mlabel)))
                    ind += 1

        y.append(lab)

        if verbose:
            if not round((inc / n_rows) * 100, 2) == round(((inc - 1) / n_rows) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- formatting sequence...({}%)'.format(round((inc / n_rows) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str
        inc += len(match)

        # table [match, order]
        # table = np.delete(np.array(table),match)

    if verbose:
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- formatting sequence...({}%)\n'.format(100))
        sys.stdout.flush()

    seq['key'] = np.array(key)
    seq['x'] = np.array(x)
    seq['y'] = np.array(y)
    return seq


def format_sequence(table, pivot, labels=None, covariates=None, identifiers=None, order=None, verbose=True):
    # TODO: move to datautility
    warn = False
    if covariates is None:
        covariates = range(table.shape[-1])
    col = np.array(covariates)

    if identifiers is not None:
        identifiers = np.array(identifiers)

    table = np.array(table)
    table[np.where(table == '')] = 'nan'

    if hasattr(pivot, '__iter__'):
        pivot_ind = table.shape[1]
        table = np.append(table,
                          np.array(['~'.join(table[i, pivot]) for i in range(len(table))], dtype=str).reshape((-1, 1)),
                          1)
        pivot = pivot_ind
    try:
        tbl_order = np.array(table[:, pivot], dtype=np.float32)
    except ValueError:
        tbl_order = np.array(table[:, pivot], dtype=str)

    _, piv = np.unique(tbl_order, return_index=True)
    p = table[piv, pivot]

    seq = dict()
    key = []
    x = []
    y = []
    id = []

    import sys

    if verbose:
        output_str = '-- formatting sequence...({}%)'.format(0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str

    n_rows = len(table)

    inc = 0
    for i in p:

        match = np.argwhere(np.array(table[:, pivot]) == i).ravel()
        filtered_table = table[match]

        ki = i
        if isinstance(ki, str):
            ki = ki.split('~')

        key.append(ki)

        if order is None:
            ordering = list(range(len(match)))
        else:
            if not hasattr(order,'__iter__'):
                order = [order]

            lex_order = []
            for j in range(len(order)):
                ord_j = len(order)-1-j
                try:
                    lex_order.append(np.array(filtered_table[:, order[ord_j]],dtype=np.float32))
                except ValueError:
                    lex_order.append(filtered_table[:, order[ord_j]])

            ordering = np.lexsort(tuple(lex_order))

        # print(np.array(filtered_table[:, col.reshape((-1))], dtype=np.float32)[ordering].reshape((-1, len(col))))
        # exit(1)
        try:
            x.append(np.array(filtered_table[:, col.reshape((-1))], dtype=str)[ordering].reshape((-1, len(col))))
        except ValueError:
            warn = True
            ft = np.array(filtered_table[:, col.reshape((-1))], dtype=str)
            # ft_shape = ft.shape
            # ft = pd.to_numeric(ft.ravel(),errors='coerce').reshape(ft_shape)
            # ft[np.isnan(ft)] = 0
            x.append(ft[ordering].reshape((-1, len(col))))

        if identifiers is not None:
            # print(np.array(table[match, identifiers], dtype=str))
            # print(match)
            # print(identifiers)
            # print(np.array(table[match, identifiers.reshape((-1, 1))], dtype=str))
            id.append(np.array(filtered_table[:, identifiers.ravel()], dtype=str)[ordering].reshape(
                (-1, len(identifiers))))

        lab = None
        if labels is not None:
            labels = np.array(labels)
            if not hasattr(labels[0], '__iter__'):
                if type(labels[0]) == np.str_:
                    lab = {0: np.full((len(match), len(col)), labels[0])}
                else:
                    lab = {0: np.array(filtered_table[:, np.array(labels).reshape((-1, 1))],
                                       dtype=np.float32).T[ordering].reshape((-1, len(labels)))}
            else:
                lab = dict()
                ind = 0
                for j in labels:
                    mlabel = np.array(j).reshape((-1))
                    is_str = False
                    try:
                        np.array(mlabel[0], dtype=np.int)
                    except ValueError:
                        is_str = True

                    if is_str:
                        lab[ind] = np.full((len(match), len(col)), mlabel[0])
                    else:
                        try:
                            lab[ind] = np.array(filtered_table[:, np.array(mlabel, dtype=int).reshape((-1))],
                                                dtype=np.float32)[ordering].reshape((-1, len(mlabel)))
                        except ValueError:
                            ftarr = filtered_table[:, np.array(mlabel, dtype=int).reshape((-1))]
                            for fti in range(len(ftarr)):
                                try:
                                    _ = np.array(ftarr[fti],dtype=np.float32)
                                except ValueError:
                                    ftarr[fti] = 'nan'
                            lab[ind] = np.array(ftarr,
                                                dtype=np.float32)[ordering].reshape((-1, len(mlabel)))
                    ind += 1

        y.append(lab)

        if verbose:
            if not round((inc / n_rows) * 100, 2) == round(((inc - 1) / n_rows) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- formatting sequence...({}%)'.format(round((inc / n_rows) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str
        inc += len(match)

    if verbose:
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- formatting sequence...({}%)\n'.format(100))
        sys.stdout.flush()

    seq['key'] = np.array(key)
    seq['x'] = np.array(x)
    seq['y'] = np.array(y)
    seq['iden'] = np.array(id)

    try:
        for i in range(len(seq['x'])):
            seq['x'][i] = np.array(seq['x'][i], dtype=np.float32)
    except ValueError:
        pass

    if verbose and warn:
        print('-- WARNING: String type found during sequence formatting')

    return seq


def format_sequence_from_file(filename, pivot, labels=None, covariates=None, identifiers=None, order=None,
                              verbose=True, header=True):
    if not os.path.exists('temp/'):
        os.makedirs('temp/')
    tmp_filename = 'temp/tmp_data.csv'
    data = pd.read_csv(filename)
    _, headers = du.read_csv(filename, 2)

    sortby = headers[pivot].tolist()
    if order is not None:
        if not hasattr(order, '__iter__'):
            sortby.append(headers[order])
        else:
            for o in order:
                sortby.append(headers[o])

    # sort by user, assignment, problem, and action and write the new data to file
    if verbose:
        print('-- preparing to format sequence from file...')
    data.sort_values(by=sortby).to_csv(tmp_filename, index=False)
    filename = tmp_filename
    # release the memory for the pandas dataframe, it is no longer needed
    data = None
    format_seq = False
    csvarr = []

    seq = None
    id = None
    n_lines = len(open(filename).readlines())
    with open(filename, 'r', errors='replace') as f:
        f_lines = csv.reader(f)

        if verbose:
            output_str = '-- formatting sequence...({}%)'.format(0)
            sys.stdout.write(output_str)
            sys.stdout.flush()
            old_str = output_str
        i = 0
        for line in f_lines:
            if (header and i == 0) or len(line) == 0:
                i += 1
                continue
            elif i % 2048 == 0:
                format_seq = True

            line = np.array(line)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            if '~'.join(np.array(line)[pivot]) != id:
                if format_seq and id is not None:
                    ar = np.array(csvarr).reshape([-1, len(line)])

                    fs = format_sequence(ar, pivot, labels, covariates, identifiers, order, False)

                    if seq is None:
                        seq = dict()
                        for k in fs.keys():
                            seq[k] = []

                    # length = []
                    # fslength = []
                    for k in seq.keys():
                        for fsk in fs[k]:
                            seq[k].append(fsk)

                        # length.append(len(seq[k]))
                        # fslength.append(len(fs[k]))

                    # print('{} {}'.format(length, fslength))
                    # if np.mean(length) != length[0]:
                    #     print(fs['x'])
                    #     print(fs['x'].shape)
                    #     exit(1)
                    # else:
                    #     length = []
                    #     fslength = []
                    #     for k in seq.keys():
                    #
                    #         try:
                    #             seq[k] = np.append(seq[k], fs[k].tolist(), axis=0)
                    #
                    #         except ValueError:
                    #             print(k)
                    #             print(du.ndims(seq[k]))
                    #             print(du.ndims(fs[k]))
                    #             du.write_csv(flatten_sequence(seq['iden']),'resources/affect_obs/seqk.csv')
                    #             du.write_csv(flatten_sequence(fs['iden']), 'resources/affect_obs/fsk.csv')
                    #             seq[k] = np.append(seq[k], fs[k], axis=0)
                    #             exit(1)
                    #
                    #     print('{} {}'.format(length,fslength))
                    #     if np.mean(length) != length[0]:
                    #         print(fs['x'])
                    #         print(fs['x'].shape)
                    #         exit(1)

                    csvarr = []
                    format_seq = False

                id = '~'.join(np.array(line)[pivot])

            csvarr.append(line)

            if verbose and not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- formatting sequence...({}%)'.format(round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1

        if len(csvarr) > 0:
            ar = np.array(csvarr)
            fs = format_sequence(ar, pivot, labels, covariates, identifiers, order, False)
            if seq is None:
                seq = fs
            else:
                for k in seq.keys():
                    seq[k] = np.append(seq[k], fs[k])

        if verbose:
            sys.stdout.write('\r' + (' ' * len(old_str)))
            sys.stdout.write('\r-- formatting sequence...({}%)\n'.format(100))
            sys.stdout.flush()

    for k in seq.keys():
        seq[k] = np.array(seq[k])

    return seq


def load_csv_to_sequence(filename, pivot, labels, columns, tag):
    import shutil

    print('cleaning temp folder...')
    shutil.rmtree('temp/')
    if not os.path.exists('temp/'):
        os.makedirs('temp/')

    csvarr = []
    n_lines = len(open(filename).readlines())
    inc = 0
    with open(filename, 'r', errors='replace') as f:
        f_lines = csv.reader(f)

        output_str = '-- loading {}...({}%)'.format(filename, 0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str
        i = 0
        header = None

        id = None
        for row in f_lines:
            line = np.array(row)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            if header is None:
                header = line
                continue

            if not hasattr(pivot, '__iter__'):
                pivot = [pivot]

            if len(csvarr) > 10000 and '_'.join([str(line[piv]) for piv in pivot]) != id:
                if id is not None:
                    ar = np.array(csvarr)

                    seq = format_data(ar, pivot, labels, columns, None, True, False)

                    flat = format_data(ar, pivot, [Network.NORMALIZED_INPUT], columns, None, False, False)

                    lb_name = str(inc)
                    inc += 1

                    # print(id)

                    np.save('temp/seq_k_' + lb_name + '.npy', seq['key'])
                    np.save('temp/seq_x_' + lb_name + '.npy', seq['x'])
                    np.save('temp/seq_y_' + lb_name + '.npy', seq['y'])

                    np.save('temp/flat_k_' + lb_name + '.npy', flat['key'])
                    np.save('temp/flat_x_' + lb_name + '.npy', flat['x'])
                    np.save('temp/flat_y_' + lb_name + '.npy', flat['y'])

                    csvarr = []

                id = '_'.join([str(line[piv]) for piv in pivot])

            csvarr.append(line)

            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()

        filenames = np.array(du.getfilenames('temp', '.npy'))

        data = dict()
        data['flat_k'] = filenames[np.argwhere(['flat' in f and '_k_' in f for f in filenames]).ravel()]
        data['flat_x'] = filenames[np.argwhere(['flat' in f and '_x_' in f for f in filenames]).ravel()]
        data['flat_y'] = filenames[np.argwhere(['flat' in f and '_y_' in f for f in filenames]).ravel()]

        data['seq_k'] = filenames[np.argwhere(['seq' in f and '_k_' in f for f in filenames]).ravel()]
        data['seq_x'] = filenames[np.argwhere(['seq' in f and '_x_' in f for f in filenames]).ravel()]
        data['seq_y'] = filenames[np.argwhere(['seq' in f and '_y_' in f for f in filenames]).ravel()]

        for d in data:
            output_str = '-- building {}...({}%)'.format(d, 0)
            sys.stdout.write(output_str)
            sys.stdout.flush()
            old_str = output_str

            ar = []
            i = 0
            n = len(data[d]) + 1
            for f in data[d]:
                x = np.load(f)
                if len(x.shape) == 3:
                    ar.append(x.reshape((-1, x.shape[-1])))
                else:
                    ar.extend(x)

                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- building {}...({}%)'.format(d, round((i / n) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str
                i += 1

            sys.stdout.write('\r' + (' ' * len(old_str)))
            sys.stdout.write('\r-- building {}...({}%)\n'.format(d, 100))
            sys.stdout.flush()

            print('Writing {}...'.format(d + '_' + tag + '.npy'))
            np.save(d + '_' + tag + '.npy', np.array(ar))


def generate_folds(key, folds=5):
    fold_ar = np.zeros(len(key))
    u = np.unique(key)
    # print(u)
    u = u[np.random.shuffle(list(range(len(u))))].reshape(-1)
    # print(u)
    f = []
    _ = [f.extend(np.array(list(range(folds)))[np.random.shuffle(list(range(folds)))].reshape(-1)) for _ in
         range(int(np.ceil(len(u) / folds)))]
    f = np.array(f)[:len(u)]
    # f = np.random.randint(0,folds,len(u))

    if len(u) < folds:
        warnings.warn('The number of unique values must be greater than the number of folds.')
        # print(key)
        # raise ValueError('The number of unique values must be greater than the number of folds.')

    for i in range(1, folds):
        val = u[np.argwhere(np.array(f) == i).ravel()]
        for j in val:
            fold_ar[np.argwhere(np.array(key) == j)] = i

    return np.array(fold_ar, dtype=np.int32)


def fold_by_key(key_ar, key=0, folds=5):
    kar = np.array(key_ar).reshape((len(key_ar), -1))

    return np.insert(kar, 0, np.array(generate_folds(
        np.array(['~'.join(np.array([i], dtype=str).ravel()) for i in kar[:, np.array([key]).ravel()]]).reshape((-1)),
        folds), dtype=str), axis=1)


def stratified_fold_by_key(key_ar, key=0, folds=5, strata=None):
    kar = np.array(key_ar).reshape((len(key_ar), -1))

    if strata is not None:
        if not hasattr(strata,'__iter__') or len(np.array(strata).shape) != 1:
            raise ValueError('strata must be a 1 dimensional list or array')

        if len(strata) != len(key_ar):
            raise ValueError('A stratum must be provided for each sample')
        strata = np.array(strata,dtype=str)
        f = np.zeros((len(key_ar)))
        for s in np.unique(strata):
            loc = np.argwhere(strata == str(s)).ravel()
            stratum = kar[loc,:]
            # print(loc)
            f[loc] = generate_folds(np.array(
                ['~'.join(np.array([i], dtype=str).ravel()) for i in stratum[:, np.array([key]).ravel()]]).reshape(
                (-1)), folds)
            # exit(1)

        return np.insert(kar, 0, np.array(f, dtype=str), axis=1)

    return np.insert(kar, 0, np.array(generate_folds(
        np.array(['~'.join(np.array([i], dtype=str).ravel()) for i in kar[:, np.array([key]).ravel()]]).reshape((-1)),
        folds), dtype=str), axis=1)


def stratify_multi_label(sequence_y, labels):
    desc = describe_multi_label(sequence_y)
    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    m = dict()
    for j in range(len(labels)):
        m[labels[j]] = {'sum': None, 'n': 0}
    for i in range(desc['n_samples']):
        for j in range(len(labels)):
            n = np.sum(1-np.isnan(sequence_y[i][labels[j]][:,0]))
            if n == 0:
                continue

            if m[labels[j]]['sum'] is None:
                m[labels[j]]['sum'] = np.nansum(sequence_y[i][labels[j]],axis=0)
            else:
                m[labels[j]]['sum'] = np.nansum(np.append(m[labels[j]]['sum'].reshape((1,-1)),
                                                          np.nansum(sequence_y[i][labels[j]],
                                                                    axis=0).reshape((-1,m[labels[j]]['sum'].shape[-1])),
                                                          axis=0), axis=0)

            m[labels[j]]['n'] += n

            # print(m[labels[j]]['sum'],m[labels[j]]['n'])

    for j in range(len(labels)):
        m[labels[j]]['mean'] = m[labels[j]]['sum']/m[labels[j]]['n']

    strata = []
    for i in range(desc['n_samples']):
        s = np.array([])
        for j in range(len(labels)):
            n = np.sum(1 - np.isnan(sequence_y[i][labels[j]][:, 0]))
            if n == 0:
                s = np.append(s,np.zeros_like(m[labels[j]]['mean']))
            else:
                # print(m[labels[j]]['mean'].)


                # print(np.apply_along_axis(np.greater_equal,0,{'x1': }))
                # print(sequence_y[i][labels[j]] >= m[labels[j]]['mean'])
                ind = np.argwhere(1-np.isnan(sequence_y[i][labels[j]][:,0])==1).ravel()
                s = np.append(s,np.nanmax(np.array(np.greater_equal(
                    sequence_y[i][labels[j]][ind], np.tile(
                        m[labels[j]]['mean'].ravel(),
                        len(ind)).reshape(-1,len(m[labels[j]]['mean']))),
                    dtype=np.float32), axis=0))
        strata.append(''.join(np.array(np.array(s,dtype=int),dtype=str)))
    return strata



def fill_input_multi_label(sequence_y, sequence_x, network):
    desc = describe_multi_label(sequence_y)

    for j in range(desc['n_label_sets']):
        if type(sequence_y[0][j][0][0]) is np.str_:
            print('Building label set {} from input...'.format(j))
            if sequence_y[0][j][0][0] == Network.RAW_INPUT:
                target = network.raw_input
            elif sequence_y[0][j][0][0] == Network.NORMALIZED_INPUT:
                target = network.norm_input
            else:
                raise ValueError('Invalid label values for label set {}.'.format(j))

            for i in range(desc['n_samples']):
                fd = dict()
                fd[network.layers[0]['z']] = [sequence_x[i]]
                sequence_y[i][j] = network.session.run(target, feed_dict=fd)[0]
    return sequence_y


def extract_from_multi_label(sequence_y, labels):
    desc = describe_multi_label(sequence_y)

    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    seq_y = []

    for i in range(desc['n_samples']):
        set = dict()
        for j in range(len(labels)):
            set[j] = sequence_y[i][labels[j]]
        seq_y.append(set)

    return np.array(seq_y)


def reverse_multi_label(sequence_y, labels):
    desc = describe_multi_label(sequence_y)

    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    seq_y = []

    for i in range(desc['n_samples']):
        set = dict()
        for j in range(len(labels)):
            set[j] = np.flip(sequence_y[i][labels[j]], axis=0)
        seq_y.append(set)

    return np.array(seq_y)


def merge_multi_label(sequence_y1, sequence_y2=None):
    desc1 = describe_multi_label(sequence_y1)

    if sequence_y2 is not None:
        desc2 = describe_multi_label(sequence_y2)

        if not desc1['n_samples'] == desc2['n_samples']:
            raise ValueError('The sequence labels must have an equal number of samples to merge.')

        seq_y = []

        for i in range(desc1['n_samples']):
            set = dict()
            for j in range(desc1['n_label_sets']):
                set[j] = sequence_y1[i][j]
            for j in range(desc2['n_label_sets']):
                set[desc1['n_label_sets'] + j] = sequence_y2[i][j]

            seq_y.append(set)
    else:
        seq_y = []

        for i in range(desc1['n_samples']):
            set = dict()
            lb_seq = None
            for j in range(desc1['n_label_sets']):
                if lb_seq is None:
                    lb_seq = sequence_y1[i][j]
                else:
                    lb_seq = np.append(lb_seq, sequence_y1[i][j], axis=1)

            set[0] = lb_seq

            seq_y.append(set)

    return np.array(seq_y)


def one_hot_multi_label(sequence_y, labels):
    desc = describe_multi_label(sequence_y)

    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    for i in labels:
        if not desc['n_labels'][i] == 1:
            raise ValueError('The supplied label {} has a length of {}. A one-hot encoding can '
                             'only be applied to labels of size 1.'.format(i, desc['n_labels'][i]))

    flat_lb = flatten_sequence(extract_from_multi_label(sequence_y, labels))

    cl = []
    for i in range(len(labels)):
        u = np.unique(flat_lb[:, i])
        u = u[np.argwhere(1 - np.isnan(u)).ravel()]
        cl.append(u)

    seq_y = []

    for i in range(desc['n_samples']):
        set = dict()
        for j in range(desc['n_label_sets']):
            if j in labels:
                seq_lb = []
                for k in sequence_y[i][labels[j]]:
                    # print(k)
                    seq_lb.append(
                        np.array([np.nan if np.any([np.isnan(m) for m in k]) else int(k[0] == cl[j][s]) for s in range(len(cl[j]))]))
                set[j] = seq_lb
            else:
                set[j] = sequence_y[i][j]
        seq_y.append(set)

    return np.array(seq_y)


def ravel_multi_label(sequence_y):
    desc = describe_multi_label(sequence_y)

    seq_y = []
    warn = True

    for i in range(desc['n_samples']):
        set = dict()
        ind = 0

        for j in range(desc['n_label_sets']):
            # print(sequence_y[i][j][0])
            if warn and type(sequence_y[i][j][0][0]) is np.str_:
                print('WARNING - label set {} corresponds to the input vector and cannot be raveled.'.format(j))
                warn = False
                continue
            for k in range(desc['n_labels'][j]):
                set[ind] = np.array(sequence_y[i][j][:, k]).reshape((-1, 1))
                ind += 1

        seq_y.append(set)

    return np.array(seq_y)


def find_in_multi_label(sequence_y, value):
    desc = describe_multi_label(sequence_y)

    f_ind = []

    for i in range(desc['n_samples']):
        for j in range(desc['n_label_sets']):
            for k in range(len(sequence_y[i][j])):
                for m in range(len(sequence_y[i][j][k])):
                    if sequence_y[i][j][k][m] == value:
                        f_ind.append([i,j,k,m])

    return f_ind


def find_and_replace_in_multi_label(sequence_y, find_value, replace_value, replace_all_classes=False):
    desc = describe_multi_label(sequence_y)

    f_ind = []

    for i in range(desc['n_samples']):
        for j in range(desc['n_label_sets']):
            for k in range(len(sequence_y[i][j])):
                if replace_all_classes:
                    if find_value in sequence_y[i][j][k]:
                        for m in range(len(sequence_y[i][j][k])):
                            sequence_y[i][j][k][m] = replace_value
                    else:
                        for m in range(len(sequence_y[i][j][k])):
                            if sequence_y[i][j][k][m] == find_value:
                                sequence_y[i][j][k][m] = replace_value

    return f_ind


def replace_in_multi_label(sequence_y, indices, replace_value):
    f_ind = []

    for i in indices:
        if not hasattr(i, '__iter__'):
            i = [i]

        if len(i) == 4:
            try:
                sequence_y[i[0]][i[1]][i[2]][i[3]] = replace_value
            except IndexError:
                raise IndexError(
                    'Supplied index array [{}] is out of range for the sequence object.'.format(','.join(i)))
        elif len(i) == 3:
            try:
                for j in range(len(sequence_y[i[0]][i[1]][i[2]])):
                    sequence_y[i[0]][i[1]][i[2]][j] = replace_value
            except IndexError:
                raise IndexError(
                    'Supplied index array [{}] is out of range for the sequence object.'.format(','.join(i)))
        elif len(i) == 2:
            try:
                for j in range(len(sequence_y[i[0]][i[1]])):
                    for k in range(len(sequence_y[i[0]][i[1]][j])):
                        sequence_y[i[0]][i[1]][j][k] = replace_value
            except IndexError:
                raise IndexError(
                    'Supplied index array [{}] is out of range for the sequence object.'.format(','.join(i)))
        elif len(i) == 1:
            try:
                for j in range(len(sequence_y[i[0]])):
                    for k in range(len(sequence_y[i[0]][j])):
                        for m in range(len(sequence_y[i[0]][j][k])):
                            sequence_y[i[0]][j][k][m] = replace_value
            except IndexError:
                raise IndexError(
                    'Supplied index array [{}] is out of range for the sequence object.'.format(','.join(i)))
        else:
            raise IndexError('The supplied index array is in an invalid format.')

    return f_ind


def use_last_multi_label(sequence_y, labels):
    desc = describe_multi_label(sequence_y)

    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    seq_y = []

    for i in range(desc['n_samples']):
        set = dict()
        for j in range(desc['n_label_sets']):
            if j in labels:
                seq_lb = np.full_like(sequence_y[i][j], np.nan)
                seq_lb[-1] = sequence_y[i][j][-1]
                set[j] = seq_lb
            else:
                set[j] = sequence_y[i][j]
        seq_y.append(set)

    return np.array(seq_y)


def offset_multi_label(sequence_y, labels, offset=1):
    desc = describe_multi_label(sequence_y)

    labels = np.array([labels]).ravel()

    if max(labels) >= desc['n_label_sets']:
        raise ValueError('The label index is larger than the number of label sets.')

    seq_y = []

    for i in range(desc['n_samples']):
        set = dict()
        for j in range(desc['n_label_sets']):
            if j in labels:
                seq_lb = np.full_like(sequence_y[i][j], np.nan)
                if offset >= 0:
                    for k in range(len(sequence_y[i][j])):
                        if k+offset < len(sequence_y[i][j]):
                            seq_lb[k] = sequence_y[i][j][k+offset]
                else:
                    for k in range(len(sequence_y[i][j]), 0, -1):
                        if (k-1)+offset >= 0:
                            seq_lb[k-1] = sequence_y[i][j][(k-1) + offset]

                set[j] = seq_lb
            else:
                set[j] = sequence_y[i][j]
        seq_y.append(set)

    return np.array(seq_y)


def sequence_levenshtein(sequence_x, index=None):
    import Levenshtein as lev
    # if no index is given, replace all text features

    flat_x = flatten_sequence(sequence_x[:200])
    if index is None:
        index = []
        for i in range(len(sequence_x[0][0])):
            if du.infer_if_string(flat_x[:,i]):
                index.append(i)

    if not hasattr(index, '__iter__'):
        index = [index]

    for i in range(len(sequence_x)):
        for k in index:
            sequence_x[i][0][k] = 0
            for j in range(1,len(sequence_x[i])):
                sequence_x[i][j][k] = lev.ratio(str(sequence_x[i][j][k]),str(sequence_x[i][j-1][k]))

    return sequence_x


def sequence_impute_missing(sequence_x, value=0):
    for i in range(len(sequence_x)):
        for k in range(len(sequence_x[i])):
            arr = np.array(sequence_x[i][k], dtype=str)
            if 'nan' in arr:
                sequence_x[i][k][np.argwhere(arr == 'nan').ravel()] = value
            elif '' in arr:
                sequence_x[i][k][np.argwhere(arr == '').ravel()] = value
            elif '.' in arr:
                sequence_x[i][k][np.argwhere(arr == '.').ravel()] = value
            elif 'NA' in arr:
                sequence_x[i][k][np.argwhere(arr == 'NA').ravel()] = value
    return sequence_x


def sequence_one_hot(sequence_x, index=None, class_list=None):
    # if no index is given, replace all text features
    flat_x = flatten_sequence(sequence_x[:200])
    if index is None:
        index = []
        for i in range(len(sequence_x[0][0])):
            if du.infer_if_string(flat_x[:,i]):
                index.append(i)

    if not hasattr(index, '__iter__'):
        index = [index]

    index.sort(reverse=True)
    for k in index:
        if class_list is None:
            class_list = np.unique(flat_x[:, k])
            # print(class_list)
            # exit(1)
        sequence_x = np.array(sequence_x).tolist()
        for i in range(len(sequence_x)):
            sequence_x[i] = np.array(du.one_hot(sequence_x[i], class_list, k, True))
    return np.array(sequence_x)


def describe_multi_label(sequence_y, print_description=False, print_descriptives=False):
    desc = dict()

    desc['n_samples'] = len(sequence_y)
    desc['n_label_sets'] = len(sequence_y[0])
    desc['n_labels'] = []

    for i in range(desc['n_label_sets']):
        desc['n_labels'].append(len(sequence_y[0][i][0]))

    if print_descriptives:
        fl_y = flatten_sequence(sequence_y)
    inc = 0
    if print_description:
        print("{:=<40}".format(''))
        print("{:=<40}".format('======  Label Description  '))
        print("{:=<40}".format(''))
        print("-- Number of Samples: {}".format(desc['n_samples']))
        print("-- Number of Label Sets: {}".format(desc['n_label_sets']))
        for i in range(desc['n_label_sets']):
            if print_descriptives and desc['n_labels'][i] < 6:
                m = np.nanmean(fl_y[:, inc:inc + desc['n_labels'][i]], axis=0)
                mstr = '[{:<.3f}'.format(m[0])
                for mi in range(1, len(m)):
                    mstr += ', {:<.3f}'.format(m[mi])
                mstr += ']'
            else:
                mstr = '[ ... ]'

            if print_descriptives:
                print("---- {}: {} Label{} :: {}".format(str(i + 1), desc['n_labels'][i],
                                                         '' if desc['n_labels'][i] == 1 else 's', mstr))
            else:
                print("---- {}: {} Label{}".format(str(i + 1), desc['n_labels'][i],
                                                   '' if desc['n_labels'][i] == 1 else 's'))
            inc += desc['n_labels'][i]

        print("{:=<40}\n".format(''))

    return desc


def split_file_with_pivot(filename, outfile, pivot, target_rows=2048, header=True, verbose=True):
    if not os.path.exists('temp/'):
        os.makedirs('temp/')
    tmp_filename = 'temp/tmp_data.csv'
    data = pd.read_csv(filename)
    _, headers = du.read_csv(filename, 2)

    sortby = headers[pivot].tolist()

    # sort by user, assignment, problem, and action and write the new data to file
    if verbose:
        print('-- preparing to split file...')
    data.sort_values(by=sortby).to_csv(tmp_filename, index=False)
    filename = tmp_filename
    # release the memory for the pandas dataframe, it is no longer needed
    data = None
    format_seq = False
    csvarr = []

    file_index = 0
    seq = None
    id = None
    n_lines = len(open(filename).readlines())
    created_files = []
    with open(filename, 'r', errors='replace') as f:
        f_lines = csv.reader(f)

        if verbose:
            output_str = '-- splitting file...({}%)'.format(0)
            sys.stdout.write(output_str)
            sys.stdout.flush()
            old_str = output_str
        i = 0
        for line in f_lines:
            if (header and i == 0) or len(line) == 0:
                i += 1
                continue
            elif i % target_rows == 0:
                format_seq = True

            line = np.array(line)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            if '~'.join(np.array(line)[pivot]) != id:
                if format_seq and id is not None:
                    ar = np.array(csvarr).reshape([-1, len(line)])

                    outfile_name = outfile+'_pt_' + str(file_index)
                    du.write_csv(ar, outfile_name, headers)
                    file_index += 1

                    created_files.append(outfile_name)

                    csvarr = []
                    format_seq = False

                id = '~'.join(np.array(line)[pivot])

            csvarr.append(line)

            if verbose and not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- splitting file...({}%)'.format(round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1

        if len(csvarr) > 0:
            ar = np.array(csvarr).reshape([-1, len(line)])
            outfile_name = outfile + '_pt_' + str(file_index)
            du.write_csv(ar, outfile_name, headers)
            file_index += 1
            created_files.append(outfile_name)

        if verbose:
            sys.stdout.write('\r' + (' ' * len(old_str)))
            sys.stdout.write('\r-- splitting file...({}%)\n'.format(100))
            sys.stdout.flush()

    return created_files


