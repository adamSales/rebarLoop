import numpy as np
import datautility as du
import evaluationutility as eu
import tf_network as tfnet
from tf_network import Optimizer, Cost, Network, Normalization
import tensorflow as tf
import os

RESOURCE_DIR = 'resources/REBAR_data/'
TEMP_DIR = RESOURCE_DIR + 'tmp/'

REMNANT_FILENAME = 'remnant/remnant_data_deep_learning_model.csv'

if not os.path.exists('resources/'):
    os.makedirs('resources/')
if not os.path.exists('resources/{}'.format('REBAR_data')):
    os.makedirs('resources/{}'.format('REBAR_data'))

if not os.path.exists('resources/REBAR_data/{}'.format('remnant')):
    os.makedirs('resources/REBAR_data/{}'.format('remnant'))

if not os.path.exists('resources/REBAR_data/{}'.format('experimental')):
    os.makedirs('resources/REBAR_data/{}'.format('experimental'))

if not os.path.exists('resources/REBAR_data/{}'.format('model_predictions')):
    os.makedirs('resources/REBAR_data/{}'.format('model_predictions'))


def train_rebar_model(model_name='rebar'):
    data, headers = du.read_csv(RESOURCE_DIR + REMNANT_FILENAME, max_rows=100)
    du.print_descriptives(data, headers)

    outputlabel = 'rebar'

    key = [1, 2, 0]
    iden = [4]
    cov = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    lab = [[3]]
    order = 7

    seq = tfnet.format_sequence_from_file(RESOURCE_DIR + REMNANT_FILENAME,
                                          key, lab, cov, iden, order)
    np.save('seq_k_' + outputlabel + '.npy', seq['key'])
    np.save('seq_x_' + outputlabel + '.npy', seq['x'])
    np.save('seq_y_' + outputlabel + '.npy', seq['y'])
    np.save('seq_i_' + outputlabel + '.npy', seq['iden'])

    max_epochs = 200
    hidden = [50]
    batch = [64]
    keep = [.5]
    step = [1e-3]
    threshold = [0.0001]
    optimizer = [Optimizer.ADAM]

    seq = dict()

    seq['x'] = np.load('seq_x_' + outputlabel + '.npy', allow_pickle=True)
    seq['y'] = np.load('seq_y_' + outputlabel + '.npy', allow_pickle=True)
    seq['key'] = np.load('seq_k_' + outputlabel + '.npy', allow_pickle=True).reshape((-1, 3))
    seq['iden'] = np.load('seq_i_' + outputlabel + '.npy', allow_pickle=True)

    for i in range(len(seq['x'])):
        for j in range(len(seq['x'][i])):
            seq['x'][i][j] = np.array(seq['x'][i][j], dtype=np.float32)

    seq['x'] = tfnet.sequence_impute_missing(seq['x'])
    np.save('seq_x_' + outputlabel + '.npy', seq['x'])
    n_cov = len(seq['x'][0][0])

    seq['y'] = tfnet.use_last_multi_label(seq['y'], 0)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    net = Network(model_name).add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
    net.add_lstm_layer(hidden[0], peepholes=True)

    net.begin_multi_output(cost_methods=[Cost.BINARY_CROSS_ENTROPY])
    net.add_dropout_layer(1, keep=keep[0], activation=tf.nn.sigmoid)
    net.end_multi_output()

    net.set_default_cost_method(Cost.CROSS_ENTROPY)
    net.set_optimizer(optimizer[0])

    net.train(x=seq['x'],
              y=seq['y'],
              step=step[0],
              use_validation=True,
              max_epochs=max_epochs, threshold=threshold[0], batch=batch[0])

    net.save_model_weights()


def evaluate_rebar_model():
    data, headers = du.read_csv(RESOURCE_DIR + REMNANT_FILENAME, max_rows=100)
    du.print_descriptives(data, headers)

    outputlabel = 'rebar'

    key = [1, 2, 0]
    iden = [4]
    cov = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    lab = [[3]]
    order = 7

    seq = tfnet.format_sequence_from_file(RESOURCE_DIR + REMNANT_FILENAME,
                                          key, lab, cov, iden, order)
    np.save('seq_k_' + outputlabel + '.npy', seq['key'])
    np.save('seq_x_' + outputlabel + '.npy', seq['x'])
    np.save('seq_y_' + outputlabel + '.npy', seq['y'])
    np.save('seq_i_' + outputlabel + '.npy', seq['iden'])

    print('replacing missing values...')
    for i in range(len(seq['x'])):
        for j in range(len(seq['x'][i])):
            seq['x'][i][j] = np.array(seq['x'][i][j], dtype=np.float32)

    seq['x'] = tfnet.sequence_impute_missing(seq['x'])
    np.save('seq_x_' + outputlabel + '.npy', seq['x'])

    seq['y'] = tfnet.use_last_multi_label(seq['y'], 0)

    print('folding dataset...')
    seq['key'] = tfnet.stratified_fold_by_key(seq['key'].reshape((-1, 3)),[0,2],10)
    np.save('seq_k_' + outputlabel + '.npy', seq['key'])

    n_cov = len(seq['x'][0][0])
    seq = dict()

    seq['x'] = np.load('seq_x_' + outputlabel + '.npy', allow_pickle=True)
    seq['y'] = np.load('seq_y_' + outputlabel + '.npy', allow_pickle=True)
    seq['key'] = np.load('seq_k_' + outputlabel + '.npy', allow_pickle=True).reshape((-1, 4))
    seq['iden'] = np.load('seq_i_' + outputlabel + '.npy', allow_pickle=True)

    max_epochs = 200
    hidden = [50]
    batch = [64]
    keep = [.5]
    step = [1e-3]
    threshold = [0.0001]
    optimizer = [Optimizer.ADAM]

    fold_auc = []

    for i in np.unique(seq['key'][:,0]):
        training = np.argwhere(np.array(seq['key'][:,0]) != i).ravel()
        test = np.argwhere(np.array(seq['key'][:,0]) == i).ravel()

        tf.reset_default_graph()
        tf.set_random_seed(1)
        np.random.seed(1)

        net = Network('rebar_eval').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
        net.add_lstm_layer(hidden[0], peepholes=True)

        net.begin_multi_output(cost_methods=[Cost.BINARY_CROSS_ENTROPY])
        net.add_dropout_layer(1, keep=keep[0], activation=tf.nn.sigmoid)
        net.end_multi_output()

        net.set_default_cost_method(Cost.CROSS_ENTROPY)
        net.set_optimizer(optimizer[0])

        net.train(x=seq['x'][training],
                  y=seq['y'][training],
                  step=step[0],
                  use_validation=True,
                  max_epochs=max_epochs, threshold=threshold[0], batch=batch[0])

        p = np.array(tfnet.flatten_sequence(net.predict(seq['x'][test])[0]),dtype=np.float32)
        a = np.array(tfnet.flatten_sequence(seq['y'][test]),dtype=np.float32)

        fold_auc.append(eu.auc(a,p))
        print(fold_auc[-1])

    print(np.nanmean(fold_auc))




def apply_rebar_model(filename, model_name='rebar', directory=RESOURCE_DIR,
                      outfile=RESOURCE_DIR + 'model_predictions/rebar_predictions.csv'):
    data, headers = du.read_csv(directory + filename, max_rows=100)
    du.print_descriptives(data, headers)

    key = [1, 2, 0]
    iden = [4]
    cov = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    lab = [[3]]
    order = 7

    seq = tfnet.format_sequence_from_file(directory + filename,
                                          key, lab, cov, iden, order)

    hidden = [50]
    keep = [.5]
    optimizer = [Optimizer.ADAM]

    for i in range(len(seq['x'])):
        for j in range(len(seq['x'][i])):
            seq['x'][i][j] = np.array(seq['x'][i][j], dtype=np.float32)

    seq['x'] = tfnet.sequence_impute_missing(seq['x'])
    n_cov = len(seq['x'][0][0])

    seq['y'] = tfnet.use_last_multi_label(seq['y'], 0)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    net = Network('rebar').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
    net.add_lstm_layer(hidden[0], peepholes=True)

    net.begin_multi_output(cost_methods=[Cost.BINARY_CROSS_ENTROPY])
    net.add_dropout_layer(1, keep=keep[0], activation=tf.nn.sigmoid)
    net.end_multi_output()

    net.set_default_cost_method(Cost.CROSS_ENTROPY)
    net.set_optimizer(optimizer[0])

    net.build()
    net.restore_model_weights()

    du.write_csv(tfnet.flatten_sequence(np.array([np.array([t[-1]]) for t in net.predict(seq['x'])[0]]),
                                        seq['key']), outfile,
                 ['user_id', 'target_assignment_id', 'target_sequence_id', 'pcomplete'])

    return outfile