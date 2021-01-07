import datautility as du
import numpy as np
import tf_network as tfnet
import evaluationutility as eu
from tf_network import Network, Cost, Normalization, Optimizer

import tensorflow as tf


def train_apply_rebar_model(outputlabel='rebar_ft'):
    data, headers = du.read_csv('rebar_ft_training.csv', 100)
    du.print_descriptives(data, headers)

    data, headers = du.read_csv('rebar_ft_test_clean.csv', 100)
    du.print_descriptives(data, headers)
    # exit(1)
    # uncomment this block to regenerate data from csv files

    key = [1]
    label = [[13], [15], [12]]
    cov = [7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    iden = [0, 2, 3, 6]
    sortby = [4]
    #
    # seq = tfnet.format_sequence_from_file('rebar_ft_training.csv',key,label,cov,iden,sortby)
    # seq['key'] = tfnet.fold_by_key(seq['key'], -1, 10)
    # seq['y'] = tfnet.offset_multi_label(seq['y'],2,-1)
    # np.save('seq_k_' + outputlabel + '.npy', seq['key'])
    # np.save('seq_x_' + outputlabel + '.npy', seq['x'])
    # np.save('seq_y_' + outputlabel + '.npy', seq['y'])
    # np.save('seq_i_' + outputlabel + '.npy', seq['iden'])

    key = [33, 1]
    label = [[13], [15], [12]]
    cov = [7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    iden = [38, 34, 2, 3]
    sortby = [4]

    # seq = tfnet.format_sequence_from_file('rebar_ft_test_clean.csv', key, label, cov, iden, sortby)
    # seq['y'] = tfnet.offset_multi_label(seq['y'], 2, -1)
    # np.save('seq_k_' + outputlabel + '_t.npy', seq['key'])
    # np.save('seq_x_' + outputlabel + '_t.npy', seq['x'])
    # np.save('seq_y_' + outputlabel + '_t.npy', seq['y'])
    # np.save('seq_i_' + outputlabel + '_t.npy', seq['iden'])
    # exit(1)

    max_epochs = 200
    hidden = 50
    batch = 64
    keep = .5
    step = 5e-4
    threshold = .001
    optimizer = Optimizer.ADAM

    seq = dict()

    seq['x'] = np.load('seq_x_' + outputlabel + '.npy')
    seq['y'] = np.load('seq_y_' + outputlabel + '.npy')
    seq['key'] = np.load('seq_k_' + outputlabel + '.npy')
    seq['iden'] = np.load('seq_i_' + outputlabel + '.npy')

    seq['x'] = tfnet.sequence_impute_missing(seq['x'])

    n_cov = len(seq['x'][0][0])
    seq['y'] = tfnet.extract_from_multi_label(seq['y'], [0, 1])
    desc = tfnet.describe_multi_label(seq['y'], True)

    exp = np.unique(np.load('seq_k_' + outputlabel + '_t.npy').reshape((-1, len(key)))[:, 0])

    seqt = dict()
    seqt['x'] = np.load('seq_x_' + outputlabel + '_t.npy')
    seqt['y'] = np.load('seq_y_' + outputlabel + '_t.npy')
    seqt['key'] = np.load('seq_k_' + outputlabel + '_t.npy').reshape((-1, len(key)))
    seqt['iden'] = np.load('seq_i_' + outputlabel + '_t.npy')

    seqt['x'] = tfnet.sequence_impute_missing(seqt['x'])
    seqt['y'] = tfnet.extract_from_multi_label(seqt['y'], [0, 1])

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    # build the model
    net = Network('rebar').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
    net.add_lstm_layer(hidden, peepholes=True, activation=tf.nn.leaky_relu)
    net.begin_multi_output(cost_methods=[Cost.BINARY_CROSS_ENTROPY,
                                         Cost.RMSE])
    net.add_dropout_layer(1, keep=keep, activation=tf.nn.sigmoid)
    net.add_dropout_layer(1, keep=keep, activation=tf.identity)
    net.end_multi_output()

    # set defaults
    net.set_default_cost_method(Cost.BINARY_CROSS_ENTROPY)
    net.set_optimizer(optimizer)

    # train the model and save the trained weights
    net.train(x=seq['x'],
              y=seq['y'],
              step=step,
              use_validation=True,
              max_epochs=max_epochs, threshold=threshold, batch=batch)
    # net.save_model_weights()
    #
    # net.build()
    # net.restore_model_weights()
    for e in range(len(exp)):
        print(exp[e])
        test = np.argwhere(seqt['key'][:, 0] == exp[e]).ravel()

        pred = net.predict(x=seqt['x'][test], batch=1024)

        fold_pred = tfnet.flatten_sequence(seqt['y'][test], key=seqt['key'][test], identifier=seqt['iden'][test])

        for p in range(len(pred)):
            tr_max = 1  # np.nanmax(tr, axis=0)
            tr_min = 0  # np.nanmin(tr, axis=0)

            upperbound = 1 if p == 0 else 0.333

            # and apply it to the experimental predictions
            fold_pred = np.append(fold_pred,
                                  ((tfnet.flatten_sequence(pred[p]) - tr_min) / (tr_max - tr_min)) * upperbound,
                                  axis=1)

        hdr = np.array(headers)[iden]
        hdr = np.append(hdr, np.array(headers)[key])

        hdr = np.append(hdr, ['complete', 'inv_mastery', 'p_complete', 'p_inv_mastery'])
        du.write_csv(fold_pred, 'rebar_ft_predictions.csv', hdr if e == 0 else None, append=e > 0)
    net.session.close()


if __name__ == '__main__':
    train_apply_rebar_model()
