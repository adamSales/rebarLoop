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

    for i in cov:
        print(headers[i])
    exit(1)
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

    auc = {'complete': [], 'inv_mastery': []}
    rmse = {'complete': [], 'inv_mastery': []}


    for i in range(10):
        training = np.argwhere(np.array(seq['key'][:,0],dtype=np.float32) != i).ravel()
        testing = np.argwhere(np.array(seq['key'][:,0],dtype=np.float32) == i).ravel()

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
        net.train(x=seq['x'][training],
                  y=seq['y'][training],
                  step=step,
                  use_validation=True,
                  max_epochs=max_epochs, threshold=threshold, batch=batch)

        pred = net.predict(x=seq['x'][testing], batch=1024)

        p_complete = tfnet.flatten_sequence(pred[0])[:,0]
        complete = tfnet.flatten_sequence(tfnet.extract_from_multi_label(seq['y'][testing],0))[:,0]

        p_inv_mastery = tfnet.flatten_sequence(pred[1])[:, 0]
        inv_mastery = tfnet.flatten_sequence(tfnet.extract_from_multi_label(seq['y'][testing], 1))[:, 0]

        auc['complete'].append(eu.auc(complete,p_complete))

        rmse['complete'].append(eu.rmse(complete, p_complete))
        rmse['inv_mastery'].append(eu.rmse(inv_mastery, p_inv_mastery))

        print('Fold AUC (complete): {:<.3f}'.format(auc['complete'][-1]))
        print('Fold RMSE (complete): {:<.3f}'.format(rmse['complete'][-1]))
        print('Fold RMSE (inv mastery): {:<.3f}'.format(rmse['inv_mastery'][-1]))

        net.session.close()

    print('Average AUC (complete): {:<.3f}'.format(np.mean(auc['complete'])))
    print('Average RMSE (complete): {:<.3f}'.format(np.mean(rmse['complete'])))
    print('Average RMSE (inv mastery): {:<.3f}'.format(np.mean(rmse['inv_mastery'])))



if __name__ == '__main__':
    train_apply_rebar_model()
