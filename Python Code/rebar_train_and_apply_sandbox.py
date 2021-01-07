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

    # seq = tfnet.format_sequence_from_file('rebar_ft_test_tr.csv', key, label, cov, iden, sortby)
    # seq['y'] = tfnet.offset_multi_label(seq['y'], 2, -1)
    # np.save('seq_k_' + outputlabel + '_tr.npy', seq['key'])
    # np.save('seq_x_' + outputlabel + '_tr.npy', seq['x'])
    # np.save('seq_y_' + outputlabel + '_tr.npy', seq['y'])
    # np.save('seq_i_' + outputlabel + '_tr.npy', seq['iden'])
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

    # seq = tfnet.sequence_truncate(seq, 3)

    n_cov = len(seq['x'][0][0])

    # seq['y'] = tfnet.find_and_replace_in_multi_label(tfnet.extract_from_multi_label(seq['y'], [0]),1,np.nan)
    # y = tfnet.extract_from_multi_label(seq['y'], [0])
    # seq['y'] = tfnet.merge_multi_label(tfnet.extract_from_multi_label(np.load('seq_y_' + outputlabel + '.npy'), [0]),
    # #                                    tfnet.find_and_replace_in_multi_label(
    # #                                        tfnet.extract_from_multi_label(np.load('seq_y_' + outputlabel + '.npy'), [0]), 1, np.nan,
    # #                                        replace_all_classes=True))

    # print(seq['y'][:10])
    # exit(1)


    # seq['y'] = tfnet.merge_multi_label(tfnet.extract_from_multi_label(seq['y'], [0, 1]),
    #                                    tfnet.extract_from_multi_label(seq['y'], [0]))
    seq['y'] = tfnet.extract_from_multi_label(seq['y'],[0,1])
    # seq['y'] = tfnet.extract_from_multi_label(seq['y'], [0])

    desc = tfnet.describe_multi_label(seq['y'], True)

    seqr = dict()
    seqr['x'] = np.load('seq_x_' + outputlabel + '_tr.npy')
    seqr['y'] = np.load('seq_y_' + outputlabel + '_tr.npy')
    seqr['key'] = np.load('seq_k_' + outputlabel + '_tr.npy').reshape((-1, len(key)))
    exp = np.unique(np.load('seq_k_' + outputlabel + '_t.npy').reshape((-1, len(key)))[:, 0])

    # print(seq['key'].shape)
    seqr['iden'] = np.load('seq_i_' + outputlabel + '_tr.npy')

    seqr['x'] = tfnet.sequence_impute_missing(seqr['x'])
    seqr['y'] = tfnet.extract_from_multi_label(seqr['y'], [0, 1])
    # seqr['y'] = tfnet.merge_multi_label(tfnet.extract_from_multi_label(seqr['y'], [0, 1]),
    #                                    tfnet.extract_from_multi_label(seqr['y'], [0]))

    seqt = dict()
    seqt['x'] = np.load('seq_x_' + outputlabel + '_t.npy')
    seqt['y'] = np.load('seq_y_' + outputlabel + '_t.npy')
    seqt['key'] = np.load('seq_k_' + outputlabel + '_t.npy').reshape((-1, len(key)))

    # print(seq['key'].shape)
    seqt['iden'] = np.load('seq_i_' + outputlabel + '_t.npy')

    seqt['x'] = tfnet.sequence_impute_missing(seqt['x'])
    seqt['y'] = tfnet.extract_from_multi_label(seqt['y'], [0, 1])
    # seqt['y'] = tfnet.merge_multi_label(tfnet.extract_from_multi_label(seqt['y'], [0, 1]),
    #                                    tfnet.extract_from_multi_label(seqt['y'], [0]))




    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    # build the model
    net = Network('rebar_fts').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
    # net.add_lstm_layer(hidden, peepholes=True)
    # net.add_lstm_layer(hidden, peepholes=False)
    net.add_lstm_layer(hidden, peepholes=True, activation=tf.nn.leaky_relu)
    net.begin_multi_output(cost_methods=[Cost.BINARY_CROSS_ENTROPY,
                                         Cost.RMSE])
    net.add_dropout_layer(1, keep=keep, activation=tf.nn.sigmoid)
    net.add_dropout_layer(1, keep=keep, activation=tf.identity)
    # net.add_dropout_layer(1, keep=keep, activation=tf.nn.sigmoid)
    net.end_multi_output()

    # set defaults
    net.set_default_cost_method(Cost.BINARY_CROSS_ENTROPY)
    net.set_optimizer(optimizer)

    # train the model and save the trained weights
    # net.train(x=seq['x'],
    #           y=seq['y'],
    #           step=step,
    #           use_validation=True,
    #           max_epochs=max_epochs, threshold=threshold, batch=batch)
    # net.save_model_weights()
    #
    net.build()
    net.restore_model_weights()
    for e in range(len(exp)):
        print(exp[e])

        training = np.argwhere(seqr['key'][:, 0] == exp[e]).ravel()
        test = np.argwhere(seqt['key'][:, 0] == exp[e]).ravel()


        # print(exp)

        # exit(1)


        net.restore_model_weights()



        # print(len(training))
        #
        if len(training) > 0:
            step = 1e-6
            max_epochs = 20
            batch = 64

            net.train(x=seqr['x'][training],
                      y=seqr['y'][training],
                      step=step,
                      use_validation=True,
                      max_epochs=max_epochs, threshold=threshold, batch=batch)

        tr_pred = net.predict(x=seq['x'], batch=1024)

        # the model is trained... now apply to the experimental set

        # seq['y'] = tfnet.extract_from_multi_label(seq['y'], [0])

        # seq = tfnet.sequence_truncate(seq, 3)

        pred = net.predict(x=seqt['x'][test], batch=1024)

        fold_pred = tfnet.flatten_sequence(seqt['y'][test], key=seqt['key'][test], identifier=seqt['iden'][test])

        # for each set of outputs, use min/max scaling from the training set to reduce majority class bias
        for p in range(len(pred)):
            # get prediction min/max from the training set...
            tr = tfnet.flatten_sequence(tr_pred[p])
            tr_max = 1#np.nanmax(tr, axis=0)
            tr_min = 0#np.nanmin(tr, axis=0)

            upperbound = 1 if p == 0 else 0.333

            # and apply it to the experimental predictions
            fold_pred = np.append(fold_pred,
                                  ((tfnet.flatten_sequence(pred[p]) - tr_min) / (
                                      tr_max - tr_min))*upperbound,
                                  axis=1)

        hdr = np.array(headers)[iden]
        hdr = np.append(hdr, np.array(headers)[key])
        # hdr = np.append(hdr, ['complete', 'p_complete'])
        hdr = np.append(hdr, ['complete','inv_mastery','p_complete','p_inv_mastery'])
        # hdr = np.append(hdr, ['complete', 'inv_mastery','prior_complete', 'p_complete', 'p_inv_mastery','p_prior_complete'])
        du.write_csv(fold_pred, 'rebar_ft_predictions_n2.csv', hdr if e == 0 else None, append=e > 0)
        # 2 is with the extra training min max 0 1
        # 3 is without extra training min max 0 1
        # 4 is with extra training min max tr
        # 5 is without extra training min max tr
    net.session.close()


if __name__ == '__main__':
    train_apply_rebar_model()
