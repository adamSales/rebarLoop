---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.1
kernelspec:
  display_name: virtenv
  language: python
  name: virtenv
---

# Setup

+++

## Files

Project base directory:

```{code-cell} ipython3
projectdir = '/home/rep/'
```

**Input files:**

```{code-cell} ipython3
remnantdatafilename = projectdir + 'data/original/Study1 Original 22 Experiments/LogData_Remnant_Study1.csv'
rctdatafilename     = projectdir + 'data/original/Study1 Original 22 Experiments/LogData_Experimental_Study1.csv'
```

**Output files:**

```{code-cell} ipython3
outputdatafilename  = projectdir + 'data/processed/model_predictions1.csv'
```

## Packages and scripts

Installed packages:

```{code-cell} ipython3
import numpy as np
import tensorflow as tf
import shutil
```

Scripts (in the `analysis/scripts` directory)

```{code-cell} ipython3
import scripts.datautility as du
import scripts.evaluationutility as eu
import scripts.tf_network2 as tfnet
from   scripts.tf_network2 import Network, Cost, Normalization, Optimizer
```

## Load data

```{code-cell} ipython3
data, headers = du.read_csv(remnantdatafilename)
du.print_descriptives(data, headers)

data, headers = du.read_csv(rctdatafilename)
du.print_descriptives(data, headers)
```

# Build and train model

```{code-cell} ipython3
seq = dict()

key = [1]
label = [[13], [15], [12]]
cov = [7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
iden = [0, 2, 3, 6]
sortby = [4]

seq = tfnet.format_sequence_from_file(remnantdatafilename,key,label,cov,iden,sortby)
print('formatting identifiers...')
seq['key'] = tfnet.fold_by_key(seq['key'], -1, 10)
print('formatting output labels...')
seq['y'] = tfnet.offset_multi_label(seq['y'],2,-1)
print('formatting feature columns...')
seq['x'] = tfnet.sequence_impute_missing(seq['x'])
print('done!')
```

```{code-cell} ipython3
seqt = dict()

key = [33, 1]
label = [[13], [15], [12]]
cov = [7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
iden = [38, 34, 2, 3]
sortby = [4]

tf.compat.v1.disable_eager_execution()

seqt = tfnet.format_sequence_from_file(rctdatafilename, key, label, cov, iden, sortby)
print('formatting output labels...')
seqt['y'] = tfnet.offset_multi_label(seqt['y'], 2, -1)
seqt['y'] = tfnet.extract_from_multi_label(seqt['y'], [0, 1])
print('formatting feature columns...')
seqt['x'] = tfnet.sequence_impute_missing(seqt['x'])
print('done!')
```

```{code-cell} ipython3
max_epochs = 200
hidden = 50
batch = 64
keep = .5
step = 5e-4
threshold = .001
optimizer = Optimizer.ADAM

tf.compat.v1.disable_eager_execution()

n_cov = len(seq['x'][0][0])
seq['y'] = tfnet.extract_from_multi_label(seq['y'], [0, 1])
desc = tfnet.describe_multi_label(seq['y'], True)

exp = np.unique(seqt['key'].reshape((-1, len(key)))[:, 0])
```

```{code-cell} ipython3
tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(1)
np.random.seed(1)

# build the model
net = Network('study1_model').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
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

# Save
net.save_model_weights() 
```

```{code-cell} ipython3
for e in range(len(exp)):
    print('formatting model predictions for experiment {}'.format(exp[e]))
    test = np.argwhere(seqt['key'].reshape((-1, len(key)))[:, 0] == exp[e]).ravel()

    pred = net.predict(x=seqt['x'][test], batch=1024)

    fold_pred = tfnet.flatten_sequence(seqt['y'][test], key=seqt['key'].reshape((-1, len(key)))[test], identifier=seqt['iden'][test])

    for p in range(len(pred)):
        tr_max = 1
        tr_min = 0
        upperbound = 1 if p == 0 else 0.333

        # and apply it to the experimental predictions
        fold_pred = np.append(fold_pred,
                              ((tfnet.flatten_sequence(pred[p]) - tr_min) / (tr_max - tr_min)) * upperbound,
                              axis=1)

    hdr = np.array(headers)[iden]
    hdr = np.append(hdr, np.array(headers)[key])

    hdr = np.append(hdr, ['complete', 'inv_mastery', 'p_complete', 'p_inv_mastery'])
    du.write_csv(fold_pred, outputdatafilename, hdr if e == 0 else None, append=e > 0)
net.session.close()
shutil.rmtree('./temp')
print('done!')
```
