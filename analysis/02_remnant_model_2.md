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
remnantdatafilename = projectdir + 'data/original/Study2 11 New Experiments/LogData_Remnant_Study2.csv'
rctdatafilename     = projectdir + 'data/original/Study2 11 New Experiments/LogData_Experimental_Study2.csv'
```

**Output files:**

```{code-cell} ipython3
outputdatafilename  = projectdir + 'data/processed/model_predictions2.csv'
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
key = [1,2,0]
iden = [4]
cov = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
lab = [[3]]
order = 7

seq = tfnet.format_sequence_from_file(remnantdatafilename,
                                          key,lab,cov,iden,order)
print('formatting feature columns...')
for i in range(len(seq['x'])):
    for j in range(len(seq['x'][i])):
        try:
            seq['x'][i][j] = np.array(seq['x'][i][j], dtype=np.float32)
        except TypeError:
            print('error...')
            print(seq['x'][i])
            exit(1)
seq['x'] = tfnet.sequence_impute_missing(seq['x'])

print('formatting output labels...')
seq['y'] = tfnet.use_last_multi_label(seq['y'], 0)

print('done!')
```

```{code-cell} ipython3
key = [1,2,0]
iden = [4]
cov = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
lab = [[3]]
order = 7

seqex = tfnet.format_sequence_from_file(rctdatafilename,
                                          key,lab,cov,iden,order)
print('formatting feature columns...')
for i in range(len(seqex['x'])):
    for j in range(len(seqex['x'][i])):
        try:
            seqex['x'][i][j] = np.array(seqex['x'][i][j], dtype=np.float32)
        except TypeError:
            print('error...')
            print(seqex['x'][i])
            exit(1)
seqex['x'] = tfnet.sequence_impute_missing(seqex['x'])

print('formatting output labels...')
seqex['y'] = tfnet.use_last_multi_label(seqex['y'], 0)

print('done!')
```

```{code-cell} ipython3
n_cov = len(seq['x'][0][0])

max_epochs = 200
use_validation = True
hidden = [50]
batch = [64]
layers = [1]
keep = [.5]
step = [1e-3]
perf = []
threshold = [0.0001]
optimizer = [Optimizer.ADAM]

tf.compat.v1.disable_eager_execution()

tfnet.describe_multi_label(seq['y'], True)
```

```{code-cell} ipython3
tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(1)
np.random.seed(1)

net = Network('study2_model').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
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
```

```{code-cell} ipython3
print('writing predictions to file...')
du.write_csv(tfnet.flatten_sequence(np.array([np.array([t[-1]]) for t in net.predict(seqex['x'])[0]]),
                                        seqex['key'].reshape((-1,3))), outputdatafilename,
             ['user_id','target_assignment_id','target_sequence_id','pcomplete'])
net.session.close()
shutil.rmtree('./temp')
print('done!')
```
