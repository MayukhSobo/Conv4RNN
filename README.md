# Conv4RNN
Sentence classification using Conv1d instead of LSTMs on Amazon Food Review Dataset

# Medium Blog [Here](https://medium.com/@theDataGeek/cnn-for-rnns-a-gentle-approach-to-use-cnns-for-nlp-53ab80768d43?source=friends_link&sk=96de5a35150fdf3aa2d0863b0faaa3b3)


# Command Line Options
```sh
$ python app.py --help

Usage: app.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  active           Get all the active devices in the machine
  buildvocabulary  Build the vocabulary the convert the dataset into train...
  buildword2vec    Builds the word vector using pretraied vector for the...
  fit              Build and fit the network mentioned by the config.yaml...
```

#### To check help for sub-commands, 
```sh
$ python app.py fit --help

Usage: app.py fit [OPTIONS]

  Build and fit the network mentioned by the config.yaml for the number of
  epochs with mentioned batch_size and at mentioned learning_rate with the
  optimizer mentioned.

Options:
  -tp, --train_path PATH          Path for the training data set  [required]
  -vp, --test_path PATH           Path for the validation data set  [required]
  -f, --fmt TEXT                  Default format of the dataset. Helpful to
                                  choose between sparse and dense
  -tn, --train_name TEXT          Name for the trainig dataset
  -vn, --validation_name TEXT     Name for the validation dataset
  -tny, --train_name_y TEXT       Name for the trainig labels
  -vny, --validation_name_y TEXT  Name for the validation labels
  -ld, --logdir PATH              Path for the model to store the logs
                                  [required]
  --help                          Show this message and exit.
```

### Control the network architecture from `arch.yaml` file 

```
---
name: model_architecture_config
date: 2019-03-03
paper: https://arxiv.org/pdf/1408.5882.pdf

arch:
  cnn:
    units:         100         # Number of convNets in each layers
    min_filter:    3           # Minimum size of the filter
    max_filter:    5           # Maximum size of the filter
    kernel_l2_reg: 0           # L2 regularisation for each kernel in conv layer

  layers:
    dropout:       0.5         # Dropout rate in the layer

  fit:
    device:        cuda        # Device to build the model
    epochs:        6           # Number of epochs
    batch_size:    50          # Number of datapoint in the batch
    optimizer:     adam        # Optimizer to use
    backend:       tensorflow  # Neural Network Framework used
    learning_rate: 0.01        # The initial learning rate

  data:
    root_path:     ./data/     # Root data path
    vocab_size:    15000       # Vocabulary size
    embedding:     300         # Size of the embedding output

  initialisation:
    embedding:     False        # Use pretrained embedding for initialisation
    bin_path:      ./data/GoogleNews-vectors-negative300.bin
```

### To use the pretrained Word2Vec to initialize the embedding layer

- Make the `embedding` is True.
- Make sure that in `bin_path` the word2vec binary is present.

### To change the device 

- Make `device = cpu` or `device = cuda` for GPU.


## Steps to the run

#### Build the vocabulary
```sh
$ python app.py buildvocabulary -f ./data -c True -s True -ts 0.3
```

#### Build the word2vec embedding
```sh
$ python app.py buildword2vec -p ./data
```

#### Finally Train the model
```
$ mkdir TFLOGS
$ python app.py fit -tp ./data/train -vp ./data/valid -ld ./TFLOGS
```

#### To see the tensorboard
```
$ tensorboard --logdir=./TFLOGS
```

and then open `https://localhost:6006` in your browser


## How to process the data
- Create two seperate files with names one with name `rt-polarity.pos` with all the positive reviews each in one new line and another file named `rt-polarity.neg` with all the negative reviews each in one new line.
- Put all these files into a directory say `data`
- Change the `root_path` in the `arch.yaml` file


## Reflections

- I need to further work on the time based splitting feature
- Create a low RAM consuming data iterator which doesn't load the complete data but loads only the smaller batch
- Implement a PyTorch version for the same code.