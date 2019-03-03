import click
import os
import numpy as np
from sklearn.externals import joblib
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@click.group()
@click.pass_context
def main(ctx):
    with open('arch.yaml', "r") as stream:
        config = yaml.load(stream)
    ctx.obj['CONF'] = config
    pass

@main.command()
@click.option(
    '-d',
    '--device',
    required=True,
    type=click.Choice(['CPU', 'GPU']),
    help='Mention the device type GPU|CPU'
)
def active(device):
    """
    Get all the active devices in the machine
    """
    print(_get_available_dev(device))


@main.command()
@click.option(
    '-f',
    '--base-path',
    default='./data/',
    type=click.Path(exists=True),
    help="Path where the data is present in files equal to the class labels"
)
@click.option(
    '-c',
    '--clean',
    default=True,
    type=bool,
    help="If the cleaning of the data needs to be performed"
)
@click.option(
    '-s',
    '--shuffle',
    default=True,
    type=bool,
    help="If the dataset needs to be shuffled"
)
@click.option(
    '-ts',
    '--test-size',
    default=0.3,
    type=float,
    help="Fraction of the test set"
)
@click.pass_context
def buildVocabulary(ctx, base_path, clean, shuffle, test_size):
    """
    Build the vocabulary the convert the dataset into
    train and test where the data is represented into
    word to rank/id matrix.
    """
    max_vocab = ctx.obj['CONF']['arch']['data']['vocab_size']
    _buildVocabulary(base_path, clean, max_vocab, shuffle, test_size)

def _buildVocabulary(base_path, clean, max_vocab, shuffle, test_size):
    from preprocess import TextReader
    from sklearn.model_selection import train_test_split
    # TODO: This needs to be dynamic
    suffix = {'rt-polarity.pos': 1, 'rt-polarity.neg': 0}
    tr = TextReader(data_dir=base_path,
                    suffix_labels=suffix)
    print(f'Found datafiles with the following class labels {tr.data_files}')
    if tr.prepare_data(clean=clean, max_vocab=max_vocab):
        X, y = tr.get_ranked_features(shuffle=shuffle)
    print(f'Created training data of shape {X.shape}')
    print(f'Created training label of shape {y.shape}')
    if not os.path.exists(os.path.join(base_path, 'train')):
        os.mkdir(os.path.join(base_path, 'train'))

    if not os.path.exists(os.path.join(base_path, 'valid')):
        os.mkdir(os.path.join(base_path, 'valid'))
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42)
    np.save(os.path.join(base_path, 'train', 'X_train'), X_train)
    np.save(os.path.join(base_path, 'train', 'y_train'), y_train)

    np.save(os.path.join(base_path, 'valid', 'X_valid'), X_valid)
    np.save(os.path.join(base_path, 'valid', 'y_valid'), y_valid)

    print(f'Saved the train and test frames in {base_path}')


@main.command()
@click.option(
    '-p',
    '--path',
    required=True,
    type=click.Path(exists=True),
    help="Path of the ranked vocabulary file"
)
@click.pass_context
def buildWord2Vec(ctx, path):
    """
    Builds the word vector using pretraied vector
    for the ranked list of words mentioned.
    """
    from preprocess import get_embedding_vector
    if not os.path.isfile(os.path.join(path, 'ranks')):
        raise IOError('Ranked vocabulary file not found')
    config = ctx.obj['CONF']
    get_embedding_vector(config, os.path.join(path, 'ranks'))


@main.command()
@click.option(
    '-tp',
    '--train_path',
    required=True,
    type=click.Path(exists=True),
    help="Path for the training data set"
)
@click.option(
    '-vp',
    '--test_path',
    required=True,
    type=click.Path(exists=True),
    help="Path for the validation data set"
)
@click.option(
    '-f',
    '--fmt',
    default='npy',
    type=str,
    help="Default format of the dataset. Helpful to choose between sparse and dense"
)
@click.option(
    '-tn',
    '--train_name',
    default='X_train',
    type=str,
    help="Name for the trainig dataset"
)
@click.option(
    '-vn',
    '--validation_name',
    default='X_valid',
    type=str,
    help="Name for the validation dataset"
)
@click.option(
    '-tny',
    '--train_name_y',
    default='y_train',
    type=str,
    help="Name for the trainig labels"
)
@click.option(
    '-vny',
    '--validation_name_y',
    default='y_valid',
    type=str,
    help="Name for the validation labels"
)
@click.option(
    '-ld',
    '--logdir',
    required=True,
    type=click.Path(exists=True),
    help="Path for the model to store the logs"
)

@click.pass_context
def fit(ctx, train_path, test_path, fmt,
        train_name, validation_name, train_name_y,
        validation_name_y, logdir):
    """
    Build and fit the network mentioned by the config.yaml
    for the number of epochs with mentioned batch_size and
    at mentioned learning_rate with the optimizer mentioned.
    """
    _fit(ctx.obj['CONF'], train_path, test_path, fmt,
        train_name, validation_name, train_name_y,
        validation_name_y, logdir)

def _fit(config, train_path, test_path, fmt,
        train_name, validation_name, train_name_y,
        validation_name_y, logdir):
    from train import CNNText
    cnnText = CNNText(
        config=config,
        train_path=train_path,
        valid_path=test_path,
        fmt=fmt,
        train_name=train_name,
        validation_name=validation_name,
        train_name_y=train_name_y,
        validation_name_y=validation_name_y,
     )
#     print(learning_rate)

    cnnText.train(logdir=logdir)

#     print(cnnText)


def _get_available_dev(d):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == d]

# def _process_word_vectors(base_path, suffix, pretrained=False, **kwargs):
#     from preprocess import TextReader
#     import pandas as pd
#     tr = TextReader(data_dir=base_path,
#                     suffix_labels=suffix)
#     print(tr.data_files)
#     if tr.prepare_data(clean=True, max_vocab=15000):
#         X, y = tr.get_ranked_features()
#     word_vectors_df = None
#     if pretrained:
#         model = kwargs.get('model')
#         if model is None:
#             raise ValueError('Model can not be None')
#         wv = tr.get_embedding_vector(model)
#         word_vectors = {}
#         for word, vector in wv:
#             word_vectors[tr.get_rank(word)] = vector
#         word_vectors_df = pd.DataFrame.from_dict(word_vectors, orient='index')
#     return X, y, word_vectors_df

if __name__ == '__main__':
    main(obj={})
