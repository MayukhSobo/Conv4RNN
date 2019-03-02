import click
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@click.group()
def main():
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
    '-v',
    '--max-vocab',
    required=True,
    type=int,
    help="If the cleaning of the data needs to be performed"
)
@click.option(
    '-s',
    '--shuffle',
    default=True,
    type=bool,
    help="If the dataset needs to be shuffled"
)
def buildVocabulary(base_path, clean, max_vocab, shuffle):
    """
    Build the vocabulary the convert the dataset into
    train and test where the data is represented into
    word to rank/id matrix.
    """
    from preprocess import TextReader
    import pandas as pd
    # TODO: This needs to be dynamic
    suffix = {'rt-polarity.pos': 1, 'rt-polarity.neg': 0}
    tr = TextReader(data_dir=base_path, 
                    suffix_labels=suffix)
    print(f'Found datafiles with the following class labels {tr.data_files}')
    if tr.prepare_data(clean=clean, max_vocab=max_vocab):
        X, y = tr.get_ranked_features(shuffle=shuffle)
    print(f'Created training data of shape {X.shape}')
    print(f'Created training label of shape {y.shape}')


@main.command()
@click.option(
    '-p',
    '--path',
    required=True,
    type=click.Path(exists=True),
    help="Path of the pretrained word vector"
)
def buildWord2Vec():
    pass

def _get_available_dev(d):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == d]

def _process_word_vectors(base_path, suffix, pretrained=False, **kwargs):
    from preprocess import TextReader
    import pandas as pd
    tr = TextReader(data_dir=base_path, 
                    suffix_labels=suffix)
    print(tr.data_files)
    if tr.prepare_data(clean=True, max_vocab=15000):
        X, y = tr.get_ranked_features()
    word_vectors_df = None
    if pretrained:
        model = kwargs.get('model')
        if model is None:
            raise ValueError('Model can not be None')
        wv = tr.get_embedding_vector(model)
        word_vectors = {}
        for word, vector in wv:
            word_vectors[tr.get_rank(word)] = vector
        word_vectors_df = pd.DataFrame.from_dict(word_vectors, orient='index')
    return X, y, word_vectors_df

if __name__ == '__main__':
    main()