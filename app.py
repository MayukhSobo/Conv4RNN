from preprocess import TextReader
import pandas as pd
from tensorflow.python.client import device_lib
import click
# from train import CNNText


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
def get_available_dev(device):
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == device]

def process_word_vectors(base_path, suffix, pretrained=False, **kwargs):
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