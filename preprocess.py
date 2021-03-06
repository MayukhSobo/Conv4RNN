import numpy as np
import os
from collections import defaultdict, Counter
import re
from tqdm import tqdm
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

class TextReader:
    """
    A class meant to load the text data from 
    files distinctively identifiable for different
    class labels, clean the text and use pretrained
    word vectors to convert into suitable word vectors.
    """
    def __init__(self, data_dir, suffix_labels):
        self.path = data_dir
        self.ranks = None
        self.raw_labeled_data = defaultdict(list)
        self.word_fequency = None
        self.max_text_length = 0
        self.data_files = {}
        self.X = None
        self.y = None
        for file, label in suffix_labels.items():
            if not os.path.isfile(os.path.join(data_dir, file)):
                raise IOError(f'Data files are not found in {data_dir}')
            else:
                self.data_files[os.path.join(data_dir, file)] = label
    
    def clean_text(self, text, stopwords):
        """
        Cleaning the text
        """
        text = [x for x in word_tokenize(text) if x not in stopwords]
        text = " ".join(text)
        return text.strip().lower()
    
    def prepare_data(self, clean=True, **kwargs):
        all_words = []
        for file_path, class_label in self.data_files.items():
            # lines = []
            with open(file_path, 'r', encoding='latin-1') as infile:
                for line in infile:
                    if not clean:
                        cleaned_line = line
                    else:
                        stopwords = kwargs.get('stopwords', [])
                        cleaned_line = self.clean_text(line, stopwords)

                    # lines.append(cleaned_line)
                    tokens = cleaned_line.split()
                    self.max_text_length = max(self.max_text_length, len(tokens))
                    all_words.extend(tokens)
                    self.raw_labeled_data[(file_path, class_label)].append(cleaned_line)
        
        self.word_fequency = Counter(all_words)
        return self.store_ranking(kwargs.get('max_vocab'))
    
    def store_ranking(self, max_vocab=None):
        ranks = [*map(lambda x: x[0], self.word_fequency.most_common(max_vocab - 2))]
        ranks.insert(0, PAD_TOKEN)
        ranks.insert(0, UNK_TOKEN)
#         if not os.path.exists(os.path.join(self.path, 'data')):
#             os.makedirs(directory)
        joblib.dump(ranks, os.path.join(self.path, 'ranks'))
        print(f'Created token ranks {os.path.join(self.path, "ranks")} of size {len(ranks)}')
        # np.save(os.path.join(self.path, 'ranks'), ranks)
        return True
    
    def get_rank(self, token):
        if self.ranks is None:
            self.ranks = joblib.load(os.path.join(self.path, 'ranks'))
        try:
            return self.ranks.index(token)
        except ValueError:
            return 0
            
    def get_ranked_features(self):
        if self.X is not None and self.y is not None:
            return self.X, self.y
        X = []
        y = []
        all_data = tqdm(self.raw_labeled_data.items())
        for path_label, corpus in all_data:
            path, label = path_label
            all_data.set_description_str(desc=f"Processing: {path}", refresh=True)
            for doc in corpus:
                tokens = doc.split()
                ranks = [self.get_rank(token) for token in tokens]
                pad_left = (self.max_text_length - len(tokens)) // 2
                pad_right = int(np.ceil((self.max_text_length - len(tokens)) / 2.0))
                ranks = np.pad(ranks, pad_width=(pad_left, pad_right), 
                               mode='constant', constant_values=(1, 1))
                y.append(label)
                X.append(ranks)
        X = np.array(X, dtype=int)
        y = np.array(y, dtype=int)
        return X, y
    
def get_embedding_vector(config, rankfile):
    """
    Get the embedding vector from the Gensim model.
    We can use pretrained word vectors like Google News.
    """
    from gensim.models import KeyedVectors
    from pyemd import emd
    ranks = joblib.load(rankfile)
    vocab_size = config["arch"]["data"]["vocab_size"]
    emb_size = config["arch"]["data"]["embedding"]
    bin_path = config["arch"]["initialisation"]["bin_path"]
    rank_matrix = np.zeros((vocab_size, emb_size))
    if vocab_size != len(ranks):
        raise RuntimeError(f'Length of the rank vocabulary({len(ranks)}) != vocab size({vocab_size})')
    else:
        print('Loading the pretrained word vector binary')
        model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        for idx, word in tqdm(enumerate(ranks), total=vocab_size):
            if model.__contains__(word):
                rank_matrix[idx: ] = model[word]
            else:
                rank_matrix[idx: ] = np.random.uniform(-0.25, 0.25, model.vector_size)
        path = os.path.dirname(rankfile)
        np.save(os.path.join(path, 'rank_matrix'), rank_matrix)
        print(f'Saved pretrained rank matrix at {path} of shape {rank_matrix.shape}')