import numpy as np
import os
from collections import defaultdict, Counter
import re
from nltk.tokenize import word_tokenize

class TextReader:
    
    def __init__(self, data_dir, suffix_labels):
        self.path = data_dir
        self.ranks = None
        self.data_files = {}
        for file, label in suffix_labels.items():
            if not os.path.exists(os.path.join(data_dir, file)) or not \
            os.path.isfile(os.path.join(data_dir, file)):
                raise IOError(f'Data files are not found in {data_dir}')
            else:
                self.data_files[os.path.join(data_dir, file)] = label
    
    def clean_text(self, text, stopwords):
        """
        Cleaning the text
        """
        text = " ".join(filter(lambda x: all([x.isalpha(), x not in stopwords]), 
                               word_tokenize(text)))
        return text.strip().lower()
    
    def prepare_data(self, clean=True, **kwargs):
        max_sent_len = 0
        all_words = []
        labeled_data = defaultdict(list)
        for file_path, class_label in self.data_files.items():
            lines = []
            with open(file_path, 'r', encoding='latin-1') as infile:
                for line in infile:
                    if not clean:
                        cleaned_line = line
                    else:
                        stopwords = kwargs.get('stopwords', [])
                        cleaned_line = self.clean_text(line, stopwords)

                    lines.append(cleaned_line)
                    tokens = cleaned_line.split()
                    max_sent_len = max(max_sent_len, len(tokens))
                    all_words.extend(tokens)
                    labeled_data[class_label].append(cleaned_line)
        TextReader.store_ranking(Counter(all_words), self.path)
        return labeled_data, max_sent_len
    
    @staticmethod
    def store_ranking(wordFreq, path, max_vocab=None):
        ranks = [*map(lambda x: x[0], wfreq.most_common(max_vocab))]
        np.save(os.path.join(path, 'ranks'), ranks)
        return True
    
    def get_rank(self, token):
        if self.ranks is None:
            self.ranks = np.load('./data/mr/ranks.npy')
        return np.where(self.ranks == token)[0][0]
            
        
    
    def get_statistics(self):
        print(f'Total number of datapoints: {len(self)}')
