class TextReader:
    
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
        all_words = []
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
                    self.max_text_length = max(self.max_text_length, len(tokens))
                    all_words.extend(tokens)
                    self.raw_labeled_data[class_label].append(cleaned_line)
        
        self.word_fequency = Counter(all_words)
        return self.store_ranking(kwargs.get('max_vocab'))
    
    def store_ranking(self, max_vocab=None):
        ranks = [*map(lambda x: x[0], self.word_fequency.most_common(max_vocab))]
        np.save(os.path.join(self.path, 'ranks'), ranks)
        return True
    
    def get_rank(self, token):
        if self.ranks is None:
            self.ranks = np.load(os.path.join(self.path, 'ranks.npy'))
        try:
            return int(np.where(self.ranks == token)[0][0]) + 1
        except IndexError:
            return 0
            
    def get_ranked_features(self):
        if self.X is not None and self.y is not None:
            return self.X, self.y
        X = []
        y = []
        for label, corpus in self.raw_labeled_data.items():
            for doc in tqdm(corpus):
                tokens = doc.split()
                ranks = [self.get_rank(token) for token in tokens]
                pad_left = (self.max_text_length - len(tokens)) // 2
                pad_right = int(np.ceil((self.max_text_length - len(tokens)) / 2.0))
                ranks = np.pad(ranks, pad_width=(pad_left, pad_right), 
                               mode='constant', constant_values=(-1, -1))
                y.append(label)
                X.append(ranks)
        return np.array(X, dtype=int), np.array(y, dtype=int)
