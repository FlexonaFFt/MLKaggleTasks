"""Fixed shape/function TF-IDF-SVD blocks; all learned transforms fit on train only."""

import re

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

FUNCTION_WORDS_TEXT='a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'

def shape_text(text):
    # Preserve character count/case, punctuation and whitespace; suppress letters/digits.
    return re.sub(r'\d','#',re.sub(r'[^\W\d_]',lambda m:'A' if m[0].isupper() else 'a',text))

def function_text(text):
    words=set(FUNCTION_WORDS_TEXT.split())
    return ' '.join(t.lower() if t.lower() in words else ('CONTENT' if t.isalnum() else t)
        for t in re.findall(r'\w+|[^\w\s]',text))

VECTORIZERS = {'shape_char': {'min_df': 3, 'max_df': 0.995, 'sublinear_tf': True, 'norm': 'l2', 'analyzer': 'char', 'ngram_range': [3, 5], 'lowercase': False, 'max_features': 20000}, 'function_word': {'min_df': 3, 'max_df': 0.995, 'sublinear_tf': True, 'norm': 'l2', 'analyzer': 'word', 'ngram_range': [1, 3], 'lowercase': False, 'token_pattern': '(?u)\\b\\w+\\b|[^\\w\\s]', 'max_features': 20000}}

def masked_svd_blocks(train_texts, query_texts):
    train_blocks, query_blocks, audit = [], [], []
    for name, cfg in VECTORIZERS.items():
        masker = shape_text if name == 'shape_char' else function_text
        fit_values = [masker(text) for text in train_texts]
        query_values = [masker(text) for text in query_texts]
        options = dict(cfg)
        options['ngram_range'] = tuple(options['ngram_range'])
        options['dtype'] = np.float32
        vectorizer = TfidfVectorizer(**options)
        sparse_fit = vectorizer.fit_transform(fit_values)
        assert sparse_fit.shape[1] > 16
        svd = TruncatedSVD(n_components=16, n_iter=5, random_state=20261017)
        svd.fit(sparse_fit)
        train_block = svd.transform(vectorizer.transform(fit_values)).astype(np.float32)
        query_block = svd.transform(vectorizer.transform(query_values)).astype(np.float32)
        assert train_block.shape == (len(train_texts), 16)
        assert query_block.shape == (len(query_texts), 16)
        assert np.isfinite(train_block).all() and np.isfinite(query_block).all()
        train_blocks.append(train_block)
        query_blocks.append(query_block)
        audit.append(dict(view=name, vocabulary_size=len(vectorizer.vocabulary_),
            fit_rows=len(train_texts), query_rows=len(query_texts), components=16,
            fit_scope='official_train_only',
            explained_variance_ratio_sum=float(svd.explained_variance_ratio_.sum())))
    return np.concatenate(train_blocks, axis=1), np.concatenate(query_blocks, axis=1), audit

