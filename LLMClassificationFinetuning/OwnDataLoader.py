import ast 
import re 

import pandas as pd 
import numpy as np 

from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass 
class DataPaths:
    train_path: str 
    test_path: str 


class DataLoader:
    def __init__(self, paths: DataPaths):
        self.paths = paths 

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(self.paths.train_path)
        test = pd.read_csv(self.paths.test_path)
        return train, test 


class TextParser:
    @staticmethod
    def parse_list(text):
        if not isinstance(text, str): return []
        try: return ast.literal_eval(text)
        except Exception: return []

    @staticmethod
    def normilize(text: str) -> str:
        if not isinstance(text, str): return ''
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text 

    def join_turns(self, text):
        turns = self.parse_list(text)
        if not turns: return ""
        joined = " <TURN> ".join(turns)
        return self.normilize(joined)


"""Важный класс для построения фичей"""
class FeatureBuilder:
    def __init__(self):
        self.parser = TextParser()

    def add_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy() 
        df["prompt_text"] = df["prompt"].apply(self.parser.join_turns)
        df["resp_a_text"] = df["response_a"].apply(self.parser.join_turns)
        df["resp_b_text"] = df["response_b"].apply(self.parser.join_turns)
        df["pair_text"] = "A: " + df["resp_a_text"] + " <SEP> B: " + df["resp_b_text"]
        return df 

    def add_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def count_tokens(s):
            if not s: return 0
            return len(s.split())

        for col in ["prompt_text", "resp_a_text", "resp_b_text"]:
            df[col + "_n_chars"] = df[col].str.len()
            df[col + "_n_tokens"] = df[col].apply(count_tokens)

        df["len_diff_chars"] = df["resp_a_text_n_chars"] - df["resp_b_text_n_chars"]
        df["len_diff_tokens"] = df["resp_a_text_n_tokens"] - df["resp_b_text_n_tokens"]
        df["abs_len_diff_chars"] = df["len_diff_chars"].abs()
        df["abs_len_diff_tokens"] = df["len_diff_tokens"].abs()
        return df 


class TextVectorizer:
    def __init__(self, max_features=50000, ngram_range=(1, 2), min_df=3):
        self.vectorizer = {
            "prompt_text": TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df),
            "resp_a_text": TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df),
            "resp_b_text": TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df),
            "pair_text": TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df),
        }

    def fit_transform(self, df: pd.DataFrame):
        mats = []
        for col, vec in self.vectorizer.items():
            mats.append(vec.fit_transform(df[col]))
        return hstack(mats)

    def transform(self, df: pd.DataFrame):
        mats = []
        for col, vec in self.vectorizer.items():
            mats.append(vec.transform(df[col]))
        return hstack(mats)


class DatasetBuilder:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols 
        self.vectorizer = TextVectorizer()

    def build_train(self, df: pd.DataFrame):
        X_text = self.vectorizer.fit_transform(df)
        X_num = csr_matrix(df[self.feature_cols].values)
        X = hstack([X_text, X_num])
        y = df[["winner_model_a", "winner_model_b", "winner_tie"]].values
        return X, y 

    def build_test(self, df: pd.DataFrame):
        X_test = self.vectorizer.transform(df)
        X_num = csr_matrix(df[self.feature_cols].values)
        X = hstack([X_test, X_num])
        return X 
