from OwnDataLoader import DataPaths, DataLoader, DatasetBuilder, FeatureBuilder
from mlcore.LogReg_model import LogRegModel, LogRegConfig

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class Solution:
    def preprocess_data(self, train_path: str, test_path: str):
        paths = DataPaths(train_path=train_path, test_path=test_path)
        loader = DataLoader(paths=paths)
        train_df, test_df = loader.load()

        builder = FeatureBuilder()
        train_df = builder.add_text_fields(train_df)
        train_df = builder.add_numeric_features(train_df)

        test_df = builder.add_text_fields(test_df)
        test_df = builder.add_numeric_features(test_df)

        return train_df, test_df 

    def build_features(self, train_df, test_df):
        feature_cols = [
            "prompt_text_n_chars", "resp_a_text_n_chars", "resp_b_text_n_chars",
            "prompt_text_n_tokens", "resp_a_text_n_tokens", "resp_b_text_n_tokens",
            "len_diff_chars", "len_diff_tokens",
            "abs_len_diff_chars", "abs_len_diff_tokens",
        ]

        dataset = DatasetBuilder(feature_cols=feature_cols)
        X_train, y_train = dataset.build_train(train_df)
        X_test = dataset.build_test(test_df)
        return X_train, y_train, X_test 

    def train_and_validate(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
        )

        model = LogRegModel(LogRegConfig)
        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_val)
        loss = log_loss(y_val, val_proba)
        print("val log loss: ", loss)
        return model

    def main(self):
        train_df, test_df = self.preprocess_data(
            train_path="datasets/train.csv",
            test_path="datasets/test.csv"
        )

        print("train shape:", train_df.shape)
        print("test shape:", test_df.shape)
        print("")

        X_train, y_train, X_test = self.build_features(train_df, test_df)
        _ = self.train_and_validate(X_train, y_train)
        print("pipeline finished")


if __name__ == '__main__':
    solve = Solution()
    solve.main()