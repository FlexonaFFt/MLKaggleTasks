from OwnDataLoader import DataPaths, DataLoader, DatasetBuilder, FeatureBuilder
from mlcore.LogReg_model import LogRegModel, LogRegConfig
from makesub import SubmissionBuilder

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

    def train_eval_and_submit(self, X_train, y_train, X_test, test_df):
        model = LogRegModel(LogRegConfig(verbose=1))
        loss, test_proba = model.train_eval_predict(X_train, y_train, X_test)
        print("val log loss:", loss)

        sub_builder = SubmissionBuilder()
        sub_builder.build(test_df, test_proba, path="submission.csv")
        print("submission saved")
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
        _ = self.train_eval_and_submit(X_train, y_train, X_test, test_df)
        print("pipeline finished")


if __name__ == '__main__':
    solve = Solution()
    solve.main()
