import pandas as pd 

class SubmissionBuilder:
    def __init__(self, id_col: str = 'id'):
        self.id_col = id_col 
        self.target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    def build(self, test_df: pd.DataFrame, proba, path: str = 'submission.csv'):
        if len(proba) != len(test_df):
            raise ValueError("proba rows must match test_df rows")

        sub = pd.DataFrame(proba, columns=self.target_cols)
        sub.insert(0, self.id_col, test_df[self.id_col].values)
        sub.to_csv(path, index=False)
        return sub