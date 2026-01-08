from dataclasses import dataclass
import pandas as pd 


@dataclass
class SubmissionConfig:
    id_col: str = 'id'
    target_col: str = 'Tm'
    output_path: str = 'submission.csv'


class SubmissionMaker:
    def __init__(self, cfg: SubmissionConfig):
        self.cfg = cfg

    def make(self, ids: pd.Series, preds) -> pd.DataFrame:
        sub = pd.DataFrame({
            self.cfg.id_col: ids,
            self.cfg.target_col: preds
        })
        return sub

    def save(self, sub_df: pd.DataFrame):
        sub_df.to_csv(self.cfg.output_path, index=False)