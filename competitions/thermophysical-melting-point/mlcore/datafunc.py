import pandas as pd 
import numpy as np

from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class DataConfig:
    train_path: str 
    test_path: str 
    target_col: str = 'Tm'
    id_col: str = 'id'

    smiles_col: str = "SMILES"
    smiles_mode: str = 'ignore' # "ignore", "simple", "char_counts", "rdkit"
    missing_strategy: str = 'zero' # "zero" или "median"
    log_features: bool = True
    log_target: bool = True
    rdkit_radius: int = 2
    rdkit_nbits: int = 2048
    use_mordred: bool = False
    use_3d: bool = False
    rdkit_3d_max_iters: int = 200
    rdkit_3d_seed: int = 0
    use_group_features: bool = True
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1"
    chemberta_max_length: int = 256
    chemberta_batch_size: int = 32


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config 

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        return train_df, test_df 


class DataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_cols: List[str] = []
        self.constant_cols: List[str] = []
        self.median_map: Optional[pd.Series] = None
        self.smiles_feature_cols: List[str] = []
        self._chem_tokenizer = None
        self._chem_model = None

    def fit(self, train_df: pd.DataFrame) -> "DataPreprocessor":
        drop_cols = [self.config.target_col, self.config.id_col]
        if self.config.smiles_col in train_df.columns:
            drop_cols.append(self.config.smiles_col)

        base_features = [c for c in train_df.columns if c not in drop_cols]
        if not self.config.use_group_features:
            base_features = [c for c in base_features if not c.startswith("Group ")]
        X = train_df[base_features].copy()

        if self.config.smiles_mode != "ignore" and self.config.smiles_col in train_df.columns:
            smiles_feats = self._smiles_features(train_df[self.config.smiles_col])
            self.smiles_feature_cols = smiles_feats.columns.tolist()
            X = pd.concat([X, smiles_feats], axis=1)

        nunique = X.nunique(dropna=False)
        self.constant_cols = nunique[nunique <= 1].index.tolist()
        self.feature_cols = [c for c in X.columns if c not in self.constant_cols]
        if self.config.missing_strategy == "median":
            self.median_map = X[self.feature_cols].median()
        else:
            self.median_map = None

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_cols:
            raise ValueError("Call fit() before transform().")

        drop_cols = [self.config.target_col, self.config.id_col]
        if self.config.smiles_col in df.columns:
            drop_cols.append(self.config.smiles_col)

        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
        if self.config.smiles_mode != "ignore" and self.config.smiles_col in df.columns:
            smiles_feats = self._smiles_features(df[self.config.smiles_col])
            for col in self.smiles_feature_cols:
                if col not in smiles_feats.columns:
                    smiles_feats[col] = 0
            smiles_feats = smiles_feats[self.smiles_feature_cols]
            X = pd.concat([X, smiles_feats], axis=1)

        X = X.drop(columns=[c for c in self.constant_cols if c in X.columns])
        X = X.reindex(columns=self.feature_cols)

        if self.config.missing_strategy == "median" and self.median_map is not None:
            X = X.fillna(self.median_map)
        else:
            X = X.fillna(0)

        return X

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def _smiles_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        if self.config.smiles_mode == "simple":
            return self._simple_smiles_features(smiles_series)
        if self.config.smiles_mode == "char_counts":
            return self._char_count_features(smiles_series)
        if self.config.smiles_mode == "rdkit":
            return self._rdkit_features(smiles_series)
        if self.config.smiles_mode == "chemberta":
            return self._chemberta_features(smiles_series)
        return pd.DataFrame(index=smiles_series.index)

    def _simple_smiles_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        s = smiles_series.fillna("")
        feats = pd.DataFrame(index=s.index)
        feats["smiles_len"] = s.str.len()
        feats["smiles_atoms"] = s.str.count(r"[A-Za-z]")
        feats["smiles_branches"] = s.str.count(r"\(")
        feats["smiles_double"] = s.str.count("=")
        feats["smiles_triple"] = s.str.count("#")
        feats["smiles_aromatic"] = s.str.count(r"[bcnops]")
        feats["smiles_Cl"] = s.str.count("Cl")
        feats["smiles_Br"] = s.str.count("Br")
        feats["smiles_N"] = s.str.count("N")
        feats["smiles_O"] = s.str.count("O")
        feats["smiles_S"] = s.str.count("S")
        return feats

    def _char_count_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        s = smiles_series.fillna("")
        chars = ["C", "N", "O", "S", "P", "F", "I", "B", "c", "n", "o", "s", "(", ")", "=", "#", "[", "]", "+", "-"]
        feats = pd.DataFrame(index=s.index)
        feats["smiles_len"] = s.str.len()
        for ch in chars:
            col = f"smiles_{ch}"
            feats[col] = s.str.count(repr(ch).strip("'"))
        return feats

    def _rdkit_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            from rdkit.Chem import Descriptors
            from rdkit.Chem import rdMolDescriptors
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
            from rdkit.Chem import AllChem
        except ImportError as exc:
            raise ImportError("RDKit is required for smiles_mode='rdkit'.") from exc
        RDLogger.DisableLog("rdApp.*")

        s = smiles_series.fillna("")
        feats = []
        mols = []
        mols_3d = []
        for smi in s:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
            mol_3d = None
            if self.config.use_3d and mol is not None:
                mol_h = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = self.config.rdkit_3d_seed
                res = AllChem.EmbedMolecule(mol_h, params)
                if res == 0:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=self.config.rdkit_3d_max_iters)
                    mol_3d = mol_h
            mols_3d.append(mol_3d)
            if mol is None:
                feats.append({
                    "rdkit_mol_wt": np.nan,
                    "rdkit_logp": np.nan,
                    "rdkit_tpsa": np.nan,
                    "rdkit_hbd": np.nan,
                    "rdkit_hba": np.nan,
                    "rdkit_rings": np.nan,
                    "rdkit_rot_bonds": np.nan,
                    "rdkit_heavy_atoms": np.nan,
                    "rdkit_fr_csp3": np.nan,
                    "rdkit_asphericity": np.nan,
                    "rdkit_eccentricity": np.nan,
                    "rdkit_inertial_shape": np.nan,
                    "rdkit_npr1": np.nan,
                    "rdkit_npr2": np.nan,
                    "rdkit_radius_gyr": np.nan,
                    "rdkit_spherocity": np.nan,
                })
                continue

            if self.config.use_3d and mol_3d is not None:
                asphericity = rdMolDescriptors.CalcAsphericity(mol_3d)
                eccentricity = rdMolDescriptors.CalcEccentricity(mol_3d)
                inertial_shape = rdMolDescriptors.CalcInertialShapeFactor(mol_3d)
                npr1 = rdMolDescriptors.CalcNPR1(mol_3d)
                npr2 = rdMolDescriptors.CalcNPR2(mol_3d)
                radius_gyr = rdMolDescriptors.CalcRadiusOfGyration(mol_3d)
                spherocity = rdMolDescriptors.CalcSpherocityIndex(mol_3d)
            else:
                asphericity = np.nan
                eccentricity = np.nan
                inertial_shape = np.nan
                npr1 = np.nan
                npr2 = np.nan
                radius_gyr = np.nan
                spherocity = np.nan

            feats.append({
                "rdkit_mol_wt": Descriptors.MolWt(mol),
                "rdkit_logp": Descriptors.MolLogP(mol),
                "rdkit_tpsa": Descriptors.TPSA(mol),
                "rdkit_hbd": Descriptors.NumHDonors(mol),
                "rdkit_hba": Descriptors.NumHAcceptors(mol),
                "rdkit_rings": Descriptors.RingCount(mol),
                "rdkit_rot_bonds": Descriptors.NumRotatableBonds(mol),
                "rdkit_heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "rdkit_fr_csp3": Descriptors.FractionCSP3(mol),
                "rdkit_asphericity": asphericity,
                "rdkit_eccentricity": eccentricity,
                "rdkit_inertial_shape": inertial_shape,
                "rdkit_npr1": npr1,
                "rdkit_npr2": npr2,
                "rdkit_radius_gyr": radius_gyr,
                "rdkit_spherocity": spherocity,
            })

        desc_df = pd.DataFrame(feats, index=s.index)

        fp_rows = []
        morgan_count_rows = []
        maccs_rows = []
        atompair_rows = []
        morgan_bit_gen = GetMorganGenerator(
            radius=self.config.rdkit_radius,
            fpSize=self.config.rdkit_nbits,
        )
        morgan_count_gen = GetMorganGenerator(
            radius=self.config.rdkit_radius,
            fpSize=self.config.rdkit_nbits,
            countSimulation=True,
        )
        atompair_gen = GetAtomPairGenerator(fpSize=self.config.rdkit_nbits)
        for smi in s:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp_rows.append([0] * self.config.rdkit_nbits)
                morgan_count_rows.append([0] * self.config.rdkit_nbits)
                maccs_rows.append([0] * 167)
                atompair_rows.append([0] * self.config.rdkit_nbits)
                continue
            fp = morgan_bit_gen.GetFingerprint(mol)
            fp_rows.append(list(fp))
            morgan_count = morgan_count_gen.GetFingerprint(mol)
            count_vec = [0] * self.config.rdkit_nbits
            if hasattr(morgan_count, "GetNonzeroElements"):
                for idx, val in morgan_count.GetNonzeroElements().items():
                    count_vec[idx] = val
            else:
                for idx, val in enumerate(list(morgan_count)):
                    count_vec[idx] = val
            morgan_count_rows.append(count_vec)
            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            maccs_rows.append(list(maccs))
            atompair = atompair_gen.GetFingerprint(mol)
            atompair_rows.append(list(atompair))

        fp_cols = [f"mfp_{i}" for i in range(self.config.rdkit_nbits)]
        fp_df = pd.DataFrame(fp_rows, columns=fp_cols, index=s.index)

        morgan_count_cols = [f"mfp_count_{i}" for i in range(self.config.rdkit_nbits)]
        morgan_count_df = pd.DataFrame(morgan_count_rows, columns=morgan_count_cols, index=s.index)
        maccs_cols = [f"maccs_{i}" for i in range(167)]
        maccs_df = pd.DataFrame(maccs_rows, columns=maccs_cols, index=s.index)
        atompair_cols = [f"ap_{i}" for i in range(self.config.rdkit_nbits)]
        atompair_df = pd.DataFrame(atompair_rows, columns=atompair_cols, index=s.index)

        feature_parts = [desc_df, fp_df, morgan_count_df, maccs_df, atompair_df]

        if self.config.use_mordred:
            try:
                from mordred import Calculator, descriptors
                calc = Calculator(descriptors, ignore_3D=not self.config.use_3d)
                mordred_mols = [m3d if (self.config.use_3d and m3d is not None) else m for m, m3d in zip(mols, mols_3d)]
                mordred_df = calc.pandas(mordred_mols)
                mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")
                mordred_df.index = s.index
                feature_parts.append(mordred_df)
            except ImportError as exc:
                raise ImportError("Mordred is required for use_mordred=True.") from exc

        return pd.concat(feature_parts, axis=1)

    def _chemberta_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        try:
            import os
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError("Transformers and torch are required for smiles_mode='chemberta'.") from exc

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(1)

        if self._chem_tokenizer is None or self._chem_model is None:
            self._chem_tokenizer = AutoTokenizer.from_pretrained(self.config.chemberta_model)
            self._chem_model = AutoModel.from_pretrained(self.config.chemberta_model)
            self._chem_model.eval()

        device = torch.device("cpu")
        self._chem_model.to(device)

        s = smiles_series.fillna("").tolist()
        batch_size = self.config.chemberta_batch_size
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(s), batch_size):
                batch = s[i : i + batch_size]
                enc = self._chem_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.chemberta_max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = self._chem_model(**enc)
                last_hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = (summed / counts).cpu().numpy()
                embeddings.append(pooled)

        emb = np.vstack(embeddings)
        emb_cols = [f"chemberta_{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb, columns=emb_cols, index=smiles_series.index)
