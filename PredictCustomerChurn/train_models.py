import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('PredictCustomerChurn/featured_train.csv')
test = pd.read_csv('PredictCustomerChurn/featured_test.csv')

customer_ids = test['customerID']
test = test.drop('customerID', axis=1)

y = train['Churn']
X = train.drop('Churn', axis=1)

feature_cols = X.columns.tolist()
X_test = test[feature_cols]

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Features: {len(feature_cols)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# 1. Baseline - LogisticRegression
print("\n=== Logistic Regression (Baseline) ===")
from sklearn.pipeline import Pipeline
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
])
lr_scores = cross_val_score(lr_pipe, X, y, cv=skf, scoring='roc_auc')
results['baseline_lr'] = {
    'cv_roc_auc_mean': round(float(lr_scores.mean()), 5),
    'cv_roc_auc_std': round(float(lr_scores.std()), 5)
}
print(f"ROC-AUC: {lr_scores.mean():.5f} ± {lr_scores.std():.5f}")

# 2. XGBoost
print("\n=== XGBoost ===")
xgb = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss', random_state=42,
    n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
)
xgb_scores = cross_val_score(xgb, X, y, cv=skf, scoring='roc_auc')
results['xgboost'] = {
    'cv_roc_auc_mean': round(float(xgb_scores.mean()), 5),
    'cv_roc_auc_std': round(float(xgb_scores.std()), 5)
}
print(f"ROC-AUC: {xgb_scores.mean():.5f} ± {xgb_scores.std():.5f}")

# 3. LightGBM
print("\n=== LightGBM ===")
lgbm = LGBMClassifier(
    random_state=42, verbose=-1,
    n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
)
lgbm_scores = cross_val_score(lgbm, X, y, cv=skf, scoring='roc_auc')
results['lgbm'] = {
    'cv_roc_auc_mean': round(float(lgbm_scores.mean()), 5),
    'cv_roc_auc_std': round(float(lgbm_scores.std()), 5)
}
print(f"ROC-AUC: {lgbm_scores.mean():.5f} ± {lgbm_scores.std():.5f}")

# 4. CatBoost
print("\n=== CatBoost ===")
cat = CatBoostClassifier(
    random_state=42, verbose=0,
    iterations=300, learning_rate=0.05, depth=6
)
cat_scores = cross_val_score(cat, X, y, cv=skf, scoring='roc_auc')
results['catboost'] = {
    'cv_roc_auc_mean': round(float(cat_scores.mean()), 5),
    'cv_roc_auc_std': round(float(cat_scores.std()), 5)
}
print(f"ROC-AUC: {cat_scores.mean():.5f} ± {cat_scores.std():.5f}")

# 5. Voting Ensemble
print("\n=== Voting Ensemble ===")
xgb_v = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss', random_state=42,
    n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
)
lgbm_v = LGBMClassifier(
    random_state=42, verbose=-1,
    n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
)
cat_v = CatBoostClassifier(
    random_state=42, verbose=0,
    iterations=300, learning_rate=0.05, depth=6
)
voting = VotingClassifier(
    estimators=[('xgb', xgb_v), ('lgbm', lgbm_v), ('catboost', cat_v)],
    voting='soft'
)
voting_scores = cross_val_score(voting, X, y, cv=skf, scoring='roc_auc')
results['voting_ensemble'] = {
    'cv_roc_auc_mean': round(float(voting_scores.mean()), 5),
    'cv_roc_auc_std': round(float(voting_scores.std()), 5)
}
print(f"ROC-AUC: {voting_scores.mean():.5f} ± {voting_scores.std():.5f}")

# Find best model
all_models = {k: v['cv_roc_auc_mean'] for k, v in results.items()}
best_name = max(all_models, key=all_models.get)
best_score = all_models[best_name]
results['best_model'] = best_name
results['best_cv_roc_auc'] = best_score

print(f"\n=== Best Model: {best_name} with ROC-AUC: {best_score:.5f} ===")

# Train best model on full train and predict
print("\nTraining best model on full data...")
model_map = {
    'xgboost': xgb,
    'lgbm': lgbm,
    'catboost': cat,
    'voting_ensemble': voting,
    'baseline_lr': lr_pipe
}
best_model = model_map[best_name]
best_model.fit(X, y)
test_proba = best_model.predict_proba(X_test)[:, 1]

# Save submission
submission = pd.DataFrame({'customerID': customer_ids, 'Churn': test_proba})
submission.to_csv('PredictCustomerChurn/submission.csv', index=False)
print(f"Submission saved: {submission.shape[0]} rows")

# Save results
with open('PredictCustomerChurn/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("model_results.json saved")

print("\n=== ALL DONE ===")
print(json.dumps(results, indent=2))
