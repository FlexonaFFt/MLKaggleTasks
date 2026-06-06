# EDA Report — Predict Customer Churn

## 1. Dataset Overview
- Train shape: (594194, 21)
- Test shape: (254655, 20)
- Train columns: ['id', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

### Column types (train)
- id: int64
- gender: object
- SeniorCitizen: int64
- Partner: object
- Dependents: object
- tenure: int64
- PhoneService: object
- MultipleLines: object
- InternetService: object
- OnlineSecurity: object
- OnlineBackup: object
- DeviceProtection: object
- TechSupport: object
- StreamingTV: object
- StreamingMovies: object
- Contract: object
- PaperlessBilling: object
- PaymentMethod: object
- MonthlyCharges: float64
- TotalCharges: float64
- Churn: object

## 2. Missing Values
### Train
No null values detected (before TotalCharges conversion)
### Test
No null values detected (before TotalCharges conversion)

## 3. Duplicates
- Train duplicates (excl id): 0
- Test duplicates (excl id): 0

## 4. Target Distribution (Churn)
Churn
No     460377
Yes    133817
- Churn rate: 22.52%

## 5. TotalCharges Conversion
- train: NaN before=0, after coerce=0
- test: NaN before=0, after coerce=0
- Median TotalCharges (train): 1433.65

## 6. Correlations with Churn
- tenure: -0.4185
- MonthlyCharges: 0.2730
- SeniorCitizen: 0.2364
- TotalCharges: -0.2184

## 7. Feature Engineering
Added: tenure_group, monthly_to_total_ratio, avg_monthly_charge, no_online_security, no_tech_support, month_to_month, electronic_check, fiber_optic, high_risk_score, is_senior_alone

## 8. Encoding
Categorical columns to encode: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
- Final train features shape: (594194, 44)
- Final test features shape: (254655, 44)
- Train columns sample: ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'monthly_to_total_ratio', 'avg_monthly_charge', 'no_online_security', 'no_tech_support', 'month_to_month', 'electronic_check']... (44 total)
