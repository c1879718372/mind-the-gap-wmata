# ==== 0) Imports ====
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, confusion_matrix, classification_report)

import matplotlib.pyplot as plt
import seaborn as sns

# ==== 1) Load your two source tables (update paths if needed) ====
incidents = pd.read_csv('/content/elevator_incidents_parsed.csv')            # outage events (with timestamps)
entries   = pd.read_csv('/content/Entries by Year_Full Data_data (2).csv')   # station entries by date

# ==== 2) Light cleaning & “station × date × shift” modeling table ====
for col in ['DateOutOfServ', 'EstimatedReturnToService']:
    if col in incidents.columns:
        incidents[col] = pd.to_datetime(incidents[col], errors='coerce')

# keep elevators/escalators only
incidents = incidents[incidents['UnitType'].isin(['ELEVATOR','ESCALATOR'])].copy()

# daily flag: outage happened this day at this station?
incidents['date'] = incidents['DateOutOfServ'].dt.date
daily_flag = (incidents.groupby(['StationCode','date'])
                        .size()
                        .reset_index(name='outage_cnt'))
daily_flag['outage_any'] = (daily_flag['outage_cnt'] > 0).astype(int)

# normalize entries table (rename to 'entries' and extract date)
entries['Date'] = pd.to_datetime(entries['Date'], errors='coerce')
entries['date'] = entries['Date'].dt.date
entries = entries.rename(columns={'Entries':'entries'})
entries = entries[['StationCode','date','entries']].copy()

# merge to get station×date rows
df = (daily_flag.merge(entries, on=['StationCode','date'], how='left')
               .fillna({'entries':0}))

# rolling 14-day outage rate per station (as a feature)
tmp = df.sort_values(['StationCode','date']).copy()
tmp['outage_roll14'] = (tmp.groupby('StationCode')['outage_any']
                          .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean()))
tmp['outage_roll14'] = tmp['outage_roll14'].fillna(0.0)

# duplicate rows to create two “shifts” per day (demo); in a fuller build, slice by actual hours
long = []
for sh in ['AM_peak','PM_peak']:
    t = tmp.copy()
    t['shift'] = sh
    long.append(t)
data = pd.concat(long, ignore_index=True)

# target = next-window outage (shifted label by station×shift)
data = data.sort_values(['StationCode','date','shift'])
data['y_next'] = (data.groupby(['StationCode','shift'])['outage_any']
                    .shift(-1))
data = data.dropna(subset=['y_next']).copy()
data['y_next'] = data['y_next'].astype(int)

# calendar features
data['dow']   = pd.to_datetime(data['date']).weekday
data['month'] = pd.to_datetime(data['date']).month

cat_cols = ['StationCode','shift','dow','month']
num_cols = ['entries','outage_roll14']

# ==== 3) Time-based split: earliest 70% dates → train; next 20% → val; newest 10% → test ====
data = data.sort_values('date')
unique_dates = np.array(sorted(data['date'].unique()))
n = len(unique_dates)
train_cut = unique_dates[:int(0.7*n)]
val_cut   = unique_dates[int(0.7*n):int(0.9*n)]
test_cut  = unique_dates[int(0.9*n):]

train = data[data['date'].isin(train_cut)]
val   = data[data['date'].isin(val_cut)]
test  = data[data['date'].isin(test_cut)]

X_train = train[cat_cols+num_cols]
y_train = train['y_next'].values
X_val   = val[cat_cols+num_cols]
y_val   = val['y_next'].values
X_test  = test[cat_cols+num_cols]
y_test  = test['y_next'].values

# ==== 4) Pipeline: OneHot + Standardize + Logistic Regression ====
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

logreg = LogisticRegression(
    penalty='l2', C=1.0, max_iter=200, solver='lbfgs', class_weight=None
)

pipe = Pipeline(steps=[('prep', preprocess),
                      ('clf', logreg)])

pipe.fit(X_train, y_train)

# ==== 5) Evaluation helpers ====
def evaluate(model, X, y, name='VAL'):
    proba = model.predict_proba(X)[:,1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y, proba)
    ap    = average_precision_score(y, proba)   # PR-AUC
    brier = brier_score_loss(y, proba)
    cm    = confusion_matrix(y, pred)
    print(f'[{name}] AUROC={auc:.3f} | PR-AUC={ap:.3f} | Brier={brier:.3f}')
    print(f'[{name}] Confusion Matrix:\n{cm}')
    print(f'[{name}] Classification Report:\n{classification_report(y, pred, digits=3)}')
    return {'auroc':auc, 'pr_auc':ap, 'brier':brier, 'cm':cm}

metrics_val  = evaluate(pipe, X_val,  y_val,  'VAL')
metrics_test = evaluate(pipe, X_test, y_test, 'TEST')

# ==== 6) Plots: ROC, PR, top |coef| ====
proba_test = pipe.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, proba_test)
prec, recall, _ = precision_recall_curve(y_test, proba_test)

plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.title('ROC (Test)'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.show()

plt.figure(); plt.plot(recall, prec)
plt.title('Precision–Recall (Test)'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.show()

# top coefficients (absolute value)
ohe = pipe.named_steps['prep'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feat_names = np.r_[cat_feature_names, np.array(num_cols)]
coef = pipe.named_steps['clf'].coef_.ravel()
top_idx = np.argsort(np.abs(coef))[::-1][:15]

plt.figure(figsize=(8,6))
sns.barplot(x=np.abs(coef[top_idx]), y=feat_names[top_idx], orient='h')
plt.title('Top |coef| Features — Logistic Regression')
plt.xlabel('|Coefficient|'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()
