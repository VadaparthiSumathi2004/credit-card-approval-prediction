"""Train classifiers and export evaluation charts/metrics for the demo dashboard."""
from pathlib import Path
import json, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / 'clean_dataset.csv'
MODEL_DIR = BASE / 'models'; MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR = BASE / 'static' / 'generated'; CHART_DIR.mkdir(parents=True, exist_ok=True)
NUM = ['Debt','BankCustomer','YearsEmployed','PriorDefault','Employed','CreditScore','Income']
CAT = ['Industry']; FEATURES = NUM + CAT; TARGET = 'Approved'; SEED = 42

def load_data():
    df = pd.read_csv(DATA).replace('?', np.nan)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing: raise ValueError(f'Dataset missing columns: {missing}')
    df = df[FEATURES + [TARGET]].copy()
    for c in NUM + [TARGET]: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[TARGET]); df = df[df[TARGET].isin([0,1])]
    if df[TARGET].nunique() != 2: raise ValueError('Approved must contain both 0 and 1 classes.')
    return df

def preprocessor():
    return ColumnTransformer([
      ('numeric', Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]), NUM),
      ('categorical', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]), CAT)
    ])

def save_charts(results, best, best_pipeline, X_test, y_test):
    names=list(results); vals=[results[n]['accuracy'] for n in names]
    plt.figure(figsize=(8,4.5)); plt.bar(names, vals); plt.ylim(0,1); plt.ylabel('Accuracy'); plt.title('Model accuracy comparison'); plt.xticks(rotation=15,ha='right'); plt.tight_layout(); plt.savefig(CHART_DIR/'model_comparison.png',dpi=140); plt.close()
    pred=best_pipeline.predict(X_test); cm=confusion_matrix(y_test,pred,labels=[0,1])
    plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title(f'Confusion matrix — {best}'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.xticks([0,1],['Rejected','Approved']); plt.yticks([0,1],['Rejected','Approved'])
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    plt.tight_layout(); plt.savefig(CHART_DIR/'confusion_matrix.png',dpi=140); plt.close()
    from sklearn.metrics import RocCurveDisplay
    plt.figure(figsize=(5,4)); RocCurveDisplay.from_estimator(best_pipeline,X_test,y_test); plt.title(f'ROC curve — {best}'); plt.tight_layout(); plt.savefig(CHART_DIR/'roc_curve.png',dpi=140); plt.close()
    try:
        pi=permutation_importance(best_pipeline,X_test,y_test,n_repeats=8,random_state=SEED,scoring='roc_auc',n_jobs=1)
        imp=pd.Series(pi.importances_mean,index=FEATURES).sort_values()
        plt.figure(figsize=(7,4.5)); imp.plot(kind='barh'); plt.xlabel('Mean decrease in ROC-AUC'); plt.title('Permutation feature importance'); plt.tight_layout(); plt.savefig(CHART_DIR/'feature_importance.png',dpi=140); plt.close()
    except Exception as e: warnings.warn(f'Feature importance chart skipped: {e}')

def main():
    df=load_data(); X=df[FEATURES]; y=df[TARGET].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
    candidates={
      'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),
      'Decision Tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=4,class_weight='balanced',random_state=SEED),
      'Random Forest':RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)
    }
    try:
      from xgboost import XGBClassifier
      candidates['XGBoost']=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=1)
    except ImportError: warnings.warn('XGBoost unavailable; skipped.')
    results={}; fitted={}
    for name,clf in candidates.items():
      pipe=Pipeline([('preprocess',preprocessor()),('classifier',clf)]); pipe.fit(X_train,y_train)
      pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
      results[name]={'accuracy':round(float(accuracy_score(y_test,pred)),4),'precision':round(float(precision_score(y_test,pred,zero_division=0)),4),'recall':round(float(recall_score(y_test,pred,zero_division=0)),4),'f1':round(float(f1_score(y_test,pred,zero_division=0)),4),'roc_auc':round(float(roc_auc_score(y_test,prob)),4),'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist(),'classification_report':classification_report(y_test,pred,target_names=['Rejected','Approved'],output_dict=True,zero_division=0)}
      fitted[name]=pipe; print(f'{name}: accuracy={results[name]["accuracy"]:.3f}, F1={results[name]["f1"]:.3f}, AUC={results[name]["roc_auc"]:.3f}')
    best=max(results,key=lambda n:(results[n]['roc_auc'],results[n]['f1']))
    artifact={'pipeline':fitted[best],'model_name':best,'features':FEATURES,'threshold':.5,'target_mapping':{'0':'Rejected','1':'Approved'}}
    joblib.dump(artifact,MODEL_DIR/'credit_approval_pipeline.joblib')
    report={'best_model':best,'selection_metric':'ROC-AUC (F1 tie-breaker)','target_mapping':artifact['target_mapping'],'features':FEATURES,'test_rows':len(X_test),'dataset_rows':len(df),'results':results}
    (MODEL_DIR/'model_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_charts(results,best,fitted[best],X_test,y_test)
    print(f'Best model: {best}; dashboard charts saved to {CHART_DIR}')
if __name__=='__main__': main()
