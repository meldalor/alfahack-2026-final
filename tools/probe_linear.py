"""Offline comparison of additive baselines on the agent's unchanged temporal folds.
Usage: python tools/probe_linear.py <completed-run-directory>
Never reads hidden targets and never overwrites submission.csv.
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ag_features import prepare, detect_time_split
from ag_models import make_plan
from ag_task import build_spec, read_table
from ag_metrics import score

base=Path(sys.argv[1])
files=json.loads((base/'agent_work/spec.json').read_text())['files']
train,test=(read_table(Path(files[k])) for k in ('train','test'))
spec=build_spec(base/'task.md',train,test,None,None,files)
if spec.task_type!='regression': raise SystemExit('Regression probe only')
prep=prepare(train,test,spec)
plan=make_plan(prep.X,prep.y,spec.task_type,detect_time_split(prep.X,prep.Xt,prep.cat_cols))
cat=list(prep.cat_cols);num=[c for c in prep.X if c not in cat]
X=prep.X.copy()
for c in cat: X[c]=X[c].astype('string').fillna('__missing__').astype(str)
for name,model in [('ridge',Ridge(alpha=100)),('huber',HuberRegressor(alpha=.1,max_iter=500,tol=1e-4))]:
    vals=[];start=time.monotonic()
    for tr,va in plan.folds:
        pre=ColumnTransformer([
            ('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,min_frequency=10),cat),
        ])
        m=make_pipeline(pre,TransformedTargetRegressor(regressor=model,transformer=StandardScaler()))
        m.fit(X.iloc[tr],prep.y[tr]);p=m.predict(X.iloc[va])
        vals.append(score(spec.metric,spec.task_type,prep.y[va],p))
    print(json.dumps({'model':name,'metric':spec.metric,'fold_scores':vals,'mean':float(np.mean(vals)),
                      'seconds':round(time.monotonic()-start,2)}),flush=True)
