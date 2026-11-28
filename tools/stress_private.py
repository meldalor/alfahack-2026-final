"""Private simulations. Run python tools/stress_private.py --out runs/private."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
CASES = {'fraud': ('PR-AUC','ROC-AUC'), 'multiclass': ('LogLoss','F1-macro'),
         'regression': ('RMSLE','MAPE'), 'text': ('PR-AUC',), 'large': ('MAE',),
         'snowflake_jsonb': ('MAE',), 'graph_comm': ('PR-AUC',), 'panel_30m': ('MAE',)}
METRIC = dict(zip(('PR-AUC','ROC-AUC','LogLoss','F1-macro','RMSLE','MAPE','MAE'),
                  ('average_precision','roc_auc','logloss','f1_macro','rmsle','mape','mae')))


def generate(name, n=4000, seed=42):
    r=np.random.default_rng(seed); k=int(n*.8)
    x=r.normal(size=n).astype('float32'); df=pd.DataFrame({'row_id':np.arange(n),'signal':x}); aux={}
    if name=='fraud':
        m=max(20,n//10); risk=r.normal(size=m); c=r.integers(m,size=n)
        t=pd.date_range('2025-01-01',periods=n,freq='17min')
        df['client_id'],df['timestamp']=c,t
        y=r.binomial(1,1/(1+np.exp(3-1.2*x-1.4*risk[c])))
        aux['clients']=pd.DataFrame({'id':np.arange(m),'risk_band':risk})
        aux['transactions']=pd.DataFrame({'client_id':np.repeat(c,2),'created_at':np.repeat(t.to_numpy(),2)+np.tile(np.array([-10,0],dtype='timedelta64[m]'),n),'amount':np.repeat(np.exp(risk[c]),2)+r.random(2*n)})
    elif name=='multiclass':
        p=np.array([16,14,13,12,10,9,8,7,5,3,2,.2]); p=p/p.sum(); q=p[::-1].copy(); q[-1]=.025; q=q/q.sum()
        y=np.r_[r.choice(12,k,p=p),r.choice(12,n-k,p=q)]; y[:12]=np.arange(12)
        df['signal']=(y+r.normal(0,.65,n)).astype('float32'); df['secondary']=(np.sin(y)+r.normal(0,.2,n)).astype('float32')
        df['segment']=r.choice(['A','B','C'],n)
    elif name=='regression':
        y=np.where(r.random(n)<.18,0,np.exp(2+1.4*x+r.normal(0,.45,n)))
        df['dirty_rate']=[f'{abs(v)*10:.1f}%'.replace('.',',') if i%7 else 'N/A' for i,v in enumerate(x)]
        df['group']=r.choice(['A','B','C'],n)
    elif name=='text':
        y=r.binomial(1,.35,n); topics=np.where(y,'подозрительная операция неизвестное списание','обычная заявка изменение адреса')
        words=r.choice(['срочно','пожалуйста','сегодня','вчера','повторно','клиент'],(n,5))
        df['description']=[f'{a} {" ".join(b)} прошу рассмотреть обращение номер {i}' for i,(a,b) in enumerate(zip(topics,words))]
        df['timestamp']=pd.date_range('2025-01-01',periods=n,freq='23min'); df['office_id']=r.integers(20,size=n)
        aux['offices']=pd.DataFrame({'office_id':np.arange(20),'capacity':r.integers(1,50,20)})
    elif name=='large':
        df['secondary']=r.normal(size=n).astype('float32'); df['group_id']=r.integers(1000,size=n,dtype='int32')
        aux['groups']=pd.DataFrame({'group_id':np.arange(1000,dtype='int32'),'offset':r.normal(size=1000).astype('float32')})
        y=(4*x+2*df.secondary.to_numpy()+aux['groups'].offset.to_numpy()[df.group_id]+r.normal(size=n)).astype('float32')
    elif name=='snowflake_jsonb':
        n_c,n_a,n_r=80,30,10
        reg_gdp=r.normal(5.0,2.0,n_r)
        acc_reg=r.integers(n_r,size=n_a); acc_lim=r.uniform(100,1000,n_a)
        crd_acc=r.integers(n_a,size=n_c); crd_tier=r.integers(1,5,n_c)
        c_idx=r.integers(n_c,size=n); a_idx=crd_acc[c_idx]; r_idx=acc_reg[a_idx]
        scores_j=np.round(r.normal(0,2.0,n),3); tiers_j=r.choice(['gold','silver','bronze'],n); vips_j=r.random(n)<0.3
        vocab=np.array(['tech','sports','music','food','travel','auto'])
        u_tags,o_tags,ov_cnt=[],[],np.zeros(n,dtype='float32')
        for i in range(n):
            tu=r.choice(vocab,size=int(r.integers(1,4)),replace=False)
            to=r.choice(vocab,size=int(r.integers(1,4)),replace=False)
            u_tags.append('{'+','.join(tu)+'}'); o_tags.append(';'.join(to))
            ov_cnt[i]=float(len(set(tu)&set(to)))
        df['card_ref']=[f'C{i}' for i in c_idx]
        df['meta_json']=[f'{{"score": {s}, "tier": "{t}", "vip": {str(bool(v)).lower()}}}' for s,t,v in zip(scores_j,tiers_j,vips_j)]
        df['user_tags']=u_tags; df['offer_tags']=o_tags
        aux['cards']=pd.DataFrame({'card_code':[f'C{i}' for i in range(n_c)],'acct_ref':[f'A{a}' for a in crd_acc],'card_tier':crd_tier})
        aux['accounts']=pd.DataFrame({'acct_code':[f'A{i}' for i in range(n_a)],'reg_ref':[f'R{rg}' for rg in acc_reg],'credit_limit':acc_lim})
        aux['regions']=pd.DataFrame({'reg_code':[f'R{i}' for i in range(n_r)],'gdp_idx':reg_gdp,'back_card':[f'C{i}' for i in range(n_r)]})
        aux['__fks__']=[
            {'src_table':'train','src_col':'card_ref','dst_table':'cards','dst_col':'card_code'},
            {'src_table':'cards','src_col':'acct_ref','dst_table':'accounts','dst_col':'acct_code'},
            {'src_table':'accounts','src_col':'reg_ref','dst_table':'regions','dst_col':'reg_code'},
            {'src_table':'regions','src_col':'back_card','dst_table':'cards','dst_col':'card_code'},
        ]
        y=(3.0*reg_gdp[r_idx]+1.5*scores_j+2.0*ov_cnt+0.5*x+r.normal(0,0.4,n)).astype('float32')
    elif name=='graph_comm':
        n_u,n_m=max(80,min(300,n//10)),max(30,min(100,n//25))
        u_comm=np.arange(n_u)%4; m_comm=np.arange(n_m)%4
        u_arr=r.integers(n_u,size=n); is_anom=(r.random(n)<0.18).astype(int)
        m_arr=np.empty(n,dtype=int)
        for i in range(n):
            cand=np.flatnonzero(m_comm!=u_comm[u_arr[i]]) if is_anom[i] else np.flatnonzero(m_comm==u_comm[u_arr[i]])
            m_arr[i]=int(r.choice(cand))
        df['user_id']=[f'U{i}' for i in u_arr]; df['merchant_id']=[f'M{i}' for i in m_arr]
        y=is_anom
    elif name=='panel_30m':
        n_g=4; n_slots=max(48*45,n)
        stamps=pd.date_range('2025-01-01 00:00:00',periods=n_slots,freq='30min')
        t_idx=np.arange(n_slots); diurn=4.0*np.sin(2*np.pi*t_idx/48.0)
        p_rows=[]; val_mat=np.zeros((n_g,n_slots),dtype='float32')
        for g in range(n_g):
            ar=np.zeros(n_slots,dtype='float64'); eps=r.normal(0,0.6,n_slots)
            for s_i in range(1,n_slots): ar[s_i]=0.85*ar[s_i-1]+eps[s_i]
            val_mat[g]=(diurn+ar+g*0.5).astype('float32')
            p_rows.append(pd.DataFrame({'district_id':f'G{g}','timestamp':stamps,'val':val_mat[g],'load':(val_mat[g]*0.7+r.normal(0,0.3,n_slots)).astype('float32')}))
        aux['sens']=pd.concat(p_rows,ignore_index=True)
        ev_slots=np.linspace(48*10,n_slots-2,n).astype(int)
        ev_g=r.integers(n_g,size=n); ev_min=r.integers(1,29,size=n)
        df['district_id']=[f'G{g}' for g in ev_g]
        df['timestamp']=stamps[ev_slots]+pd.to_timedelta(ev_min,unit='m')
        y=(2.5*val_mat[ev_g,ev_slots]+0.6*x+r.normal(0,0.4,n)).astype('float32')
    else: raise ValueError(name)
    tr,te=df.iloc[:k].copy(),df.iloc[k:].copy(); tr['target']=y[:k]
    sample=pd.DataFrame({'row_id':r.permutation(te.row_id.to_numpy())})
    for col in ([f'class_{i}' for i in range(12)] if name=='multiclass' else ['target']): sample[col]=0.
    return tr,te,aux,sample,np.asarray(y[k:])


def validate(pred,sample,te,truth,name):
    from sklearn.metrics import average_precision_score,roc_auc_score,log_loss,f1_score,mean_absolute_error
    if list(pred)!=list(sample) or len(pred)!=len(sample) or not pred.row_id.is_unique or not pred.row_id.equals(sample.row_id): raise ValueError('wrong schema or ID order')
    p=pred.drop(columns='row_id').to_numpy(float); y=pd.Series(truth,index=te.row_id).loc[pred.row_id].to_numpy()
    if not np.isfinite(p).all(): raise ValueError('NaN/Inf predictions')
    if name in ('fraud','text','multiclass','graph_comm') and ((p<0).any() or (p>1).any()): raise ValueError('invalid probabilities')
    if name=='multiclass':
        if not np.allclose(p.sum(1),1,atol=1e-6): raise ValueError('probabilities do not sum to 1')
        scores={'LogLoss':log_loss(y,p,labels=np.arange(12)),'F1-macro':f1_score(y,p.argmax(1),labels=np.arange(12),average='macro',zero_division=0)}
    elif name in ('fraud','text','graph_comm'): scores={'PR-AUC':average_precision_score(y,p[:,0]),'ROC-AUC':roc_auc_score(y,p[:,0])}
    elif name=='regression':
        if (p<0).any(): raise ValueError('negative predictions')
        scores={'RMSLE':np.sqrt(np.mean((np.log1p(y)-np.log1p(p[:,0]))**2)),'MAPE':np.mean(np.abs(y-p[:,0])/np.maximum(abs(y),1e-9))}
    else: scores={'MAE':mean_absolute_error(y,p[:,0])}
    return {k:float(v) for k,v in scores.items()},bool(np.all(np.ptp(p,axis=0)<1e-10))


def dump(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def worker(args):
    if hasattr(os, 'sched_setaffinity'):
        cpus=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,cpus[:args.threads])
    from ag_relational import enrich
    from ag_features import prepare,op_text_oof,detect_time_split
    from ag_task import build_spec
    from unittest.mock import patch
    import ag_database,solution,importlib.util,gc
    d=args.out; name=args.scenario; metric=args.metric
    tr,te,aux,sample,y=generate(name,args.rows,args.seed)
    fks=aux.pop('__fks__',None)
    task_type='Multiclass classification; probabilities class_0 through class_11' if name=='multiclass' else 'Binary classification; probability of class 1' if name in ('fraud','text','graph_comm') else 'Regression'
    (d/'task.md').write_text(f'ID: row_id. Target: target. {task_type}. Основная метрика: {metric}.\n',encoding='utf-8')
    issues=[]; checks={'issues':issues,'train_rows':len(tr),'test_rows':len(te)}
    def check(ok,msg):
        if not ok: issues.append(msg)
    rel_meta={'foreign_keys':list(fks)} if fks else {}
    a,b=enrich(tr,te,aux,meta=rel_meta)
    check(a.row_id.equals(tr.row_id.reset_index(drop=True)) and b.row_id.equals(te.row_id.reset_index(drop=True)),'join changed rows/order')
    checks['enriched_columns']=len(b.columns)
    if name=='fraud':
        check('clients__risk_band' in a,'mismatched key client_id -> clients.id not joined')
        check('transactions__hist_count' in a,'timed child aggregation absent')
        if 'transactions__hist_count' in b:
            expected=((aux['transactions'].client_id==te.client_id.iloc[0])&(aux['transactions'].created_at<te.timestamp.iloc[0])).sum()
            check(b.transactions__hist_count.iloc[0]==expected,'child count differs from strict past oracle')
    if name=='snowflake_jsonb':
        for req_c in ('cards__card_tier','accounts__credit_limit','regions__gdp_idx'):
            check(req_c in a,f'3-hop snowflake column missing: {req_c}')
    if name!='large':
        spec=build_spec(d/'task.md',a,b,sample,None,{})
        check(spec.metric==METRIC[metric],f'wrong metric parsed: {spec.metric}')
        if name=='multiclass':
            check(spec.output_kind=='proba_multi' and len(spec.classes)==12,'12-class contract not recognized')
            checks['train_class_counts']={str(i):int(v) for i,v in tr.target.value_counts().items()}; checks['test_class_counts']={str(i):int(v) for i,v in pd.Series(y).value_counts().items()}
        prep=prepare(a,b,spec)
        if name=='regression':
            check('dirty_rate' in prep.X and 'dirty_rate' not in prep.cat_cols,'dirty number not numeric')
            if 'dirty_rate' in prep.X: check(pd.isna(prep.X.dirty_rate.iloc[0]) and np.isclose(prep.X.dirty_rate.iloc[1],round(abs(tr.signal.iloc[1])*10,1)),'dirty number parsed incorrectly')
        if name=='snowflake_jsonb':
            check('meta_json__json__score' in prep.X,'flat JSONB key not unpacked')
            check('user_tags__ov__offer_tags' in prep.X,'Postgres array / semicolon tag overlap missing')
        if name=='graph_comm':
            gnn_cols=[c for c in prep.X.columns if c.startswith('gnn__')]
            checks['gnn_features']=len(gnn_cols)
            check(len(gnn_cols)>=5,'2-fold OOF Zero-GNN did not activate on community graph')
        if name=='panel_30m':
            from ag_models import make_plan
            from ag_nowcast import build_nowcast_features
            tc=detect_time_split(prep.X,prep.Xt,prep.cat_cols); check(tc is not None,'30m panel time split not detected')
            if tc:
                pl=make_plan(prep.X,prep.y,'regression',time_col=tc)
                nc_tr1,nc_te1=build_nowcast_features(rel_meta,prep.y,pl,'regression',n_jobs=args.threads,use_lgb_top=False,max_seconds=30.0)
                check(nc_te1 is not None and 'nc__sens__val' in nc_te1,'30m nowcast did not fire')
                if nc_te1 is not None:
                    sens_p=aux['sens'].copy()
                    t_cut=pd.to_datetime(te['timestamp']).dt.floor('30min').min()
                    sens_p.loc[pd.to_datetime(sens_p['timestamp'])>=t_cut,'val']=99999.0
                    rm2={}; enrich(tr,te,{'sens':sens_p},meta=rm2)
                    _,nc_te2=build_nowcast_features(rm2,prep.y,pl,'regression',n_jobs=args.threads,use_lgb_top=False,max_seconds=30.0)
                    delta=float(np.nanmax(np.abs(nc_te1.iloc[0].to_numpy(float)-nc_te2.iloc[0].to_numpy(float))))
                    checks['future_panel_poison_delta']=delta; check(delta<1e-5,'30m nowcast leaked future panel')
        if name=='text':
            tc=detect_time_split(prep.X,prep.Xt,prep.cat_cols); check(tc is not None and bool(prep.raw_text),'time split/text not detected')
            if tc and prep.raw_text:
                periods=np.repeat(np.arange(4),int(np.ceil(len(a)/4)))[:len(a)]
                def op(raw,labels): return op_text_oof(raw,labels,spec.task_type,len(spec.classes),periods=periods)
                oa,ob=op(prep.raw_text,prep.y); check(np.isfinite(oa.to_numpy()).all() and len(ob)==len(b),'invalid text OOF')
                changed=prep.y.copy(); changed[periods>=2]=1-changed[periods>=2]; pa,_=op(prep.raw_text,changed)
                check(np.allclose(oa.loc[periods<=2],pa.loc[periods<=2]),'future target leakage in text OOF')
                raw={key:(s.mask(periods>=2,'другой будущий текст '*10),t.map(lambda _: 'новый словарь '*10)) for key,(s,t) in prep.raw_text.items()}
                qa,_=op(raw,prep.y); delta=float(np.max(np.abs(oa.loc[periods==1].to_numpy()-qa.loc[periods==1].to_numpy())))
                checks['future_text_poison_delta']=delta; check(delta<1e-7,'future text changes past OOF (vocabulary/IDF leakage)')
        del prep
    if importlib.util.find_spec('pyarrow') is None:
        checks['input_mode']='enrich_csv'; a.to_csv(d/'train.csv',index=False); b.to_csv(d/'test.csv',index=False); sample.to_csv(d/'sample_submission.csv',index=False)
        dump(d/'checks.json',checks); del a,b,tr,te,aux,sample,y; gc.collect(); solution.run(d/'task.md')
    else:
        checks['input_mode']='mock_extract_parquet'; dump(d/'checks.json',checks); del a,b,y; gc.collect()
        os.environ['PGDATABASE']='synthetic_private'
        def _mock_extract(out_dir, meta=None, **kw):
            if isinstance(meta, dict) and fks:
                meta['foreign_keys'] = list(fks)
            return {'train':tr,'test':te,'sample_submission':sample,**aux}
        with patch.object(ag_database,'extract',side_effect=_mock_extract): solution.run(d/'task.md')


def run_case(args,name,metric):
    d=args.out/f'{name}_{metric.lower()}'; d.mkdir(exist_ok=False); n=args.large_rows if name=='large' else args.rows
    env={k:v for k,v in os.environ.items() if not k.startswith(('PG','OPENAI_','AGENT_'))}
    env.update(AGENT_TIME_BUDGET=str(args.budget),AGENT_HARD_LIMIT=str(args.hard_limit if args.hard_limit is not None else args.budget+20))
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[var]=str(args.threads)
    cmd=[sys.executable,'-m','tools.stress_private','--worker','--scenario',name,'--metric',metric,'--out',str(d),'--rows',str(n),'--seed',str(args.seed),'--threads',str(args.threads)]
    start=time.monotonic(); peak=0; timed_out=False
    with (d/'stdout.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); p=psutil.Process(proc.pid)
        while proc.poll() is None:
            rss=0
            try:
                for ch in [p]+p.children(recursive=True):
                    try: rss+=ch.memory_info().rss
                    except psutil.Error: pass
            except psutil.Error: pass
            peak=max(peak,rss)
            if time.monotonic()-start>args.budget+args.grace:
                timed_out=True; os.killpg(proc.pid,signal.SIGKILL); break
            time.sleep(.1)
        rc=proc.wait()
    res={'scenario':name,'metric':metric,'seconds':round(time.monotonic()-start,2),'peak_rss_gib':round(peak/2**30,4),'exit_code':rc,'timeout':timed_out,'issues':[]}
    try:
        res.update(json.loads((d/'checks.json').read_text())); _,te,_,sample,y=generate(name,n,args.seed)
        scores,constant=validate(pd.read_csv(d/'submission.csv'),sample,te,y,name); res.update(scores=scores,constant=constant,submission_valid=True)
        if constant: res['issues'].append('constant/fallback only')
        spec=json.loads((d/'agent_work/spec.json').read_text());
        if spec['metric']!=METRIC[metric]: res['issues'].append('agent metric mismatch')
        f=d/'agent_work/final.json'
        if f.exists(): res['final']=json.loads(f.read_text())
        else: res['issues'].append('final ensemble not confirmed (final.json absent)')
    except Exception as exc: res['issues'].append(f'{type(exc).__name__}: {exc}'); res.setdefault('submission_valid',False)
    if rc or timed_out: res['issues'].append('process failed or external timeout')
    if peak>=24*2**30: res['issues'].append('peak RSS >=24 GiB')
    logtext=(d/'stdout.log').read_text(errors='replace')
    res['watchdog']='WATCHDOG:' in logtext
    if res['watchdog']: res['issues'].append('watchdog stopped agent before normal completion')
    res['logged_errors']=[l for l in logtext.splitlines() if 'ОШИБКА' in l or 'Traceback (most recent call last)' in l][:30]
    if res['logged_errors']: res['issues'].append('logged errors; inspect stdout.log')
    res['status']='FAIL' if res['issues'] else 'PASS'; dump(d/'result.json',res); print(json.dumps(res,ensure_ascii=False),flush=True); return res


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=ROOT/'runs/private'); p.add_argument('--budget',type=float,default=90); p.add_argument('--grace',type=float,default=90,help='external timeout margin after budget'); p.add_argument('--hard-limit',type=float,help='agent watchdog seconds (default budget+20)'); p.add_argument('--threads',type=int,default=4); p.add_argument('--rows',type=int,default=4000); p.add_argument('--large-rows',type=int,default=2_000_000); p.add_argument('--seed',type=int,default=42); p.add_argument('--scenario',choices=CASES); p.add_argument('--metric'); p.add_argument('--primary-only',action='store_true'); p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(); a.out=a.out.resolve()
    if min(a.rows,a.large_rows)<500 or a.budget<=0 or a.threads<1 or a.grace<21: p.error('invalid resource limits')
    if a.hard_limit is not None and not a.budget<a.hard_limit<a.budget+a.grace: p.error('hard limit must lie between budget and external deadline')
    if a.metric and (not a.scenario or a.metric not in CASES[a.scenario]): p.error('metric requires matching scenario')
    if a.worker: worker(a); return
    a.out.mkdir(parents=True,exist_ok=True); results=[]
    for name in ([a.scenario] if a.scenario else CASES):
        for metric in ([a.metric] if a.metric else CASES[name][:1] if a.primary_only else CASES[name]):
            results.append(run_case(a,name,metric)); dump(a.out/'summary.json',results)
    sys.exit(int(any(r['status']!='PASS' for r in results)))

if __name__=='__main__': main()
