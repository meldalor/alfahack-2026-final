"""Regression checks for final-stage relational input and submission contract."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ag_relational import enrich
from ag_task import TaskSpec, parse_metric, build_spec
from ag_models import make_plan, lgb_objective
from solution import SubmissionWriter, HARD_LIMIT, TIME_BUDGET


class FinalTests(unittest.TestCase):
    def test_transitive_join_and_order(self):
        tr = pd.DataFrame({'event_id':[9, 2], 'ad_id':[20, 10], 'y':[1, 0]})
        te = pd.DataFrame({'event_id':[8], 'ad_id':[10]})
        tables = {'campaigns':pd.DataFrame({'campaign_id':[100, 200], 'budget':[3, 8]}),
                  'ads':pd.DataFrame({'ad_id':[10,20], 'campaign_id':[100,200]})}
        a, b = enrich(tr, te, tables)
        self.assertEqual(a.event_id.tolist(), [9,2])
        self.assertEqual(a.campaigns__budget.tolist(), [8,3])
        self.assertEqual(b.campaigns__budget.tolist(), [3])
        self.assertNotIn('y', b)

    def test_composite_key(self):
        tr = pd.DataFrame({'district_id':['a','a'], 'timestamp':['t1','t2'], 'y':[1,2]})
        te = pd.DataFrame({'district_id':['b'], 'timestamp':['t2']})
        dim = pd.DataFrame({'district_id':['a','a','b','b'], 'timestamp':['t1','t2','t1','t2'], 'v':[1,2,3,4]})
        a,b = enrich(tr,te,{'weather':dim})
        self.assertEqual(a.weather__v.tolist(), [1,2])
        self.assertEqual(b.weather__v.tolist(), [4])

    def test_duplicate_key_is_not_joined(self):
        tr=pd.DataFrame({'k_id':[1,2], 'y':[0,1]});te=tr.drop(columns='y')
        a,b=enrich(tr,te,{'bad':pd.DataFrame({'k_id':[1,1], 'value':[2,3]})})
        self.assertEqual(len(a),2);self.assertNotIn('bad__value',a)

    def test_submission_template_order(self):
        spec=TaskSpec('', '', id_col='event_id',target='y',sub_columns=['event_id','y'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'submission.csv'
            writer=SubmissionWriter(path,spec,pd.DataFrame({'event_id':[3,1,2]}),pd.DataFrame({'event_id':[2,3,1],'y':[0,0,0]}))
            writer.write(np.array([30.,10.,20.]))
            result=pd.read_csv(path)
            self.assertEqual(result.event_id.tolist(),[2,3,1])
            self.assertEqual(result.y.tolist(),[20,30,10])

    def test_id_without_template(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'task.md';path.write_text('Target: y. ID: event_id. Metric: MAE.')
            tr=pd.DataFrame({'event_id':[1,2,3], 'y':[1.,2.,3.]})
            te=pd.DataFrame({'event_id':[5,4]})
            spec=build_spec(path,tr,te,None,None,{})
            self.assertEqual(spec.id_col,'event_id')
            self.assertEqual(spec.sub_columns,['event_id','y'])

    def test_dirty_dimension_retains_only_unambiguous_keys(self):
        dim=pd.DataFrame({'k_id':list(range(100))+[0], 'v':list(range(100))+[999]})
        tr=pd.DataFrame({'k_id':[0,1], 'y':[0,1]});te=pd.DataFrame({'k_id':[2]})
        a,b=enrich(tr,te,{'dim':dim})
        self.assertTrue(pd.isna(a.dim__v.iloc[0]))
        self.assertEqual(a.dim__v.iloc[1],1)
        self.assertEqual(b.dim__v.iloc[0],2)

    def test_temporal_lookup_is_backward_bounded_and_keeps_order(self):
        dim=pd.DataFrame({'district_id':['a']*4, 'timestamp':pd.date_range('2025-01-01',periods=4,freq='h').astype(str), 'v':[10,20,30,40]})
        # Interval [t, t+1h) completes at t+1h: at 02:15 last completed is 01:00 (20), NEVER 02:00 (30)!
        tr=pd.DataFrame({'district_id':['a','a','a'], 'timestamp':['2025-01-01 02:15:00','2025-01-01 00:30:00','2025-01-01 01:00:00'], 'y':[0,1,2]})
        te=pd.DataFrame({'district_id':['a','a','a'], 'timestamp':['2025-01-01 04:00:00','2025-01-01 03:59:00','2025-01-01 05:00:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertEqual(a.sensor__v.iloc[0],20)      # 02:15 -> 01:00-02:00 completed (20)
        self.assertTrue(pd.isna(a.sensor__v.iloc[1])) # 00:30 -> 00:00-01:00 NOT completed yet -> NaN
        self.assertEqual(a.sensor__v.iloc[2],10)      # 01:00 -> 00:00-01:00 just completed (10)
        self.assertEqual(b.sensor__v.iloc[0],40)      # 04:00 -> 03:00-04:00 just completed (40)
        self.assertEqual(b.sensor__v.iloc[1],30)      # 03:59 -> 02:00-03:00 completed (30), NOT 03:00!
        self.assertTrue(pd.isna(b.sensor__v.iloc[2])) # 05:00 -> 03:00-04:00 is >= 1h stale -> NaN
        self.assertEqual(a.timestamp.tolist(),tr.timestamp.tolist())

    def test_temporal_lookup_does_not_cross_groups(self):
        stamps=pd.date_range('2025-01-01',periods=4,freq='h').astype(str).tolist()
        dim=pd.DataFrame({'district_id':['a']*4+['b']*4,'timestamp':stamps*2,'v':[1,2,3,4,101,102,103,104]})
        tr=pd.DataFrame({'district_id':['b','a'],'timestamp':['2025-01-01 02:10:00']*2,'y':[0,1]})
        te=pd.DataFrame({'district_id':['c'],'timestamp':['2025-01-01 02:10:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertEqual(a.sensor__v.tolist(),[102,2])  # 02:10 -> completed 01:00-02:00 (102 / 2)
        self.assertTrue(pd.isna(b.sensor__v.iloc[0]))

    def test_duplicate_composite_keys_are_removed(self):
        stamps=pd.date_range('2025-01-01',periods=60,freq='h').astype(str).tolist()
        dim=pd.DataFrame({'district_id':['a']*60+['b']*60,'timestamp':stamps*2,'v':list(range(120))})
        dim=pd.concat([dim,pd.DataFrame({'district_id':['a'],'timestamp':[stamps[0]],'v':[999]})],ignore_index=True)
        tr=pd.DataFrame({'district_id':['a','b'],'timestamp':[stamps[0]]*2,'y':[0,1]})
        te=pd.DataFrame({'district_id':['a'],'timestamp':[stamps[1]]})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertTrue(pd.isna(a.sensor__v.iloc[0]))
        self.assertEqual(a.sensor__v.iloc[1],60)
        self.assertEqual(b.sensor__v.iloc[0],1)

    def test_full_datetime_preferred_over_day_of_year(self):
        from ag_features import detect_time_split
        a=pd.DataFrame({'when__doy':[1,2,3], 'when__ord':[1000,1001,1002]})
        b=pd.DataFrame({'when__doy':[4,5], 'when__ord':[1003,1004]})
        self.assertEqual(detect_time_split(a,b,[]),'when__ord')

    def test_text_features_cannot_see_same_or_future_period_labels(self):
        from ag_features import op_text_oof
        text=pd.Series(['apple bright good' if i%2 else 'lemon dark sour' for i in range(120)])
        raw={'review':(text,pd.Series(['apple bright good','lemon dark sour']))}
        periods=np.repeat(np.arange(3),40)
        y=np.arange(120,dtype=float)%7
        a,_=op_text_oof(raw,y,'regression',1,periods=periods)
        changed=y.copy();changed[40:]+=1000
        b,_=op_text_oof(raw,changed,'regression',1,periods=periods)
        np.testing.assert_allclose(a.iloc[:80],b.iloc[:80])
        self.assertGreater(np.abs(a.iloc[80:].values-b.iloc[80:].values).mean(),100)
        # Early periods may have one observed class; class columns must remain aligned.
        classes=np.repeat([0,1,2],40)
        c,t=op_text_oof(raw,classes,'multiclass',3,periods=periods)
        self.assertEqual(c.shape,(120,3))
        np.testing.assert_allclose(c.iloc[40:80,0],1.)
        np.testing.assert_allclose(c.sum(axis=1),1.)
        np.testing.assert_allclose(t.sum(axis=1),1.,rtol=1e-6)

    def test_subsample_preserves_original_time_boundaries(self):
        from ag_models import subset_plan
        X=pd.DataFrame({'t':np.arange(1000)})
        plan=make_plan(X,np.arange(1000),'regression','t')
        rows=np.sort(np.random.default_rng(3).choice(1000,600,replace=False))
        small=subset_plan(plan,rows)
        for (tr,va),(st,sv) in zip(plan.folds,small.folds):
            np.testing.assert_array_equal(rows[st],np.intersect1d(rows,tr))
            np.testing.assert_array_equal(rows[sv],np.intersect1d(rows,va))
        np.testing.assert_array_equal(small.oof_mask,plan.oof_mask[rows])

    def test_database_task_does_not_discover_neighbour_template(self):
        from ag_task import find_files
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);own=root/'database_task';peer=root/'other_task'
            own.mkdir();peer.mkdir()
            task=own/'task.md';task.write_text('Read PostgreSQL database')
            (peer/'sample_submission.csv').write_text('id,y\n1,0\n')
            self.assertEqual(find_files(task,task.read_text(),search_parent=False),{})
            (own/'sample_submission.csv').write_text('id,y\n2,0\n')
            self.assertEqual(Path(find_files(task,task.read_text(),search_parent=False)['sample']),own/'sample_submission.csv')

    def test_metric_and_deadline(self):
        self.assertEqual(parse_metric('Метрика: PR-AUC')[0],'average_precision')
        self.assertEqual(lgb_objective('binary','average_precision',2)['metric'],'average_precision')
        self.assertLess(TIME_BUDGET,HARD_LIMIT);self.assertLess(HARD_LIMIT,7200)

    def test_chronological_validation(self):
        X=pd.DataFrame({'t':np.repeat(np.arange(100),10)})
        plan=make_plan(X,np.arange(len(X)),'regression','t')
        self.assertEqual(plan.mode,'refit')
        for tr,va in plan.folds:
            self.assertLess(X.t.iloc[tr].max(),X.t.iloc[va].min())

    def test_temporal_ffill_uses_only_past_values(self):
        stamps=pd.date_range('2025-01-01',periods=5,freq='h').astype(str).tolist()
        dim=pd.DataFrame({'district_id':['a']*5,'timestamp':stamps,'v':[np.nan,10.,np.nan,np.nan,99.]})
        tr=pd.DataFrame({'district_id':['a','a'],'timestamp':['2025-01-01 01:30:00','2025-01-01 03:30:00'],'y':[0.,1.]})
        te=pd.DataFrame({'district_id':['a'],'timestamp':['2025-01-01 04:30:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertTrue(pd.isna(a.sensor__v.iloc[0]))  # 01:30 -> 00:00 is NaN, never uses future 01:00 (10.0)!
        self.assertEqual(a.sensor__v.iloc[1],10.)      # 03:30 -> 02:00 outage filled from completed 01:00 (10.0)
        self.assertEqual(b.sensor__v.iloc[0],10.)      # 04:30 -> 03:00 outage filled from completed 01:00, never 04:00 (99.0)!
        self.assertEqual(a.sensor__raw_nulls.iloc[1],1.)

    def test_interval_join_matches_active_events_only(self):
        tr=pd.DataFrame({'event_id':['E1','E2'],'district_id':['d1','d1'],
                         'timestamp':['2025-01-01 10:30:00','2025-01-01 15:00:00'],'y':[1.,2.]})
        te=pd.DataFrame({'event_id':['E3'],'district_id':['d2'],'timestamp':['2025-01-01 10:30:00']})
        ev=pd.DataFrame({'event_id':['C1'],'district_id':['d1'],
                         'start_timestamp':['2025-01-01 10:00:00'],'end_timestamp':['2025-01-01 12:00:00'],
                         'attendance':[5000],'impact':['High']})
        a,b=enrich(tr,te,{'city_events':ev})
        self.assertEqual(a.city_events__active_count.tolist(),[1.,0.])
        self.assertEqual(a.city_events__attendance.iloc[0],5000.)
        self.assertEqual(b.city_events__active_count.iloc[0],0.)

    def test_fit_linear_regression_and_binary(self):
        from ag_models import fit_linear, ModelConfig
        rng=np.random.default_rng(0)
        X=pd.DataFrame({'a':rng.normal(size=120),'b':rng.normal(size=120),
                        'c':pd.Categorical(rng.choice(['x','y','z'],120))})
        Xt=X.iloc[:20].copy()
        y_reg=2.5*X['a'].values - 1.5*X['b'].values + 5.0
        spec_r=TaskSpec('','','id','y',task_type='regression',metric='mae')
        plan_r=make_plan(X,y_reg,'regression',n_folds=3)
        for kind in ('ridge','lasso','huber'):
            res=fit_linear(X,y_reg,Xt,plan_r,spec_r,ModelConfig(),n_jobs=2,time_limit=60,kind=kind)
            self.assertIsNotNone(res)
            self.assertLess(res.score,0.2)
            self.assertEqual(len(res.test),len(Xt))
        y_cls=(y_reg>5.0).astype(int)
        spec_c=TaskSpec('','','id','y',task_type='binary',metric='average_precision',classes=[0,1])
        plan_c=make_plan(X,y_cls,'binary',n_folds=3)
        res_c=fit_linear(X,y_cls,Xt,plan_c,spec_c,ModelConfig(),n_jobs=2,time_limit=60,kind='logreg')
        self.assertIsNotNone(res_c)
        self.assertGreater(res_c.score,0.95)
        self.assertEqual(res_c.test.shape,(len(Xt),2))

    def test_one_to_many_static_and_causal_aggregation(self):
        # Static one-to-many child table (no timestamps)
        tr=pd.DataFrame({'event_id':[1,2],'user_id':['u1','u2'],'y':[0,1]})
        te=pd.DataFrame({'event_id':[3],'user_id':['u1']})
        hist=pd.DataFrame({'user_id':['u1','u1','u2'],'amount':[10.,30.,5.]})
        a,b=enrich(tr,te,{'orders':hist})
        self.assertEqual(a.orders__count.tolist(),[2.,1.])
        self.assertEqual(a.orders__amount__mean.tolist(),[20.,5.])
        self.assertEqual(b.orders__amount__max.iloc[0],30.)
        # Causal temporal one-to-many: strictly child_time < event_time (no future leakage)
        tr2=pd.DataFrame({'event_id':[1,2],'user_id':['u1','u1'],
                          'timestamp':['2025-01-02 00:00:00','2025-01-04 00:00:00'],'y':[0,1]})
        te2=pd.DataFrame({'event_id':[3],'user_id':['u1'],'timestamp':['2025-01-05 00:00:00']})
        tx=pd.DataFrame({'user_id':['u1','u1','u1'],
                         'timestamp':['2025-01-01 00:00:00','2025-01-03 00:00:00','2025-01-06 00:00:00'],
                         'amt':[10.,50.,999.]})
        a2,b2=enrich(tr2,te2,{'tx':tx})
        self.assertEqual(a2.tx__hist_count.tolist(),[1.,2.])
        self.assertEqual(a2.tx__amt__hist_mean.tolist(),[10.,30.])
        self.assertEqual(b2.tx__amt__hist_max.iloc[0],50.)  # future 999.0 at Jan 06 excluded!

    def test_cross_impute_outages(self):
        from ag_features import cross_impute_outages
        rng=np.random.default_rng(7)
        n=400
        z=rng.normal(size=n)
        X=pd.DataFrame({'a__x':(2.0*z+1.0).astype('float32'),'a__raw_nulls':np.zeros(n,dtype='float32'),
                        'b__z':z.astype('float32')})
        Xt=pd.DataFrame({'a__x':np.array([0.0,0.0],dtype='float32'),'a__raw_nulls':np.array([1.0,0.0],dtype='float32'),
                         'b__z':np.array([1.5,1.5],dtype='float32')})
        X2,Xt2=cross_impute_outages(X,Xt,[])
        # Row 0 in Xt has raw_nulls=1 -> reconstructed toward 0.7*(2*1.5+1)+0.3*0 = 2.8; row 1 untouched (0.0)
        self.assertGreater(Xt2.a__x.iloc[0],2.5)
        self.assertEqual(Xt2.a__x.iloc[1],0.0)

    def test_resolve_db_tables_nonstandard_names(self):
        from solution import _resolve_db_tables
        tables={'events_train':pd.DataFrame(),'events_test':pd.DataFrame(),
                'sample_sub':pd.DataFrame(),'metadata':pd.DataFrame(),'users':pd.DataFrame()}
        rm=_resolve_db_tables(tables,'Predict churn on events_train and events_test')
        self.assertEqual(rm['train'],'events_train')
        self.assertEqual(rm['test'],'events_test')
        self.assertEqual(rm['sample'],'sample_sub')
        self.assertEqual(rm['dictionary'],'metadata')

    def test_join_mismatched_key_dtypes_and_custom_time_names(self):
        # int64 key in train/test vs padded string key in dimension + custom created_at time column (strictly completed!)
        tr=pd.DataFrame({'event_id':[1,2],'user_id':[10,20],'created_at':['2025-01-01 02:30:00','2025-01-01 03:30:00'],'y':[0,1]})
        te=pd.DataFrame({'event_id':[3],'user_id':[10],'created_at':['2025-01-01 04:30:00']})
        users=pd.DataFrame({'user_id':[' 10 ','20'],'tier':['gold','silver']})
        stamps=pd.date_range('2025-01-01',periods=4,freq='h').astype(str).tolist()
        sensor=pd.DataFrame({'user_id':['10']*4,'created_at':stamps,'val':[1.,2.,3.,4.]})
        a,b=enrich(tr,te,{'users':users,'sensor':sensor})
        self.assertEqual(a.users__tier.tolist(),['gold','silver'])
        self.assertEqual(a.sensor__val.iloc[0],2.)  # 02:30 -> completed 01:00-02:00 (2.0), NOT 02:00 (3.0)
        self.assertEqual(b.sensor__val.iloc[0],4.)  # 04:30 -> completed 03:00-04:00 (4.0)

    def test_dirty_numeric_strings_and_date_diffs(self):
        from ag_features import prepare
        tr=pd.DataFrame({'id':[1,2,3,4,5,6],
                         'price':['10,5','12,0','N/A','null','15,5','20,0'],
                         'pct':['10%','20%','-','40%','50%','60%'],
                         'timestamp':['2025-05-10']*6,
                         'start_date':['2025-05-01','2025-05-05','2025-05-02','2025-05-03','2025-05-04','2025-05-06'],
                         'y':[1.,2.,3.,4.,5.,6.]})
        te=tr.drop(columns=['y']).iloc[:2].copy()
        spec=TaskSpec('','',id_col='id',target='y',task_type='regression',metric='mae')
        prep=prepare(tr,te,spec)
        self.assertTrue(pd.api.types.is_numeric_dtype(prep.X['price']))
        self.assertTrue(pd.api.types.is_numeric_dtype(prep.X['pct']))
        self.assertIn('timestamp__minus__start_date',prep.X.columns)
        self.assertAlmostEqual(float(prep.X['timestamp__minus__start_date'].iloc[0]),9.0,places=3)

    def test_multiclass_prefix_proba_and_rare_class_cv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'task.md';p.write_text('Multiclass classification, metric: logloss')
            tr=pd.DataFrame({'id':range(11),'x':range(11),'y':[0,0,0,0,0,1,1,1,1,1,2]})  # class 2 has 1 member!
            te=pd.DataFrame({'id':[100,101],'x':[1,2]})
            smp=pd.DataFrame({'id':[101,100],'class_0':[0.,0.],'class_1':[0.,0.],'class_2':[0.,0.]})
            spec=build_spec(p,tr,te,smp,None,{})
            self.assertEqual(spec.output_kind,'proba_multi')
            plan=make_plan(tr[['x']],tr['y'].values,'multiclass',n_folds=3)
            self.assertEqual(len(plan.folds),3)
            out_p=Path(d)/'submission.csv'
            w=SubmissionWriter(out_p,spec,te,smp)
            w.write(np.array([[0.7,0.2,0.1],[0.1,0.3,0.6]]))
            res=pd.read_csv(out_p)
            self.assertEqual(res.id.tolist(),[101,100])
            self.assertAlmostEqual(float(res.class_2.iloc[0]),0.6)

    def test_auc_pr_parsing_and_rank_blend(self):
        from ag_ensemble import rank_blend
        from ag_models import FitResult
        self.assertEqual(parse_metric('Основная метрика: AUC-PR')[0],'average_precision')
        self.assertEqual(parse_metric('Metric: precision-recall AUC')[0],'average_precision')
        y=np.array([0,0,0,0,1,1,1,1])
        m=np.ones(8,bool)
        r1=FitResult('m1',np.column_stack([1-np.linspace(0.01,0.09,8),np.linspace(0.01,0.09,8)]),
                     np.array([[0.9,0.1],[0.2,0.8]]),0.9,[],[],None,1.0)
        r2=FitResult('m2',np.column_stack([1-np.linspace(0.1,0.9,8),np.linspace(0.1,0.9,8)]),
                     np.array([[0.8,0.2],[0.1,0.9]]),0.9,[],[],None,1.0)
        spec=TaskSpec('','','id','y',task_type='binary',metric='average_precision',classes=[0,1])
        rb=rank_blend([r1,r2],y,m,spec)
        self.assertIsNotNone(rb)
        self.assertTrue(((rb['test']>=0)&(rb['test']<=1)).all())

    def test_causal_rfm_and_db_schema_report(self):
        tr=pd.DataFrame({'event_id':[1,2],'user_id':['u1','u1'],
                         'timestamp':['2025-01-02 00:00:00','2025-01-05 00:00:00'],'y':[0,1]})
        te=pd.DataFrame({'event_id':[3],'user_id':['u1'],'timestamp':['2025-01-06 00:00:00']})
        tx=pd.DataFrame({'user_id':['u1','u1','u1'],
                         'timestamp':['2025-01-01 00:00:00','2025-01-03 00:00:00','2025-01-10 00:00:00'],
                         'amt':[10.,50.,999.]})
        rep=[]
        a,b=enrich(tr,te,{'tx':tx},report=rep)
        self.assertTrue(any('one-to-many causal' in r for r in rep))
        self.assertAlmostEqual(float(a.tx__hist_recency_days.iloc[0]),1.0,places=3)
        self.assertAlmostEqual(float(a.tx__hist_recency_days.iloc[1]),2.0,places=3)
        self.assertAlmostEqual(float(a.tx__amt__hist_trend.iloc[1]),20.0,places=3)  # last 50 - mean 30 = +20
        self.assertAlmostEqual(float(b.tx__hist_recency_days.iloc[0]),3.0,places=3)  # Jan 06 - Jan 03; Jan 10 ignored!

    def test_fit_mlp_regression_and_binary(self):
        from ag_models import fit_mlp, ModelConfig
        rng=np.random.default_rng(1)
        X=pd.DataFrame({'a':rng.normal(size=240),'b':rng.normal(size=240),
                        'c':pd.Categorical(rng.choice(['x','y','z'],240))})
        Xt=X.iloc[:20].copy()
        y_reg=3.0*X['a'].values - 2.0*X['b'].values + 1.0*(X['a'].values**2)
        spec_r=TaskSpec('','',id_col='id',target='y',task_type='regression',metric='mae')
        plan_r=make_plan(X,y_reg,'regression',n_folds=3)
        rr=fit_mlp(X,y_reg,Xt,plan_r,spec_r,ModelConfig(),n_jobs=2,time_limit=60)
        self.assertIsNotNone(rr)
        self.assertLess(rr.score,1.0)
        self.assertEqual(len(rr.test),len(Xt))
        y_cls=(y_reg>0.5).astype(int)
        spec_c=TaskSpec('','',id_col='id',target='y',task_type='binary',metric='roc_auc',classes=[0,1])
        plan_c=make_plan(X,y_cls,'binary',n_folds=3)
        rc=fit_mlp(X,y_cls,Xt,plan_c,spec_c,ModelConfig(),n_jobs=2,time_limit=60)
        self.assertIsNotNone(rc)
        self.assertGreater(rc.score,0.85)
        self.assertEqual(rc.test.shape,(len(Xt),2))

    def test_stack_blend_regression_and_binary(self):
        from ag_ensemble import stack_blend
        from ag_models import FitResult
        rng=np.random.default_rng(2)
        n=400
        y=rng.normal(10.0,3.0,size=n)
        m=np.ones(n,bool)
        rs=[FitResult(f'm{i}',y+rng.normal(0,0.8+0.2*i,size=n),np.full(10,10.0),0.8,[],[],None,1.0) for i in range(3)]
        spec_r=TaskSpec('','',id_col='id',target='y',task_type='regression',metric='mae')
        sb=stack_blend(rs,y,m,spec_r)
        self.assertIsNotNone(sb)
        self.assertLess(sb['score'],0.75)
        yc=(y>10.0).astype(int)
        def mk_p(s):
            p=1/(1+np.exp(-s))
            return np.column_stack([1-p,p])
        rs_c=[FitResult(f'c{i}',mk_p((y-10.0)+rng.normal(0,1.0,size=n)),mk_p(np.linspace(-1,1,10)),0.9,[],[],None,1.0) for i in range(3)]
        spec_c=TaskSpec('','',id_col='id',target='y',task_type='binary',metric='roc_auc',classes=[0,1])
        sbc=stack_blend(rs_c,yc,m,spec_c)
        self.assertIsNotNone(sbc)
        self.assertEqual(sbc['test'].shape,(10,2))
        self.assertTrue(np.isfinite(sbc['oof'][m]).all())

    def test_completed_interval_never_leaks_current_hour(self):
        # Exact organizer rule: hourly record stamped 10:00 covers [10:00, 11:00) and MUST NOT be used at 10:00 or 10:30!
        stamps=pd.date_range('2025-01-01 00:00:00',periods=250,freq='h').astype(str).tolist()
        vals=[float(i) for i in range(250)]
        vals[210]=999999.0  # poison value at hour index 210 (current uncompleted hour for call at 210:30)
        vals[211]=888888.0  # future hour
        dim=pd.DataFrame({'district_id':['d1']*250,'timestamp':stamps,'v':vals})
        t_cur=pd.Timestamp('2025-01-01 00:00:00')+pd.Timedelta(hours=210)
        tr=pd.DataFrame({'event_id':[1,2],'district_id':['d1','d1'],
                         'timestamp':[str(t_cur),str(t_cur+pd.Timedelta(minutes=30))],'y':[1.,2.]})
        te=pd.DataFrame({'event_id':[3],'district_id':['d1'],
                         'timestamp':[str(t_cur+pd.Timedelta(minutes=59))]})
        a,b=enrich(tr,te,{'weather':dim})
        # All 3 calls in [t_cur, t_cur+59m] must receive hour 209 (209.0), NEVER 210 (999999.0) in ANY column!
        self.assertEqual(a.weather__v.tolist(),[209.0,209.0])
        self.assertEqual(b.weather__v.iloc[0],209.0)
        for df_chk in (a,b):
            for c in df_chk.columns:
                if c.startswith('weather__') and pd.api.types.is_numeric_dtype(df_chk[c]):
                    self.assertLess(float(df_chk[c].max()),1000.0,f'Leak detected in {c}')

    def test_binary_runlen_completed_interval_no_leak(self):
        stamps=pd.date_range('2025-01-01 00:00:00',periods=30,freq='h').astype(str).tolist()
        # Storm active at hours 10, 11, 12, 13 (runlen=1,2,3,4), off at 14, on at 15 (runlen=1)
        flags=[0]*30
        for h in (10,11,12,13,15): flags[h]=1
        dim=pd.DataFrame({'district_id':['d1']*30,'timestamp':stamps,'storm_flag':flags})
        tr=pd.DataFrame({'event_id':[1,2,3],'district_id':['d1']*3,
                         'timestamp':['2025-01-01 13:30:00','2025-01-01 14:30:00','2025-01-01 15:30:00'],'y':[1.,2.,3.]})
        te=pd.DataFrame({'event_id':[4],'district_id':['d1'],'timestamp':['2025-01-01 16:30:00']})
        a,b=enrich(tr,te,{'weather':dim})
        # At 13:30 -> completed 12:00 (runlen=3, NEVER 13:00 runlen=4!)
        # At 14:30 -> completed 13:00 (runlen=4); at 15:30 -> completed 14:00 (runlen=0); at 16:30 -> completed 15:00 (runlen=1)
        self.assertEqual(a.weather__storm_flag__runlen.tolist(),[3.0,4.0,0.0])
        self.assertEqual(b.weather__storm_flag__runlen.iloc[0],1.0)

    def test_causal_stream_zero_future_or_tie_leak_and_density_guard(self):
        from ag_features import causal_stream_features
        rng=np.random.default_rng(11)
        ts_tr=pd.date_range('2025-01-01 00:00:00',periods=400,freq='30min')
        ts_te=pd.date_range('2025-01-09 08:00:00',periods=200,freq='30min')
        # Add exact timestamp tie in test (rows 50 and 51 share exact timestamp!)
        ts_te=ts_te.tolist()
        ts_te[51]=ts_te[50]
        tr=pd.DataFrame({'event_id':range(400),'timestamp':ts_tr.astype(str),
                         'flag':rng.integers(0,2,400),'kind':rng.choice(['A','B'],400),'y':rng.normal(size=400)})
        te=pd.DataFrame({'event_id':range(400,600),'timestamp':[str(x) for x in ts_te],
                         'flag':rng.integers(0,2,200),'kind':rng.choice(['A','B'],200)})
        s_tr1,s_te1=causal_stream_features(tr,te,exclude={'y','event_id'})
        self.assertGreater(s_tr1.shape[1],10)
        # Poison row 50 (tie) and all rows > 50 in test -> row 50 AND row 51 MUST NOT change by a single bit!
        te_poison=te.copy()
        te_poison.loc[50:,'flag']=1 - te_poison.loc[50:,'flag']
        te_poison.loc[50:,'kind']='B'
        s_tr2,s_te2=causal_stream_features(tr,te_poison,exclude={'y','event_id'})
        pd.testing.assert_frame_equal(s_tr1,s_tr2)
        pd.testing.assert_frame_equal(s_te1.iloc[:52],s_te2.iloc[:52])
        # Density-mismatch guard: test 10x sparser than train -> must cleanly return 0 columns
        ts_sparse=pd.date_range('2025-02-01 00:00:00',periods=200,freq='5h')
        te_sp=te.copy(); te_sp['timestamp']=ts_sparse.astype(str)
        st_a,st_b=causal_stream_features(tr,te_sp,exclude={'y','event_id'})
        self.assertEqual(st_a.shape[1],0); self.assertEqual(st_b.shape[1],0)

    def test_cross_group_sys_mean_and_nowcast_zero_future_leak(self):
        from ag_features import detect_time_split, prepare
        from ag_nowcast import build_nowcast_features
        rng=np.random.default_rng(7)
        n_hr=24*30  # 30 days hourly x 6 districts = 4320 rows
        stamps=pd.date_range('2025-01-01 00:00:00',periods=n_hr,freq='h')
        dists=[f'D{i}' for i in range(6)]
        rows=[]
        for d_i,d in enumerate(dists):
            hr=stamps.hour.values
            sig=10.0 + 3.0*np.sin(2*np.pi*hr/24.0) + d_i + rng.normal(0,0.3,n_hr)
            rows.append(pd.DataFrame({'district_id':d,'timestamp':stamps.astype(str),'spd':sig,'vol':sig*2.0+1.0}))
        panel=pd.concat(rows,ignore_index=True)
        # Events in days 10..27 (train, 600 rows -> >=50 rows/period for temporal folds) and days 28..30 (test)
        ev_tr_ts=pd.date_range('2025-01-10 00:30:00',periods=600,freq='40min')
        ev_te_ts=pd.date_range('2025-01-28 00:30:00',periods=40,freq='1h')
        tr=pd.DataFrame({'event_id':range(600),'district_id':[dists[i%6] for i in range(600)],
                         'timestamp':ev_tr_ts.astype(str),'y':rng.normal(20,3,600)})
        te=pd.DataFrame({'event_id':range(600,640),'district_id':[dists[i%6] for i in range(40)],
                         'timestamp':ev_te_ts.astype(str)})
        meta1={}
        a1,b1=enrich(tr,te,{'traffic':panel},meta=meta1)
        self.assertIn('traffic__spd__sys_mean',a1.columns)
        self.assertIn('traffic__spd__sys_dev',a1.columns)
        self.assertEqual(len(meta1['panels']),1)
        spec=TaskSpec('','',id_col='event_id',target='y',task_type='regression',metric='mae')
        prep=prepare(a1,b1,spec)
        tcol=detect_time_split(prep.X,prep.Xt,prep.cat_cols)
        plan=make_plan(prep.X,prep.y,'regression',time_col=tcol)
        nc_tr1,nc_te1=build_nowcast_features(meta1,prep.y,plan,'regression',n_jobs=2,use_lgb_top=False,max_seconds=30.0)
        self.assertIsNotNone(nc_tr1)
        self.assertIn('nc__traffic__spd',nc_tr1.columns)
        self.assertEqual(int(nc_te1.isna().sum().sum()),0)
        # Poison panel at and after test row 20 hour -> test row 20 nowcast & sys_mean MUST NOT change!
        probe_hr=pd.Timestamp(ev_te_ts[20]).floor('h')
        panel_p=panel.copy()
        panel_p.loc[pd.to_datetime(panel_p['timestamp'])>=probe_hr,['spd','vol']]=99999.0
        meta2={}
        a2,b2=enrich(tr,te,{'traffic':panel_p},meta=meta2)
        nc_tr2,nc_te2=build_nowcast_features(meta2,prep.y,plan,'regression',n_jobs=2,use_lgb_top=False,max_seconds=30.0)
        self.assertAlmostEqual(float(b1.loc[20,'traffic__spd__sys_mean']),float(b2.loc[20,'traffic__spd__sys_mean']),places=5)
        self.assertAlmostEqual(float(nc_te1.loc[20,'nc__traffic__spd']),float(nc_te2.loc[20,'nc__traffic__spd']),places=5)

    def test_temporal_autoresearch_operators_zero_test_leak(self):
        from ag_features import op_freq, op_group_dev
        rng=np.random.default_rng(19)
        X=pd.DataFrame({'cat':pd.Categorical(rng.choice(['a','b','c'],200)),'num':rng.normal(10,2,200)})
        Xt=pd.DataFrame({'cat':pd.Categorical(rng.choice(['a','b','c'],60)),'num':rng.normal(10,2,60)})
        Xt_p=Xt.copy(); Xt_p.loc[10:,'num']=999999.0; Xt_p.loc[10:,'cat']='c'
        fa1,fb1=op_freq(X,Xt,['cat'],train_only=True)
        fa2,fb2=op_freq(X,Xt_p,['cat'],train_only=True)
        pd.testing.assert_frame_equal(fa1,fa2)
        pd.testing.assert_frame_equal(fb1.iloc[:10],fb2.iloc[:10])
        ga1,gb1=op_group_dev(X,Xt,[('cat','num')],train_only=True)
        ga2,gb2=op_group_dev(X,Xt_p,[('cat','num')],train_only=True)
        pd.testing.assert_frame_equal(ga1,ga2)
        pd.testing.assert_frame_equal(gb1.iloc[:10],gb2.iloc[:10])

    def test_binary_flags_stay_numeric_tag_matching_and_exact_date_diffs(self):
        from ag_features import prepare
        ts_s=pd.date_range('2025-03-01',periods=40,freq='D')
        cr_s=[t - pd.Timedelta(days=31 + (i % 5)) for i, t in enumerate(ts_s)]
        tr=pd.DataFrame({
            'id':range(40),
            'ts':ts_s.astype(str),
            'created':[str(t) for t in cr_s],
            'flag01':[i%2 for i in range(40)],
            'ad_tags':['sport, music, tech' if i%2==0 else 'travel, food, art' for i in range(40)],
            'user_tags':['music, tech' if i%2==0 else 'sport, art' for i in range(40)],
            'y':np.linspace(1,40,40),
        })
        te=tr.drop(columns=['y']).iloc[:10].copy()
        spec=TaskSpec('','',id_col='id',target='y',task_type='regression',metric='mae',cat_hint=['flag01'])
        prep=prepare(tr,te,spec)
        self.assertNotIn('flag01',prep.cat_cols)
        self.assertTrue(pd.api.types.is_numeric_dtype(prep.X['flag01']))
        self.assertAlmostEqual(float(prep.X['ts__minus__created'].iloc[0]),31.0,places=3)
        self.assertIn('ad_tags__ov__user_tags',prep.X.columns)
        self.assertAlmostEqual(float(prep.X['ad_tags__ov__user_tags'].iloc[0]),2.0,places=4)
        self.assertAlmostEqual(float(prep.X['ad_tags__jac__user_tags'].iloc[0]),2.0/3.0,places=4)

    def test_group_stream_and_cur_hr_boundary_zero_leak(self):
        from ag_features import causal_stream_features
        rng=np.random.default_rng(23)
        ts_tr=pd.date_range('2025-01-01 00:00:00',periods=400,freq='20min')
        ts_te=pd.date_range('2025-01-06 14:00:00',periods=120,freq='20min')
        tr=pd.DataFrame({'event_id':range(400),'timestamp':ts_tr.astype(str),
                         'district_id':[f'D{i%5}' for i in range(400)],'flag':[1 if i%3==0 else 0 for i in range(400)],
                         'y':rng.normal(size=400)})
        te=pd.DataFrame({'event_id':range(400,520),'timestamp':ts_te.astype(str),
                         'district_id':[f'D{i%5}' for i in range(120)],'flag':[1 if i%3==0 else 0 for i in range(120)]})
        # Row 0 of test is at 14:00:00 with flag=1; row 1 is at 14:20:00 -> pos_in_cur_hr MUST be 1.0!
        s_tr1,s_te1=causal_stream_features(tr,te,exclude={'y','event_id'})
        self.assertIn('stream_grp__district_id__cnt_24h',s_tr1.columns)
        self.assertEqual(float(s_te1.loc[1,'stream__flag_pos_in_cur_hr']),1.0)
        # Poison test rows strictly > 40 -> rows <= 40 must not change at all
        te_p=te.copy(); te_p.loc[41:,'district_id']='D9'; te_p.loc[41:,'flag']=1-te_p.loc[41:,'flag']
        s_tr2,s_te2=causal_stream_features(tr,te_p,exclude={'y','event_id'})
        pd.testing.assert_frame_equal(s_tr1,s_tr2)
        pd.testing.assert_frame_equal(s_te1.iloc[:41],s_te2.iloc[:41])

    def test_simplex_stack_nonnegative_hgb_and_xgb_pseudohuber(self):
        from ag_ensemble import stack_blend
        from ag_models import FitResult, ModelConfig, fit_hgb, fit_xgb
        rng=np.random.default_rng(31)
        n=450
        X=pd.DataFrame({'a':rng.normal(size=n),'b':rng.normal(size=n),'c':pd.Categorical(rng.choice(['x','y','z'],n))})
        Xt=X.iloc[:30].copy()
        y=22.0 + 4.0*X['a'].values - 2.5*X['b'].values + rng.normal(0,0.8,n)
        spec=TaskSpec('','',id_col='id',target='y',task_type='regression',metric='mae')
        plan=make_plan(X,y,'regression',n_folds=3)
        r_xgb=fit_xgb(X,y,Xt,plan,spec,ModelConfig(),n_jobs=2,time_limit=60,lr=0.1,depth=4,rounds=120,
                      objective_override='pseudohuber',name='xgb_ph')
        self.assertIsNotNone(r_xgb)
        self.assertLess(r_xgb.score,2.0)  # must not explode with calibrated base_score & huber_slope
        r_hgb=fit_hgb(X,y,Xt,plan,spec,ModelConfig(),n_jobs=2,time_limit=60,max_iter=120,name='hgb')
        self.assertIsNotNone(r_hgb)
        self.assertLess(r_hgb.score,2.0)
        # Non-negative simplex stack weights (even when a noisy weak model is included)
        r_bad=FitResult('weak',y+rng.normal(0,8.0,n),np.full(30,22.0),6.0,[],[],None,1.0)
        sb=stack_blend([r_xgb,r_hgb,r_bad],y,plan.oof_mask,spec,plan=plan)
        self.assertIsNotNone(sb)
        self.assertTrue(all(w>=0.0 for w in sb['weights'].values()),f'Negative weight found: {sb["weights"]}')
        self.assertAlmostEqual(sum(sb['weights'].values()),1.0,places=2)

    def test_interval_join_drops_all_post_factum_columns(self):
        tr=pd.DataFrame({'event_id':[1,2],'district_id':['d1','d1'],
                         'timestamp':['2025-01-01 10:30:00','2025-01-01 15:00:00'],'y':[1.,2.]})
        te=pd.DataFrame({'event_id':[3],'district_id':['d1'],'timestamp':['2025-01-01 11:00:00']})
        ev=pd.DataFrame({'district_id':['d1'],'start_timestamp':['2025-01-01 10:00:00'],'end_timestamp':['2025-01-01 12:00:00'],
                         'expected_attendance':[5000],'actual_duration':[125],'final_cost':[999],'outcome_code':[1]})
        a,_=enrich(tr,te,{'city_events':ev})
        self.assertIn('city_events__expected_attendance',a.columns)
        for bad in ('city_events__actual_duration','city_events__final_cost','city_events__outcome_code'):
            self.assertNotIn(bad,a.columns)

    def test_nowcast_chrono_slicing_all_groups_key_mismatch_and_bit_exact_stream(self):
        from ag_features import detect_time_split, prepare
        from ag_nowcast import _recent_chrono_pos, build_nowcast_features
        # 1) Chronological slicing MUST preserve ALL groups even when panel is sorted by (group, time)
        n_g, n_t = 10, 500
        g_arr = np.repeat(np.arange(n_g), n_t)
        h_ns = np.tile(np.arange(n_t, dtype=np.int64) * 3600_000_000_000, n_g)
        picked = _recent_chrono_pos(np.arange(n_g * n_t), h_ns, limit=1000)
        self.assertEqual(len(picked), 1000)
        self.assertEqual(set(g_arr[picked].tolist()), set(range(n_g)))  # all 10 groups represented equally!
        self.assertEqual(int(h_ns[picked].min()), 400 * 3600_000_000_000)  # strictly last 100 timestamps
        # 2) Key dtype mismatch (int64 in train/test vs str in panel) + binary runlen + online-stream bit-exactness
        rng = np.random.default_rng(41)
        n_hr = 24 * 35  # 35 days x 6 groups = 5040 rows
        stamps = pd.date_range('2025-01-01 00:00:00', periods=n_hr, freq='h')
        rows = []
        for d in range(6):
            hr = stamps.hour.values
            flag = ((hr >= 12) & (hr <= 15)).astype(int)
            sig = 10.0 + 4.0 * flag + 2.0 * np.sin(2 * np.pi * hr / 24.0) + d + rng.normal(0, 0.2, n_hr)
            rows.append(pd.DataFrame({'district_id': str(d), 'timestamp': stamps.astype(str), 'spd': sig, 'storm': flag}))
        panel = pd.concat(rows, ignore_index=True)
        ev_tr_ts = pd.date_range('2025-01-10 00:30:00', periods=600, freq='45min')
        ev_te_ts = pd.date_range('2025-02-01 00:30:00', periods=60, freq='1h')
        tr = pd.DataFrame({'event_id': range(600), 'district_id': [i % 6 for i in range(600)],
                           'timestamp': ev_tr_ts.astype(str), 'y': rng.normal(20, 3, 600)})
        te = pd.DataFrame({'event_id': range(600, 660), 'district_id': [i % 6 for i in range(60)],
                           'timestamp': ev_te_ts.astype(str)})
        meta1 = {}
        a1, b1 = enrich(tr, te, {'weather': panel}, meta=meta1)
        spec = TaskSpec('', '', id_col='event_id', target='y', task_type='regression', metric='mae')
        prep = prepare(a1, b1, spec)
        tcol = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
        plan = make_plan(prep.X, prep.y, 'regression', time_col=tcol)
        nc_tr1, nc_te1 = build_nowcast_features(meta1, prep.y, plan, 'regression', n_jobs=2, use_lgb_top=False, max_seconds=30.0)
        self.assertIsNotNone(nc_tr1)
        self.assertEqual(int(nc_te1.isna().sum().sum()), 0)
        # Drop future test rows (> 25): online-stream predictions for rows <= 25 MUST be bit-exact!
        te_cut = te.iloc[:26].copy()
        meta2 = {}
        a2, b2 = enrich(tr, te_cut, {'weather': panel}, meta=meta2)
        nc_tr2, nc_te2 = build_nowcast_features(meta2, prep.y, plan, 'regression', n_jobs=2, use_lgb_top=False, max_seconds=30.0)
        np.testing.assert_array_equal(nc_te1.iloc[:26].to_numpy(), nc_te2.to_numpy())

    def test_hgb_object_col_consistent_mapping(self):
        from ag_models import ModelConfig, fit_hgb
        rng = np.random.default_rng(53)
        n = 300
        # Pure object column (not pd.Categorical) with strong signal, reversed order in test
        lbl_tr = rng.choice(['alpha', 'beta', 'gamma'], size=n)
        lbl_te = np.array(['gamma', 'beta', 'alpha'] * 10, dtype=object)
        val_map = {'alpha': 10.0, 'beta': 50.0, 'gamma': 90.0}
        X = pd.DataFrame({'x': rng.normal(size=n), 'obj': pd.Series(lbl_tr, dtype=object)})
        Xt = pd.DataFrame({'x': np.zeros(30), 'obj': pd.Series(lbl_te, dtype=object)})
        y = np.array([val_map[v] for v in lbl_tr]) + rng.normal(0, 0.5, n)
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        plan = make_plan(X, y, 'regression', n_folds=3)
        r = fit_hgb(X, y, Xt, plan, spec, ModelConfig(), n_jobs=2, time_limit=30, max_iter=80, name='hgb_obj')
        self.assertIsNotNone(r)
        # Predictions on test MUST follow gamma > beta > alpha!
        self.assertGreater(float(r.test[0]), 80.0)
        self.assertLess(float(r.test[2]), 20.0)

    def test_multiclass_int_target_preserved_and_float_class_cols_matched(self):
        from ag_task import build_spec
        from solution import SubmissionWriter
        tr = pd.DataFrame({'row_id': range(24), 'x': np.linspace(0, 1, 24), 'target': [i % 12 for i in range(24)]})
        te = pd.DataFrame({'row_id': range(24, 30), 'x': np.linspace(0, 1, 6)})
        a, b = enrich(tr, te, {})
        # enrich MUST keep integer target intact (not float64 from concat NaNs)
        self.assertTrue(pd.api.types.is_integer_dtype(a['target']))
        sample = pd.DataFrame({'row_id': list(range(24, 30)), **{f'class_{i}': 0.0 for i in range(12)}})
        with tempfile.TemporaryDirectory() as d:
            tp = Path(d) / 'task.md'
            tp.write_text('ID: row_id. Target: target. Multiclass classification. Metric: LogLoss.\n', encoding='utf-8')
            spec = build_spec(tp, a, b, sample, None, {})
            self.assertEqual(spec.output_kind, 'proba_multi')
            self.assertEqual(len(spec.classes), 12)
            # Also verify integer-like float targets (e.g. from CSV with NaNs) still match class_0..class_11
            a_flt = a.copy(); a_flt['target'] = a_flt['target'].astype(float)
            spec2 = build_spec(tp, a_flt, b, sample, None, {})
            self.assertEqual(spec2.output_kind, 'proba_multi')
            out_csv = Path(d) / 'submission.csv'
            w = SubmissionWriter(out_csv, spec2, b, sample)
            w.write(np.full((6, 12), 1.0 / 12.0))
            sub = pd.read_csv(out_csv)
            self.assertEqual(list(sub.columns), list(sample.columns))
            self.assertEqual(len(sub), 6)

    def test_mismatched_pk_fk_join_clients_id(self):
        # Case 1: train has row_id + client_id, clients has id
        tr = pd.DataFrame({'row_id': [10, 11, 12], 'client_id': [1, 2, 1], 'y': [0, 1, 0]})
        te = pd.DataFrame({'row_id': [13, 14], 'client_id': [2, 3]})
        clients = pd.DataFrame({'id': [1, 2, 3], 'risk_band': [0.1, 0.9, 0.5]})
        a, b = enrich(tr, te, {'clients': clients})
        self.assertIn('clients__risk_band', a.columns)
        self.assertEqual(a['clients__risk_band'].tolist(), [0.1, 0.9, 0.1])
        self.assertEqual(b['clients__risk_band'].tolist(), [0.9, 0.5])
        # Case 2: train ALSO has its own `id` column (must NOT wrongly join train.id == clients.id!)
        tr2 = pd.DataFrame({'id': [1, 2, 3], 'client_id': [3, 1, 2], 'y': [0, 1, 0]})
        te2 = pd.DataFrame({'id': [4, 5], 'client_id': [2, 1]})
        a2, b2 = enrich(tr2, te2, {'clients': clients})
        self.assertEqual(a2['clients__risk_band'].tolist(), [0.5, 0.1, 0.9])
        self.assertEqual(b2['clients__risk_band'].tolist(), [0.9, 0.1])

    def test_temporal_tfidf_zero_future_vocab_leakage(self):
        from ag_features import op_text_oof
        rng = np.random.default_rng(77)
        n = 160
        periods = np.repeat(np.arange(4), 40)
        y = rng.integers(0, 2, n)
        sa = pd.Series([f'{"fraud alert charge" if yi else "normal address update"} doc_{i}' for i, yi in enumerate(y)])
        sb = pd.Series(['fraud alert charge test'] * 20)
        oa1, _ = op_text_oof({'txt': (sa, sb)}, y, 'binary', 2, periods=periods)
        # Mutate BOTH future train text (periods >= 2) AND all test text: period 1 OOF MUST be bit-identical!
        sa_p = sa.mask(periods >= 2, 'completely_different_future_vocabulary_xyz ' * 8)
        sb_p = pd.Series(['brand_new_test_tokens_abc ' * 8] * 20)
        oa2, _ = op_text_oof({'txt': (sa_p, sb_p)}, y, 'binary', 2, periods=periods)
        delta = float(np.max(np.abs(oa1.loc[periods == 1].to_numpy() - oa2.loc[periods == 1].to_numpy())))
        self.assertLess(delta, 1e-7)

    def test_pg_boolean_conversion_and_sql_fk_multi_role_join(self):
        from ag_database import _convert_pg_booleans
        df_b = pd.DataFrame({'is_storm': ['t', 'f', None, 't'], 'val': [1.0, 2.0, 3.0, 4.0]})
        df_b = _convert_pg_booleans(df_b, ['is_storm'])
        self.assertTrue(pd.api.types.is_numeric_dtype(df_b['is_storm']))
        self.assertEqual(df_b['is_storm'].iloc[0], 1.0)
        self.assertEqual(df_b['is_storm'].iloc[1], 0.0)
        self.assertTrue(pd.isna(df_b['is_storm'].iloc[2]))
        # Explicit SQL FKs: arbitrary key name + multi-role sender_id/receiver_id -> users.acc_code
        tr = pd.DataFrame({'event_id': [1, 2], 'sender_id': ['U1', 'U2'], 'receiver_id': ['U2', 'U3'], 'y': [0, 1]})
        te = pd.DataFrame({'event_id': [3], 'sender_id': ['U3'], 'receiver_id': ['U1']})
        users = pd.DataFrame({'acc_code': ['U1', 'U2', 'U3'], 'tier': [10, 20, 30]})
        meta = {'foreign_keys': [
            {'src_table': 'train', 'src_col': 'sender_id', 'dst_table': 'users', 'dst_col': 'acc_code'},
            {'src_table': 'train', 'src_col': 'receiver_id', 'dst_table': 'users', 'dst_col': 'acc_code'},
        ]}
        a, b = enrich(tr, te, {'users': users}, meta=meta)
        self.assertIn('sender__users__tier', a.columns)
        self.assertIn('receiver__users__tier', a.columns)
        self.assertEqual(a['sender__users__tier'].tolist(), [10, 20])
        self.assertEqual(a['receiver__users__tier'].tolist(), [20, 30])
        self.assertEqual(b['sender__users__tier'].iloc[0], 30)
        self.assertEqual(b['receiver__users__tier'].iloc[0], 10)

    def test_windowed_causal_rfm_1d_7d_30d_zero_future_leak(self):
        tr = pd.DataFrame({'event_id': [1, 2, 3], 'user_id': ['u1', 'u1', 'u1'],
                           'timestamp': ['2025-01-20 12:00:00', '2025-01-25 12:00:00', '2025-02-01 12:00:00'],
                           'y': [1.0, 2.0, 3.0]})
        te = pd.DataFrame({'event_id': [4], 'user_id': ['u1'], 'timestamp': ['2025-02-02 12:00:00']})
        tx = pd.DataFrame({
            'user_id': ['u1'] * 5,
            'timestamp': [
                '2025-01-15 12:00:00',  # 18d before test (inside 30d, outside 7d)
                '2025-01-30 12:00:00',  # 3d before test (inside 7d, outside 1d)
                '2025-02-02 06:00:00',  # 6h before test (inside 1d)
                '2025-02-02 12:00:00',  # exact tie with test (MUST be excluded!)
                '2025-02-05 00:00:00',  # future (MUST be excluded!)
            ],
            'amt': [100.0, 20.0, 5.0, 9999.0, 8888.0],
        })
        _, b = enrich(tr, te, {'tx': tx})
        self.assertEqual(float(b['tx__hist_cnt_1d'].iloc[0]), 1.0)
        self.assertEqual(float(b['tx__hist_cnt_7d'].iloc[0]), 2.0)
        self.assertEqual(float(b['tx__hist_cnt_30d'].iloc[0]), 3.0)
        self.assertAlmostEqual(float(b['tx__amt__hist_sum_7d'].iloc[0]), 25.0, places=3)
        self.assertAlmostEqual(float(b['tx__amt__hist_sum_30d'].iloc[0]), 125.0, places=3)

    def test_dirty_regression_target_unix_epoch_and_dmy_dates(self):
        from ag_features import detect_time_split, prepare
        ep_tr = np.arange(1700000000, 1700000000 + 20 * 86400, 86400)
        ep_te = np.arange(1700000000 + 25 * 86400, 1700000000 + 30 * 86400, 86400)
        tr = pd.DataFrame({
            'id': range(20),
            'created_at': ep_tr,
            'ru_date': ['01.08.2025' if i < 10 else '15.09.2025' for i in range(20)],
            'x': np.arange(20, dtype=float),
            'y': ['N/A' if i == 0 else f'{10 + i},5' for i in range(20)],
        })
        te = pd.DataFrame({'id': range(20, 25), 'created_at': ep_te, 'ru_date': ['20.09.2025'] * 5, 'x': np.arange(5, dtype=float)})
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        prep = prepare(tr, te, spec)
        self.assertEqual(len(prep.y), 19)  # dirty 'N/A' row dropped cleanly; no NaN in y!
        self.assertTrue(np.isfinite(prep.y).all())
        self.assertAlmostEqual(float(prep.y[0]), 11.5, places=4)
        self.assertIn('created_at__ord', prep.X.columns)
        self.assertEqual(int(prep.X['ru_date__month'].iloc[0]), 8)  # 01.08.2025 -> August, NOT January!
        self.assertEqual(int(prep.X['ru_date__month'].iloc[-1]), 9)
        tcol = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
        self.assertEqual(tcol, 'created_at__ord')

    def test_worker_auto_rejects_temporal_llm_leakage(self):
        import os, subprocess
        worker = Path(__file__).resolve().parent.parent / 'ag_worker.py'
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            rng = np.random.default_rng(11)
            tr = pd.DataFrame({'a': rng.normal(size=100), 'b': rng.normal(size=100)})
            te = pd.DataFrame({'a': rng.normal(size=80), 'b': rng.normal(size=80)})
            tr.to_pickle(dp / 'train.pkl'); te.to_pickle(dp / 'test.pkl')
            env = {**os.environ, 'AG_CHECK_TEMPORAL_LEAK': '1'}
            # 1) Leaky code (concat train+test global mean) MUST be rejected!
            (dp / 'code.py').write_text(
                'def make_features(train, test):\n'
                '    m = pd.concat([train["a"], test["a"]]).mean()\n'
                '    return pd.DataFrame({"z": train["a"] - m}), pd.DataFrame({"z": test["a"] - m})\n',
                encoding='utf-8',
            )
            r_bad = subprocess.run([sys.executable, str(worker), str(dp)], capture_output=True, text=True, env=env)
            self.assertNotEqual(r_bad.returncode, 0)
            self.assertIn('temporal leakage detected', r_bad.stderr)
            # 2) Clean train-only code MUST pass!
            (dp / 'code.py').write_text(
                'def make_features(train, test):\n'
                '    m = train["a"].mean()\n'
                '    return pd.DataFrame({"z": train["a"] - m}), pd.DataFrame({"z": test["a"] - m})\n',
                encoding='utf-8',
            )
            r_ok = subprocess.run([sys.executable, str(worker), str(dp)], capture_output=True, text=True, env=env)
            self.assertEqual(r_ok.returncode, 0, r_ok.stderr)

    def test_calibrated_nowcast_binary_and_nonneg_bounds(self):
        from ag_features import detect_time_split, prepare
        from ag_nowcast import build_nowcast_features
        rng = np.random.default_rng(61)
        n_hr = 24 * 35
        stamps = pd.date_range('2025-01-01 00:00:00', periods=n_hr, freq='h')
        rows = []
        for d in range(6):
            hr = stamps.hour.values
            storm = ((hr >= 14) & (hr <= 16)).astype(float)
            rain = np.maximum(0.0, 8.0 * storm + rng.normal(0, 0.2, n_hr))
            rows.append(pd.DataFrame({'district_id': f'D{d}', 'timestamp': stamps.astype(str), 'rain': rain, 'storm': storm}))
        panel = pd.concat(rows, ignore_index=True)
        ev_tr_ts = pd.date_range('2025-01-10 00:30:00', periods=600, freq='45min')
        ev_te_ts = pd.date_range('2025-02-01 00:30:00', periods=60, freq='1h')
        tr = pd.DataFrame({'event_id': range(600), 'district_id': [f'D{i%6}' for i in range(600)],
                           'timestamp': ev_tr_ts.astype(str), 'y': 10.0 + 5.0 * (ev_tr_ts.hour >= 14) + rng.normal(0, 0.5, 600)})
        te = pd.DataFrame({'event_id': range(600, 660), 'district_id': [f'D{i%6}' for i in range(60)],
                           'timestamp': ev_te_ts.astype(str)})
        meta = {}
        a, b = enrich(tr, te, {'weather': panel}, meta=meta)
        spec = TaskSpec('', '', id_col='event_id', target='y', task_type='regression', metric='mae')
        prep = prepare(a, b, spec)
        tcol = detect_time_split(prep.X, prep.Xt, prep.cat_cols)
        plan = make_plan(prep.X, prep.y, 'regression', time_col=tcol)
        nc_tr, nc_te = build_nowcast_features(meta, prep.y, plan, 'regression', n_jobs=2, use_lgb_top=True, max_seconds=90.0)
        self.assertIsNotNone(nc_te)
        # Non-negative rain & [0, 1] storm probabilities MUST respect causal bounds!
        self.assertGreaterEqual(float(nc_te['nc__weather__rain'].min()), 0.0)
        self.assertGreaterEqual(float(nc_te['nc__weather__storm'].min()), 0.0)
        self.assertLessEqual(float(nc_te['nc__weather__storm'].max()), 1.0)
        if 'nc_gb__weather__storm' in nc_te.columns:
            self.assertGreaterEqual(float(nc_te['nc_gb__weather__storm'].min()), 0.0)
            self.assertLessEqual(float(nc_te['nc_gb__weather__storm'].max()), 1.0)

    def test_cls_simplex_log_odds_and_rare_class_proba_score(self):
        from ag_ensemble import blend
        from ag_metrics import proba_score
        from ag_models import FitResult
        rng = np.random.default_rng(73)
        # Rare class missing from a fold: proba_score MUST NOT crash
        y_fold = np.array([0, 0, 1, 1, 2, 2])  # class 3 absent out of 4 classes
        p_fold = np.array([[0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                           [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.6, 0.2]])
        self.assertTrue(np.isfinite(proba_score('roc_auc', y_fold, p_fold)))
        self.assertTrue(np.isfinite(proba_score('average_precision', y_fold, p_fold)))
        # Continuous classification simplex blend improves logloss over single models
        n = 300
        s = rng.normal(size=n)
        y = (s + rng.normal(0, 0.8, n) > 0).astype(int)
        p1 = 1 / (1 + np.exp(-(s + rng.normal(0, 0.9, n))))
        p2 = 1 / (1 + np.exp(-(s + rng.normal(0, 0.9, n))))
        r1 = FitResult('m1', np.column_stack([1 - p1, p1]), np.full((10, 2), 0.5), 0.5, [], [], None, 1.0)
        r2 = FitResult('m2', np.column_stack([1 - p2, p2]), np.full((10, 2), 0.5), 0.5, [], [], None, 1.0)
        spec = TaskSpec('', '', id_col='id', target='y', task_type='binary', metric='logloss', classes=[0, 1])
        b = blend([r1, r2], y, np.ones(n, bool), spec)
        self.assertIsNotNone(b)
        ll_b = proba_score('logloss', y, b['oof'])
        self.assertLess(ll_b, min(proba_score('logloss', y, r1.oof), proba_score('logloss', y, r2.oof)))

    def test_zero_mape_hurdle_and_et_object_col_consistency(self):
        from ag_ensemble import apply_zero_mape_hurdle
        from ag_metrics import score
        from ag_models import ModelConfig, fit_et
        rng = np.random.default_rng(89)
        n = 400
        x = rng.normal(size=n)
        X = pd.DataFrame({'x': x, 'obj': pd.Series(np.where(x > 0, 'pos', 'neg'), dtype=object)})
        Xt = pd.DataFrame({'x': np.array([-2.0, 2.0] * 15), 'obj': pd.Series(['neg', 'pos'] * 15, dtype=object)})
        y = np.where(x < -0.3, 0.0, np.exp(1.5 + 0.8 * x))
        plan = make_plan(X, y, 'regression', n_folds=3)
        oof = np.maximum(0.5, y + rng.normal(0, 0.5, n))  # never predicts exact 0 -> huge MAPE
        test_p = np.full(30, 5.0)
        base_mape = score('mape', 'regression', y, oof)
        hur = apply_zero_mape_hurdle(X, Xt, y, oof, test_p, plan, plan.oof_mask, n_jobs=1)
        self.assertIsNotNone(hur)
        self.assertLess(hur['score'], base_mape * 1e-4)  # massive orders-of-magnitude reduction!
        self.assertGreater(float(np.ptp(hur['test'])), 1e-6)
        # Unpredictable random zeros MUST never produce a degenerate all-zero constant submission
        y_rand0 = np.where(rng.random(n) < 0.18, 0.0, np.exp(1.5 + 0.8 * x))
        hur_r = apply_zero_mape_hurdle(X, Xt, y_rand0, oof, np.linspace(1.0, 10.0, 30), plan, plan.oof_mask, n_jobs=1)
        if hur_r is not None:
            self.assertGreater(float(np.ptp(hur_r['test'])), 1e-6)
        # Also verify fit_et maps object columns consistently between train and test
        y_obj = np.where(X['obj'] == 'pos', 100.0, 10.0)
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        r_et = fit_et(X, y_obj, Xt, plan, spec, ModelConfig(), n_jobs=1, time_limit=30, n_estimators=30)
        self.assertIsNotNone(r_et)
        self.assertLess(float(r_et.test[0]), 25.0)  # 'neg' -> ~10
        self.assertGreater(float(r_et.test[1]), 85.0)  # 'pos' -> ~100

    def test_fast_cls_simplex_and_rank_blend_large(self):
        import time
        from ag_ensemble import blend, rank_blend
        from ag_models import FitResult, FoldPlan
        rng = np.random.default_rng(101)
        n = 65_000
        y = rng.binomial(1, 0.15, size=n)
        s = rng.normal(size=n) + 0.6 * y
        p1 = 1 / (1 + np.exp(-(s + rng.normal(0, 1.0, n))))
        p2 = 1 / (1 + np.exp(-(s + rng.normal(0, 1.0, n))))
        p3 = 1 / (1 + np.exp(-(s + rng.normal(0, 1.2, n))))
        mask = np.zeros(n, bool); mask[30_000:] = True
        plan = FoldPlan(folds=[(np.arange(30_000), np.arange(30_000, 47_500)),
                               (np.arange(47_500), np.arange(47_500, n))],
                        mode='refit', oof_mask=mask, time_col='t', desc='time')
        rs = [FitResult(f'm{i}', np.column_stack([1 - p, p]), np.full((100, 2), 0.5), 0.3, [], [], None, 1.0)
              for i, p in enumerate((p1, p2, p3), 1)]
        spec = TaskSpec('', '', id_col='id', target='y', task_type='binary', metric='average_precision', classes=[0, 1])
        t0 = time.time()
        b = blend(rs, y, mask, spec, plan=plan)
        rb = rank_blend(rs, y, mask, spec, plan=plan)
        dt = time.time() - t0
        self.assertIsNotNone(b)
        self.assertIsNotNone(rb)
        self.assertLess(dt, 10.0, f'Large rank_blend+blend took {dt:.2f}s (expected <10s)')

    def test_time_cut_policy_preserves_unique_family(self):
        from ag_metrics import better, direction
        from ag_models import FitResult
        from solution import model_family
        r_main = FitResult('lgb_main', np.zeros(10), np.zeros(5), 2.25, [], [], None, 10.0, config={})
        r_hub = FitResult('lgb_huber', np.zeros(10), np.zeros(5), 2.20, [], [], None, 10.0, config={'time_cut': True})
        r_s1 = FitResult('lgb_s1', np.zeros(10), np.zeros(5), 2.35, [], [], None, 10.0, config={'time_cut': True})
        fams = {}
        for r in (r_main, r_hub, r_s1):
            fams.setdefault(model_family(r.name), []).append(r)
        kept = {}
        for f, rs in fams.items():
            uncut = [r for r in rs if not r.config.get('time_cut')]
            if uncut:
                best_u = max(uncut, key=lambda r: r.score if direction('mae') == 'max' else -r.score).score
                rs = [r for r in rs if (not r.config.get('time_cut')) or better(r.score, best_u, 'mae')]
            kept[f] = [r.name for r in rs]
        # Unique family lgb_huber (2.20) MUST survive even with time_cut=True; worse lgb_s1 seed MUST be dropped!
        self.assertEqual(kept['lgb_huber'], ['lgb_huber'])
        self.assertEqual(kept['lgb'], ['lgb_main'])

    def test_heuristic_multi_role_fk_and_rolling_cat_nunique(self):
        # 1) Heuristic multi-role without explicit SQL FK: sender_id / receiver_id -> users.id
        tr = pd.DataFrame({'event_id': [1, 2], 'sender_id': [1, 2], 'receiver_id': [2, 3], 'y': [0, 1]})
        te = pd.DataFrame({'event_id': [3], 'sender_id': [3], 'receiver_id': [1]})
        users = pd.DataFrame({'id': [1, 2, 3], 'tier': ['gold', 'silver', 'bronze']})
        a, b = enrich(tr, te, {'users': users})
        self.assertEqual(a['sender__users__tier'].tolist(), ['gold', 'silver'])
        self.assertEqual(a['receiver__users__tier'].tolist(), ['silver', 'bronze'])
        self.assertEqual(b['sender__users__tier'].iloc[0], 'bronze')
        self.assertEqual(b['receiver__users__tier'].iloc[0], 'gold')
        # 2) Strictly causal rolling categorical nunique over 1d / 7d / 30d (< event_time)
        tr2 = pd.DataFrame({'event_id': [1], 'client_id': ['c1'], 'timestamp': ['2025-02-10 12:00:00'], 'y': [1.0]})
        te2 = pd.DataFrame({'event_id': [2], 'client_id': ['c1'], 'timestamp': ['2025-02-10 12:00:00']})
        tx = pd.DataFrame({
            'client_id': ['c1'] * 6,
            'timestamp': [
                '2025-01-01 00:00:00',  # 40d ago
                '2025-01-20 12:00:00',  # 21d ago (in 30d only)
                '2025-02-05 12:00:00',  # 5d ago (in 7d & 30d)
                '2025-02-10 06:00:00',  # 6h ago (in 1d, 7d, 30d)
                '2025-02-10 12:00:00',  # EXACT tie -> MUST BE EXCLUDED!
                '2025-02-11 00:00:00',  # FUTURE -> MUST BE EXCLUDED!
            ],
            'amt': [1000., 100., 40., 20., 99999., 88888.],
            'cat': ['old', 'food', 'tech', 'food', 'POISON_TIE', 'POISON_FUT'],
        })
        a2, b2 = enrich(tr2, te2, {'tx': tx})
        self.assertEqual(float(a2['tx__cat__hist_nunique_1d'].iloc[0]), 1.0)
        self.assertEqual(float(a2['tx__cat__hist_nunique_7d'].iloc[0]), 2.0)
        self.assertEqual(float(a2['tx__cat__hist_nunique_30d'].iloc[0]), 2.0)
        self.assertEqual(float(b2['tx__cat__hist_nunique_30d'].iloc[0]), 2.0)

    def test_temporal_sys_max_and_float64_ord_precision(self):
        from ag_features import detect_time_split, prepare
        stamps = pd.date_range('2025-01-01 00:00:00', periods=20, freq='h').astype(str).tolist()
        dim = pd.DataFrame({'district_id': ['d1'] * 20 + ['d2'] * 20, 'timestamp': stamps * 2,
                            'v': [float(i) for i in range(20)] + [float(i * 10) for i in range(20)]})
        tr = pd.DataFrame({'event_id': [1, 2, 3, 4, 5, 6], 'district_id': ['d1'] * 6,
                           'timestamp': ['2025-01-01 15:30:00', '2025-01-01 15:35:00', '2025-01-01 15:40:00',
                                         '2025-01-01 15:45:00', '2025-01-01 15:50:00', '2025-01-01 15:59:50'],
                           'y': [1., 2., 3., 4., 5., 6.]})
        te = pd.DataFrame({'event_id': [7, 8], 'district_id': ['d1', 'd1'],
                           'timestamp': ['2025-01-01 16:00:00', '2025-01-01 16:05:00']})
        a, b = enrich(tr, te, {'sensor': dim})
        self.assertIn('sensor__v__sys_max', a.columns)
        self.assertAlmostEqual(float(a['sensor__v__sys_max'].iloc[0]), 140.0, places=3)
        spec = TaskSpec('', '', id_col='event_id', target='y', task_type='regression', metric='mae')
        prep = prepare(a, b, spec)
        self.assertEqual(prep.X['timestamp__ord'].dtype, np.float64)
        self.assertEqual(detect_time_split(prep.X, prep.Xt, prep.cat_cols), 'timestamp__ord')

    def test_cross_table_categorical_match_train_only(self):
        from ag_features import prepare
        tr = pd.DataFrame({
            'id': range(30),
            'ads__target_gender': ['Male', 'Female', 'All'] * 10,
            'users__gender': ['Male', 'Male', 'Female'] * 10,
            'ads__target_age': ['18-24', '25-34', '35-44'] * 10,
            'users__age': ['18-24', '35-44', '35-44'] * 10,
            'ads__unrelated': ['red', 'blue', 'green'] * 10,
            'y': np.arange(30, dtype=float),
        })
        te = tr.drop(columns=['y']).iloc[:6].copy()
        te['ads__unrelated'] = ['Male', 'Female', 'Male', 'Female', 'Male', 'Female']
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        prep = prepare(tr, te, spec)
        self.assertIn('ads__target_gender__eq__users__gender', prep.X.columns)
        self.assertIn('ads__target_age__eq__users__age', prep.X.columns)
        self.assertNotIn('ads__unrelated__eq__users__gender', prep.X.columns)
        self.assertEqual(prep.X['ads__target_gender__eq__users__gender'].iloc[:3].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(prep.X['ads__target_age__eq__users__age'].iloc[:3].tolist(), [1.0, 0.0, 1.0])

    def test_simplex_stack_classification_and_regression_metrics(self):
        from ag_ensemble import stack_blend
        from ag_models import FitResult
        rng = np.random.default_rng(41)
        n = 400
        y = rng.integers(0, 2, size=n)
        m = np.ones(n, bool)
        s_good = (2.0 * y - 1.0) + rng.normal(0, 0.9, n)
        def to_p(z):
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
            return np.column_stack([1.0 - p, p])
        r1 = FitResult('good1', to_p(s_good), to_p(rng.normal(size=20)), 0.85, [], [], None, 1.0)
        r2 = FitResult('good2', to_p(s_good + rng.normal(0, 0.4, n)), to_p(rng.normal(size=20)), 0.82, [], [], None, 1.0)
        r_bad = FitResult('noisy', to_p(rng.normal(0, 2.5, n)), to_p(rng.normal(size=20)), 0.50, [], [], None, 1.0)
        X_t = pd.DataFrame({'t': np.arange(n, dtype=float)})
        plan2 = make_plan(X_t, y, 'binary', time_col='t')
        for met in ('average_precision', 'logloss'):
            spec = TaskSpec('', '', id_col='id', target='y', task_type='binary', metric=met, classes=[0, 1])
            sb = stack_blend([r1, r2, r_bad], y, plan2.oof_mask if plan2.oof_mask.sum() >= 300 else m, spec,
                             plan=plan2 if plan2.oof_mask.sum() >= 300 else None)
            self.assertIsNotNone(sb)
            self.assertTrue(all(w >= 0.0 for w in sb['weights'].values()), f'Negative weight in {met}: {sb["weights"]}')
            self.assertAlmostEqual(sum(sb['weights'].values()), 1.0, places=2)
            self.assertLess(sb['weights'].get('noisy', 0.0), 0.15)
        yr = np.exp(rng.normal(2.0, 0.5, n))
        rr1 = FitResult('g1', np.clip(yr * rng.normal(1.0, 0.08, n), 0.1, None), np.full(10, 7.0), 0.1, [], [], None, 1.0)
        rr2 = FitResult('g2', np.clip(yr * rng.normal(1.0, 0.12, n), 0.1, None), np.full(10, 7.0), 0.15, [], [], None, 1.0)
        rr3 = FitResult('bad', np.clip(yr * rng.normal(2.0, 0.8, n), 0.1, None), np.full(10, 15.0), 0.9, [], [], None, 1.0)
        for met in ('rmsle', 'mape'):
            spec_r = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric=met)
            sbr = stack_blend([rr1, rr2, rr3], yr, m, spec_r)
            self.assertIsNotNone(sbr)
            self.assertTrue(all(w >= 0.0 for w in sbr['weights'].values()))
            self.assertAlmostEqual(sum(sbr['weights'].values()), 1.0, places=2)
            self.assertLess(sbr['weights'].get('bad', 0.0), 0.05)

    def test_worker_poison_filter_drops_leaky_features(self):
        import subprocess
        worker = Path(__file__).resolve().parents[1] / 'ag_worker.py'
        rng = np.random.default_rng(51)
        tr = pd.DataFrame({'t__ord': np.arange(100, dtype=np.float64), 'a': rng.normal(size=100), 'b': rng.normal(size=100)})
        te = pd.DataFrame({'t__ord': np.arange(100, 160, dtype=np.float64), 'a': rng.normal(size=60), 'b': rng.normal(size=60)})
        code = (
            "def make_features(train, test):\n"
            "    tr_out = pd.DataFrame(index=train.index)\n"
            "    te_out = pd.DataFrame(index=test.index)\n"
            "    tr_out['clean_sum'] = train['a'] + train['b']\n"
            "    te_out['clean_sum'] = test['a'] + test['b']\n"
            "    tr_out['leak_test_mean'] = train['a'] - test['a'].mean()\n"
            "    te_out['leak_test_mean'] = test['a'] - test['a'].mean()\n"
            "    tr_out['leak_shift_fut'] = train['a'].shift(-1).fillna(0.0)\n"
            "    te_out['leak_shift_fut'] = test['a'].shift(-1).fillna(0.0)\n"
            "    return tr_out, te_out\n"
        )
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            tr.to_pickle(dp / 'train.pkl'); te.to_pickle(dp / 'test.pkl')
            (dp / 'code.py').write_text(code, encoding='utf-8')
            res = subprocess.run([sys.executable, '-u', str(worker), str(dp), 't__ord'], capture_output=True, text=True, timeout=30)
            self.assertEqual(res.returncode, 0, res.stderr + res.stdout)
            out_tr = pd.read_pickle(dp / 'out_train.pkl')
            self.assertIn('clean_sum', out_tr.columns)
            self.assertNotIn('leak_test_mean', out_tr.columns)
            self.assertNotIn('leak_shift_fut', out_tr.columns)
            self.assertIn('POISON_DROP:', res.stdout)

    def test_downcast_and_pg_description_comments(self):
        from ag_database import _downcast_df, comments_to_dictionary
        df = pd.DataFrame({
            'id': np.array([10 ** 12, 10 ** 12 + 1], dtype=np.int64),
            'small_int': np.array([1, 2], dtype=np.int64),
            'val': np.array([1.5, 2.5], dtype=np.float64),
            'big_id': np.array([1e12, 1e12 + 1], dtype=np.float64),
        })
        dc = _downcast_df(df.copy())
        self.assertEqual(dc['id'].dtype, np.int64)
        self.assertEqual(dc['small_int'].dtype, np.int32)
        self.assertEqual(dc['val'].dtype, np.float32)
        self.assertEqual(dc['big_id'].dtype, np.float64)
        dic = comments_to_dictionary([
            {'table': 'train', 'column': 'amount', 'type': 'numeric', 'description': 'Order amount in RUB'},
            {'table': 'users', 'column': 'tier', 'type': 'text', 'description': 'User loyalty tier'},
        ], main_tables={'train', 'test'})
        self.assertIsNotNone(dic)
        self.assertIn('amount', dic['column'].tolist())
        self.assertIn('users__tier', dic['column'].tolist())

    def test_autoresearch_early_exit_after_two_no_gain_llm_rounds(self):
        from ag_features import Prepared
        from ag_research import Researcher
        from ag_utils import Budget
        rng = np.random.default_rng(77)
        X = pd.DataFrame({'a': rng.normal(size=120), 'b': rng.normal(size=120)})
        Xt = pd.DataFrame({'a': rng.normal(size=20), 'b': rng.normal(size=20)})
        y = 2.0 * X['a'].values + rng.normal(0, 0.3, 120)
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        plan = make_plan(X, y, 'regression', n_folds=3)
        prep = Prepared(X=X, Xt=Xt, y=y, cat_cols=[])

        class DummyLLM:
            enabled = True
            calls = 0
            def chat(self, *args, **kwargs):
                self.calls += 1
                return '{"analysis":"test","hypotheses":[{"title":"noop","rationale":"r","action":{"type":"params","params":{"min_split_gain":' + str(5.0 + self.calls) + '}}}]}'

        with tempfile.TemporaryDirectory() as d:
            r = Researcher(spec, prep, plan, DummyLLM(), Budget(300.0), n_jobs=2, work=Path(d))
            r.queue = []
            r.builtin = lambda: None
            r.run(deadline=1e9, max_experiments=20)
            self.assertEqual(r.llm_rounds, 2)
            self.assertEqual(r.no_gain_llm_rounds, 2)

    def test_mlp_dual_ema_swa_and_refit_bagging(self):
        from ag_models import ModelConfig, fit_mlp
        from solution import model_family
        self.assertEqual(model_family('mlp_dual'), 'mlp_dual')
        self.assertEqual(model_family('mlp_dual_s2'), 'mlp_dual')
        self.assertEqual(model_family('mlp_s3'), 'mlp')
        rng = np.random.default_rng(123)
        n = 360
        X = pd.DataFrame({
            't__ord': np.arange(n, dtype=np.float64),
            'a': rng.normal(size=n),
            'b': rng.exponential(2.0, size=n),
            'c': pd.Categorical(rng.choice(['u', 'v', 'w'], size=n)),
        })
        Xt = pd.DataFrame({
            't__ord': np.arange(n, n + 30, dtype=np.float64),
            'a': rng.normal(size=30),
            'b': rng.exponential(2.0, size=30),
            'c': pd.Categorical(rng.choice(['u', 'v', 'w'], size=30)),
        })
        y = 10.0 + 3.0 * X['a'].values - 1.5 * X['b'].values + (X['c'] == 'u').astype(float).values * 2.0 + rng.normal(0, 0.3, n)
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        plan = make_plan(X, y, 'regression', time_col='t__ord')
        r_dual = fit_mlp(X, y, Xt, plan, spec, ModelConfig(), n_jobs=2, time_limit=60, seed=0,
                         name='mlp_dual', prep_mode='dual')
        self.assertIsNotNone(r_dual)
        self.assertLess(r_dual.score, 1.0)
        self.assertEqual(len(r_dual.test), len(Xt))
        self.assertTrue(np.isfinite(r_dual.test).all())
        # Finite-difference gradient check on _NumpyTabularResNet (SwiGLU + Ws skip + s_in feature gate)
        from ag_models import _NumpyTabularResNet
        net = _NumpyTabularResNet(6, 1, loss_kind='huber', hidden=16, h2=8, dropout=0.0, seed=7)
        xb = rng.normal(size=(8, 6)).astype(np.float32)
        yb = rng.normal(size=(8,)).astype(np.float32)
        wb = np.ones(8, dtype=np.float32)
        out, cache = net._forward(xb, train=True)
        loss0, dout = net._loss_grad(out, yb, wb)
        x_raw, x, c_n1, c_a1, m1, a1d, c_nr, c_ar, ar_g, vr, mr, h1, c_n2, c_a2, m2, a2d = cache
        da2d = dout @ net.p['W3'].T
        dz2, _, _ = net._ln_bwd(net._silu_bwd(da2d, c_a2), c_n2)
        dh1 = dz2 @ net.p['W2'].T
        dar_g, dvr = dh1 * vr, dh1 * ar_g
        dzr, _, _ = net._ln_bwd(net._silu_bwd(dar_g, c_ar), c_nr)
        da1d = dh1 + dzr @ net.p['Wr'].T + dvr @ net.p['Wv'].T
        dz1, _, _ = net._ln_bwd(net._silu_bwd(da1d, c_a1), c_n1)
        g_s_in = np.sum((dz1 @ net.p['W1'].T + dout @ net.p['Ws'].T) * x_raw, axis=0)
        g_Ws = x.T @ dout
        eps = 1e-3
        for pname, idx_p, ana in [('s_in', 0, float(g_s_in[0])), ('Ws', (0, 0), float(g_Ws[0, 0]))]:
            orig = float(net.p[pname][idx_p])
            net.p[pname][idx_p] = orig + eps
            lp, _ = net._loss_grad(net._forward(xb, train=False)[0], yb, wb)
            net.p[pname][idx_p] = orig - eps
            lm, _ = net._loss_grad(net._forward(xb, train=False)[0], yb, wb)
            net.p[pname][idx_p] = orig
            num_g = (lp - lm) / (2.0 * eps)
            self.assertAlmostEqual(ana, num_g, delta=1e-2, msg=f'Grad mismatch on {pname}: {ana} vs {num_g}')

    def test_subdaily_1h_6h_causal_velocity_and_smoothed_entity_history(self):
        from ag_features import op_entity_history
        # 1) Sub-daily child transactions (every 15 min) -> strictly causal 1h/6h windows (< event_time)
        ts_tx = pd.date_range('2025-03-01 00:00:00', periods=30, freq='15min')
        tx = pd.DataFrame({
            'client_id': ['c1'] * 30,
            'created_at': ts_tx.astype(str),
            'amt': [10.0] * 25 + [99999.0] * 5,  # index 25 is at 06:15:00 -> tie/future for 06:15:00 event!
        })
        tr = pd.DataFrame({'event_id': [1, 2], 'client_id': ['c1', 'c1'],
                           'timestamp': ['2025-03-01 06:15:00', '2025-03-01 06:30:00'], 'y': [0, 1]})
        te = pd.DataFrame({'event_id': [3], 'client_id': ['c1'], 'timestamp': ['2025-03-01 06:15:00']})
        a, b = enrich(tr, te, {'tx': tx})
        self.assertIn('tx__hist_cnt_1h', a.columns)
        self.assertIn('tx__hist_cnt_6h', a.columns)
        self.assertIn('tx__amt__hist_sum_1h', a.columns)
        self.assertIn('tx__hist_burst_1h_1d', a.columns)
        # At 06:15:00, strictly past 1h window is [05:15:00, 06:15:00) -> 05:15, 05:30, 05:45, 06:00 = 4 tx, sum=40 (NEVER 99999!)
        self.assertEqual(float(a['tx__hist_cnt_1h'].iloc[0]), 4.0)
        self.assertAlmostEqual(float(a['tx__amt__hist_sum_1h'].iloc[0]), 40.0, places=3)
        self.assertAlmostEqual(float(b['tx__amt__hist_sum_1h'].iloc[0]), 40.0, places=3)
        # 2) Bayesian-smoothed causal entity history never leaks future periods
        X = pd.DataFrame({'uid': ['u1', 'u2'] * 10, 't': np.repeat(np.arange(5, dtype=float), 4)})
        Xt = pd.DataFrame({'uid': ['u1', 'u_new'], 't': [5.0, 5.0]})
        y1 = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(20)])
        y2 = y1.copy(); y2[12:] = 1.0 - y2[12:]  # flip periods 3 and 4
        h1_tr, h1_te = op_entity_history(X, Xt, y1, ['uid'], 't')
        h2_tr, _ = op_entity_history(X, Xt, y2, ['uid'], 't')
        self.assertIn('hist__smooth', h1_tr.columns)
        # Periods <= 3 must be identical under future target mutation!
        np.testing.assert_allclose(h1_tr.loc[X['t'] <= 3.0, 'hist__smooth'].values,
                                   h2_tr.loc[X['t'] <= 3.0, 'hist__smooth'].values, equal_nan=True)
        # Unseen entity u_new gets the global past mean (0.5), u1 gets smoothed > 0.5
        self.assertAlmostEqual(float(h1_te['hist__smooth'].iloc[1]), 0.5, places=3)
        self.assertGreater(float(h1_te['hist__smooth'].iloc[0]), 0.65)

    def test_fast_bin_rank_score_exact_and_large_10model_stack_blend(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        from ag_ensemble import _fast_bin_rank_score, stack_blend
        from ag_models import FitResult, FoldPlan
        rng = np.random.default_rng(131)
        # 1) Exact match vs sklearn including tied scores!
        y = rng.binomial(1, 0.15, size=2000)
        s = np.round(rng.normal(size=2000) + 0.8 * y, 1)  # deliberate ties
        self.assertAlmostEqual(_fast_bin_rank_score('average_precision', y, s),
                               float(average_precision_score(y, s)), places=11)
        self.assertAlmostEqual(_fast_bin_rank_score('roc_auc', y, s),
                               float(roc_auc_score(y, s)), places=11)
        # 2) Large 10-model PR-AUC stack_blend must NOT collapse to uniform 0.1 weights
        n = 40_000
        y2 = rng.binomial(1, 0.12, size=n)
        sig = 1.2 * y2 + rng.normal(0, 1.0, n)
        def mk(z):
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
            return np.column_stack([1.0 - p, p])
        rs = [FitResult('strong1', mk(sig + rng.normal(0, 0.4, n)), mk(rng.normal(size=50)), 0.5, [], [], None, 1.0),
              FitResult('strong2', mk(sig + rng.normal(0, 0.5, n)), mk(rng.normal(size=50)), 0.48, [], [], None, 1.0)]
        for j in range(8):
            rs.append(FitResult(f'noise{j}', mk(rng.normal(0, 1.5, n)), mk(rng.normal(size=50)), 0.12, [], [], None, 1.0))
        mask = np.zeros(n, bool); mask[10_000:] = True
        plan = FoldPlan(folds=[(np.arange(10_000), np.arange(10_000, 25_000)),
                               (np.arange(25_000), np.arange(25_000, n))],
                        mode='refit', oof_mask=mask, time_col='t', desc='time')
        spec = TaskSpec('', '', id_col='id', target='y', task_type='binary', metric='average_precision', classes=[0, 1])
        sb = stack_blend(rs, y2, mask, spec, plan=plan)
        self.assertIsNotNone(sb)
        # Strong models must hold > 85% of total weight (previously collapsed to 0.1 each!)
        w_strong = sb['weights'].get('strong1', 0.0) + sb['weights'].get('strong2', 0.0)
        self.assertGreater(w_strong, 0.85, f'Unexpected weights: {sb["weights"]}')

    def test_error_analysis_binary_ranking_and_regression_mae_bias(self):
        from ag_features import Prepared
        from ag_models import FitResult
        from ag_research import Researcher
        from ag_utils import Budget
        rng = np.random.default_rng(149)
        n = 500
        X = pd.DataFrame({'seg': pd.Categorical(rng.choice(['A', 'B'], n)), 'x': rng.normal(size=n)})
        Xt = X.iloc[:20].copy()
        # 1) Binary imbalanced PR-AUC must report top-10% lift, NEVER misleading 0.5-threshold F1=0.000
        yb = (rng.random(n) < 0.12).astype(int)
        p1 = np.clip(0.08 + 0.25 * yb + rng.normal(0, 0.05, n), 0.01, 0.45)  # all < 0.5!
        spec_b = TaskSpec('', '', id_col='id', target='y', task_type='binary', metric='average_precision', classes=[0, 1])
        plan_b = make_plan(X, yb, 'binary', n_folds=3)
        with tempfile.TemporaryDirectory() as d:
            rb = Researcher(spec_b, Prepared(X=X, Xt=Xt, y=yb, cat_cols=['seg']), plan_b, None, Budget(60.0), 1, Path(d))
            rb.best = FitResult('b', np.column_stack([1.0 - p1, p1]), np.full((20, 2), 0.5), 0.7, [], [], None, 1.0)
            ea_b = rb.error_analysis()
            self.assertIn('top-10% lift=', ea_b)
            self.assertNotIn('per-class F1:', ea_b)
            # 2) Regression MAE must report mae and bias
            yr = 10.0 + 2.0 * X['x'].values + rng.normal(0, 1.0, n)
            spec_r = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
            plan_r = make_plan(X, yr, 'regression', n_folds=3)
            rr = Researcher(spec_r, Prepared(X=X, Xt=Xt, y=yr, cat_cols=['seg']), plan_r, None, Budget(60.0), 1, Path(d))
            rr.best = FitResult('r', yr + 0.5, np.full(20, 10.0), 0.5, [], [], None, 1.0)
            ea_r = rr.error_analysis()
            self.assertIn('median residual=', ea_r)
            self.assertIn('mae=', ea_r)


class Pool5HarnessTests(unittest.TestCase):
    def test_synthetic_generators_deterministic_and_disjoint(self):
        from tools.stress_private import generate, CASES
        for name in CASES:
            with self.subTest(name=name):
                a, b, aux, sample, y = generate(name, 600, 42)
                aa, bb, aux2, sample2, y2 = generate(name, 600, 42)
                self.assertEqual(len(a) + len(b), 600)
                self.assertTrue(set(a.row_id).isdisjoint(b.row_id))
                self.assertNotIn('target', b)
                pd.testing.assert_frame_equal(a, aa)
                pd.testing.assert_frame_equal(sample, sample2)
                np.testing.assert_array_equal(y, y2)
                if name == 'multiclass':
                    self.assertEqual(a.target.nunique(), 12)
                    self.assertEqual(len(sample.columns), 13)
                if name == 'regression':
                    self.assertTrue((y == 0).any())

    def test_stress_contract_rejects_bad_order_and_probabilities(self):
        from tools.stress_private import generate, validate
        a, b, aux, sample, y = generate('multiclass', 600)
        good = sample.copy(); good.iloc[:, 1:] = 1 / 12
        scores, constant = validate(good, sample, b, y, 'multiclass')
        self.assertTrue(constant); self.assertAlmostEqual(scores['LogLoss'], np.log(12))
        with self.assertRaises(ValueError): validate(good.iloc[::-1], sample, b, y, 'multiclass')
        bad = good.copy(); bad.iloc[0, 1] = 2
        with self.assertRaises(ValueError): validate(bad, sample, b, y, 'multiclass')
        bad = good.copy(); bad.iloc[0, 1] = 0
        with self.assertRaises(ValueError): validate(bad, sample, b, y, 'multiclass')

    def test_release_whitelist_rejects_unapproved_versions(self):
        from tools.release_check import check_requirements
        reg = 'Approved packages:\nnumpy==1.26.4\npandas==2.2.3\n'
        self.assertEqual(check_requirements('numpy==1.26.4\n', reg)['used'], 1)
        for wrong in ['numpy==2.0.0', 'torch==1.0', 'numpy>=1.0', 'numpy==1.26.4\nnumpy==1.26.4']:
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                check_requirements(wrong, reg)

    def test_dual_poison_audit_with_synthetic_public_fixture(self):
        from tools.release_check import poison_audit
        from unittest.mock import patch
        from ag_features import causal_stream_features
        cutoff = pd.Timestamp('2025-08-01 06:50:00')
        times = pd.date_range(cutoff - pd.Timedelta(minutes=6500), periods=750, freq='10min')
        rng = np.random.default_rng(4)
        df = pd.DataFrame({'event_id': np.arange(750) + 10512, 'timestamp': times,
                           'district_id': 'D06', 'flag': rng.integers(0, 2, 750),
                           'kind': rng.choice(['A', 'B'], 750)})
        tr = df.iloc[:600].copy(); tr['target'] = rng.normal(size=600)
        te = df.iloc[600:].reset_index(drop=True)
        sensor_times = pd.date_range(times.min().floor('h') - pd.Timedelta(hours=12), times.max().ceil('h') + pd.Timedelta(hours=2), freq='h')
        tables = {'train': tr, 'test': te}
        for name in ('weather', 'traffic', 'air_quality', 'power_grid', 'public_transport'):
            tables[name] = pd.DataFrame({'timestamp': sensor_times, 'district_id': 'D06', 'v': rng.normal(size=len(sensor_times))})
        report = poison_audit(tables)
        self.assertEqual(set(report), {'sensors', 'events', 'combined'})
        self.assertTrue(all(v['compared'] > 0 and not v['changed'] for v in report.values()))
        def leaky(a, b, exclude=()):
            x, z = causal_stream_features(a, b, exclude=exclude)
            x['deliberate_leak'] = a.flag.to_numpy(); z['deliberate_leak'] = b.flag.to_numpy()
            return x, z
        with patch('ag_features.causal_stream_features', side_effect=leaky):
            report = poison_audit(tables)
        self.assertIn('deliberate_leak', report['events']['changed'])
        self.assertIn('deliberate_leak', report['combined']['changed'])

    def test_release_zip_rejects_bad_content(self):
        import zipfile
        from tools.release_check import audit_zip
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'solution.py').write_text('print(1)\n'); (root / 'requirements.txt').write_text('numpy==1.26.4\n')
            archive = root / 'test.zip'
            def pack(extra=None):
                with zipfile.ZipFile(archive, 'w') as z:
                    for n in ('solution.py', 'requirements.txt'): z.write(root / n, n)
                    if extra: z.writestr(extra, 'extra')
            pack(); self.assertEqual(audit_zip(archive, root)['files'], 2)
            pack('nested/a.py')
            with self.assertRaises(ValueError): audit_zip(archive, root)
            (root / 'solution.py').write_bytes(b'print(1)\r\n'); pack()
            with self.assertRaises(ValueError): audit_zip(archive, root)
            (root / 'solution.py').write_text('import sys\nsys.path.append("/x")\n'); pack()
            with self.assertRaises(ValueError): audit_zip(archive, root)
            (root / 'solution.py').write_text('token="sk-' + 'x' * 20 + '"\n'); pack()
            with self.assertRaises(ValueError): audit_zip(archive, root)

    def test_entity_key_candidates_excludes_eq_and_caps_key_length(self):
        from ag_features import entity_key_candidates
        rng = np.random.default_rng(7)
        n = 600
        df = pd.DataFrame({
            't': np.repeat(np.arange(6), 100).astype(float),
            'district_id': np.tile([f'D{i:02d}' for i in range(20)], 30),
            'kind': rng.choice(['A', 'B', 'C', 'D'], n),
            'sev': rng.choice(['L', 'M', 'H'], n),
            'aux__sensor_status': rng.choice(['ok', 'warn', 'fail'], n),
            'aux__a__eq__aux__b': rng.choice([0.0, 1.0], n),
            'aux__cond': rng.choice(['sun', 'rain', 'snow', 'fog'], n),
        })
        for j in range(10):
            df[f'aux__cat{j}'] = rng.choice(['x', 'y', 'z'], n)
        cats = [c for c in df.columns if c not in ('t', 'aux__a__eq__aux__b')]
        keys = entity_key_candidates(df, 't', cats, extra=['district_id'])
        self.assertTrue(keys)
        for k in keys:
            self.assertLessEqual(len(k), 4)
            self.assertFalse(any('__eq__' in c for c in k))
            self.assertFalse(any(c.startswith('aux__') for c in k))

    def test_resnet_chunked_predict_exact_and_stratified_es(self):
        from ag_models import _NumpyTabularResNet
        rng = np.random.default_rng(11)
        X = rng.normal(size=(400, 12)).astype(np.float32)
        y = np.zeros(400, dtype=np.float32)
        y[:12] = 1.0  # 3% minority positives
        sw = np.ones(400, dtype=np.float32)
        net = _NumpyTabularResNet(12, 1, loss_kind='bce', hidden=32, h2=16, seed=3)
        net.fit(X, y, sw, max_epochs=8, batch_size=64, patience=5, n_jobs=2)
        p_full = net.predict(X, batch_size=1000)
        p_chunk = net.predict(X, batch_size=37)
        np.testing.assert_allclose(p_full, p_chunk, atol=1e-6)
        out_nt, cache_nt = net._forward(X[:10], train=False)
        self.assertIsNone(cache_nt)
        self.assertEqual(out_nt.shape, (10, 1))
        from solution import model_family
        self.assertEqual(model_family('cat_top_s2'), 'cat_top')
        self.assertEqual(model_family('mlp_dual_s11'), 'mlp_dual')

    def test_causal_cat_nunique_exact_int64_microsecond_no_tie_leak(self):
        from ag_relational import _causal_window_cat_nunique
        # Test 200 consecutive microsecond timestamps where float64(ns) rounds UP on ~48% of values
        base = pd.Timestamp('2025-02-02 12:00:00.100000').value
        ts = np.array([base + i * 1000 for i in range(200)], dtype=np.int64)
        codes = np.arange(200, dtype=np.int64)  # 200 distinct entities, each has ONE event at exact tie ts[i]
        cats = np.zeros(200, dtype=np.int32)
        wins = [86400 * 1_000_000_000]
        # Exact tie MUST yield 0.0 for every single microsecond timestamp (strictly < t)
        out_tie = _causal_window_cat_nunique(codes, ts, cats, codes, ts, wins, q_valid=np.ones(200, bool))
        self.assertEqual(float(out_tie.max()), 0.0)
        # 1 nanosecond AFTER child event MUST yield 1.0
        out_after = _causal_window_cat_nunique(codes, ts, cats, codes, ts + 1, wins, q_valid=np.ones(200, bool))
        self.assertEqual(float(out_after.min()), 1.0)

    def test_fit_mlp_refit_never_returns_untrained_test_after_time_cut(self):
        from ag_models import FoldPlan, ModelConfig, fit_mlp
        rng = np.random.default_rng(19)
        n = 240
        t = np.arange(n, dtype=float)
        x = rng.normal(size=n)
        y = 10.0 + 4.0 * x + rng.normal(0, 0.2, n)
        X = pd.DataFrame({'t': t, 'x1': x, 'x2': x * 0.5 + rng.normal(0, 0.1, n)})
        Xt = pd.DataFrame({'t': np.arange(n, n + 40, dtype=float), 'x1': np.linspace(-2, 2, 40), 'x2': np.linspace(-1, 1, 40)})
        folds = [(np.arange(0, 140), np.arange(140, 190)), (np.arange(0, 190), np.arange(190, 240))]
        mask = np.zeros(n, bool); mask[140:] = True
        plan = FoldPlan(folds=folds, mode='refit', oof_mask=mask, time_col='t', desc='time')
        spec = TaskSpec('', '', id_col='id', target='y', task_type='regression', metric='mae')
        # Tiny time_limit forces guard.expired() right after folds; refit MUST extend deadline and learn y ~ 10 + 4*x1
        r = fit_mlp(X, y, Xt, plan, spec, ModelConfig(), n_jobs=1, time_limit=5.0, seed=0)
        self.assertIsNotNone(r)
        self.assertIsNotNone(r.test)
        corr = float(np.corrcoef(r.test, 10.0 + 4.0 * Xt['x1'].values)[0, 1])
        self.assertGreater(corr, 0.85)

    def test_worker_poison_filter_int8_and_fallback_trunc(self):
        import os, subprocess
        worker = Path(__file__).resolve().parent.parent / 'ag_worker.py'
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            rng = np.random.default_rng(23)
            tr = pd.DataFrame({'t': np.arange(60, dtype=float), 'flag8': rng.integers(0, 5, 60, dtype=np.int8), 'a': rng.normal(size=60)})
            te = pd.DataFrame({'t': np.arange(60, 100, dtype=float), 'flag8': rng.integers(0, 5, 40, dtype=np.int8), 'a': rng.normal(size=40)})
            tr.to_pickle(dp / 'train.pkl'); te.to_pickle(dp / 'test.pkl')
            # Non-constant leaky feature on narrow int8 + clean row-wise feature
            (dp / 'code.py').write_text(
                'def make_features(train, test):\n'
                '    m = float(test["flag8"].mean())\n'
                '    ok_tr = pd.DataFrame({"good": train["flag8"].astype(float) * 2.0, "leak": train["a"] + m})\n'
                '    ok_te = pd.DataFrame({"good": test["flag8"].astype(float) * 2.0, "leak": test["a"] + m})\n'
                '    return ok_tr, ok_te\n',
                encoding='utf-8',
            )
            res = subprocess.run([sys.executable, str(worker), str(dp), 't'], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn('POISON_DROP: leak', res.stdout)
            out_tr = pd.read_pickle(dp / 'out_train.pkl')
            self.assertEqual(list(out_tr.columns), ['good'])

if __name__=='__main__': unittest.main()
