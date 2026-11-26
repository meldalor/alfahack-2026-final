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
        # A conflicting hourly observation must not be selected arbitrarily.
        tr=pd.DataFrame({'district_id':['a','a','a'], 'timestamp':['2025-01-01 02:15:00','2024-12-31 23:59:00','2025-01-01 00:30:00'], 'y':[0,1,2]})
        te=pd.DataFrame({'district_id':['a','a'], 'timestamp':['2025-01-01 04:00:00','2025-01-01 03:59:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertEqual(a.sensor__v.iloc[0],30)
        self.assertTrue(pd.isna(a.sensor__v.iloc[1]))
        self.assertEqual(a.sensor__v.iloc[2],10)
        self.assertTrue(pd.isna(b.sensor__v.iloc[0]))
        self.assertEqual(b.sensor__v.iloc[1],40)
        self.assertEqual(a.timestamp.tolist(),tr.timestamp.tolist())

    def test_temporal_lookup_does_not_cross_groups(self):
        stamps=pd.date_range('2025-01-01',periods=4,freq='h').astype(str).tolist()
        dim=pd.DataFrame({'district_id':['a']*4+['b']*4,'timestamp':stamps*2,'v':[1,2,3,4,101,102,103,104]})
        tr=pd.DataFrame({'district_id':['b','a'],'timestamp':['2025-01-01 01:10:00']*2,'y':[0,1]})
        te=pd.DataFrame({'district_id':['c'],'timestamp':['2025-01-01 01:10:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertEqual(a.sensor__v.tolist(),[102,2])
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
        tr=pd.DataFrame({'district_id':['a','a'],'timestamp':['2025-01-01 00:30:00','2025-01-01 02:30:00'],'y':[0.,1.]})
        te=pd.DataFrame({'district_id':['a'],'timestamp':['2025-01-01 03:30:00']})
        a,b=enrich(tr,te,{'sensor':dim})
        self.assertTrue(pd.isna(a.sensor__v.iloc[0]))  # no past value -> stays NaN, never uses future 10.0
        self.assertEqual(a.sensor__v.iloc[1],10.)      # outage at 02:00 filled from 01:00
        self.assertEqual(b.sensor__v.iloc[0],10.)      # outage at 03:00 filled from 01:00, never 99.0
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
        # int64 key in train/test vs padded string key in dimension + custom created_at time column
        tr=pd.DataFrame({'event_id':[1,2],'user_id':[10,20],'created_at':['2025-01-01 01:30:00','2025-01-01 02:30:00'],'y':[0,1]})
        te=pd.DataFrame({'event_id':[3],'user_id':[10],'created_at':['2025-01-01 03:30:00']})
        users=pd.DataFrame({'user_id':[' 10 ','20'],'tier':['gold','silver']})
        stamps=pd.date_range('2025-01-01',periods=4,freq='h').astype(str).tolist()
        sensor=pd.DataFrame({'user_id':['10']*4,'created_at':stamps,'val':[1.,2.,3.,4.]})
        a,b=enrich(tr,te,{'users':users,'sensor':sensor})
        self.assertEqual(a.users__tier.tolist(),['gold','silver'])
        self.assertEqual(a.sensor__val.iloc[0],2.)
        self.assertEqual(b.sensor__val.iloc[0],4.)

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

if __name__=='__main__': unittest.main()
