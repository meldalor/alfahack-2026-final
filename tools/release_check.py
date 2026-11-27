"""Fail-closed release gate. Run with PG* read-only credentials and --regulation path.
No upload, no SQL writes, no changes to production source.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PIN=re.compile(r'(?im)^\s*(?:[-*]\s+)?`?([a-z0-9_.-]+(?:\[[a-z0-9_,.-]+\])?==[a-z0-9.+!-]+)`?\s*(?:#.*)?$')
SECRET_PATTERNS=[('private key',r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
                 ('API key',r'\bsk-[A-Za-z0-9_-]{12,}\b'),
                 ('IP address',r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
                 ('literal credential',r'''(?i)\b(?:PGPASSWORD|OPENAI_API_KEY|llm_key|password|api_key)\s*[=:]\s*["'][^"'\s]{5,}["']'''),
                 ('credential URL',r'(?i)\b(?:postgres(?:ql)?|https?)://[^\s:/@]+:[^\s/@]+@'),
                 ('unquoted credential',r'(?im)^\s*(?:export\s+)?(?:PGPASSWORD|OPENAI_API_KEY|llm_key)\s*=\s*[A-Za-z0-9_-]{12,}\s*$')]


def requirements(text):
    lines=[s.split('#',1)[0].strip() for s in text.splitlines() if s.split('#',1)[0].strip()]
    if any(not PIN.fullmatch(s) for s in lines): raise ValueError('requirements must contain exact pinned package lines only')
    def canon(s):
        name,version=s.split('==',1); return re.sub(r'[-_.]+','-',name.lower()),version
    pairs=[canon(s) for s in lines]
    if len(dict(pairs))!=len(pairs): raise ValueError('duplicate package pins')
    return dict(pairs)


def check_requirements(req,reg):
    actual=requirements(req); pins=PIN.findall(reg)
    if not pins: raise ValueError('no exact pins recognized in tech_regulation.txt; supply original whitelist in one-pin-per-line format')
    allowed=requirements('\n'.join(dict.fromkeys(pins)))
    bad={k:v for k,v in actual.items() if allowed.get(k)!=v}
    if bad: raise ValueError(f'unapproved package/version: {bad}')
    return {'used':len(actual),'approved':len(allowed),'unused_approved':sorted(set(allowed)-set(actual))}


def audit_zip(path,root=ROOT):
    expected={'solution.py','requirements.txt',*[p.name for p in root.glob('ag_*.py')]}
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=expected or any('/' in n or '\\' in n for n in names): raise ValueError('ZIP is not flat or exact file set differs')
        if z.testzip() is not None: raise ValueError('ZIP CRC error')
        for name in names:
            data=z.read(name)
            if data!=(root/name).read_bytes(): raise ValueError(f'{name}: differs from working tree')
            if b'\r' in data: raise ValueError(f'{name}: non-LF line endings')
            text=data.decode('utf-8')
            for kind,pattern in SECRET_PATTERNS:
                if re.search(pattern,text): raise ValueError(f'{name}: possible {kind} (value redacted)')
            if re.search(r'\bsys\s*\.\s*path\b|\bPYTHONPATH\b',text): raise ValueError(f'{name}: forbidden path manipulation/reference')
            if name.endswith('.py'): ast.parse(text,filename=name)
    return {'files':len(names),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def poison_audit(tables):
    """Audit actual Task 2 tables (names here are intentionally audit-only)."""
    import numpy as np
    import pandas as pd
    from ag_relational import enrich
    from ag_features import causal_stream_features
    names=('weather','traffic','air_quality','power_grid','public_transport')
    tr,te=tables['train'],tables['test']; t=pd.to_datetime(te.timestamp,utc=True)
    cutoff=pd.Timestamp('2025-08-01 06:50:00',tz='UTC'); idx=np.flatnonzero(t.eq(cutoff).to_numpy() & te.event_id.astype(str).eq('11162').to_numpy())
    if len(idx)!=1: raise ValueError('documented audit event not uniquely found')
    idx=int(idx[0]); hour=cutoff.floor('h')
    for name in names:
        if name not in tables or not pd.to_datetime(tables[name].timestamp,utc=True).ge(hour).any(): raise ValueError(f'missing future sensor rows: {name}')
    def snapshot(source,test):
        a,b=enrich(tr,test,source)
        if len(b)!=len(test) or not b.event_id.equals(test.event_id.reset_index(drop=True)): raise ValueError('join changed test IDs/order')
        _,stream=causal_stream_features(a,b,exclude=set(tr)-set(te)|{'event_id'})
        derived=[c for c in b if c not in test]
        if not derived or stream.empty: raise ValueError('no relational/stream features to audit')
        for name in names:
            if not any(c.startswith(name+'__') for c in derived): raise ValueError(f'{name} not joined')
        return pd.concat([b.loc[idx,derived],stream.iloc[idx]])
    def mutate(frame,mask,columns):
        out=frame.copy()
        for c in columns:
            s=out[c]
            if pd.api.types.is_numeric_dtype(s):
                vals=set(s.dropna().unique()) if s.nunique(dropna=True)<=2 else set()
                out.loc[mask,c]=1-s.loc[mask] if vals and vals<={0,1} else 987654
            else: out.loc[mask,c]='POISON_TYPE'
        return out
    baseline=snapshot(tables,te)
    poisoned=dict(tables)
    for name in names:
        frame=tables[name]; cols=[c for c in frame if c not in ('timestamp','district_id') and not c.endswith('_id')]
        poisoned[name]=mutate(frame,pd.to_datetime(frame.timestamp,utc=True).ge(hour),cols)
    cols=[c for c in te if c not in ('timestamp','event_id','district_id') and not c.endswith('_id')]
    events=mutate(te,t.ge(cutoff),cols)
    results={}
    for label,source,test in [('sensors',poisoned,te),('events',tables,events),('combined',poisoned,events)]:
        other=snapshot(source,test)
        if set(other.index)!=set(baseline.index): raise ValueError(f'{label}: feature schema changed')
        changed=[]
        for c,v in baseline.items():
            w=other[c]
            if pd.isna(v) and pd.isna(w): continue
            if pd.isna(v) or pd.isna(w) or v!=w: changed.append(c)
        results[label]={'compared':len(baseline),'changed':changed}
    return results


def db_worker():
    from ag_database import extract
    os.environ['PGDATABASE']='hack_final_task2'
    with tempfile.TemporaryDirectory() as d: results=poison_audit(extract(Path(d)))
    print('AUDIT_JSON='+json.dumps(results),flush=True)
    return int(any(r['changed'] for r in results.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--regulation',type=Path,default=ROOT/'tech_regulation.txt'); p.add_argument('--zip',type=Path); p.add_argument('--report',type=Path,default=ROOT/'runs/release_check.json'); p.add_argument('--skip-db',action='store_true',help='local-only; still NOT a release pass'); p.add_argument('--db-timeout',type=int,default=1200); p.add_argument('--db-worker',action='store_true',help=argparse.SUPPRESS); a=p.parse_args()
    if a.db_worker: return db_worker()
    report={}
    def gate(name,fn):
        try: report[name]={'status':'PASS','details':fn()}
        except Exception as exc: report[name]={'status':'FAIL','details':str(exc)}
        print(name,report[name]['status'],flush=True)
    def cmd(args,timeout=300):
        cp=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        if cp.returncode: raise RuntimeError(cp.stdout[-10000:])
        return cp.stdout[-10000:]
    gate('py_compile',lambda:cmd(['-m','py_compile',*[str(f) for f in sorted(ROOT.glob('*.py'))+sorted((ROOT/'tools').glob('*.py'))]]))
    gate('unit_tests',lambda:cmd(['-m','unittest','discover','-s','tools','-p','test*.py']))
    gate('requirements',lambda:check_requirements((ROOT/'requirements.txt').read_text(),a.regulation.read_text()))
    with tempfile.TemporaryDirectory() as d:
        path=a.zip.resolve() if a.zip else Path(d)/'solution.zip'
        def archive():
            if not a.zip: cmd(['tools/make_zip.py',str(path)])
            return audit_zip(path)
        gate('archive',archive)
    if a.skip_db: report['poison_db']={'status':'SKIP','details':'real PostgreSQL audit explicitly skipped'}
    elif not all(os.environ.get(k) for k in ('PGHOST','PGUSER','PGPASSWORD')): report['poison_db']={'status':'FAIL','details':'missing PGHOST/PGUSER/PGPASSWORD; use read-only server environment'}
    else:
        def db():
            cp=subprocess.run([sys.executable,'-m','tools.release_check','--db-worker'],cwd=ROOT,capture_output=True,text=True,timeout=a.db_timeout)
            marker=next((s for s in cp.stdout.splitlines() if s.startswith('AUDIT_JSON=')),None)
            if not marker: raise RuntimeError('database audit did not complete; inspect connectivity/schema locally')
            result=json.loads(marker.split('=',1)[1])
            if cp.returncode or any(v['changed'] for v in result.values()): raise ValueError(json.dumps(result))
            return result
        gate('poison_db',db)
    report['release_ready']=all(v['status']=='PASS' for v in report.values()); a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); return int(not report['release_ready'])

if __name__=='__main__': sys.exit(main())
