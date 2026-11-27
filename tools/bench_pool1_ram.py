"""Pool 1.4 benchmark: peak RAM on ~2M-row table (downcast + SpooledTemporaryFile + causal child rolling windows)."""
import gc
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag_database import _downcast_df, _spooled_temp
from ag_relational import enrich


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def main():
    rng = np.random.default_rng(42)
    n_child = 2_000_000
    print(f'Start RSS: {rss_gb():.3f} GB', flush=True)

    # 1. Simulate streaming COPY to SpooledTemporaryFile + read_csv + _downcast_df on 2M rows
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix='ram2m_'))
    with _spooled_temp(work) as buf:
        buf.write(b'client_id,timestamp,amount,fee,channel\n')
        chunk = 200_000
        base_ts = pd.Timestamp('2025-01-01', tz='UTC')
        for start in range(0, n_child, chunk):
            cid = rng.integers(0, 20_000, size=chunk)
            secs = rng.integers(0, 90 * 86400, size=chunk)
            ts = (base_ts + pd.to_timedelta(secs, unit='s')).strftime('%Y-%m-%d %H:%M:%S')
            amt = np.round(rng.lognormal(3.0, 1.0, size=chunk), 2)
            fee = np.round(amt * 0.02, 2)
            ch = rng.choice(['web', 'app', 'pos', 'atm'], size=chunk)
            part = pd.DataFrame({'client_id': cid, 'timestamp': ts, 'amount': amt, 'fee': fee, 'channel': ch})
            buf.write(part.to_csv(index=False, header=False).encode('utf-8'))
            del part
        buf.seek(0)
        print(f'After writing 2M CSV to SpooledTemporaryFile: RSS={rss_gb():.3f} GB ({time.time()-t0:.1f}s)', flush=True)
        df_raw = pd.read_csv(buf, low_memory=False)
    mem_before = df_raw.memory_usage(deep=True).sum() / (1024 ** 3)
    rss_before = rss_gb()
    df_dc = _downcast_df(df_raw)
    gc.collect()
    mem_after = df_dc.memory_usage(deep=True).sum() / (1024 ** 3)
    rss_after = rss_gb()
    print(f'2M table RAM: df_mem {mem_before:.3f} -> {mem_after:.3f} GB | RSS {rss_before:.3f} -> {rss_after:.3f} GB | dtypes={dict(df_dc.dtypes)}', flush=True)

    # 2. Benchmark causal one-to-many rolling windows (1d/7d/30d + 7d/30d + nunique) on 2M child rows
    n_tr, n_te = 40_000, 10_000
    tr = pd.DataFrame({
        'event_id': np.arange(n_tr, dtype=np.int32),
        'client_id': rng.integers(0, 20_000, size=n_tr, dtype=np.int32),
        'timestamp': (pd.Timestamp('2025-03-01', tz='UTC') + pd.to_timedelta(rng.integers(0, 30 * 86400, n_tr), unit='s')).strftime('%Y-%m-%d %H:%M:%S'),
        'y': rng.integers(0, 2, size=n_tr),
    })
    te = pd.DataFrame({
        'event_id': np.arange(n_tr, n_tr + n_te, dtype=np.int32),
        'client_id': rng.integers(0, 20_000, size=n_te, dtype=np.int32),
        'timestamp': (pd.Timestamp('2025-04-01', tz='UTC') + pd.to_timedelta(rng.integers(0, 10 * 86400, n_te), unit='s')).strftime('%Y-%m-%d %H:%M:%S'),
    })
    t1 = time.time()
    rep = []
    a, b = enrich(tr, te, {'transactions': df_dc}, report=rep)
    dt_enr = time.time() - t1
    print(f'enrich 2M child -> 50k events: {dt_enr:.2f}s, out_shape={a.shape}, RSS={rss_gb():.3f} GB, report={rep}', flush=True)


if __name__ == '__main__':
    main()
