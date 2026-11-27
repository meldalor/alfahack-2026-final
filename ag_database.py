"""Read-only PostgreSQL extraction via psycopg (official driver) with libpq fallback.
Connection comes from standard PG* environment variables (never logged).
"""
import ctypes as C
import ctypes.util
import gc
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_INT32_MIN = -2_147_483_648
_INT32_MAX = 2_147_483_647


def _convert_pg_booleans(df: pd.DataFrame, bool_cols: list[str]) -> pd.DataFrame:
    """PostgreSQL COPY CSV exports boolean as literal 't'/'f' strings; map them to clean numeric 1.0/0.0."""
    for c in bool_cols:
        if c not in df.columns or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype("string").str.strip().str.lower()
        df[c] = s.map({"t": 1.0, "true": 1.0, "1": 1.0, "f": 0.0, "false": 0.0, "0": 0.0}).astype("float32")
    return df


def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safe downcast (int64->int32 if in range, float64->float32 for non-IDs < 1e7) to cut peak RAM."""
    for c in df.columns:
        dt = df[c].dtype
        low = str(c).lower()
        if dt == np.int64:
            if len(df) == 0:
                df[c] = df[c].astype(np.int32)
            else:
                cmin, cmax = int(df[c].min()), int(df[c].max())
                if _INT32_MIN <= cmin and cmax <= _INT32_MAX:
                    df[c] = df[c].astype(np.int32)
        elif dt == np.float64:
            if low == 'id' or low.endswith('_id'):
                continue
            vals = df[c].to_numpy()
            fin = vals[np.isfinite(vals)]
            if len(fin) == 0 or float(np.max(np.abs(fin))) < 1e7:
                df[c] = df[c].astype(np.float32)
    return df


def _spooled_temp(tmp_dir: Path | None = None):
    d = str(tmp_dir) if (tmp_dir is not None and Path(tmp_dir).is_dir()) else None
    try:
        return tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024, dir=d, mode="w+b")
    except Exception:
        return tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024, mode="w+b")


# pg_catalog.pg_constraint works for non-owner read-only roles (e.g. hack_reader), unlike information_schema on PG
_PG_FK_SQL = (
    "SELECT ns.nspname AS src_schema, cls.relname AS src_table, sa.attname AS src_col, "
    "nd.nspname AS dst_schema, cld.relname AS dst_table, da.attname AS dst_col "
    "FROM pg_catalog.pg_constraint c "
    "JOIN pg_catalog.pg_class cls ON cls.oid = c.conrelid "
    "JOIN pg_catalog.pg_namespace ns ON ns.oid = cls.relnamespace "
    "JOIN pg_catalog.pg_class cld ON cld.oid = c.confrelid "
    "JOIN pg_catalog.pg_namespace nd ON nd.oid = cld.relnamespace "
    "JOIN LATERAL unnest(c.conkey, c.confkey) WITH ORDINALITY AS ord(src_att, dst_att, pos) ON TRUE "
    "JOIN pg_catalog.pg_attribute sa ON sa.attrelid = c.conrelid AND sa.attnum = ord.src_att "
    "JOIN pg_catalog.pg_attribute da ON da.attrelid = c.confrelid AND da.attnum = ord.dst_att "
    "WHERE c.contype = 'f' AND ns.nspname NOT IN ('pg_catalog', 'information_schema') "
    "ORDER BY ns.nspname, cls.relname, c.conname, ord.pos"
)

_FK_SQL = (
    "SELECT tc.table_schema AS src_schema, tc.table_name AS src_table, kcu.column_name AS src_col, "
    "ccu.table_schema AS dst_schema, ccu.table_name AS dst_table, ccu.column_name AS dst_col "
    "FROM information_schema.table_constraints tc "
    "JOIN information_schema.key_column_usage kcu "
    "  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
    "JOIN information_schema.constraint_column_usage ccu "
    "  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema "
    "WHERE tc.constraint_type = 'FOREIGN KEY' "
    "  AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')"
)


class _PsycopgDatabase:
    def __init__(self):
        import psycopg

        self.conn = psycopg.connect(connect_timeout=15)
        self.conn.autocommit = True
        self.query("SET default_transaction_read_only = on")
        self.query("SET statement_timeout = '300s'")

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def query(self, sql):
        with self.conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return pd.DataFrame()
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)

    def foreign_keys(self):
        for sql in (_PG_FK_SQL, _FK_SQL):
            try:
                df = self.query(sql)
                if df is not None and not df.empty:
                    return df.to_dict("records")
            except Exception:
                continue
        return []

    def table(self, schema, name, tmp_dir: Path | None = None):
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        literal = lambda s: "'" + s.replace("'", "''") + "'"
        types = self.query(
            "SELECT column_name,data_type FROM information_schema.columns WHERE table_schema="
            + literal(schema) + " AND table_name=" + literal(name)
        )
        text_types = {'text', 'character varying', 'character', 'uuid'}
        dtypes = {r.column_name: str for r in types.itertuples(index=False) if r.data_type in text_types}
        bool_cols = [r.column_name for r in types.itertuples(index=False) if r.data_type == 'boolean']
        with _spooled_temp(tmp_dir) as buf:
            with self.conn.cursor() as cur:
                with cur.copy(f'COPY (SELECT * FROM {quote(schema)}.{quote(name)}) TO STDOUT WITH CSV HEADER') as copy:
                    for chunk in copy:
                        buf.write(chunk)
            buf.seek(0)
            df = pd.read_csv(buf, low_memory=False, dtype=dtypes, keep_default_na=False, na_values=[""])
        return _downcast_df(_convert_pg_booleans(df, bool_cols))


class _LibpqDatabase:
    def __init__(self):
        path = ctypes.util.find_library('pq')
        self.lib = None
        for cand in [path, 'libpq.so.5', 'libpq.so']:
            if not cand:
                continue
            try:
                self.lib = C.CDLL(cand)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError('PostgreSQL requires psycopg or system libpq')
        for name, restype, args in [
            ('PQconnectdb', C.c_void_p, [C.c_char_p]),
            ('PQstatus', C.c_int, [C.c_void_p]),
            ('PQexec', C.c_void_p, [C.c_void_p, C.c_char_p]),
            ('PQresultStatus', C.c_int, [C.c_void_p]),
            ('PQntuples', C.c_int, [C.c_void_p]),
            ('PQnfields', C.c_int, [C.c_void_p]),
            ('PQfname', C.c_char_p, [C.c_void_p, C.c_int]),
            ('PQgetvalue', C.c_char_p, [C.c_void_p, C.c_int, C.c_int]),
            ('PQgetisnull', C.c_int, [C.c_void_p, C.c_int, C.c_int]),
            ('PQclear', None, [C.c_void_p]),
            ('PQfinish', None, [C.c_void_p]),
            ('PQgetCopyData', C.c_int, [C.c_void_p, C.POINTER(C.c_void_p), C.c_int]),
            ('PQfreemem', None, [C.c_void_p]),
        ]:
            fn = getattr(self.lib, name); fn.restype = restype; fn.argtypes = args
        self.conn = self.lib.PQconnectdb(b'connect_timeout=15')
        if not self.conn or self.lib.PQstatus(self.conn) != 0:
            self.close()
            raise RuntimeError('PostgreSQL connection failed; check PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD')
        self.query("SET default_transaction_read_only = on")
        self.query("SET statement_timeout = '300s'")

    def close(self):
        if self.conn:
            self.lib.PQfinish(self.conn); self.conn = None

    def query(self, sql):
        res = self.lib.PQexec(self.conn, sql.encode())
        try:
            if not res or self.lib.PQresultStatus(res) not in (1, 2):
                raise RuntimeError('PostgreSQL read query failed (details suppressed to protect credentials)')
            cols = [self.lib.PQfname(res, j).decode() for j in range(self.lib.PQnfields(res))]
            rows = [[None if self.lib.PQgetisnull(res, i, j) else self.lib.PQgetvalue(res, i, j).decode()
                     for j in range(len(cols))] for i in range(self.lib.PQntuples(res))]
            return pd.DataFrame(rows, columns=cols)
        finally:
            if res: self.lib.PQclear(res)

    def foreign_keys(self):
        for sql in (_PG_FK_SQL, _FK_SQL):
            try:
                df = self.query(sql)
                if df is not None and not df.empty:
                    return df.to_dict("records")
            except Exception:
                continue
        return []

    def table(self, schema, name, tmp_dir: Path | None = None):
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        literal = lambda s: "'" + s.replace("'", "''") + "'"
        types = self.query("SELECT column_name,data_type FROM information_schema.columns WHERE table_schema="
                           + literal(schema) + " AND table_name=" + literal(name))
        text_types = {'text', 'character varying', 'character', 'uuid'}
        dtypes = {r.column_name: str for r in types.itertuples(index=False) if r.data_type in text_types}
        bool_cols = [r.column_name for r in types.itertuples(index=False) if r.data_type == 'boolean']
        res = self.lib.PQexec(self.conn, f'COPY (SELECT * FROM {quote(schema)}.{quote(name)}) TO STDOUT WITH CSV HEADER'.encode())
        ok = res and self.lib.PQresultStatus(res) == 3
        if res: self.lib.PQclear(res)
        if not ok: raise RuntimeError('PostgreSQL table export failed')
        with _spooled_temp(tmp_dir) as buf:
            while True:
                ptr = C.c_void_p()
                size = self.lib.PQgetCopyData(self.conn, C.byref(ptr), 0)
                if size == -1: break
                if size < 0: raise RuntimeError('PostgreSQL export interrupted')
                try: buf.write(C.string_at(ptr, size))
                finally: self.lib.PQfreemem(ptr)
            self.lib.PQgetResult.restype = C.c_void_p
            self.lib.PQgetResult.argtypes = [C.c_void_p]
            while True:
                res = self.lib.PQgetResult(self.conn)
                if not res: break
                status = self.lib.PQresultStatus(res); self.lib.PQclear(res)
                if status != 1: raise RuntimeError('PostgreSQL export did not complete')
            buf.seek(0)
            df = pd.read_csv(buf, low_memory=False, dtype=dtypes, keep_default_na=False, na_values=[""])
        return _downcast_df(_convert_pg_booleans(df, bool_cols))


def Database():
    try:
        import psycopg  # noqa: F401
        return _PsycopgDatabase()
    except ImportError:
        return _LibpqDatabase()


def _fetch_column_comments(db, key_of: dict) -> list:
    """Read COMMENT ON COLUMN metadata from pg_description when data_dictionary is absent or incomplete."""
    try:
        df = db.query(
            "SELECT n.nspname AS table_schema, c.relname AS table_name, a.attname AS column_name, "
            "pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type, d.description "
            "FROM pg_catalog.pg_description d "
            "JOIN pg_catalog.pg_class c ON c.oid = d.objoid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_catalog.pg_attribute a ON a.attrelid = d.objoid AND a.attnum = d.objsubid "
            "WHERE d.objsubid > 0 AND n.nspname NOT IN ('pg_catalog','information_schema') "
            "AND d.description IS NOT NULL AND d.description <> '' "
            "ORDER BY n.nspname, c.relname, a.attnum"
        )
        if df is None or df.empty:
            return []
        out = []
        for r in df.itertuples(index=False):
            tk = key_of.get((str(r.table_schema), str(r.table_name)), str(r.table_name))
            out.append({
                'table': tk,
                'column': str(r.column_name),
                'type': str(r.data_type),
                'description': str(r.description),
            })
        return out
    except Exception:
        return []


def comments_to_dictionary(comments: list, main_tables=()) -> pd.DataFrame | None:
    """Build a data_dictionary-compatible DataFrame from pg_description column comments."""
    if not comments:
        return None
    main_set = {str(t) for t in (main_tables or ())}
    rows = []
    seen = set()
    for item in comments:
        tbl = str(item.get('table', ''))
        col = str(item.get('column', ''))
        dtype = str(item.get('type', ''))
        desc = str(item.get('description', '')).strip()
        if not col or not desc:
            continue
        names = [col] if tbl in main_set else [f'{tbl}__{col}', col]
        for nm in names:
            if nm not in seen:
                seen.add(nm)
                rows.append({'column': nm, 'type': dtype, 'description': desc})
    return pd.DataFrame(rows) if rows else None


def extract(work: Path, meta: dict | None = None):
    if not os.environ.get('PGDATABASE'):
        raise RuntimeError('No train/test files; set PGDATABASE and PostgreSQL connection variables')
    db = Database()
    try:
        schema = db.query(
            "SELECT table_schema,table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog','information_schema') "
            "UNION SELECT schemaname AS table_schema, matviewname AS table_name FROM pg_matviews "
            "WHERE schemaname NOT IN ('pg_catalog','information_schema') "
            "ORDER BY table_schema,table_name"
        )
        tables = {}
        key_of = {}
        for row in schema.itertuples(index=False):
            key = row.table_name
            if key in tables:
                key = f"{row.table_schema}__{row.table_name}"
            tables[key] = db.table(row.table_schema, row.table_name, tmp_dir=work)
            key_of[(row.table_schema, row.table_name)] = key
            if len(tables[key]) >= 500_000:
                gc.collect()
        if isinstance(meta, dict):
            fks = []
            seen_fk = set()
            for r in db.foreign_keys():
                src_k = key_of.get((r.get("src_schema"), r.get("src_table")), r.get("src_table"))
                dst_k = key_of.get((r.get("dst_schema"), r.get("dst_table")), r.get("dst_table"))
                sc, dc = r.get("src_col"), r.get("dst_col")
                sig = (src_k, sc, dst_k, dc)
                if src_k and dst_k and sc and dc and sig not in seen_fk:
                    seen_fk.add(sig)
                    fks.append({"src_table": src_k, "src_col": sc, "dst_table": dst_k, "dst_col": dc})
            meta["foreign_keys"] = fks
            meta["column_comments"] = _fetch_column_comments(db, key_of)
        return tables
    finally:
        db.close()
