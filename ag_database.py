"""Read-only PostgreSQL extraction via psycopg (official driver) with libpq fallback.
Connection comes from standard PG* environment variables (never logged).
"""
import ctypes as C
import ctypes.util
import os
import tempfile
from pathlib import Path

import pandas as pd


def _convert_pg_booleans(df: pd.DataFrame, bool_cols: list[str]) -> pd.DataFrame:
    """PostgreSQL COPY CSV exports boolean as literal 't'/'f' strings; map them to clean numeric 1.0/0.0."""
    for c in bool_cols:
        if c not in df.columns or pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype("string").str.strip().str.lower()
        df[c] = s.map({"t": 1.0, "true": 1.0, "1": 1.0, "f": 0.0, "false": 0.0, "0": 0.0}).astype("float32")
    return df


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
        try:
            df = self.query(_FK_SQL)
            return df.to_dict("records") if not df.empty else []
        except Exception:
            return []

    def table(self, schema, name):
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        literal = lambda s: "'" + s.replace("'", "''") + "'"
        types = self.query(
            "SELECT column_name,data_type FROM information_schema.columns WHERE table_schema="
            + literal(schema) + " AND table_name=" + literal(name)
        )
        text_types = {'text', 'character varying', 'character', 'uuid'}
        dtypes = {r.column_name: str for r in types.itertuples(index=False) if r.data_type in text_types}
        bool_cols = [r.column_name for r in types.itertuples(index=False) if r.data_type == 'boolean']
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b") as buf:
            with self.conn.cursor() as cur:
                with cur.copy(f'COPY (SELECT * FROM {quote(schema)}.{quote(name)}) TO STDOUT WITH CSV HEADER') as copy:
                    for chunk in copy:
                        buf.write(chunk)
            buf.seek(0)
            df = pd.read_csv(buf, low_memory=False, dtype=dtypes, keep_default_na=False, na_values=[""])
        return _convert_pg_booleans(df, bool_cols)


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
        try:
            df = self.query(_FK_SQL)
            return df.to_dict("records") if not df.empty else []
        except Exception:
            return []

    def table(self, schema, name):
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
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b") as buf:
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
        return _convert_pg_booleans(df, bool_cols)


def Database():
    try:
        import psycopg  # noqa: F401
        return _PsycopgDatabase()
    except ImportError:
        return _LibpqDatabase()


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
            tables[key] = db.table(row.table_schema, row.table_name)
            key_of[(row.table_schema, row.table_name)] = key
        if isinstance(meta, dict):
            fks = []
            for r in db.foreign_keys():
                src_k = key_of.get((r.get("src_schema"), r.get("src_table")), r.get("src_table"))
                dst_k = key_of.get((r.get("dst_schema"), r.get("dst_table")), r.get("dst_table"))
                if src_k and dst_k and r.get("src_col") and r.get("dst_col"):
                    fks.append({
                        "src_table": src_k, "src_col": r["src_col"],
                        "dst_table": dst_k, "dst_col": r["dst_col"],
                    })
            meta["foreign_keys"] = fks
        return tables
    finally:
        db.close()
