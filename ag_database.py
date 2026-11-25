"""Read-only PostgreSQL extraction via psycopg (official driver) with libpq fallback.
Connection comes from standard PG* environment variables (never logged).
"""
import ctypes as C
import ctypes.util
import io
import os
from pathlib import Path

import pandas as pd


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

    def table(self, schema, name):
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        literal = lambda s: "'" + s.replace("'", "''") + "'"
        types = self.query(
            "SELECT column_name,data_type FROM information_schema.columns WHERE table_schema="
            + literal(schema) + " AND table_name=" + literal(name)
        )
        text_types = {'text', 'character varying', 'character', 'uuid'}
        dtypes = {r.column_name: str for r in types.itertuples(index=False) if r.data_type in text_types}
        buf = io.BytesIO()
        with self.conn.cursor() as cur:
            with cur.copy(f'COPY {quote(schema)}.{quote(name)} TO STDOUT WITH CSV HEADER') as copy:
                for chunk in copy:
                    buf.write(chunk)
        buf.seek(0)
        return pd.read_csv(buf, low_memory=False, dtype=dtypes, keep_default_na=False, na_values=[""])


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

    def table(self, schema, name):
        quote = lambda s: '"' + s.replace('"', '""') + '"'
        literal = lambda s: "'" + s.replace("'", "''") + "'"
        types = self.query("SELECT column_name,data_type FROM information_schema.columns WHERE table_schema="
                           + literal(schema) + " AND table_name=" + literal(name))
        text_types = {'text', 'character varying', 'character', 'uuid'}
        dtypes = {r.column_name: str for r in types.itertuples(index=False) if r.data_type in text_types}
        res = self.lib.PQexec(self.conn, f'COPY {quote(schema)}.{quote(name)} TO STDOUT WITH CSV HEADER'.encode())
        ok = res and self.lib.PQresultStatus(res) == 3
        if res: self.lib.PQclear(res)
        if not ok: raise RuntimeError('PostgreSQL table export failed')
        buf = io.BytesIO()
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
        return pd.read_csv(buf, low_memory=False, dtype=dtypes, keep_default_na=False, na_values=[""])


def Database():
    try:
        import psycopg  # noqa: F401
        return _PsycopgDatabase()
    except ImportError:
        return _LibpqDatabase()


def extract(work: Path):
    if not os.environ.get('PGDATABASE'):
        raise RuntimeError('No train/test files; set PGDATABASE and PostgreSQL connection variables')
    db = Database()
    try:
        schema = db.query("SELECT table_schema,table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY table_schema,table_name")
        tables = {}
        for row in schema.itertuples(index=False):
            if row.table_name in tables:
                raise ValueError('Ambiguous table names across schemas')
            tables[row.table_name] = db.table(row.table_schema, row.table_name)
        return tables
    finally:
        db.close()
