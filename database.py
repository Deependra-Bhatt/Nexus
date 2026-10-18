"""SQLite database setup for meridian.db (spec §27). Single physical DB, logical isolation
is enforced by giving each MCP server its own repository module that only knows its own tables.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    customer_tier TEXT NOT NULL,
    kyc_status TEXT NOT NULL,
    email TEXT,
    phone TEXT
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL,
    status TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    available_balance REAL NOT NULL,
    current_balance REAL NOT NULL,
    auto_debit_restricted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    posted_at TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    direction TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    policy_type TEXT NOT NULL,
    status TEXT NOT NULL,
    premium_amount REAL NOT NULL,
    premium_frequency TEXT NOT NULL,
    auto_debit_allowed INTEGER NOT NULL DEFAULT 1,
    payment_terms_summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES policies(policy_id),
    status TEXT NOT NULL,
    filed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    risk_profile TEXT NOT NULL,
    status TEXT NOT NULL,
    total_value REAL NOT NULL,
    holdings_summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investment_products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    minimum_investment REAL NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interaction_log (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    product_line TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS escalation_flags (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    category TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    urgency TEXT,
    product_line TEXT,
    request_type TEXT,
    draft_confidence REAL,
    security_flags TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    # 30s busy timeout: concurrent requests (Streamlit sessions, MCP server processes) share
    # this file and short write locks must wait rather than fail with "database is locked".
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Table ownership per MCP server (spec §28). Enforced, not just conventional: a repository's
# connection carries a SQLite authorizer that refuses any read or write of a table the server
# does not own, so e.g. the insurance repository cannot query `customers` even by mistake.
TABLE_OWNERSHIP: dict[str, frozenset[str]] = {
    "banking": frozenset({"accounts", "transactions"}),
    "insurance": frozenset({"policies", "claims"}),
    "wealth": frozenset({"portfolios", "investment_products"}),
    "concierge_ops": frozenset(
        {"customers", "interaction_log", "escalation_flags", "audit_log"}
    ),
}

_TABLE_ACTIONS = {
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
}
_DDL_ACTIONS = {
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
}


def _authorizer_for(allowed: frozenset[str]):
    def _authorize(action, arg1, arg2, db_name, trigger):
        if action in _DDL_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action in _TABLE_ACTIONS:
            table = (arg1 or "").lower()
            if table.startswith("sqlite_") or table in allowed:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return _authorize


@contextmanager
def session(owner: str | None = None):
    """Open a connection. With `owner` (an MCP server name), the connection may only touch that
    server's tables (TABLE_OWNERSHIP); anything else raises sqlite3.DatabaseError
    ("not authorized"). Without `owner` (schema setup, seeding, tests) access is unrestricted.
    """
    conn = get_connection()
    if owner is not None:
        if owner not in TABLE_OWNERSHIP:
            conn.close()
            raise ValueError(f"Unknown table owner: {owner!r}")
        conn.set_authorizer(_authorizer_for(TABLE_OWNERSHIP[owner]))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added to escalation_flags after the first schema version (the escalation case record:
# triage context for the human team plus a workflow status). Added in place by migrate_schema(),
# so a database created by an earlier version keeps its rows and keeps working.
_ESCALATION_CASE_COLUMNS = (
    ("status", "TEXT NOT NULL DEFAULT 'open'"),
    ("urgency", "TEXT"),
    ("product_line", "TEXT"),
    ("request_type", "TEXT"),
    ("draft_confidence", "REAL"),
    ("security_flags", "TEXT"),
)
_MIGRATED_PATHS: set[str] = set()


def migrate_schema() -> None:
    """Idempotent, additive schema migration (no row is changed or dropped).

    Runs on an unrestricted connection because the per-server authorizer denies DDL by design.
    Called by init_db() and, once per database file and process, by the concierge_ops repository
    before it writes an escalation case -- so an existing data/meridian.db is upgraded on first use
    instead of failing with "no such column".
    """
    path = str(settings.DATABASE_PATH)
    if path in _MIGRATED_PATHS:
        return
    with session() as conn:
        conn.executescript(
            SCHEMA
        )  # creates any missing table (all statements are IF NOT EXISTS)
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(escalation_flags)")
        }
        for column, ddl in _ESCALATION_CASE_COLUMNS:
            if column not in existing:
                try:
                    conn.execute(
                        f"ALTER TABLE escalation_flags ADD COLUMN {column} {ddl}"
                    )
                except (
                    sqlite3.OperationalError
                ) as exc:  # another process migrated first
                    if "duplicate column" not in str(exc).lower():
                        raise
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalation_flags_request ON escalation_flags (request_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalation_flags_status ON escalation_flags (status, created_at)"
        )
    _MIGRATED_PATHS.add(path)


def init_db() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)
    migrate_schema()


def reset_db() -> None:
    """Drop and recreate all tables. Used by tests and the seed script."""
    with session() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS escalation_flags;
            DROP TABLE IF EXISTS interaction_log;
            DROP TABLE IF EXISTS investment_products;
            DROP TABLE IF EXISTS portfolios;
            DROP TABLE IF EXISTS claims;
            DROP TABLE IF EXISTS policies;
            DROP TABLE IF EXISTS transactions;
            DROP TABLE IF EXISTS accounts;
            DROP TABLE IF EXISTS customers;
            """)
        conn.executescript(SCHEMA)
    _MIGRATED_PATHS.discard(str(settings.DATABASE_PATH))
    migrate_schema()
