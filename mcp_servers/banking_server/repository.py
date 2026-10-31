"""Banking repository: owns accounts + transactions only (spec §28)."""

from database import session
from errors import InvalidIdentifierError


class BankingRepository:
    def get_account(self, account_id: str) -> dict | None:
        with session("banking") as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_accounts_for_customer(self, customer_id: str) -> list[dict]:
        with session("banking") as conn:
            rows = conn.execute(
                "SELECT * FROM accounts WHERE customer_id = ?", (customer_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_transactions(self, account_id: str, limit: int = 10) -> list[dict]:
        if self.get_account(account_id) is None:
            raise InvalidIdentifierError(f"Unknown account_id: {account_id!r}")
        with session("banking") as conn:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE account_id = ? "
                "ORDER BY posted_at DESC LIMIT ?",
                (account_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
