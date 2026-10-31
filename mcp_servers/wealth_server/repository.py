"""Wealth repository: owns portfolios + investment_products only (spec §28)."""

from database import session


class WealthRepository:
    def get_portfolio(self, portfolio_id: str) -> dict | None:
        with session("wealth") as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_investment_product(self, product_id: str) -> dict | None:
        with session("wealth") as conn:
            row = conn.execute(
                "SELECT * FROM investment_products WHERE product_id = ?", (product_id,)
            ).fetchone()
        return dict(row) if row else None
