"""Seed meridian.db with an expanded, varied dataset for comprehensive testing."""

from datetime import datetime, timedelta, timezone
from database import reset_db, session

NOW = datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def seed() -> None:
    reset_db()
    with session() as conn:
        # 1. Customers: Diverse tiers, KYC statuses, and demographics
        conn.executemany(
            "INSERT INTO customers (customer_id, customer_name, customer_tier, kyc_status, email, phone) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "CUS-20077",
                    "Arjun Mehta",
                    "premier",
                    "verified",
                    "arjun.mehta@example.com",
                    "+1-555-0101",
                ),
                (
                    "CUS-30012",
                    "Priya Shah",
                    "standard",
                    "verified",
                    "priya.shah@example.com",
                    "+1-555-0102",
                ),
                (
                    "CUS-40055",
                    "Tom Reyes",
                    "standard",
                    "pending_review",
                    "tom.reyes@example.com",
                    "+1-555-0103",
                ),
                (
                    "CUS-50101",
                    "Elena Rostova",
                    "elite",
                    "verified",
                    "elena.rostova@example.com",
                    "+1-555-0104",
                ),
                (
                    "CUS-60220",
                    "Marcus Vance",
                    "corporate",
                    "verified",
                    "marcus.vance@example.com",
                    "+1-555-0105",
                ),
                (
                    "CUS-70311",
                    "Sofia Chen",
                    "standard",
                    "rejected",
                    "sofia.chen@example.com",
                    "+1-555-0106",
                ),
                (
                    "CUS-80499",
                    "David O'Connor",
                    "premier",
                    "expired",
                    "david.oconnor@example.com",
                    "+1-555-0107",
                ),
            ],
        )

        # 2. Accounts: Checking, savings, money market, multiple currencies, and restrictions
        conn.executemany(
            "INSERT INTO accounts (account_id, customer_id, account_type, status, currency, "
            "available_balance, current_balance, auto_debit_restricted) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "ACC-20077",
                    "CUS-20077",
                    "savings",
                    "active",
                    "USD",
                    5230.44,
                    5230.44,
                    0,
                ),
                (
                    "ACC-20078",
                    "CUS-20077",
                    "checking",
                    "active",
                    "USD",
                    1200.00,
                    1340.10,
                    0,
                ),
                (
                    "ACC-30012",
                    "CUS-30012",
                    "savings",
                    "active",
                    "USD",
                    800.00,
                    800.00,
                    1,
                ),
                (
                    "ACC-50101",
                    "CUS-50101",
                    "checking",
                    "active",
                    "EUR",
                    14500.50,
                    14820.00,
                    0,
                ),
                (
                    "ACC-50102",
                    "CUS-50101",
                    "money_market",
                    "active",
                    "USD",
                    55000.00,
                    55000.00,
                    0,
                ),
                (
                    "ACC-60220",
                    "CUS-60220",
                    "checking",
                    "active",
                    "USD",
                    125400.80,
                    128900.50,
                    0,
                ),
                (
                    "ACC-70311",
                    "CUS-70311",
                    "savings",
                    "frozen",
                    "USD",
                    250.00,
                    250.00,
                    1,
                ),
                (
                    "ACC-80499",
                    "CUS-80499",
                    "checking",
                    "active",
                    "GBP",
                    3400.25,
                    3400.25,
                    0,
                ),
            ],
        )

        # 3. Transactions: Debits, credits, various descriptions and timelines
        conn.executemany(
            "INSERT INTO transactions (transaction_id, account_id, posted_at, description, amount, direction) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "TXN-00001",
                    "ACC-20077",
                    _iso(NOW - timedelta(days=1)),
                    "Grocery Mart",
                    -84.21,
                    "debit",
                ),
                (
                    "TXN-00002",
                    "ACC-20077",
                    _iso(NOW - timedelta(days=3)),
                    "Payroll Deposit",
                    3200.00,
                    "credit",
                ),
                (
                    "TXN-00003",
                    "ACC-20077",
                    _iso(NOW - timedelta(days=10)),
                    "Meridian Insurance Premium",
                    -145.00,
                    "debit",
                ),
                (
                    "TXN-00004",
                    "ACC-50101",
                    _iso(NOW - timedelta(days=2)),
                    "European Travel Expense",
                    -450.00,
                    "debit",
                ),
                (
                    "TXN-00005",
                    "ACC-50101",
                    _iso(NOW - timedelta(days=6)),
                    "Consulting Fee Income",
                    2400.00,
                    "credit",
                ),
                (
                    "TXN-00006",
                    "ACC-60220",
                    _iso(NOW - timedelta(days=1)),
                    "Corporate Software License",
                    -1250.00,
                    "debit",
                ),
                (
                    "TXN-00007",
                    "ACC-80499",
                    _iso(NOW - timedelta(days=4)),
                    "ATM Cash Withdrawal",
                    -100.00,
                    "debit",
                ),
            ],
        )

        # 4. Policies: Life, auto, home, health insurance with varying frequencies and rules
        conn.executemany(
            "INSERT INTO policies (policy_id, customer_id, policy_type, status, premium_amount, "
            "premium_frequency, auto_debit_allowed, payment_terms_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "POL-IN-30091",
                    "CUS-20077",
                    "life_insurance",
                    "active",
                    145.00,
                    "monthly",
                    1,
                    "Premiums are due monthly. Auto-debit from a linked bank account is permitted "
                    "provided the account is in good standing and not flagged as restricted.",
                ),
                (
                    "POL-AU-10044",
                    "CUS-30012",
                    "auto_insurance",
                    "active",
                    89.50,
                    "monthly",
                    0,
                    "Premiums are due monthly via manual payment only; auto-debit is not enabled for this policy tier.",
                ),
                (
                    "POL-HO-40012",
                    "CUS-50101",
                    "home_insurance",
                    "active",
                    1200.00,
                    "annually",
                    1,
                    "Annual premium billing with automatic renewal option.",
                ),
                (
                    "POL-HE-70088",
                    "CUS-60220",
                    "health_insurance",
                    "active",
                    450.00,
                    "monthly",
                    1,
                    "Comprehensive executive health plan billed monthly.",
                ),
                (
                    "POL-AU-90111",
                    "CUS-80499",
                    "auto_insurance",
                    "lapsed",
                    75.00,
                    "monthly",
                    0,
                    "Policy lapsed due to missed manual payment cycle.",
                ),
            ],
        )

        # 5. Claims: Various claim statuses and lifecycles
        conn.executemany(
            "INSERT INTO claims (claim_id, policy_id, status, filed_at) VALUES (?, ?, ?, ?)",
            [
                (
                    "CLM-50011",
                    "POL-AU-10044",
                    "under_review",
                    _iso(NOW - timedelta(days=5)),
                ),
                (
                    "CLM-50012",
                    "POL-HO-40012",
                    "approved",
                    _iso(NOW - timedelta(days=20)),
                ),
                (
                    "CLM-50013",
                    "POL-HE-70088",
                    "settled",
                    _iso(NOW - timedelta(days=45)),
                ),
                ("CLM-50014", "POL-AU-90111", "denied", _iso(NOW - timedelta(days=60))),
            ],
        )

        # 6. Portfolios: Different risk profiles and holdings
        conn.executemany(
            "INSERT INTO portfolios (portfolio_id, customer_id, risk_profile, status, total_value, "
            "holdings_summary) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "PORT-60033",
                    "CUS-20077",
                    "moderate",
                    "active",
                    48250.75,
                    "60% equities, 30% bonds, 10% cash",
                ),
                (
                    "PORT-60034",
                    "CUS-50101",
                    "aggressive",
                    "active",
                    215000.00,
                    "85% equities, 10% crypto, 5% cash",
                ),
                (
                    "PORT-60035",
                    "CUS-60220",
                    "conservative",
                    "active",
                    540000.50,
                    "20% equities, 70% bonds, 10% cash",
                ),
            ],
        )

        # 7. Investment Products: Broad spectrum of investment types and risk categories
        conn.executemany(
            "INSERT INTO investment_products (product_id, product_name, risk_level, minimum_investment, "
            "description) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    "PROD-EQ-01",
                    "Meridian Growth Equity Fund",
                    "high",
                    1000.00,
                    "A diversified growth equity fund.",
                ),
                (
                    "PROD-BD-01",
                    "Meridian Stable Bond Fund",
                    "low",
                    500.00,
                    "A conservative fixed-income fund.",
                ),
                (
                    "PROD-RE-01",
                    "Meridian Global Real Estate Trust",
                    "medium",
                    5000.00,
                    "Commercial and residential real estate portfolio.",
                ),
                (
                    "PROD-CC-01",
                    "Meridian Digital Asset Tracker",
                    "high",
                    2500.00,
                    "Regulated exposure to major digital currencies.",
                ),
                (
                    "PROD-CM-01",
                    "Meridian Commodity Index Fund",
                    "medium",
                    1500.00,
                    "Diversified basket of precious metals and energy commodities.",
                ),
            ],
        )

    print("Seeded database dataset at configured DATABASE_PATH.")


if __name__ == "__main__":
    seed()
