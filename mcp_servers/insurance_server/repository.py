"""Insurance repository: owns policies + claims only (spec §28)."""

from database import session


class InsuranceRepository:
    def get_policy(self, policy_id: str) -> dict | None:
        with session("insurance") as conn:
            row = conn.execute(
                "SELECT * FROM policies WHERE policy_id = ?", (policy_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_claim(self, claim_id: str) -> dict | None:
        # The owning customer is resolved through the claim's policy (both tables belong to the
        # insurance server) so the tool layer can enforce ownership without touching other servers' data.
        with session("insurance") as conn:
            row = conn.execute(
                "SELECT c.claim_id, c.policy_id, c.status, c.filed_at, p.customer_id AS customer_id "
                "FROM claims c JOIN policies p ON p.policy_id = c.policy_id "
                "WHERE c.claim_id = ?",
                (claim_id,),
            ).fetchone()
        return dict(row) if row else None
