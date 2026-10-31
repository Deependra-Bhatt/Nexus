"""Concierge Ops repository: owns customers, interaction_log, escalation_flags, audit_log (spec §28)."""

import json
from datetime import datetime, timezone

from database import migrate_schema, session


class ConciergeOpsRepository:
    def get_customer(self, customer_id: str) -> dict | None:
        with session("concierge_ops") as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        return dict(row) if row else None

    def log_interaction(
        self, request_id: str, customer_id: str, product_line: str, summary: str
    ) -> None:
        with session("concierge_ops") as conn:
            conn.execute(
                "INSERT INTO interaction_log (request_id, customer_id, created_at, product_line, summary) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    customer_id,
                    datetime.now(timezone.utc).isoformat(),
                    product_line,
                    summary,
                ),
            )

    def create_escalation_flag(
        self,
        request_id: str,
        customer_id: str,
        category: str,
        reason: str,
        *,
        urgency: str | None = None,
        product_line: str | None = None,
        request_type: str | None = None,
        draft_confidence: float | None = None,
        security_flags: list[str] | None = None,
    ) -> int:
        """Open an escalation case for the human team; returns its escalation_id.

        The case carries the triage context (urgency, product line, request type, the confidence
        that was evaluated and any security flags) and starts in status 'open'. Customer free
        text is deliberately not stored here: the case points to the request by request_id.
        """
        migrate_schema()  # upgrades a database created before the case columns existed
        with session("concierge_ops") as conn:
            cursor = conn.execute(
                "INSERT INTO escalation_flags (request_id, customer_id, category, reason, created_at, "
                "status, urgency, product_line, request_type, draft_confidence, security_flags) "
                "VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)",
                (
                    request_id,
                    customer_id,
                    category,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                    urgency or None,
                    product_line or None,
                    request_type or None,
                    draft_confidence,
                    json.dumps(list(security_flags)) if security_flags else None,
                ),
            )
            return int(cursor.lastrowid)

    def write_audit_log(
        self, request_id: str | None, actor: str, action: str, details: str
    ) -> None:
        with session("concierge_ops") as conn:
            conn.execute(
                "INSERT INTO audit_log (request_id, actor, action, details, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    actor,
                    action,
                    details,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
