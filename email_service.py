"""Email delivery and audit logging for FRMS alerts."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Iterable


class AlertEmailService:
    """Send alert emails and retain the outcome of every attempted delivery."""

    def __init__(self, database: Any) -> None:
        self.database = database

    def recipients(self) -> list[dict]:
        with self.database.cursor() as cur:
            cur.execute(
                """SELECT id, name, email FROM users
                   WHERE is_active AND role IN ('super_admin', 'pdrm_officer', 'patrol_unit')
                   ORDER BY name"""
            )
            return cur.fetchall()

    def send_alert(self, alert: dict, recipients: Iterable[dict], *, source: str, sent_by: int | None = None) -> int:
        recipient_list = list(recipients)
        subject = f"FRMS alert {alert['alert_id']}: {alert['full_name']}"
        body = (
            f"Missing-person alert: {alert['full_name']}\n\n"
            f"Alert ID: {alert['alert_id']}\n"
            f"Location: {alert.get('location') or 'Not provided'}\n"
            f"Similarity: {alert.get('similarity_score') or 'Not available'}%\n"
            f"Priority: {alert.get('priority') or 'high'}\n\n"
            f"{alert.get('message') or 'A CCTV face match requires review.'}"
        )
        if not recipient_list:
            self._log(alert["id"], None, None, subject, body, source, "failed", "No active alert recipients found.", sent_by)
            return 0

        delivered = 0
        for recipient in recipient_list:
            try:
                self._deliver(recipient["email"], subject, body)
            except Exception as exc:  # Every SMTP failure is part of the audit trail.
                self._log(alert["id"], recipient["id"], recipient["email"], subject, body, source, "failed", str(exc), sent_by)
            else:
                self._log(alert["id"], recipient["id"], recipient["email"], subject, body, source, "sent", None, sent_by)
                delivered += 1
        return delivered

    @staticmethod
    def _deliver(recipient: str, subject: str, body: str) -> None:
        host = os.environ.get("SMTP_HOST")
        sender = os.environ.get("SMTP_FROM")
        if not host or not sender:
            raise RuntimeError("SMTP_HOST and SMTP_FROM must be configured.")
        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)
        port = int(os.environ.get("SMTP_PORT", "587"))
        username, password = os.environ.get("SMTP_USERNAME"), os.environ.get("SMTP_PASSWORD")
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if os.environ.get("SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"}:
                smtp.starttls()
            if username:
                smtp.login(username, password or "")
            smtp.send_message(message)

    def _log(self, alert_id: int, recipient_user_id: int | None, recipient_email: str | None,
             subject: str, body: str, source: str, status: str, error_message: str | None,
             sent_by: int | None) -> None:
        with self.database.cursor() as cur:
            cur.execute(
                """INSERT INTO email_logs
                   (alert_id, recipient_user_id, recipient_email, subject, body, source, status,
                    error_message, sent_by, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())""",
                (alert_id, recipient_user_id, recipient_email, subject, body, source, status,
                 error_message, sent_by),
            )
