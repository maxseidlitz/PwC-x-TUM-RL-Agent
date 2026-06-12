"""
E-Mail-Versand mit Template-System und Personalisierung.
Unterstützt SMTP (Gmail, etc.) und SendGrid.
"""

import logging
import os
import re
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Einfaches Template-System mit {{variable}} Platzhaltern."""

    def __init__(self, templates_dir: str = "./templates"):
        self.templates_dir = Path(templates_dir)

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template nicht gefunden: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return self._substitute(template, context)

    def render_string(self, template: str, context: Dict[str, Any]) -> str:
        return self._substitute(template, context)

    @staticmethod
    def _substitute(template: str, context: Dict[str, Any]) -> str:
        def replacer(match):
            key = match.group(1).strip()
            return str(context.get(key, f"{{{{{key}}}}}"))
        return re.sub(r"\{\{(.+?)\}\}", replacer, template)


class SMTPMailer:
    def __init__(self, config: Dict[str, Any]):
        self.server = config.get("smtp_server", "smtp.gmail.com")
        self.port = int(config.get("smtp_port", 587))
        self.sender = os.environ.get("EMAIL_SENDER") or config.get("sender_email", "")
        self.password = os.environ.get("EMAIL_PASSWORD") or config.get("sender_password", "")
        self.use_tls = config.get("use_tls", True)

    def send(self, to: str, subject: str, body_html: str, body_text: str = "") -> bool:
        if not self.sender or not self.password:
            logger.error("E-Mail-Zugangsdaten fehlen (EMAIL_SENDER / EMAIL_PASSWORD).")
            return False
        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject
        if body_text:
            msg.attach(MIMEText(body_text, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html", "utf-8"))
        try:
            with smtplib.SMTP(self.server, self.port, timeout=30) as smtp:
                if self.use_tls:
                    smtp.starttls()
                smtp.login(self.sender, self.password)
                smtp.sendmail(self.sender, [to], msg.as_string())
            logger.info("E-Mail an %s gesendet: %s", to, subject)
            return True
        except smtplib.SMTPException as e:
            logger.error("SMTP-Fehler beim Senden an %s: %s", to, e)
            return False


class SendGridMailer:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.environ.get("SENDGRID_API_KEY") or config.get("api_key", "")
        self.sender = os.environ.get("EMAIL_SENDER") or config.get("sender_email", "")

    def send(self, to: str, subject: str, body_html: str, body_text: str = "") -> bool:
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail, Email, To, Content
        except ImportError:
            logger.error("sendgrid-Paket nicht installiert.")
            return False
        if not self.api_key:
            logger.error("SENDGRID_API_KEY nicht gesetzt.")
            return False
        message = Mail(
            from_email=Email(self.sender),
            to_emails=To(to),
            subject=subject,
            html_content=Content("text/html", body_html),
        )
        try:
            sg = sendgrid.SendGridAPIClient(api_key=self.api_key)
            response = sg.send(message)
            logger.info("SendGrid: E-Mail an %s gesendet (Status %s).", to, response.status_code)
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error("SendGrid-Fehler: %s", e)
            return False


class ContactManager:
    """Orchestriert Personalisierung, Versand und Status-Updates."""

    def __init__(self, config: Dict[str, Any], db):
        email_cfg = config.get("email", {})
        provider = email_cfg.get("provider", "smtp")
        if provider == "sendgrid":
            self.mailer = SendGridMailer(email_cfg)
        else:
            self.mailer = SMTPMailer(email_cfg)
        self.template_engine = TemplateEngine(
            email_cfg.get("templates_dir", "./templates")
        )
        self.db = db
        self.recipient_override = (
            os.environ.get("EMAIL_RECIPIENT") or email_cfg.get("recipient_email")
        )

    def contact_listing(self, listing, applicant: Dict[str, Any]) -> bool:
        """
        Schreibt einen Poster an. Gibt True zurück bei Erfolg.
        `applicant` enthält persönliche Daten des Bewerbers.
        """
        context = {
            "applicant_name": applicant.get("name", ""),
            "applicant_age": applicant.get("age", ""),
            "applicant_occupation": applicant.get("occupation", ""),
            "applicant_income": applicant.get("income", ""),
            "applicant_move_in": applicant.get("move_in_date", ""),
            "listing_title": listing.title or "",
            "listing_price": listing.price or "",
            "listing_address": listing.address or "",
            "listing_url": listing.url or "",
            "portal": listing.portal or "",
        }
        template_file = self._choose_template(listing.portal)
        try:
            body_html = self.template_engine.render(template_file, context)
        except FileNotFoundError:
            body_html = self._fallback_template(context)

        subject = self.template_engine.render_string(
            applicant.get("subject_template", "Anfrage: {{listing_title}}"),
            context
        )
        recipient = listing.contact_email or self.recipient_override
        if not recipient:
            logger.warning("Kein Empfänger für Listing %s.", listing.listing_id)
            return False

        success = self.mailer.send(recipient, subject, body_html)
        self.db.update_status(
            listing.listing_id,
            "kontaktiert" if success else "neu",
        )
        self.db.log_contact({
            "id": str(uuid.uuid4()),
            "listing_id": listing.listing_id,
            "method": "email",
            "template_used": template_file,
            "success": "true" if success else "false",
        })
        return success

    def send_report(self, recipient: str, stats: Dict[str, Any]) -> bool:
        context = {
            "total": stats.get("total", 0),
            "neu": stats["by_status"].get("neu", 0),
            "kontaktiert": stats["by_status"].get("kontaktiert", 0),
            "antwort_erhalten": stats["by_status"].get("antwort_erhalten", 0),
            "abgelehnt": stats["by_status"].get("abgelehnt", 0),
            "buchung": stats["by_status"].get("buchung", 0),
            "response_rate": stats.get("response_rate_percent", 0),
        }
        try:
            body_html = self.template_engine.render("report.html", context)
        except FileNotFoundError:
            body_html = self._fallback_report(context)
        return self.mailer.send(recipient, "Wohnungssuche – Tagesbericht", body_html)

    @staticmethod
    def _choose_template(portal: str) -> str:
        mapping = {
            "wg-boerse": "contact_wg.html",
            "immobilienscout": "contact_wohnung.html",
            "kleinanzeigen": "contact_wohnung.html",
        }
        return mapping.get(portal, "contact_default.html")

    @staticmethod
    def _fallback_template(ctx: Dict[str, Any]) -> str:
        return f"""
        <html><body>
        <p>Hallo,</p>
        <p>ich bin {ctx['applicant_name']}, {ctx['applicant_age']} Jahre alt,
        und arbeite als {ctx['applicant_occupation']}.</p>
        <p>Ich interessiere mich sehr für Ihr Inserat <strong>{ctx['listing_title']}</strong>
        auf {ctx['portal']} und würde mich freuen, wenn wir einen Besichtigungstermin
        vereinbaren könnten.</p>
        <p>Mit freundlichen Grüßen,<br>{ctx['applicant_name']}</p>
        </body></html>
        """

    @staticmethod
    def _fallback_report(ctx: Dict[str, Any]) -> str:
        return f"""
        <html><body>
        <h2>Wohnungssuche – Tagesbericht</h2>
        <table border="1" cellpadding="8">
        <tr><td>Gesamt gefunden</td><td>{ctx['total']}</td></tr>
        <tr><td>Neu</td><td>{ctx['neu']}</td></tr>
        <tr><td>Kontaktiert</td><td>{ctx['kontaktiert']}</td></tr>
        <tr><td>Antwort erhalten</td><td>{ctx['antwort_erhalten']}</td></tr>
        <tr><td>Abgelehnt</td><td>{ctx['abgelehnt']}</td></tr>
        <tr><td>Buchung</td><td>{ctx['buchung']}</td></tr>
        <tr><td>Response-Rate</td><td>{ctx['response_rate']}%</td></tr>
        </table>
        </body></html>
        """
