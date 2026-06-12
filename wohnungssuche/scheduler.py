"""
Job-Scheduling für Crawler, Kontaktierung und Reporting.
Verwendet APScheduler mit Retry-Logik und strukturiertem Logging.
"""

import logging
import os
import signal
import sys
import time
from typing import Dict, Any

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from crawler import CrawlerManager
from database import Database
from mailer import ContactManager

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Einfache Environment-Variable-Substitution ${VAR}
    import re
    def env_sub(match):
        var = match.group(1)
        return os.environ.get(var, match.group(0))
    raw = re.sub(r"\$\{([^}]+)\}", env_sub, raw)
    return yaml.safe_load(raw)


def setup_logging(log_config: Dict[str, Any]) -> None:
    import logging.handlers
    os.makedirs(os.path.dirname(log_config.get("file", "./logs/crawler.log")), exist_ok=True)
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = log_config.get("file")
    if log_file:
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_config.get("max_bytes", 10_485_760),
                backupCount=log_config.get("backup_count", 5),
                encoding="utf-8",
            )
        )
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# ── Job-Funktionen ────────────────────────────────────────────────────────────

def run_crawlers(config: Dict[str, Any], db: Database) -> None:
    logger.info("=== Crawler-Job gestartet ===")
    manager = CrawlerManager(config)
    listings = manager.run_all()
    new_count = db.bulk_add_listings(listings)
    logger.info("Crawler-Job abgeschlossen: %d neu, %d gesamt.", new_count, len(listings))
    _notify_slack(
        config,
        f":house: Crawler: {new_count} neue Listings gefunden ({len(listings)} gesamt)."
    )


def run_contacts(config: Dict[str, Any], db: Database) -> None:
    logger.info("=== Kontaktierungs-Job gestartet ===")
    sched_cfg = config.get("scheduler", {})
    max_contacts = sched_cfg.get("max_contacts_per_run", 10)
    contact_delay = sched_cfg.get("contact_delay_seconds", 30)
    new_listings = db.get_new_listings(limit=max_contacts)
    if not new_listings:
        logger.info("Keine neuen Listings zum Kontaktieren.")
        return

    applicant = _load_applicant(config)
    mailer = ContactManager(config, db)
    sent = 0
    for listing in new_listings:
        success = mailer.contact_listing(listing, applicant)
        if success:
            sent += 1
            logger.info("Kontaktiert: %s (%s)", listing.title, listing.portal)
        time.sleep(contact_delay)
    logger.info("Kontaktierungs-Job: %d/%d erfolgreich.", sent, len(new_listings))


def run_report(config: Dict[str, Any], db: Database) -> None:
    logger.info("=== Report-Job gestartet ===")
    stats = db.get_stats()
    recipient = (
        os.environ.get("EMAIL_RECIPIENT")
        or config.get("email", {}).get("recipient_email")
    )
    if not recipient:
        logger.warning("Report: kein Empfänger konfiguriert.")
        return
    mailer = ContactManager(config, db)
    mailer.send_report(recipient, stats)
    logger.info("Report gesendet an %s. Stats: %s", recipient, stats)


def _load_applicant(config: Dict[str, Any]) -> Dict[str, Any]:
    """Liest Bewerberprofil aus config oder Environment-Variablen."""
    applicant_cfg = config.get("applicant", {})
    return {
        "name": os.environ.get("APPLICANT_NAME") or applicant_cfg.get("name", "Max Mustermann"),
        "age": os.environ.get("APPLICANT_AGE") or applicant_cfg.get("age", "28"),
        "occupation": os.environ.get("APPLICANT_OCCUPATION") or applicant_cfg.get("occupation", "Software-Entwickler"),
        "income": os.environ.get("APPLICANT_INCOME") or applicant_cfg.get("income", "3.000 € netto"),
        "move_in_date": os.environ.get("APPLICANT_MOVE_IN") or applicant_cfg.get("move_in_date", "01.08.2025"),
        "subject_template": applicant_cfg.get("subject_template", "Anfrage zu Ihrer Wohnung: {{listing_title}}"),
    }


def _notify_slack(config: Dict[str, Any], message: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL") or config.get("notifications", {}).get("slack_webhook")
    if not webhook or webhook.startswith("$"):
        return
    try:
        import requests
        requests.post(webhook, json={"text": message}, timeout=10)
    except Exception as e:
        logger.debug("Slack-Benachrichtigung fehlgeschlagen: %s", e)


# ── Scheduler-Setup ───────────────────────────────────────────────────────────

def create_scheduler(config: Dict[str, Any]) -> BlockingScheduler:
    db = Database(config.get("database", {}))
    sched_cfg = config.get("scheduler", {})
    crawler_interval = sched_cfg.get("crawler_interval", 3600)
    contact_interval = sched_cfg.get("contact_interval", 1800)
    report_interval = sched_cfg.get("report_interval", 86400)

    scheduler = BlockingScheduler(timezone="Europe/Berlin")

    def _job_listener(event):
        if event.exception:
            logger.error("Job '%s' fehlgeschlagen: %s", event.job_id, event.exception)
            _notify_slack(config, f":x: Job `{event.job_id}` fehlgeschlagen: {event.exception}")

    scheduler.add_listener(_job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    scheduler.add_job(
        run_crawlers,
        "interval",
        seconds=crawler_interval,
        id="crawler",
        kwargs={"config": config, "db": db},
        max_instances=1,
        coalesce=True,
        next_run_time=__import__("datetime").datetime.now(),  # sofort starten
    )
    scheduler.add_job(
        run_contacts,
        "interval",
        seconds=contact_interval,
        id="contacts",
        kwargs={"config": config, "db": db},
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_report,
        "interval",
        seconds=report_interval,
        id="report",
        kwargs={"config": config, "db": db},
        max_instances=1,
        coalesce=True,
    )
    return scheduler


def main() -> None:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    setup_logging(config.get("logging", {}))
    logger.info("Wohnungssuche-Scheduler startet …")

    scheduler = create_scheduler(config)

    def _shutdown(signum, frame):
        logger.info("Signal %s empfangen – beende Scheduler.", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler beendet.")


if __name__ == "__main__":
    main()
