"""
Datenbankschema und Datenbankoperationen für die Wohnungssuche.
Unterstützt SQLite (lokal) und PostgreSQL (Produktion).
"""

import os
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, String, Decimal, DateTime,
    Text, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)
Base = declarative_base()


class Listing(Base):
    __tablename__ = "listings"

    listing_id = Column(String(255), primary_key=True)
    portal = Column(String(50), nullable=False)   # wg-boerse, immobilienscout, kleinanzeigen
    title = Column(String(500))
    price = Column(Decimal(10, 2))
    size_sqm = Column(Decimal(6, 1))
    address = Column(String(500))
    description = Column(Text)
    contact_name = Column(String(255))
    contact_email = Column(String(255))
    url = Column(String(1000), nullable=False)
    image_url = Column(String(1000))
    status = Column(String(50), default="neu")    # neu, kontaktiert, antwort_erhalten, abgelehnt, buchung
    created_at = Column(DateTime, default=datetime.utcnow)
    contacted_at = Column(DateTime)
    response_at = Column(DateTime)
    notes = Column(Text)

    __table_args__ = (
        UniqueConstraint("portal", "url", name="uq_portal_url"),
        Index("idx_status", "status"),
        Index("idx_portal", "portal"),
        Index("idx_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "portal": self.portal,
            "title": self.title,
            "price": float(self.price) if self.price else None,
            "size_sqm": float(self.size_sqm) if self.size_sqm else None,
            "address": self.address,
            "url": self.url,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "contacted_at": self.contacted_at.isoformat() if self.contacted_at else None,
            "response_at": self.response_at.isoformat() if self.response_at else None,
        }

    def __repr__(self) -> str:
        return f"<Listing {self.portal}:{self.listing_id} '{self.title}' {self.price}€>"


class ContactLog(Base):
    __tablename__ = "contact_log"

    id = Column(String(36), primary_key=True)
    listing_id = Column(String(255), nullable=False)
    contacted_at = Column(DateTime, default=datetime.utcnow)
    method = Column(String(50))       # email, platform_message
    template_used = Column(String(100))
    success = Column(String(10))      # true, false
    error_message = Column(Text)

    __table_args__ = (Index("idx_cl_listing_id", "listing_id"),)


class Database:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("Datenbank initialisiert.")

    def _create_engine(self):
        db_type = self.config.get("type", "sqlite")
        if db_type == "sqlite":
            path = self.config.get("path", "./wohnungssuche.db")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            url = f"sqlite:///{path}"
        elif db_type == "postgresql":
            host = self.config["host"]
            port = self.config.get("port", 5432)
            name = self.config["name"]
            user = self.config["user"]
            password = self.config["password"]
            url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            raise ValueError(f"Unbekannter Datenbanktyp: {db_type}")
        return create_engine(url, echo=False)

    @contextmanager
    def session(self) -> Session:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Listings ─────────────────────────────────────────────────────────────

    def add_listing(self, listing_data: Dict[str, Any]) -> bool:
        """Gibt True zurück wenn neu, False wenn bereits vorhanden."""
        with self.session() as s:
            existing = s.query(Listing).filter_by(
                listing_id=listing_data["listing_id"]
            ).first()
            if existing:
                return False
            s.add(Listing(**listing_data))
            logger.debug("Neues Listing gespeichert: %s", listing_data.get("listing_id"))
            return True

    def bulk_add_listings(self, listings: List[Dict[str, Any]]) -> int:
        """Fügt mehrere Listings ein, überspringt Duplikate. Gibt Anzahl neuer zurück."""
        new_count = 0
        for data in listings:
            if self.add_listing(data):
                new_count += 1
        return new_count

    def get_listings(
        self,
        status: Optional[str] = None,
        portal: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Listing]:
        with self.session() as s:
            q = s.query(Listing)
            if status:
                q = q.filter(Listing.status == status)
            if portal:
                q = q.filter(Listing.portal == portal)
            q = q.order_by(Listing.created_at.desc())
            if limit:
                q = q.limit(limit)
            return q.all()

    def get_new_listings(self, limit: int = 50) -> List[Listing]:
        return self.get_listings(status="neu", limit=limit)

    def update_status(self, listing_id: str, status: str, **kwargs) -> bool:
        with self.session() as s:
            listing = s.query(Listing).filter_by(listing_id=listing_id).first()
            if not listing:
                return False
            listing.status = status
            if status == "kontaktiert":
                listing.contacted_at = datetime.utcnow()
            elif status == "antwort_erhalten":
                listing.response_at = datetime.utcnow()
            for key, value in kwargs.items():
                if hasattr(listing, key):
                    setattr(listing, key, value)
            return True

    def listing_exists(self, listing_id: str) -> bool:
        with self.session() as s:
            return s.query(Listing).filter_by(listing_id=listing_id).count() > 0

    # ── Statistiken ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self.session() as s:
            total = s.query(Listing).count()
            by_status = {}
            for status in ["neu", "kontaktiert", "antwort_erhalten", "abgelehnt", "buchung"]:
                by_status[status] = s.query(Listing).filter_by(status=status).count()
            by_portal = {}
            for portal in ["wg-boerse", "immobilienscout", "kleinanzeigen"]:
                by_portal[portal] = s.query(Listing).filter_by(portal=portal).count()
            contacted = by_status.get("kontaktiert", 0) + by_status.get("antwort_erhalten", 0)
            responses = by_status.get("antwort_erhalten", 0)
            conversion = round(responses / contacted * 100, 1) if contacted > 0 else 0.0
            return {
                "total": total,
                "by_status": by_status,
                "by_portal": by_portal,
                "response_rate_percent": conversion,
            }

    # ── Contact-Log ───────────────────────────────────────────────────────────

    def log_contact(self, log_data: Dict[str, Any]) -> None:
        with self.session() as s:
            s.add(ContactLog(**log_data))
