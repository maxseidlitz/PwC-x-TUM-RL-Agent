"""
Crawler für WG-Börse (wg-gesucht.de), ImmobilienScout24 und Kleinanzeigen.
Verwendet Playwright für JS-rendering und BeautifulSoup für statisches HTML.
"""

import hashlib
import logging
import re
import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# User-Agents rotieren um Blockierungen zu vermeiden
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


def make_listing_id(portal: str, url: str) -> str:
    return hashlib.sha256(f"{portal}:{url}".encode()).hexdigest()[:32]


def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    digits = re.sub(r"[^\d,.]", "", text).replace(",", ".")
    try:
        return float(digits)
    except ValueError:
        return None


def parse_size(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+[,.]?\d*)\s*m", text)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


class BaseCrawler(ABC):
    def __init__(self, config: Dict[str, Any], filters: Dict[str, Any]):
        self.config = config
        self.filters = filters
        self.delay = config.get("delay_seconds", 3)
        self.max_pages = config.get("max_pages", 5)
        self._robots: Optional[RobotFileParser] = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _sleep(self, extra: float = 0.0) -> None:
        jitter = random.uniform(0.5, 1.5)
        time.sleep(self.delay * jitter + extra)

    def _check_robots(self, url: str) -> bool:
        """Gibt False zurück wenn robots.txt den Zugriff verbietet."""
        try:
            if self._robots is None:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
                self._robots = RobotFileParser(robots_url)
                self._robots.read()
            return self._robots.can_fetch("*", url)
        except Exception:
            return True  # im Zweifel erlauben

    def _get_page(self, url: str, session: requests.Session) -> Optional[BeautifulSoup]:
        if not self._check_robots(url):
            logger.warning("robots.txt verbietet: %s", url)
            return None
        try:
            resp = session.get(url, headers=self._get_headers(), timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error("Fehler beim Abrufen von %s: %s", url, e)
            return None

    def _matches_filters(self, price: Optional[float]) -> bool:
        if price is None:
            return True
        p_min = self.filters.get("price_min", 0)
        p_max = self.filters.get("price_max", 99999)
        return p_min <= price <= p_max

    @abstractmethod
    def scrape(self) -> List[Dict[str, Any]]:
        """Gibt eine Liste von Listing-Dicts zurück."""
        ...


# ── WG-Gesucht (WG-Börse) ────────────────────────────────────────────────────

class WGGesuchtCrawler(BaseCrawler):
    """
    Scraper für wg-gesucht.de.
    Hinweis: Login ist für vollständige Kontaktdaten notwendig.
    Diese Implementierung scrapt öffentlich zugängliche Listenansichten.
    """

    PORTAL = "wg-boerse"

    def scrape(self) -> List[Dict[str, Any]]:
        listings = []
        base_url = self.config.get("base_url", "https://www.wg-gesucht.de")
        search_url = self.config.get(
            "search_url",
            "https://www.wg-gesucht.de/wg-zimmer-und-1-zimmer-wohnungen-in-Muenchen.90.0+3.1.0.html"
        )
        with requests.Session() as session:
            for page in range(self.max_pages):
                page_url = f"{search_url}?page_number={page}" if page > 0 else search_url
                logger.info("[WG-Gesucht] Seite %d: %s", page + 1, page_url)
                soup = self._get_page(page_url, session)
                if not soup:
                    break
                items = self._parse_listings(soup, base_url)
                if not items:
                    logger.info("[WG-Gesucht] Keine weiteren Listings auf Seite %d.", page + 1)
                    break
                listings.extend(items)
                self._sleep()
        logger.info("[WG-Gesucht] %d Listings gefunden.", len(listings))
        return listings

    def _parse_listings(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        results = []
        # WG-Gesucht Listeneinträge haben typischerweise class "wgg_card"
        cards = soup.find_all("div", class_=re.compile(r"wgg_card|offer_list_item"))
        for card in cards:
            try:
                result = self._parse_card(card, base_url)
                if result and self._matches_filters(result.get("price")):
                    results.append(result)
            except Exception as e:
                logger.debug("[WG-Gesucht] Fehler beim Parsen einer Karte: %s", e)
        return results

    def _parse_card(self, card, base_url: str) -> Optional[Dict[str, Any]]:
        # Titel
        title_el = card.find("h3") or card.find("a", class_=re.compile(r"headline"))
        title = title_el.get_text(strip=True) if title_el else "Unbekannt"

        # URL
        link_el = card.find("a", href=True)
        if not link_el:
            return None
        href = link_el["href"]
        url = href if href.startswith("http") else f"{base_url}{href}"

        # Preis
        price_el = card.find(class_=re.compile(r"price|miete|kaltmiete"))
        price = parse_price(price_el.get_text() if price_el else "")

        # Größe
        size_el = card.find(class_=re.compile(r"size|qm|zimmer"))
        size = parse_size(size_el.get_text() if size_el else "")

        # Adresse
        addr_el = card.find(class_=re.compile(r"address|location|stadtteil"))
        address = addr_el.get_text(strip=True) if addr_el else ""

        listing_id = make_listing_id(self.PORTAL, url)
        return {
            "listing_id": listing_id,
            "portal": self.PORTAL,
            "title": title,
            "price": price,
            "size_sqm": size,
            "address": address,
            "url": url,
            "status": "neu",
        }


# ── ImmobilienScout24 ────────────────────────────────────────────────────────

class ImmobilienScout24Crawler(BaseCrawler):
    """
    Scraper für immobilienscout24.de.
    Nutzt Playwright für JS-gerenderte Inhalte.
    """

    PORTAL = "immobilienscout"

    def scrape(self) -> List[Dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
            return self._scrape_with_playwright()
        except ImportError:
            logger.warning("Playwright nicht installiert – Fallback auf requests.")
            return self._scrape_with_requests()

    def _scrape_with_playwright(self) -> List[Dict[str, Any]]:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

        listings = []
        search_url = self.config.get(
            "search_url",
            "https://www.immobilienscout24.de/Suche/de/bayern/muenchen/wohnung-mieten"
        )
        price_min = self.filters.get("price_min", 400)
        price_max = self.filters.get("price_max", 1200)
        params = f"?pricefrom={price_min}&priceto={price_max}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="de-DE",
            )
            page = context.new_page()
            for page_num in range(1, self.max_pages + 1):
                url = f"{search_url}{params}&pagenumber={page_num}"
                logger.info("[IS24] Seite %d: %s", page_num, url)
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    self._sleep(1.0)
                    # Cookie-Banner wegklicken falls vorhanden
                    try:
                        page.click("button:has-text('Alle akzeptieren')", timeout=3000)
                    except PlaywrightTimeout:
                        pass
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")
                    items = self._parse_listings(soup)
                    if not items:
                        break
                    listings.extend(items)
                except PlaywrightTimeout as e:
                    logger.error("[IS24] Timeout Seite %d: %s", page_num, e)
                    break
                self._sleep()
            browser.close()
        logger.info("[IS24] %d Listings gefunden.", len(listings))
        return listings

    def _scrape_with_requests(self) -> List[Dict[str, Any]]:
        listings = []
        search_url = self.config.get(
            "search_url",
            "https://www.immobilienscout24.de/Suche/de/bayern/muenchen/wohnung-mieten"
        )
        with requests.Session() as session:
            for page_num in range(1, self.max_pages + 1):
                url = f"{search_url}?pagenumber={page_num}"
                soup = self._get_page(url, session)
                if not soup:
                    break
                items = self._parse_listings(soup)
                if not items:
                    break
                listings.extend(items)
                self._sleep()
        return listings

    def _parse_listings(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        results = []
        base_url = "https://www.immobilienscout24.de"
        # IS24 nutzt data-testid oder article-Tags
        cards = soup.find_all("article") or soup.find_all(
            "li", attrs={"data-testid": re.compile(r"result")}
        )
        for card in cards:
            try:
                result = self._parse_card(card, base_url)
                if result and self._matches_filters(result.get("price")):
                    results.append(result)
            except Exception as e:
                logger.debug("[IS24] Karte Parse-Fehler: %s", e)
        return results

    def _parse_card(self, card, base_url: str) -> Optional[Dict[str, Any]]:
        link_el = card.find("a", href=re.compile(r"/expose/"))
        if not link_el:
            return None
        href = link_el.get("href", "")
        url = href if href.startswith("http") else f"{base_url}{href}"

        title_el = card.find(attrs={"data-testid": "result-list-entry-title"}) or card.find("h5")
        title = title_el.get_text(strip=True) if title_el else "Unbekannt"

        price_el = card.find(attrs={"data-testid": re.compile(r"primary-price|price")})
        price = parse_price(price_el.get_text() if price_el else "")

        size_el = card.find(attrs={"data-testid": re.compile(r"area|size")})
        size = parse_size(size_el.get_text() if size_el else "")

        addr_el = card.find(attrs={"data-testid": re.compile(r"address|location")})
        address = addr_el.get_text(strip=True) if addr_el else ""

        listing_id = make_listing_id(self.PORTAL, url)
        return {
            "listing_id": listing_id,
            "portal": self.PORTAL,
            "title": title,
            "price": price,
            "size_sqm": size,
            "address": address,
            "url": url,
            "status": "neu",
        }


# ── Kleinanzeigen ─────────────────────────────────────────────────────────────

class KleinanzeigenCrawler(BaseCrawler):
    """Scraper für kleinanzeigen.de (vormals eBay Kleinanzeigen)."""

    PORTAL = "kleinanzeigen"

    def scrape(self) -> List[Dict[str, Any]]:
        listings = []
        base_url = self.config.get("base_url", "https://www.kleinanzeigen.de")
        price_min = self.filters.get("price_min", 400)
        price_max = self.filters.get("price_max", 1200)
        search_url = self.config.get(
            "search_url",
            "https://www.kleinanzeigen.de/s-wohnung-mieten/muenchen/c203l1091"
        )
        with requests.Session() as session:
            for page in range(1, self.max_pages + 1):
                # Kleinanzeigen Paginierung: /seite:{n}/
                if page == 1:
                    page_url = f"{search_url}/preis:{price_min}:{price_max}"
                else:
                    page_url = f"{search_url}/seite:{page}/preis:{price_min}:{price_max}"
                logger.info("[Kleinanzeigen] Seite %d: %s", page, page_url)
                soup = self._get_page(page_url, session)
                if not soup:
                    break
                items = self._parse_listings(soup, base_url)
                if not items:
                    break
                listings.extend(items)
                self._sleep()
        logger.info("[Kleinanzeigen] %d Listings gefunden.", len(listings))
        return listings

    def _parse_listings(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        results = []
        cards = soup.find_all("article", class_=re.compile(r"aditem"))
        for card in cards:
            try:
                result = self._parse_card(card, base_url)
                if result and self._matches_filters(result.get("price")):
                    results.append(result)
            except Exception as e:
                logger.debug("[Kleinanzeigen] Karte Parse-Fehler: %s", e)
        return results

    def _parse_card(self, card, base_url: str) -> Optional[Dict[str, Any]]:
        link_el = card.find("a", class_=re.compile(r"ellipsis|aditem-main"))
        if not link_el:
            link_el = card.find("a", href=re.compile(r"/s-anzeige/"))
        if not link_el:
            return None
        href = link_el.get("href", "")
        url = href if href.startswith("http") else f"{base_url}{href}"

        title_el = card.find(class_=re.compile(r"ellipsis|aditem-title"))
        title = title_el.get_text(strip=True) if title_el else "Unbekannt"

        price_el = card.find(class_=re.compile(r"price|aditem-price"))
        price = parse_price(price_el.get_text() if price_el else "")

        desc_el = card.find(class_=re.compile(r"aditem-description|description"))
        description = desc_el.get_text(strip=True) if desc_el else ""
        size = parse_size(description)

        addr_el = card.find(class_=re.compile(r"aditem-details|location"))
        address = addr_el.get_text(strip=True) if addr_el else ""

        listing_id = make_listing_id(self.PORTAL, url)
        return {
            "listing_id": listing_id,
            "portal": self.PORTAL,
            "title": title,
            "price": price,
            "size_sqm": size,
            "address": address,
            "description": description,
            "url": url,
            "status": "neu",
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CrawlerManager:
    def __init__(self, config: Dict[str, Any]):
        self.crawler_config = config.get("crawler", {})
        self.filters = config.get("filters", {})

    def run_all(self) -> List[Dict[str, Any]]:
        all_listings: List[Dict[str, Any]] = []
        crawlers = [
            ("wg_boerse", WGGesuchtCrawler),
            ("immobilienscout", ImmobilienScout24Crawler),
            ("kleinanzeigen", KleinanzeigenCrawler),
        ]
        for key, CrawlerClass in crawlers:
            portal_cfg = self.crawler_config.get(key, {})
            if not portal_cfg.get("enabled", True):
                logger.info("Crawler '%s' deaktiviert, übersprungen.", key)
                continue
            logger.info("Starte Crawler: %s", key)
            try:
                crawler = CrawlerClass(portal_cfg, self.filters)
                listings = crawler.scrape()
                all_listings.extend(listings)
            except Exception as e:
                logger.error("Crawler '%s' fehlgeschlagen: %s", key, e, exc_info=True)
        logger.info("Gesamt: %d Listings über alle Portale.", len(all_listings))
        return all_listings
