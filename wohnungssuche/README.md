# Wohnungssuche München – Automatisierungssystem

Automatisiertes Scraping, Kontaktierung und Tracking für WG-Zimmer und Wohnungen in München.

## Architektur

```
wohnungssuche/
├── crawler.py          # Scraper (WG-Gesucht, ImmobilienScout24, Kleinanzeigen)
├── database.py         # SQLAlchemy ORM, SQLite/PostgreSQL
├── mailer.py           # SMTP / SendGrid, Template-Engine
├── scheduler.py        # APScheduler – steuert alle Jobs
├── dashboard.py        # CLI-Dashboard & CSV-Export
├── config.yaml         # Alle Einstellungen (keine Secrets hier!)
├── .env.example        # Vorlage für Umgebungsvariablen
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── templates/
    ├── contact_wg.html
    ├── contact_wohnung.html
    ├── contact_default.html
    └── report.html
```

## Setup

### 1. Python-Umgebung

```bash
cd wohnungssuche
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Konfiguration

```bash
cp .env.example .env
# .env bearbeiten: E-Mail, Bewerberprofil etc. eintragen
```

`config.yaml` anpassen:
- Preisrange (`price_min` / `price_max`)
- Crawler aktivieren/deaktivieren
- Scheduling-Intervalle

### 3. Starten

**Scheduler (läuft dauerhaft):**
```bash
python scheduler.py
```

**Nur einmalig crawlen:**
```bash
python -c "
from scheduler import load_config
from crawler import CrawlerManager
from database import Database
config = load_config()
db = Database(config['database'])
mgr = CrawlerManager(config)
listings = mgr.run_all()
new = db.bulk_add_listings(listings)
print(f'{new} neue Listings gespeichert')
"
```

**Dashboard:**
```bash
python dashboard.py stats
python dashboard.py list --status neu
python dashboard.py list --status kontaktiert --verbose
python dashboard.py update          # Status manuell ändern
python dashboard.py export --output meine_listings.csv
```

## Docker-Deployment

```bash
cp .env.example .env
# .env ausfüllen

docker compose up -d
docker compose logs -f
```

## Gmail App-Passwort einrichten

1. Google-Konto → Sicherheit → 2-Faktor-Authentifizierung aktivieren
2. App-Passwörter → Neues App-Passwort erstellen → "E-Mail" / "Eigenes Gerät"
3. Generierten Code als `EMAIL_PASSWORD` in `.env` eintragen

## Status-Felder

| Status | Bedeutung |
|---|---|
| `neu` | Gerade gecrawlt, noch nicht kontaktiert |
| `kontaktiert` | Anfrage gesendet |
| `antwort_erhalten` | Positive Rückmeldung |
| `abgelehnt` | Abgelehnt oder kein Interesse |
| `buchung` | Wohnung gebucht / Vertrag |

## Konfigurierbare Intervalle

| Job | Standard | Einstellung |
|---|---|---|
| Crawler | 1 Stunde | `scheduler.crawler_interval` |
| Kontaktierung | 30 Minuten | `scheduler.contact_interval` |
| Tagesbericht | 24 Stunden | `scheduler.report_interval` |

## Hinweise

- **robots.txt** wird vor jedem Request geprüft.
- **Rate-Limiting**: Randomisierte Delays zwischen Requests (`delay_seconds` + Jitter).
- **Secrets** niemals in `config.yaml` – ausschließlich über `.env` / Umgebungsvariablen.
- Bei Produktiv-Deployment auf Raspberry Pi oder Cloud-VM empfiehlt sich Docker.
