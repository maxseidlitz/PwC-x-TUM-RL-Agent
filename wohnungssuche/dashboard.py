"""
CLI-Dashboard und CSV-Export für die Wohnungssuche-Datenbank.
Zeigt Statistiken an und erlaubt manuelle Status-Updates.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Optional

import yaml


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    import re
    def env_sub(m):
        return os.environ.get(m.group(1), m.group(0))
    raw = re.sub(r"\$\{([^}]+)\}", env_sub, raw)
    return yaml.safe_load(raw)


def print_stats(db) -> None:
    stats = db.get_stats()
    print("\n" + "=" * 50)
    print("  WOHNUNGSSUCHE MÜNCHEN – STATISTIK")
    print("=" * 50)
    print(f"  Gesamt Listings:       {stats['total']}")
    print()
    print("  Status:")
    for status, count in stats["by_status"].items():
        bar = "█" * count if count <= 40 else "█" * 40 + "+"
        print(f"    {status:<20} {count:>4}  {bar}")
    print()
    print("  Portal:")
    for portal, count in stats["by_portal"].items():
        print(f"    {portal:<20} {count:>4}")
    print()
    print(f"  Response-Rate:         {stats['response_rate_percent']}%")
    print("=" * 50 + "\n")


def print_listings(listings, verbose: bool = False) -> None:
    if not listings:
        print("Keine Listings gefunden.")
        return
    print(f"\n{'Nr':<4} {'Portal':<15} {'Preis':>7} {'Größe':>7} {'Status':<18} Titel")
    print("-" * 85)
    for i, l in enumerate(listings, 1):
        price = f"{float(l.price):.0f}€" if l.price else "–"
        size = f"{float(l.size_sqm):.0f}m²" if l.size_sqm else "–"
        title = (l.title or "")[:40]
        print(f"{i:<4} {l.portal:<15} {price:>7} {size:>7} {l.status:<18} {title}")
        if verbose:
            print(f"     URL: {l.url}")
            if l.address:
                print(f"     Adresse: {l.address}")
            print()
    print()


def export_csv(db, output_path: str = "export.csv") -> None:
    listings = db.get_listings()
    fieldnames = [
        "listing_id", "portal", "title", "price", "size_sqm",
        "address", "url", "status", "created_at", "contacted_at", "response_at",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for l in listings:
            writer.writerow({k: getattr(l, k, "") or "" for k in fieldnames})
    print(f"Export gespeichert: {output_path} ({len(listings)} Einträge)")


def update_status_interactive(db) -> None:
    listings = db.get_listings()
    print_listings(listings)
    try:
        nr = int(input("Nummer des Listings wählen (0 = abbrechen): "))
        if nr == 0:
            return
        listing = listings[nr - 1]
    except (ValueError, IndexError):
        print("Ungültige Eingabe.")
        return
    print(f"\nAktueller Status: {listing.status}")
    statuses = ["neu", "kontaktiert", "antwort_erhalten", "abgelehnt", "buchung"]
    for i, s in enumerate(statuses, 1):
        print(f"  {i}. {s}")
    try:
        choice = int(input("Neuer Status (Nummer): "))
        new_status = statuses[choice - 1]
    except (ValueError, IndexError):
        print("Ungültige Eingabe.")
        return
    db.update_status(listing.listing_id, new_status)
    print(f"Status aktualisiert: {listing.listing_id} → {new_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wohnungssuche München – Dashboard")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="Statistik anzeigen")
    lst = sub.add_parser("list", help="Listings anzeigen")
    lst.add_argument("--status", help="Nach Status filtern")
    lst.add_argument("--portal", help="Nach Portal filtern")
    lst.add_argument("--verbose", "-v", action="store_true")
    sub.add_parser("update", help="Status manuell ändern")
    exp = sub.add_parser("export", help="CSV-Export")
    exp.add_argument("--output", default="export.csv")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)
    from database import Database
    db = Database(config.get("database", {}))

    if args.command == "stats":
        print_stats(db)
    elif args.command == "list":
        listings = db.get_listings(
            status=getattr(args, "status", None),
            portal=getattr(args, "portal", None),
        )
        print_listings(listings, verbose=getattr(args, "verbose", False))
    elif args.command == "update":
        update_status_interactive(db)
    elif args.command == "export":
        export_csv(db, args.output)


if __name__ == "__main__":
    main()
