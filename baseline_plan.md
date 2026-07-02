# Baseline Plan: Base Stock Policy

## Ziel

Implementierung einer regelbasierten Einkaufsstrategie (Base Stock Policy) als Vergleichsmaßstab für den PPO-Agenten. Damit kann gezeigt werden, um wieviel Prozent der RL-Agent in Gesamtkosten und Service Level gegenüber der klassischen Methode gewinnt.

---

## Hintergrund

### Was ist eine Base Stock Policy?

Die Base Stock Policy ist die klassische Einkaufsstrategie aus der Praxis. Sie folgt einer simplen, fixen Regel:

> "Bestelle immer genug, um Lagerbestand + Pipeline auf einen fixen Zielwert S aufzufüllen."

```
order_qty = max(0, S - (aktuelles_lager + pipeline_summe))
```

### Warum ist das die richtige Baseline?

- Industriestandard für Inventory-RL-Studien
- Gleiche Kostenstruktur wie PPO (Holding, Ordering, Lost Sales)
- Deterministisch und nachvollziehbar — kein ML
- Infrastruktur im Code bereits vorbereitet (`base_stock_results`)

### Kernunterschied zum PPO-Agenten

| | Base Stock | PPO Agent |
|---|---|---|
| Bestellmenge | Immer gleich (fix S) | Jede Woche neu |
| Reagiert auf Forecast | Nein | Ja |
| Reagiert auf Pipeline | Nein | Ja |
| Lernfähig | Nein | Ja |

---

## Demand-Erfassung

Die Base Stock Policy sieht dieselben Daten wie der PPO-Agent — fairer Vergleich:

- **Historische Periode** → echte `demand_data` aus dem Excel (Demand-Sheet)
- **Planungsperiode (Zukunft)** → `future_forecast` aus dem Forecast-Sheet

---

## S-Wert Bestimmung

S wird **einmal** zu Beginn aus den historischen Daten berechnet und bleibt dann konstant.

**Basisformel:**
```
S = avg_demand × (lead_time + 1)
```

Wir testen **drei S-Werte gleichzeitig**, damit PPO nicht nur gegen eine schlecht gewählte Baseline gewinnt:

| Variante | Formel | Farbe im Chart |
|---|---|---|
| Konservativ | `S = avg_demand × lead_time` | Grün |
| Mittel (Basis) | `S = avg_demand × (lead_time + 1)` | Lila |
| Aggressiv | `S = avg_demand × (lead_time + 2)` | Braun |

---

## Implementierungsplan

### Schritt 1 — Funktion `run_base_stock_policy()` in `inventory_ppo.py`

Neue Funktion die dieselbe Step-Logik wie `SingleEchelonEnv` repliziert, aber statt PPO-Modell die Base-Stock-Regel verwendet. Gibt Records im **exakt gleichen Format** zurück wie die PPO-Simulation.

**Inputs:** `S`, `demand_data`, `week_labels`, `lead_time`, `initial_inventory`, Kostenparameter

**Output:** `records` (Liste von Dicts, gleiche Keys wie PPO-Records)

### Schritt 2 — `base_stock_results` befüllen

In `run_training_pipeline()`, nach dem PPO-Training, die Baseline für alle drei S-Werte berechnen und das globale Array befüllen:

```python
s_base = int(np.mean(demand_data) * (lead_time + 1))
for S in [s_base - avg_demand, s_base, s_base + avg_demand]:
    bs_records = run_base_stock_policy(S, demand_data, week_labels, ...)
    base_stock_results.append((S, bs_records))
```

### Schritt 3 — Keine weiteren Änderungen nötig

Charts und Excel-Export funktionieren **automatisch** — der Code prüft bereits `if base_stock_results:` und:

- Panel 1 (Inventory): PPO-Linie vs. Base-Stock-Linien
- Panel 4 (Cumulative Cost): Kumulierte Kosten aller Strategien im Vergleich
- Excel-Sheet "Policy Comparison": KPI-Vergleich aller Policies

---

## Erwartetes Ergebnis

Nach der Implementierung zeigt das Dashboard:

- **PPO** (blau) vs. **Base Stock S=klein** (grün) vs. **Base Stock S=mittel** (lila) vs. **Base Stock S=groß** (braun)
- KPI-Vergleich: Gesamtkosten, Service Level, Ø Lagerbestand, Bestellmenge
- Excel-Export mit "Policy Comparison"-Sheet
