# Baseline Planı: Base Stock Politikası

## Amaç

PPO ajanıyla karşılaştırma için kural tabanlı bir satın alma stratejisi (Base Stock Politikası) uygulanması. Bu sayede RL ajanının toplam maliyet ve hizmet düzeyi açısından klasik yönteme göre yüzde kaç daha iyi performans gösterdiği kanıtlanabilir.

---

## Arka Plan

### Base Stock Politikası Nedir?

Base Stock Politikası, pratikte kullanılan klasik satın alma stratejisidir. Basit ve sabit bir kurala dayanır:

> "Stok + pipeline'ı sabit bir hedef değer S'ye tamamlayacak kadar her zaman sipariş ver."

```
order_qty = max(0, S - (mevcut_stok + pipeline_toplamı))
```

### Neden Doğru Baseline Bu?

- Envanter RL çalışmaları için endüstri standardı
- PPO ile aynı maliyet yapısı (Holding, Ordering, Lost Sales)
- Deterministik ve anlaşılır — ML yok
- Kod altyapısı zaten hazır (`base_stock_results`)

### PPO Ajanıyla Temel Fark

| | Base Stock | PPO Ajan |
|---|---|---|
| Sipariş miktarı | Her zaman aynı (sabit S) | Her hafta yeniden hesaplanır |
| Tahmine tepki | Hayır | Evet |
| Pipeline'a tepki | Hayır | Evet |
| Öğrenebilir | Hayır | Evet |

---

## Talep Verisi Kullanımı

Base Stock Politikası, PPO ajanıyla aynı verileri görür — adil karşılaştırma:

- **Geçmiş dönem** → Excel'den gerçek `demand_data` (Demand-Sheet)
- **Planlama dönemi (gelecek)** → Forecast-Sheet'ten `future_forecast`

---

## S Değerinin Belirlenmesi

S, geçmiş verilerden **bir kez** hesaplanır ve sabit kalır.

**Temel formül:**
```
S = ort_talep × (temin_süresi + 1)
```

PPO'nun yalnızca kötü seçilmiş bir baseline'a karşı kazanmaması için **aynı anda üç farklı S değeri** test edilir:

| Varyant | Formül | Grafik Rengi |
|---|---|---|
| Muhafazakâr | `S = ort_talep × temin_süresi` | Yeşil |
| Orta (Baz) | `S = ort_talep × (temin_süresi + 1)` | Mor |
| Agresif | `S = ort_talep × (temin_süresi + 2)` | Kahverengi |

---

## Uygulama Planı

### Adım 1 — `inventory_ppo.py`'a `run_base_stock_policy()` Fonksiyonu Ekleme

`SingleEchelonEnv` ile aynı adım mantığını kullanan, ancak PPO modeli yerine Base Stock kuralını uygulayan yeni bir fonksiyon. PPO simülasyonuyla **tamamen aynı formatta** kayıt döndürür.

**Girdiler:** `S`, `demand_data`, `week_labels`, `lead_time`, `initial_inventory`, maliyet parametreleri

**Çıktı:** `records` (PPO kayıtlarıyla aynı anahtarlara sahip dict listesi)

### Adım 2 — `base_stock_results` Doldurulması

PPO eğitiminin ardından `run_training_pipeline()` içinde üç S değeri için baseline hesaplanır ve global dizi doldurulur:

```python
s_base = int(np.mean(demand_data) * (lead_time + 1))
for S in [s_base - avg_demand, s_base, s_base + avg_demand]:
    bs_records = run_base_stock_policy(S, demand_data, week_labels, ...)
    base_stock_results.append((S, bs_records))
```

### Adım 3 — Başka Değişiklik Gerekmez

Grafikler ve Excel dışa aktarımı **otomatik olarak** çalışır — kod zaten `if base_stock_results:` kontrolü yapıyor ve şunları gösteriyor:

- Panel 1 (Inventory): PPO çizgisi vs. Base Stock çizgileri
- Panel 4 (Cumulative Cost): Tüm stratejilerin kümülatif maliyet karşılaştırması
- Excel "Policy Comparison" sayfası: Tüm politikaların KPI karşılaştırması

---

## Beklenen Sonuç

Uygulamanın ardından dashboard şunları gösterecek:

- **PPO** (mavi) vs. **Base Stock S=küçük** (yeşil) vs. **Base Stock S=orta** (mor) vs. **Base Stock S=büyük** (kahverengi)
- KPI karşılaştırması: Toplam maliyet, hizmet düzeyi, ortalama stok, sipariş miktarı
- "Policy Comparison" sayfasıyla Excel dışa aktarımı
