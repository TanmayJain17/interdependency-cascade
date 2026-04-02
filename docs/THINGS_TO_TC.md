# Feature & Parameter Reference — Values Used vs. Real Data Needed

**Project:** GNN-based Flood-Induced Cascading Infrastructure Failure Prediction  
**Scope:** Lower Manhattan heterogeneous infrastructure graph  
**Last updated:** Week 3 (graph construction + PyG conversion)

---

## 1. NODE FEATURES (dim=8 per type, used in `convert_to_pyg.py`)

### 1.1 POWER SUBSTATIONS (14 nodes)

| # | Feature | Current Value | Source | Real Data Needed? | Where to Get Real Data |
|---|---------|--------------|--------|-------------------|----------------------|
| 0 | `lat` | Real coordinates | HIFLD via Rutgers ArcGIS | ✅ Already real | — |
| 1 | `lon` | Real coordinates | HIFLD via Rutgers ArcGIS | ✅ Already real | — |
| 2 | `elevation` | **3.0 m** (flat default) | Assumed LM average | ⚠️ Need real per-node | USGS 3DEP 1-meter DEM: `https://apps.nationalmap.gov/downloader/` — download DEM tile for Manhattan, sample elevation at each node's lat/lon |
| 3 | `flood_depth` | **0.0 m** (placeholder) | Placeholder | ⚠️ GISSR overlay (NEXT STEP) | Dr. Miura's GISSR model output — Sandy/Ida scenarios |
| 4 | `is_active` | 1.0 if STATUS="IN SERVICE", else 0.0 | HIFLD STATUS field | ✅ Already real | — |
| 5 | `max_volt_kv` | Derived from connected line VOLT_CLASS: 345→345kV, 220-287→250kV, 100-161→138kV, UNDER 100→69kV, NOT AVAILABLE→**138kV assumed** | HIFLD VOLT_CLASS on transmission lines | ⚠️ Partially real | Con Edison OASIS filings or ask Dr. Lin for bus voltage data |
| 6 | `degree_centrality` | Computed from graph topology | NetworkX `degree_centrality()` | ✅ Already real (derived) | — |
| 7 | `is_external` | 1.0 if substation is outside LM bbox | Graph construction logic | ✅ Already real | — |

**Missing power features that matter for pandapower (Issue 6 — for Dr. Lin):**
- Bus voltage magnitude (kV) — not just voltage class
- Line impedance (R, X in ohms/km)
- Line thermal capacity (MVA)
- Generator output at each bus (MW)
- Load demand at each bus (MW)

---

### 1.2 TELECOM CLUSTERS (224 nodes)

| # | Feature | Current Value | Source | Real Data Needed? | Where to Get Real Data |
|---|---------|--------------|--------|-------------------|----------------------|
| 0 | `lat` | Real (cluster centroid) | OpenCelliD | ✅ Already real | — |
| 1 | `lon` | Real (cluster centroid) | OpenCelliD | ✅ Already real | — |
| 2 | `elevation` | **3.0 m** (flat default) | Assumed | ⚠️ Need real | USGS 3DEP DEM (same as power) |
| 3 | `flood_depth` | **0.0 m** (placeholder) | Placeholder | ⚠️ GISSR overlay | Dr. Miura's GISSR |
| 4 | `tower_count` | Real count per 500m grid cell | OpenCelliD grouped count | ✅ Already real | — |
| 5 | `has_lte` | 1.0 if any tower in cluster has radio="LTE" | OpenCelliD `radio` field | ✅ Already real | — |
| 6 | `has_5g` | 1.0 if any tower has radio="NR" or "5G" | OpenCelliD `radio` field | ✅ Already real (though 5G coverage in OpenCelliD may be incomplete) | — |
| 7 | `battery_backup_hrs` | **6.0 hours** (all clusters) | FCC recommendation median of 4-8h range | ⚠️ Assumed uniform | FCC NPRM 15-98 filings; carrier-specific backup power plans (AT&T, T-Mobile, Verizon file with FCC); or use sensitivity analysis with 4h/6h/8h |

---

### 1.3 HOSPITALS (8 nodes)

| # | Feature | Current Value | Source | Real Data Needed? | Where to Get Real Data |
|---|---------|--------------|--------|-------------------|----------------------|
| 0 | `lat` | Real coordinates | NYC Facilities Database | ✅ Already real | — |
| 1 | `lon` | Real coordinates | NYC Facilities Database | ✅ Already real | — |
| 2 | `elevation` | **3.0 m** (flat default) | Assumed | ⚠️ Need real | USGS 3DEP DEM |
| 3 | `flood_depth` | **0.0 m** (placeholder) | Placeholder | ⚠️ GISSR overlay | Dr. Miura's GISSR |
| 4 | `bed_count` | **200** (all hospitals) | Assumed NYC acute care average | ⚠️ Need per-hospital | NYS DOH Health Facility Profile: `https://profiles.health.ny.gov/hospital/` — search by facility name, bed count is listed. Also: CMS Hospital Compare API |
| 5 | `generator_fuel_hrs` | **96 hours** (all hospitals) | NFPA 110 Level 1 minimum (hospitals must have 96h fuel) | ⚠️ Minimum, not actual | Facility-level emergency plans (unlikely to be public). 96h is the defensible floor — actual capacity varies 96-240h. Use 96h as conservative default |
| 6 | `water_storage_days` | **1.0 day** (all hospitals) | Engineering estimate | ⚠️ Assumed | Facility-level data; typically 1-3 days. Could vary by hospital size |
| 7 | `is_critical` | **1.0** (all hospitals) | All hospitals treated as critical | ✅ Reasonable assumption | — (all hospitals are critical by definition in disaster scenarios) |

---

### 1.4 SUBWAY STATIONS (75 nodes)

| # | Feature | Current Value | Source | Real Data Needed? | Where to Get Real Data |
|---|---------|--------------|--------|-------------------|----------------------|
| 0 | `lat` | Real coordinates | MTA GTFS data | ✅ Already real | — |
| 1 | `lon` | Real coordinates | MTA GTFS data | ✅ Already real | — |
| 2 | `elevation` | **3.0 m** (street-level default) | Assumed | ⚠️ Need real | USGS 3DEP DEM for street-level elevation |
| 3 | `flood_depth` | **0.0 m** (placeholder) | Placeholder | ⚠️ GISSR overlay | Dr. Miura's GISSR — her subway flooding paper has this data specifically |
| 4 | `num_routes` | Real count of daytime routes at station | MTA `Daytime Routes` field | ✅ Already real | — |
| 5 | `ada_accessible` | 1.0 if ADA ≥ 1 (full access) | MTA `ADA` field (0/1/2) | ✅ Already real | — |
| 6 | `depth_below_surface` | **15.0 m** (all stations) | NYC subway average depth estimate | ⚠️ Assumed uniform | MTA Capital Construction data; some available in: Munoz, C.L. "Depth of NYC subway stations" report; or OpenStreetMap `level` tags on some stations; or Dr. Miura may have this from her subway flooding paper |
| 7 | `has_flood_gates` | **0.0** (all stations) | Most stations lack flood protection | ⚠️ Assumed all unprotected | MTA Climate Adaptation reports; post-Sandy resilience projects installed flex-gates at ~37 stations — list may be in MTA capital budget documents or in: MTA 2017 "Resilience Program Progress Report" |

---

### 1.5 WATER INFRASTRUCTURE (7 nodes)

| # | Feature | Current Value | Source | Real Data Needed? | Where to Get Real Data |
|---|---------|--------------|--------|-------------------|----------------------|
| 0 | `lat` | Real coordinates | NYC FacDB + OSM + DEP | ✅ Already real | — |
| 1 | `lon` | Real coordinates | NYC FacDB + OSM + DEP | ✅ Already real | — |
| 2 | `elevation` | **3.0 m** (flat default) | Assumed | ⚠️ Need real | USGS 3DEP DEM |
| 3 | `flood_depth` | **0.0 m** (placeholder) | Placeholder | ⚠️ GISSR overlay | Dr. Miura's GISSR |
| 4 | `is_treatment_plant` | 1.0 if subtype contains "TREATMENT" or "CONTROL" | FacDB FACTYPE field | ✅ Already real | — |
| 5 | `is_pump` | 1.0 if subtype contains "PUMP" | FacDB FACTYPE field | ✅ Already real | — |
| 6 | `capacity_proxy` | **50 MGD** (all nodes) | Generic estimate | ⚠️ Assumed uniform | NYC DEP Annual Report — actual plant capacities are published. The 14 hardcoded plants already have real `capacity_mgd` values in `nyc_water_infra_data.py` — wire these into the feature vector |
| 7 | `is_external` | 1.0 if outside LM bbox | Graph construction logic | ✅ Already real | — |

---

## 2. EDGE FEATURES (dim=4, used in `convert_to_pyg.py`)

| # | Feature | Current Value | Source | Notes |
|---|---------|--------------|--------|-------|
| 0 | `weight` | Varies: inverse-distance for dependencies, voltage-based for power lines | Computed from coordinates + VOLT_CLASS | ✅ Real (derived) |
| 1 | `distance_m` | Real Euclidean distance (UTM) between node coords | Computed from lat/lon | ✅ Real |
| 2 | `buffer_hours` | See Section 3 below | Engineering standards + assumptions | ⚠️ See details |
| 3 | `coupling_strength` | `1.0 / (1.0 + buffer_hours)` | Derived from buffer_hours | ✅ Derived |

---

## 3. BUFFER DURATIONS (used in `build_graph_v3.py` edge attributes)

These are the delay (in hours) before a failed upstream node causes failure at the downstream node.

| Edge Type | Buffer (hours) | Justification | Confidence | How to Improve |
|-----------|---------------|---------------|------------|---------------|
| `power_line` | **0.0** | Power failure propagates at speed of electricity | ✅ High | — (physically correct) |
| `subway_line` | **0.0** | Station flooding propagates immediately to adjacent stations via tunnels | ✅ High | — |
| `power → telecom` | **6.0** | Cell tower battery backup: FCC recommends 8h, real-world median 4-8h, used 6h | ⚠️ Medium | FCC NPRM 15-98; carrier filings. Varies by carrier and site. Consider distribution: rural=less, urban=more |
| `power → hospital` | **96.0** | NFPA 110 Type 10 Level 1: hospitals must maintain 96h generator fuel | ⚠️ Medium-High | NFPA 110 is regulatory minimum — actual could be higher. 96h is defensible lower bound |
| `power → subway` | **0.0** | Subway traction power is immediate — no battery backup for train operations | ✅ High | — |
| `power → water` | **3.0** | Pump stations have 2-4h diesel backup; used median 3h | ⚠️ Medium | NYC DEP emergency plans; varies by pump station. Some critical pumps have longer backup |
| `telecom → power` (SCADA) | **2.0** | SCADA systems have local RTU cache; operators lose remote visibility in ~1-2h | ⚠️ Low-Medium | Con Edison SCADA architecture docs (not public). Ask Dr. Lin |
| `water → hospital` | **24.0** | Hospital stored water reserves: ~1 day typical | ⚠️ Low | Facility-specific; varies enormously (12h to 72h). 24h is a reasonable median |
| `subway → infra` (repair) | **0.0** | Not a failure propagation delay — it's an access constraint | ✅ N/A | This is recovery layer, not cascade |

---

## 4. OTHER CONSTANTS & ASSUMPTIONS

| Parameter | Value | Used In | Justification | Where to Improve |
|-----------|-------|---------|---------------|-----------------|
| Telecom grid cell size | 500m (~0.005°) | `build_graph_v3.py` GRID | Balance between spatial resolution and node count | Sensitivity analysis: try 250m, 500m, 750m |
| Power dependency assignment | Nearest substation | `build_graph_v3.py` 4c | Every building draws from closest substation's distribution network | Con Edison service territory maps (proprietary). Ask Dr. Lin |
| Water flow radius | 5,000m | `build_graph_v3.py` 4d | Pump stations connect to treatment plants within 5km | NYC DEP sewer shed maps: `https://data.cityofnewyork.us/` search "drainage area" |
| LM bounding box (lat) | 40.700 – 40.755 | All scripts | Southern tip to ~34th St | Matches Dr. Miura's GISSR flood model coverage |
| LM bounding box (lon) | -74.020 – -73.970 | All scripts | Hudson River to East River | Same |
| NYC bounding box | 40.49–40.92, -74.27–-73.70 | Download scripts | Full 5 boroughs + NJ border | — |
| Default elevation | 3.0m above sea level | `convert_to_pyg.py` | Lower Manhattan average | USGS DEM |
| Voltage class → kV mapping | 345→345, 220-287→250, 100-161→138, UNDER 100→69, N/A→138 | `convert_to_pyg.py` | Midpoint of HIFLD ranges | HIFLD metadata docs; Con Edison OASIS |

---

## 5. PRIORITY ORDER FOR REPLACING DEFAULTS WITH REAL DATA

### Priority 1 — Do this week (directly impacts cascade simulation)
1. **flood_depth** → GISSR overlay (next step in pipeline)
2. **elevation** → USGS 3DEP DEM download + sampling

### Priority 2 — Do before paper submission (improves model accuracy)
3. **bed_count** → NYS DOH hospital profiles (10 min of web scraping)
4. **depth_below_surface** → Check Dr. Miura's subway flooding paper data
5. **has_flood_gates** → MTA resilience report (list of protected stations)
6. **capacity_proxy for water** → Already have real values in DEP hardcoded data — wire them in

### Priority 3 — Nice to have (marginal improvement)
7. **battery_backup_hrs** → FCC filings or use sensitivity analysis
8. **generator_fuel_hrs** → Leave at 96h (NFPA minimum is defensible)
9. **water_storage_days** → Facility-specific, unlikely to find public data

### Priority 4 — Requires Dr. Lin's input
10. **Electrical parameters** (impedance, capacity, load) → Issue 6, discuss Friday

---

## 6. FEATURE COMPLETENESS SCORECARD

| Node Type | Total Features | From Real Data | From Defaults | % Real |
|-----------|---------------|----------------|---------------|--------|
| Power | 8 | 5 (lat, lon, is_active, max_volt*, degree_centrality, is_external) | 3 (elevation, flood_depth, max_volt fallback) | 63% |
| Telecom | 8 | 5 (lat, lon, tower_count, has_lte, has_5g) | 3 (elevation, flood_depth, battery_backup) | 63% |
| Hospital | 8 | 2 (lat, lon) | 6 (elevation, flood_depth, bed_count, generator, water_storage, is_critical**) | 25% |
| Subway | 8 | 4 (lat, lon, num_routes, ada_accessible) | 4 (elevation, flood_depth, depth, flood_gates) | 50% |
| Water | 8 | 4 (lat, lon, is_treatment, is_pump, is_external) | 3 (elevation, flood_depth, capacity) | 50% |

*max_volt is real when derived from connected line VOLT_CLASS, default when no lines connect
**is_critical=1.0 is an assumption, but a defensible one for all hospitals

**Overall: ~50% of node features are from real data.** After GISSR overlay (flood_depth) and DEM (elevation), this jumps to ~70%.