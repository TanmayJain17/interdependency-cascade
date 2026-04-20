# NYC Infrastructure Cascade Analysis

Predicting flood-induced cascading failures across NYC critical infrastructure using a heterogeneous graph pipeline and graph neural networks.

**Lab:** Climate, Energy, and Risk Analytics Lab, NYU Tandon School of Engineering
**Advisors:** Prof. Yuki Miura (primary), Prof. Yuzhang Lin (power systems)
**Researcher:** Tanmay Jain, NYU Courant Institute

---

## Overview

This project models cascading failures across six interdependent urban infrastructure systems — power, water, telecom, subway, hospitals, and fuel — under three NYC DEP flood scenarios. The pipeline combines physics-based simulation with graph neural networks following the I³ framework.

### Research question

Given a flood scenario over NYC, which infrastructure nodes fail directly, and which fail through cascading dependencies propagated through the interconnected network?

### Week 6 key findings

Under the Extreme 2080 scenario (500-year storm with 2080 sea level rise):
- 298 nodes fail directly from flooding
- 797 total nodes fail after 96 hours of cascade propagation
- Amplification ratio of 2.67x, consistent with published cascade literature (Brunner et al. ~2.16x reference)
- Four Manhattan hospitals — Bellevue, NYU Langone, Mount Sinai Beth Israel, Mount Sinai NYEE — fail through cascade in 98.9% of Monte Carlo runs despite not being directly flooded. These are the same hospitals that evacuated during Hurricane Sandy in 2012.

---

## Pipeline architecture

Five stages:

1. **Data acquisition** — download infrastructure data from HIFLD, MTA, OpenStreetMap, NYSERDA, OpenCelliD, and NYC DEP
2. **Graph construction** — build heterogeneous directed graph (6,231 nodes, 13,709 edges, 9 edge types)
3. **Flood overlay** — apply DEP flood scenarios via spatial join
4. **Monte Carlo simulation** — HAZUS-based fragility sampling plus cascade propagation (1000 runs per scenario)
5. **GNN prediction** — GraphTransformer + RGCN following I³ framework (in progress)

---

## Repository structure

```
.
├── src/
│   ├── data_acquisition/    # Download scripts (one per infrastructure type)
│   ├── graph/               # Graph construction, PyTorch Geometric conversion
│   ├── flood/               # DEP flood overlay pipeline
│   ├── simulation/          # Fragility curves, Monte Carlo, cascade simulation
│   └── visualization/       # Static (matplotlib) and interactive (folium) maps
├── data/
│   ├── power/               # Substations, transmission lines
│   ├── fuel/                # Gas stations, petroleum terminals
│   ├── healthcare/          # Hospitals
│   ├── transit/             # Subway stations
│   ├── water/               # Water infrastructure
│   ├── telecom/             # Cell towers (clustered)
│   ├── graph/               # Constructed heterogeneous graphs
│   ├── flood/               # Flood-tagged graphs
│   └── simulation/          # Aggregated cascade results (raw runs gitignored)
└── outputs/                 # Generated figures (gitignored)
```

Large raw data files and per-run simulation outputs are gitignored. They are regeneratable by running the download and simulation scripts.

---

## Setup

### Environment

```bash
conda create -n infra-cascade python=3.11
conda activate infra-cascade
pip install -r requirements.txt
```

Key packages: `networkx`, `geopandas`, `shapely`, `torch`, `torch-geometric`, `folium`, `matplotlib`, `pandas`.

### Run the pipeline

All scripts run from the project root.

```bash
# Stage 1: download raw data
python3 src/data_acquisition/download_power.py
python3 src/data_acquisition/download_hospitals.py
python3 src/data_acquisition/download_subway.py
python3 src/data_acquisition/download_water.py
python3 src/data_acquisition/download_fuel.py
python3 src/data_acquisition/download_telecom.py
python3 src/data_acquisition/fetch_dep_flood_maps.py

# Stage 2: build the heterogeneous graph
python3 src/graph/build_graph_nyc.py
python3 src/graph/convert_to_pyg_nyc.py

# Stage 3: apply DEP flood overlay
python3 src/flood/flood_overlay_v3.py nyc

# Stage 4: run cascade Monte Carlo across all three DEP scenarios
python3 src/simulation/multi_scenario_runner.py

# Stage 5: generate visualizations
python3 src/visualization/visualize_cascade_static_city.py
python3 src/visualization/visualize_cascade_interactive_city.py
```

---

## Current state (Week 6)

**Complete:**
- Citywide heterogeneous graph (6,231 nodes, 9 edge types, 13,709 directed edges)
- Three DEP flood scenarios applied via spatial join with geometric validation
- HAZUS-based Monte Carlo fragility sampling (1000 runs per scenario, 3000 total)
- Inter-infrastructure cascade propagation through dependency edges with buffer hours
- Static and interactive citywide visualizations

**In progress:**
- Intra-infrastructure power grid cascade via pandapower (awaiting electrical parameters)
- GraphTransformer + RGCN implementation for Stage 5 prediction
- Multi-hazard overlay combining DEP scenarios with Sandy GISSR storm surge

**Known limitations:**
- DEP flood maps exclude storm surge by design — surge-vulnerable infrastructure (FDR corridor, Staten Island coast) is under-represented
- Current cascade is rule-based with buffer hours rather than physics-based — pandapower integration pending
- Redundancy currently modeled as OR-gate rather than AND-gate for backup systems

---

## Data sources

| Infrastructure | Source | NYC count |
|---|---|---|
| Power substations | HIFLD (Rutgers mirror) | 208 |
| Transmission lines | HIFLD (Rutgers mirror) | 186 |
| Hospitals | NYC Facilities Database | 74 |
| Subway stations | MTA GTFS feed | 496 |
| Water infrastructure | NYC FacDB + DEP plants | 178 |
| Gas stations | OpenStreetMap + NYS ArcGIS | 1,181 |
| Petroleum terminals | NYSERDA Terminal Resiliency Assessment | 6 |
| Cell towers | OpenCelliD (clustered to 500m grid) | 4,150 |
| Flood scenarios | NYC DEP Stormwater Flood Maps | 3 scenarios |

---

## Contact

Tanmay Jain — Graduate Research Assistant, NYU Courant / CUSP
Primary advisor: Prof. Yuki Miura
