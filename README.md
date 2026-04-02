# Flood-Induced Cascading Infrastructure Failure Prediction using GNNs

This project builds a heterogeneous Graph Neural Network (GNN) framework to predict cascading infrastructure failures in NYC Lower Manhattan under hurricane-scale flood events. We construct a directed heterogeneous infrastructure graph (328 nodes, 752 edges, 5 infrastructure types) from six real NYC data sources, overlay flood simulation depths from Dr. Yuki Miura's GISSR model validated against Hurricane Sandy ground truth, and convert the result to PyTorch Geometric HeteroData format for downstream GNN training. Sandy validation achieves 68% recall overall and 100% recall on critical infrastructure nodes.

---

## Pipeline

```
Stage 1: Data Acquisition          Stage 2: Graph Construction
  download_power.py         ──►      build_graph.py
  download_telecom.py               (328 nodes, 752 edges,
  download_hospitals.py              5 node types, 13 edge types)
  download_subway.py                       │
  download_water.py                        ▼
        │                         Stage 3: Flood Overlay
        │                           flood_overlay.py
        │                         (GISSR Sandy depths,
        │                          195/328 nodes flooded)
        │                                  │
        │                                  ▼
        │                         Stage 4: Validation
        │                           validate_sandy.py
        │                         (68% recall, 100% critical)
        │                                  │
        │                                  ▼
        └──────────────────────► Stage 5: PyG Conversion
                                    convert_to_pyg.py
                                  (HeteroData, ready for GNN)
```

---

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/TanmayJain17/interdependency-cascade.git
cd interdependency-cascade

# 2. Create conda environment
conda create -n infra-gnn python=3.10
conda activate infra-gnn
pip install -r requirements.txt

# 3. Clone the GISSR flood simulation repo (required for flood overlay)
git clone https://github.com/ym2540/GIS_FloodSimulation.git

# 4. Clone the NYC flood layers reference repo (optional, for reference)
git clone https://github.com/mebauer/nyc-flood-layers.git
```

---

## Reproduce the Pipeline

Run all scripts from the **project root** (`~/Desktop/RA/`):

```bash
# Stage 1: Download infrastructure data
python src/data_acquisition/download_power.py
python src/data_acquisition/download_telecom.py
python src/data_acquisition/download_hospitals.py
python src/data_acquisition/download_subway.py
python src/data_acquisition/download_water.py

# Stage 2: Build heterogeneous infrastructure graph
python src/graph/build_graph.py

# Stage 3: Overlay GISSR flood simulation depths
python src/flood/flood_overlay.py

# Stage 4: Validate against Hurricane Sandy ground truth
python src/flood/validate_sandy.py

# Stage 5: Convert to PyTorch Geometric HeteroData
python src/graph/convert_to_pyg.py

# Visualize
python src/visualization/visualize_graph.py
python src/visualization/visualize_map.py
```

---

## Current Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Data Acquisition (5 infra types) | Done |
| 2 | Heterogeneous Graph Construction | Done |
| 3 | GISSR Flood Overlay | Done |
| 4 | Sandy Ground-Truth Validation | Done |
| 5 | PyG HeteroData Conversion | Done |
| 6 | GNN Model Training | In Progress |

---

## Data Sources

| Layer | Source | Coverage |
|-------|--------|----------|
| Power substations | HIFLD (Homeland Infra. Foundation-Level Data) | NYC-wide |
| Telecom towers | OpenCelliD (MCC 310) | NYC-wide |
| Hospitals / healthcare | NYC Facilities Database | NYC-wide |
| Subway stations | MTA GTFS open data | NYC-wide |
| Water infrastructure | NYC Open Data (DEP) | NYC-wide |
| Flood simulation | GISSR model (Dr. Yuki Miura, Columbia) | Lower Manhattan |
| Sandy validation | NYC OEM Sandy Inundation Zone | NYC-wide |

---

## Graph Statistics

- **Nodes:** 328 (Power: 14, Telecom: 224, Hospital: 8, Subway: 75, Water: 7)
- **Edges:** 752 directed (13 heterogeneous edge types)
- **Flooded nodes (Sandy cold storm):** 195 / 328 (59.5%)
- **Node features:** 8 per node type (elevation, flood_depth, capacity, age, etc.)
- **Edge features:** 4 (weight, distance_m, buffer_hours, coupling_strength)

---

## Authors

- **Tanmay Jain** — NYU Courant Institute of Mathematical Sciences
- **Advisor:** Dr. Yuki Miura — NYU Tandon School of Engineering (GISSR flood model)
- **Advisor:** Dr. Yuzhang Lin — NYU Tandon School of Engineering
