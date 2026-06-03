# BWSC operational layers — gap notice

The Week 12 handoff anticipated three BWSC operational sub-layers:

| Asset | Handoff count |
| --- | --- |
| Pumping stations | 9 |
| Outfalls | 267 |
| Tide gates | 201 |

After exhausting the public source priority order (data.boston.gov
BWSC org page, BWSC's own ArcGIS org services5.arcgis.com/ji3WHeqN0AysEK5g,
BostonGIS services.arcgis.com/sFnw0xNflSi8J0uh, MassGIS catalog,
ArcGIS Online public search), no canonical layer for any of these
three asset types is publicly served. The 2023 BWSC Coastal
Stormwater Discharge Analysis report contains ~88 coastal-vulnerable
outfall IDs with no lat/lon coordinates — insufficient for spatial
use.

**What this folder ships instead** (the best public proxy):

- `massdep_cso_outfalls.geojson` — MassDEP's CSO layer (BWSC + MWRA + Cambridge)
- `epa_r1_cso_outfalls.geojson` — EPA Region 1's 2022 CSO Locations (overlapping but independent inventory)
- `npdes_facilities_outfalls.geojson` — EPA national NPDES facility-level dischargers in Boston (includes MWRA Deer Island)

**Pump stations and tide gates are NOT in this folder.** OSM coverage
of municipal water infrastructure is too sparse to use as a proxy,
and hardcoding requires an authoritative inventory we do not have.

**Path forward:** request the BWSC GIS extract through a data-sharing
agreement. The handoff numbers (9/267/201) are consistent with
BWSC's internal GIS database — that is the canonical source.
