import geopandas as gpd
from shapely.validation import make_valid

print("Loading Sandy Inundation Zone...")
sandy = gpd.read_file("data/flood/sandy_inundation_zone.geojson")
print(f"  Features: {len(sandy)}")

# Fix invalid geometries
sandy["geometry"] = sandy.geometry.apply(lambda g: make_valid(g) if g is not None else g)
sandy = sandy[sandy.geometry.notna()].reset_index(drop=True)
print(f"  Valid geometries: {len(sandy)}")

print("\nLoading flood-tagged nodes...")
nodes = gpd.read_file("data/flood/lm_infra_nodes_flood.geojson")
nodes = nodes.set_crs("EPSG:4326", allow_override=True)
print(f"  Nodes: {len(nodes)}")

# Spatial join: which nodes fall inside ANY Sandy polygon
print("\nSpatial join...")
joined = gpd.sjoin(nodes, sandy, how="left", predicate="within")
nodes["sandy_observed"] = joined.index_right.notna().groupby(level=0).any()

# Confusion matrix
print("\n" + "=" * 60)
print("VALIDATION: GISSR Simulation vs Observed Sandy Inundation")
print("=" * 60)

gissr_yes = nodes["sandy_inundated"] == True
obs_yes   = nodes["sandy_observed"] == True

tp = (gissr_yes & obs_yes).sum()
fp = (gissr_yes & ~obs_yes).sum()
fn = (~gissr_yes & obs_yes).sum()
tn = (~gissr_yes & ~obs_yes).sum()

print(f"\n                    Observed Sandy")
print(f"                    YES      NO")
print(f"  GISSR YES      {tp:>5d}   {fp:>5d}")
print(f"  GISSR NO       {fn:>5d}   {tn:>5d}")
print(f"\n  True positives:   {tp}  (both say flooded)")
print(f"  False positives:  {fp}  (GISSR says flooded, Sandy says dry)")
print(f"  False negatives:  {fn}  (GISSR says dry, Sandy says flooded)")
print(f"  True negatives:   {tn}  (both agree: dry)")

if (tp + fn) > 0:
    print(f"\n  Recall:    {tp/(tp+fn):.1%}  (of observed-flooded, how many did GISSR catch)")
if (tp + fp) > 0:
    print(f"  Precision: {tp/(tp+fp):.1%}  (of GISSR-flooded, how many were really flooded)")
accuracy = (tp + tn) / (tp + fp + fn + tn)
print(f"  Accuracy:  {accuracy:.1%}")

print(f"\nBy infrastructure type:")
print(f"{'Type':<12} {'GISSR':>6} {'Sandy':>6} {'Both':>5} {'GISSR only':>11} {'Sandy only':>11}")
print("-" * 58)
for itype in ["power", "telecom", "hospital", "subway", "water"]:
    sub = nodes[nodes["infra_type"] == itype]
    g = (sub["sandy_inundated"] == True).sum()
    s = (sub["sandy_observed"] == True).sum()
    both = ((sub["sandy_inundated"] == True) & (sub["sandy_observed"] == True)).sum()
    g_only = ((sub["sandy_inundated"] == True) & (sub["sandy_observed"] == False)).sum()
    s_only = ((sub["sandy_inundated"] == False) & (sub["sandy_observed"] == True)).sum()
    print(f"{itype:<12} {g:>6} {s:>6} {both:>5} {g_only:>11} {s_only:>11}")

nodes.to_file("data/flood/lm_infra_nodes_validated.geojson", driver="GeoJSON")
print(f"\nSaved -> data/flood/lm_infra_nodes_validated.geojson")