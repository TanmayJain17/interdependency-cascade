"""
plot_intra_amplifier_shift.py - Week-14 Step 4 figure (Dr. Lin).

The visual version of the net-new result: the within-network (intra) cascade is
a largely non-redundant failure channel. Built entirely from artifacts already
on disk - no new cascade run:
  outputs/cascade/intra_dead_nodes.csv      (intra-added nodes, alpha=0.2)
  outputs/economic/node_failure_downtime.csv (inter-cascade q0 per node)
  data/flood/nyc_infra_graph_all_flood.graphml (subway betweenness = load)

Panel A: of the intra-added nodes, how many are NET-NEW (inter q0<0.05, the
         inter-cascade never reaches them) vs already-failing, per network.
Panel B: the new subway amplifiers - net-new overload nodes ranked by
         betweenness LOAD (centrality didn't predict inter amplification, but
         betweenness-as-load drives this distinct intra-overload channel).

PROVISIONAL: gc_2026 magnitudes shift on corrected depths; the qualitative
finding (intra is non-redundant, reaches hubs the dependency cascade misses)
is structural.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.flood.depth_interface import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "cascade"


def _station_label(node_id: str) -> str:
    s = node_id.replace("subway_", "")
    *name, lines = s.split("_") if "_" in s else (s, "")
    pretty = " ".join(w.capitalize() for w in name).replace(" St", " St").replace(" Av", " Av")
    return f"{pretty} ({lines.upper()})" if lines else pretty


def build_figure() -> Path:
    intra = pd.read_csv(OUT / "intra_dead_nodes.csv")
    added = intra[intra["cause"].isin(["intra_overload", "intra_flow"])].copy()
    q0 = pd.read_csv(REPO_ROOT / "outputs" / "economic" / "node_failure_downtime.csv")
    q0map = dict(zip(q0["node_id"].astype(str), q0["q0"]))
    added["inter_q0"] = added["node_id"].astype(str).map(q0map).fillna(0.0)
    added["net_new"] = added["inter_q0"] < 0.05

    # subway betweenness load (same quantity the overload solver uses)
    G = nx.read_graphml(REPO_ROOT / "data" / "flood" / "nyc_infra_graph_all_flood.graphml")
    sub_nodes = [n for n, d in G.nodes(data=True) if d.get("infra_type") == "subway"]
    U = G.subgraph(sub_nodes).to_undirected()
    bet = nx.betweenness_centrality(U, normalized=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Within-network (intra) cascade is a largely non-redundant failure channel",
                 fontsize=13, fontweight="bold")

    # --- Panel A: net-new vs already-failing, per network ---
    nets = [("subway", "intra_overload", "Subway\n(Motter-Lai overload)"),
            ("water", "intra_flow", "Water\n(directed backflow)")]
    labels, newv, oldv = [], [], []
    for infra, cause, lab in nets:
        s = added[added["infra_type"] == infra]
        labels.append(lab)
        newv.append(int(s["net_new"].sum()))
        oldv.append(int((~s["net_new"]).sum()))
    y = range(len(labels))
    axA.barh(y, newv, color="#d1495b", label="net-new (inter never reaches)")
    axA.barh(y, oldv, left=newv, color="#bcbcbc", label="already failing via inter")
    for i, (nv, ov) in enumerate(zip(newv, oldv)):
        axA.text(nv / 2, i, str(nv), va="center", ha="center", color="white", fontweight="bold")
        if ov:
            axA.text(nv + ov / 2, i, str(ov), va="center", ha="center", color="black")
    axA.set_yticks(list(y)); axA.set_yticklabels(labels)
    axA.set_xlabel("intra-added nodes"); axA.legend(loc="lower right", fontsize=9)
    axA.set_title("47 of 53 intra-added nodes are net-new (89%)", fontsize=11)
    axA.invert_yaxis()

    # --- Panel B: new subway amplifiers by betweenness load ---
    nn = added[(added["infra_type"] == "subway") & added["net_new"]].copy()
    nn["load"] = nn["node_id"].astype(str).map(bet).fillna(0.0)
    nn = nn.sort_values("load", ascending=True).tail(12)
    axB.barh(range(len(nn)), nn["load"], color="#30638e")
    axB.set_yticks(range(len(nn)))
    axB.set_yticklabels([_station_label(n) for n in nn["node_id"]], fontsize=8)
    axB.set_xlabel("betweenness load (shortest-path throughput)")
    axB.set_title("New subway amplifiers: high-load hubs that\nfail only via load redistribution", fontsize=11)

    fig.text(0.5, 0.005,
             "gc_2026 (provisional): magnitudes shift on corrected depths; the qualitative finding is structural. "
             "Betweenness centrality did not predict inter-cascade amplification, but betweenness-as-load drives this intra channel.",
             ha="center", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    png = OUT / "intra_amplifier_shift.png"
    fig.savefig(png, dpi=160)
    fig.savefig(OUT / "intra_amplifier_shift.pdf")
    plt.close(fig)
    return png


if __name__ == "__main__":
    p = build_figure()
    print(f"saved: {p}")
    print(f"saved: {p.with_suffix('.pdf')}")
