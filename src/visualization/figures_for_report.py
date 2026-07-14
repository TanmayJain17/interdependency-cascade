import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Colorblind-friendly palette (Okabe-Ito)
CB_COLORS = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'yellow': '#F0E442', 'pink': '#CC79A7', 'vermilion': '#D55E00', 'skyblue': '#56B4E9'
}

def generate_figure_1():
    """Figure 1: Pipeline overview schematic (Improved Layout)"""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')
    
    stages = [
        "Hazard Inputs\n(NYC DEP / GeoClaw)",
        "Depth Correction\n(eta - ground via DEM)",
        "HAZUS Fragility &\nMonte Carlo Seeding",
        "Joint Inter+Intra\nCascade Engine",
        "Per-node / Per-timestep\nLabels",
        "CascadeGNN\nSurrogate"
    ]
    
    # Draw boxes and arrows with better spacing
    box_width = 1.7
    spacing = 2.2
    for i, text in enumerate(stages):
        rect = patches.Rectangle((i*spacing, 0.5), box_width, 1.2, fill=True, color=CB_COLORS['skyblue'], alpha=0.3, ec="black")
        ax.add_patch(rect)
        ax.text(i*spacing + (box_width/2), 1.1, text, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(stages) - 1:
            ax.arrow(i*spacing + box_width, 1.1, spacing - box_width - 0.1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
            
    # Divider for World 1 vs World 2
    ax.axvline(x=10.6, color=CB_COLORS['vermilion'], linestyle='--', linewidth=2)
    ax.text(4.5, 2.0, "WORLD 1: Physics Simulator ('Teacher', Hours)", ha='center', fontsize=11, fontweight='bold', color=CB_COLORS['vermilion'])
    ax.text(11.8, 2.0, "WORLD 2: ('Student', ms)", ha='center', fontsize=11, fontweight='bold', color=CB_COLORS['vermilion'])
    
    plt.xlim(-0.5, 13.5)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    plt.savefig("Figure_1_Pipeline_v2.png", dpi=300)
    plt.close()

def generate_figure_2():
    """Figure 2: Depth correction, before vs. after (Legend Fixed)"""
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    scenarios = ['gc_2026\n(Before)', 'gc_2026\n(After)', 'gc_2050\n(After)', 'gc_2080\n(After)']
    nodes = [373, 267, 368, 484]
    depths = [3.09, 0.90, 1.06, 1.27]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, nodes, width, label='Flooded Nodes', color=CB_COLORS['blue'])
    ax1.set_ylabel('Count of Flooded Nodes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylim(0, 550) # Give headroom for legend
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, depths, width, label='Median Depth (m)', color=CB_COLORS['orange'])
    ax2.set_ylabel('Median Depth (m)')
    ax2.set_ylim(0, 3.8) # Give headroom
    
    ax2.axhline(y=1.0, color=CB_COLORS['green'], linestyle='--', label='GISSR Sandy Median (~1.0m)')
    
    # Unified legend placed outside to avoid overlap
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.title('Figure 2: Depth Correction Impact')
    plt.tight_layout()
    plt.savefig("Figure_2_Depth_Correction_v2.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_figure_3():
    """Figure 3: Six-scenario cascade results (Annotation Fixed)"""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    scenarios = ['moderate\n_current', 'moderate\n_2050', 'extreme\n_2080', 'geoclaw\n_2026', 'geoclaw\n_2050', 'geoclaw\n_2080']
    direct = [33.9, 61.5, 298.4, 206.6, 295.5, 416.0]
    direct_err = [2.9, 3.3, 7.3, 4.3, 4.4, 4.8]
    total = [102.6, 137.0, 812.3, 1424.7, 1564.4, 1880.1]
    total_err = [18.8, 18.6, 91.2, 80.1, 45.2, 92.5]
    amplification = ["3.04x", "2.23x", "2.72x", "6.90x", "5.29x", "4.52x"]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax.bar(x[:3] - width/2, direct[:3], width, yerr=direct_err[:3], label='Direct Failures (Pluvial)', color=CB_COLORS['skyblue'], capsize=3)
    ax.bar(x[:3] + width/2, total[:3], width, yerr=total_err[:3], label='Total Failures (Pluvial)', color=CB_COLORS['blue'], capsize=3)
    
    ax.bar(x[3:] - width/2, direct[3:], width, yerr=direct_err[3:], label='Direct Failures (Surge)', color=CB_COLORS['orange'], capsize=3)
    ax.bar(x[3:] + width/2, total[3:], width, yerr=total_err[3:], label='Total Failures (Surge)', color=CB_COLORS['vermilion'], capsize=3)
    
    for i in range(len(scenarios)):
        ax.text(x[i] + width/2, total[i] + total_err[i] + 40, amplification[i], ha='center', va='bottom', fontweight='bold')
    
    # Moved benchmark annotation to top left to avoid bar overlap
    ax.text(0.1, 1800, "Brunner et al. (2024) Benchmark: ~3.16x Amplification", ha='left', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_ylabel('Number of Failures')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.88))
    
    plt.title('Figure 3: Direct vs Total Failures by Scenario')
    plt.tight_layout()
    plt.savefig("Figure_3_Cascade_Results_v2.png", dpi=300)
    plt.close()

def generate_figure_4():
    """Figure 4: Gowanus cascade chain (Text Overlap Fixed)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Part A: Timeline
    ax1.axis('off')
    ax1.text(0.5, 0.95, "(a) Gowanus Emergent Chain", ha='center', fontsize=12, fontweight='bold')
    events = [
        "t = 0\nCoastal substations flood",
        "t = 6-23h\nTerminal on backup power",
        "t ≈ 24h\nBackup dies -> inter-cascade",
        "t ≈ 24h\n~280 stations starve (intra-flow)"
    ]
    for i, event in enumerate(events):
        ax1.text(0.5, 0.75 - i*0.2, event, ha='center', bbox=dict(boxstyle="round,pad=0.4", fc=CB_COLORS['skyblue'], ec="black", alpha=0.5))
        if i < len(events)-1:
            ax1.arrow(0.5, 0.65 - i*0.2, 0, -0.06, head_width=0.04, head_length=0.03, fc='black', ec='black')
    
    # Part B: Stacked Bar
    causes = ['Flood', 'Inter-Cascade', 'Intra-Overload', 'Intra-Flow']
    values = [207, 887, 23, 337]
    bottom = 0
    colors = [CB_COLORS['blue'], CB_COLORS['orange'], CB_COLORS['green'], CB_COLORS['pink']]
    
    for i in range(len(causes)):
        ax2.bar(['geoclaw_2026\n(Run 0)'], [values[i]], bottom=bottom, label=causes[i], color=colors[i], width=0.4)
        
        # Adjust text for the tiny Intra-Overload slice
        if values[i] < 50:
            ax2.text(0.25, bottom + values[i]/2, f"{causes[i]}: {values[i]}", ha='left', va='center', color='black', fontweight='bold')
            ax2.plot([0.15, 0.23], [bottom + values[i]/2, bottom + values[i]/2], color='black', linewidth=1)
        else:
            ax2.text(0, bottom + values[i]/2, f"{causes[i]}: {values[i]}", ha='center', va='center', color='white', fontweight='bold')
            
        bottom += values[i]
        
    ax2.set_ylabel('Number of Failed Nodes')
    ax2.set_title('(b) Failure Cause Decomposition')
    ax2.set_xlim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig("Figure_4_Gowanus_v2.png", dpi=300)
    plt.close()

def generate_figure_5():
    """Figure 5: Held-out PR-AUC vs. timestep (Minor polish)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    timesteps = [6, 24, 48, 96]
    inter_only = [0.959, 0.999, 0.997, 0.999]
    joint = [0.945, 0.993, 0.977, 0.982]
    
    ax.plot(timesteps, inter_only, marker='o', markersize=8, linewidth=2.5, label='Inter-only labels', color=CB_COLORS['blue'])
    ax.plot(timesteps, joint, marker='s', markersize=8, linewidth=2.5, label='Joint inter+intra labels', color=CB_COLORS['orange'])
    
    ax.set_ylim(0.90, 1.00)
    ax.set_xlabel('Timestep (Hours)', fontsize=11)
    ax.set_ylabel('PR-AUC Score', fontsize=11)
    ax.set_xticks(timesteps)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title('Figure 5: CascadeGNN Held-out PR-AUC', fontsize=12)
    plt.tight_layout()
    plt.savefig("Figure_5_PRAUC_v2.png", dpi=300)
    plt.close()

def generate_figure_6():
    """Figure 6: Per-type failures at t=96h, extreme_2080 (Minor polish)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    types = ['Power', 'Telecom', 'Hospital', 'Subway', 'Water', 'Fuel']
    failures = [13.0, 495.8, 25.0, 128.3, 35.4, 114.8]
    
    y_pos = np.arange(len(types))
    
    bars = ax.barh(y_pos, failures, color=CB_COLORS['green'], edgecolor='black', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types, fontsize=11)
    ax.set_xlabel('Average Failures', fontsize=11)
    ax.set_title('Figure 6: Per-Type Failures at t=96h (extreme_2080)', fontsize=12)
    
    # Add some padding to x-axis so text doesn't hit the border
    ax.set_xlim(0, max(failures) * 1.15)
    
    for i, v in enumerate(failures):
        ax.text(v + 8, i, f"{v:.1f}", va='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig("Figure_6_Per_Type_v2.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating updated figures...")
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print("All updated figures saved successfully!")