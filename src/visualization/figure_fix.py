import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Global typography setup
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

# Data definition
scenarios = [
    'Moderate\n(Current)', 'Moderate\n(2050)', 'Extreme\n(2080)',
    'GeoClaw\n(2026)', 'GeoClaw\n(2050)', 'GeoClaw\n(2080)'
]

direct_failures = [35, 60, 300, 206, 295, 415]
total_failures  = [106, 134, 816, 1421, 1561, 1876]
total_errors    = [15, 20, 90, 80, 50, 95]
multipliers     = ['3.04x', '2.23x', '2.72x', '6.90x', '5.29x', '4.52x']

# Color palette
c_pluvial_dir = '#5BACE6'  # Light Blue
c_pluvial_tot = '#2265A8'  # Dark Blue
c_surge_dir   = '#E59E06'  # Amber/Gold
c_surge_tot   = '#D35400'  # Rust/Orange

x = np.arange(len(scenarios))
width = 0.36

fig, ax = plt.subplots(figsize=(13, 8), dpi=300)

# 1. Plot Bars
for i in range(len(scenarios)):
    if i < 3:  # Pluvial Group
        ax.bar(x[i] - width/2, direct_failures[i], width, color=c_pluvial_dir)
        ax.bar(x[i] + width/2, total_failures[i], width, color=c_pluvial_tot, 
               yerr=total_errors[i], capsize=4, error_kw={'ecolor': 'black', 'lw': 1.5})
    else:      # Surge Group
        ax.bar(x[i] - width/2, direct_failures[i], width, color=c_surge_dir)
        ax.bar(x[i] + width/2, total_failures[i], width, color=c_surge_tot, 
               yerr=total_errors[i], capsize=4, error_kw={'ecolor': 'black', 'lw': 1.5})

# 2. Multiplier Text Annotations above Total Failure bars
for i in range(len(scenarios)):
    y_pos = total_failures[i] + total_errors[i] + 40
    ax.text(x[i] + width/2, y_pos, multipliers[i], ha='center', va='bottom', 
            fontsize=13, fontweight='bold', color='black')

# 3. Vertical Divider Line between groups
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# 4. Section Badges (Top of graph)
ax.text(1.0, 2150, 'PLUVIAL SCENARIOS', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#1A4E80',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1'))

ax.text(4.0, 2150, 'COASTAL SURGE SCENARIOS', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#A04000',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF9E7', edgecolor='#F9E79F'))

# 5. Benchmark Annotation Box placed in open space above Moderate Pluvial bars
ax.text(0.7, 1250, 'Brunner et al. (2024) Benchmark:\n~3.16x Amplification', 
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#B0BEC5', linewidth=1.2))

# 6. Title and Axis Setup
ax.set_title('Figure 3: Direct vs Total Failures by Scenario', pad=45, fontweight='bold', fontsize=18)
ax.set_ylabel('Number of Failures', labelpad=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontweight='medium')
ax.set_ylim(0, 2300)

# Gridlines
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=':', alpha=0.6, color='gray')

# 7. Simplified Legend placed outside above top frame
legend_elements = [
    Patch(facecolor='#4080BF', label='Direct Failures'),
    Patch(facecolor='#1B4F82', label='Total Failures')
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, 
          frameon=True, facecolor='white', edgecolor='#CFD8DC', fontsize=13)

plt.tight_layout()
plt.savefig('figure3_clean_no_overlap.png', dpi=300)
plt.show()