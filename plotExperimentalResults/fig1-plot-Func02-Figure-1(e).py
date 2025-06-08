import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 120,
    'savefig.dpi': 120,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'text.usetex':True,
})


network_structures_maxpooling = [
    "eDCNN-maxpooling (L=8, S=5, b=0.03)",
    "eDCNN-maxpooling (L=8, S=7, b=0.03)",
    "eDCNN-maxpooling (L=8, S=11, b=0.03)",
    "eDCNN-maxpooling (L=8, S=15, b=0.03)",
    "eDCNN-maxpooling (L=8, S=17, b=0.03)",
    "eDCNN-maxpooling (L=8, S=21, b=0.03)"
]


prediction_results_maxpooling = [
    0.00981606,
    0.01035955,
    0.01225488,
    0.01326073,
    0.01148136,
    0.01510463
]

# Data for eDCNN-zeropadding
network_structures_zeropadding = [
    "eDCNN-zeropadding(L=8, S=5, b=0.03)",
    "eDCNN-zeropadding(L=8, S=7, b=0.03)",
    "eDCNN-zeropadding(L=8, S=11, b=0.03)",
    "eDCNN-zeropadding(L=8, S=15, b=0.03)",
    "eDCNN-zeropadding(L=8, S=17, b=0.03)",
    "eDCNN-zeropadding(L=8, S=21, b=0.03)"
]


prediction_results_zeropadding = [
    0.01084637,
    0.01070921,
    0.02840706,
    0.0187854,
    0.02640004,
    0.02927771
]

filter_sizes = [5, 7, 11, 15, 17, 21]

# Filter selected sizes
selected_filter_sizes = [5, 7, 17, 21]

# Get indices of selected filter sizes
selected_indices = [i for i, size in enumerate(filter_sizes) if size in selected_filter_sizes]

# Filter data for selected sizes
filtered_sizes = [filter_sizes[i] for i in selected_indices]
filtered_maxpooling = [prediction_results_maxpooling[i] for i in selected_indices]
filtered_zeropadding = [prediction_results_zeropadding[i] for i in selected_indices]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set bar width and positions
bar_width = 0.35
x = np.arange(len(filtered_sizes))

# Set colors
red_color = '#d62728'   
green_color = '#2ca02c'   

# Create bars
rects1 = ax.bar(x - bar_width/2, filtered_maxpooling, bar_width, 
                label='eDCNN-maxpooling(L=8, bias=0.03)', color=red_color)
rects2 = ax.bar(x + bar_width/2, filtered_zeropadding, bar_width,
                label='eDCNN-zeropadding(L=8, bias=0.03)', color=green_color)

# Add labels and title
ax.set_xlabel('The length of filter (s)')
ax.set_ylabel('Test RMSE')
ax.set_xticks(x)
ax.set_xticklabels(filtered_sizes)

# Add a second x-axis label to indicate size categories
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([x[1]/2, (x[2] + x[3]/2)])
ax2.set_xticklabels(['Small filter length', 'Large filter length'])

ax.legend()
ax.grid(True, linestyle='--', alpha=0.7, axis='y')

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

plt.savefig('./func2-results/func2-noisefree-diffFilterSize-comparison.png')
plt.savefig('./func2-results/func2-noisefree-diffFilterSize-comparison.pdf')

plt.show()



