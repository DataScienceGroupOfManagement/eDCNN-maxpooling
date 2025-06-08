import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'text.usetex':True,
})

network_labels = [
    'DFCN(L=2, width=4), param.= 429',
    'DFCN(L=2, width=10), param.=1131',
    'DFCN(L=2, width=2d+10), param.=65731',
    'cDCNN(L=10, s=5, bias=0.03), param.=121',
    'cDCNN-fc(L=5, s=3, bias=0.03), param.=8301',
    'eDCNN-zeropadding(L=4, s=5, bias=0.03), param.=141',
    'eDCNN-zeropadding(L=8, s=5, bias=0.03), param.=181',
    'eDCNN-maxpooling(L=4, s=5, bias=0.03), param.=56',
    'eDCNN-maxpooling(L=8, s=5, bias=0.03), param.=86',
]

color_scheme = [
    '#1f77b4',  # 蓝色系 - DFCN width=4
    '#4c96d7',  # 蓝色系 - DFCN width=10
    '#cce6ff',  # 蓝色系 - DFCN width=2d+10
    '#ff7f0e',  # Orange - cDCNN
    '#ECE70F',   #   - cDCNN-fc
    '#2ca02c',  # 绿色系 - eDCNN-zeropadding
    '#98df8a',  # 绿色系 - eDCNN-zeropadding
    '#d62728',  # 红色系 - eDCNN-maxpooling
    '#ff7f7f',  # 红色系 - eDCNN-maxpooling
]

data_by_noise_level = {
    '$\\delta$=0.01': [0.00871298, 0.012261778,0.014897587,  0.0316608, 0.01776218,  0.01803188, 0.0098501, 0.00990322, 0.01002562,],
    '$\\delta$=0.1':  [0.04578139, 0.097460526, 0.115884473, 0.03780396, 0.07030722,  0.05263059, 0.02205947,  0.01706803, 0.0151545, ],
    '$\\delta$=0.3':  [0.13856474, 0.337367022, 0.317980876, 0.06716904, 0.32100847,  0.05885089, 0.07898909, 0.04718488, 0.06496545,],
    '$\\delta$=0.5':  [0.25123852, 0.58867659, 0.544982463,0.08619233,0.63043972,  0.06358376, 0.11717337, 0.08413939, 0.1090181,  ]
}


noise_levels = list(data_by_noise_level.keys()) # These will be on the X-axis
num_networks = len(network_labels)
num_noise_levels = len(noise_levels)

performances_by_network = []
for i in range(num_networks):
    network_performance = [data_by_noise_level[noise][i] for noise in noise_levels]
    performances_by_network.append(network_performance)

bar_width = 0.1  # Width of each bar; adjusted for more bars per group
index = np.arange(num_noise_levels) # Positions for noise level groups on X-axis
fig, ax = plt.subplots(figsize=(14, 8)) # Adjust figure size to accommodate labels and legend

# Plot bars for each network architecture
for i in range(num_networks):
    bar_positions = index + (i - num_networks / 2 + 0.5) * bar_width
    rects = ax.bar(bar_positions, performances_by_network[i], bar_width, 
                  label=network_labels[i], color=color_scheme[i],  alpha=1.0)

    # Add text annotations for each bar in this group
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., # x-position: center of the bar
                height + 0.005,                     # y-position: slightly above the top of the bar
                f'{height:.5f}',                    # text: the height value, formatted to 5 decimal places
                ha='center',                        # horizontal alignment
                va='bottom',                        # vertical alignment
                fontsize=13,                         # smaller font size for data labels
                rotation=90)                        # rotate text to be vertical
                                                    # Removed xytext and textcoords parameters

ax.set_xlabel('Noise Levels', fontsize=18)
ax.set_ylabel('Test RMSE', fontsize=18)
ax.set_xticks(index)
ax.set_xticklabels(noise_levels, fontsize=14)
ax.legend( fontsize=16, title_fontsize=14, loc='upper left')

# Add grid lines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

current_ylim = ax.get_ylim()
ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

plt.tight_layout()

plt.savefig('./func2-results/func2-noisydata-comparison.pdf')
plt.savefig('./func2-results/func2-noisydata-comparison.png')

plt.show()




