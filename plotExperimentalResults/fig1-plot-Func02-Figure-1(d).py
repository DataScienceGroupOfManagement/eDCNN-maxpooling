import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
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


DFCN_width_legends = ['DFCN(L=2,width=2)',
                     'DFCN(L=2,width=4)',
                     'DFCN(L=2,width=6)',
                     'DFCN(L=2,width=8)',
                     'DFCN(L=2,width=10)',
                     'DFCN(L=2,width=20)',
                     'DFCN(L=2,width=40)',
                     'DFCN(L=2,width=60)']

DFCN_width_num_of_parameters = [211, 429, 655, 889, 1131, 2461, 5721, 9781]

DFCN_width_RMSE = [0.05756155, 0.00802427, 0.00982634,
                  0.00892485, 0.0103979, 0.01053366, 0.01264698, 0.01133383]


# Data
eDCNN_maxpooling_networks = [
    "eDCNN-maxpooling(L=2, S=5, b=0.03)",
    "eDCNN-maxpooling(L=4, S=5, b=0.03)",
    "eDCNN-maxpooling(L=6, S=5, b=0.03)",
    "eDCNN-maxpooling(L=8, S=5, b=0.03)",
  ]
eDCNN_maxpooling_num_of_parameters = [41, 56, 71, 86,]
eDCNN_maxpooling_rmse_results = [0.01126485,0.0097693,0.00944106,0.00981856,]

eDCNN_zeropadding_networks = [
'eDCNN-zeropadding(L=2, S=5, b=0.03)',
'eDCNN-zeropadding(L=4, S=5, b=0.03)',
'eDCNN-zeropadding(L=6, S=5, b=0.03)',
'eDCNN-zeropadding(L=8, S=5, b=0.03)',
]

eDCNN_zeropadding_num_of_parameters = [121, 141, 161, 181 ]
eDCNN_zeropadding_rmse_results = [0.00746653, 0.00788369, 0.01055301, 0.01084637]

# Plot both DFCN and eDCNN results on the same figure
plt.figure(figsize=(10,6))
ax = plt.gca() # Get current axes

# Plot DFCN results
plt.plot(DFCN_width_num_of_parameters, DFCN_width_RMSE, color='b', linestyle='-', marker='o', label='DFCN(L=2) with different width')
plt.fill_between(DFCN_width_num_of_parameters, DFCN_width_RMSE, color='skyblue', alpha=0.3)
# Extract width from legend for text annotation
dfcn_widths = [int(legend.split('width=')[1].split(')')[0]) for legend in DFCN_width_legends]
for i, rmse in enumerate(DFCN_width_RMSE):
    plt.text(DFCN_width_num_of_parameters[i], rmse + 0.002, f'width={dfcn_widths[i]}', ha='center', va='bottom', fontsize=10, color='blue')


# Plot eDCNN-maxpooling results
plt.plot(eDCNN_maxpooling_num_of_parameters, eDCNN_maxpooling_rmse_results, color='r', linestyle='-', marker='o', label='eDCNN-maxpooling(s=5, bias=0.03) with different depth')
plt.fill_between(eDCNN_maxpooling_num_of_parameters, eDCNN_maxpooling_rmse_results, color='mistyrose', alpha=0.3)
# Extract depth (L) from network names for text annotation
edcnn_maxpooling_depths = [int(name.split('L=')[1].split(',')[0]) for name in eDCNN_maxpooling_networks]
for i, rmse in enumerate(eDCNN_maxpooling_rmse_results):
    plt.text(eDCNN_maxpooling_num_of_parameters[i], rmse + 0.002, f'L={edcnn_maxpooling_depths[i]}', ha='center', va='bottom', fontsize=10, color='red')

plt.axhline(y=0.01, color='blue', linestyle='--', alpha=0.8)

# Plot eDCNN-zeropadding results
plt.plot(eDCNN_zeropadding_num_of_parameters, eDCNN_zeropadding_rmse_results, color='g', linestyle='-', marker='o', label='eDCNN-zeropadding(s=5, bias=0.03) with different depth')
plt.fill_between(eDCNN_zeropadding_num_of_parameters, eDCNN_zeropadding_rmse_results, color='lightgreen', alpha=0.3) # Changed fill color for better distinction
# Extract depth (L) from network names for text annotation
edcnn_zeropadding_depths = [int(name.split('L=')[1].split(',')[0]) for name in eDCNN_zeropadding_networks]
for i, rmse in enumerate(eDCNN_zeropadding_rmse_results):
    plt.text(eDCNN_zeropadding_num_of_parameters[i], rmse + 0.002, f'L={edcnn_zeropadding_depths[i]}', ha='center', va='bottom', fontsize=10,color='g')

# Set x-axis to log scale
ax.set_xscale('log')

ax.set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200, 500, 1000, 5000, 10000]) # Adjusted ticks for better spacing in lower range
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # Use scalar formatter for ticks

ax.set_xlim([40, 10000]) # Adjusted limits to focus more on the relevant range

plt.ylim([0, 0.066])
plt.xlabel('Number of Parameters')
plt.ylabel('Test RMSE')
# plt.title('DFCN vs eDCNN Model Performance Comparison')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('./func2-results/func2-noisefree-RMSEvsParameters.png')
plt.savefig('./func2-results/func2-noisefree-RMSEvsParameters.pdf')

plt.show()


