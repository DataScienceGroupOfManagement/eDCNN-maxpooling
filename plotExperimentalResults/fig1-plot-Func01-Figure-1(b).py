import matplotlib.pyplot as plt
import numpy as np

DFCN_legends = ['DFCN(L=2,width=5)',
                'DFCN(L=2,width=7)',
                'DFCN(L=2,width=9)',
                'DFCN(L=2,width=11)',
                'DFCN(L=2,width=13)',
                'DFCN(L=2,width=15)',
                'DFCN(L=2,width=17)',
                'DFCN(L=2,width=18)']
num_of_parameters = [61, 99, 145, 199, 261, 331, 409, 451]
rmse_values = [0.056275799, 0.069419906, 0.026323774, 0.03133125, 0.021104716, 0.022525422, 0.020717381, 0.018490976]

eDCNN_depth_legends = ['eDCNN(L=4,S=11)',
                       'eDCNN(L=6,S=11)',
                       'eDCNN(L=8,S=11)',
                       'eDCNN(L=10,S=11)',
                       'eDCNN(L=12,S=11)',
                       'eDCNN(L=14,S=11,)',
                       'eDCNN(L=16,S=11)',]

eDCNN_depth_num_of_parameters = [65, 97, 128, 160, 191, 223, 254]
eDCNN_depth_RMSE = [0.03408081, 0.03443208, 0.01272163,
0.016430818, 0.01306256, 0.01998702, 0.01358106]

eDCNN_zeropadding_depth_legends = ['eDCNN(L=4,S=11)', 
                      'eDCNN(L=6,S=11)', 
                      'eDCNN(L=8,S=11)', 
                      'eDCNN(L=10,S=11)', 
                      'eDCNN(L=12,S=11)', 
                      'eDCNN(L=14,S=11)', 
                      'eDCNN(L=16,S=11)'] 

eDCNN_zeropadding_depth_num_of_parameters = [93, 137, 181, 225, 269, 313, 357] 

eDCNN_zeropadding_depth_RMSE = [0.02419549, 0.02938098, 0.02559973, 
                   0.08211731, 0.02674585, 0.07799264, 0.0289004]


plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
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

# Plot both DFCN and eDCNN results on the same figure
plt.figure(figsize=(10,6))

# Plot DFCN results
plt.plot(num_of_parameters, rmse_values, color='#1f77b4', linestyle='-', marker='o', label='DFCN(L=2) with different width')
plt.fill_between(num_of_parameters, rmse_values, color='#1f77b4', alpha=0.3)
for i, txt in enumerate(DFCN_legends):
    # plt.text(num_of_parameters[i], rmse_values[i]+0.002, f'{rmse_values[i]:.4f} with width={5+i*2}', ha='center', va='bottom', fontsize=8)
    plt.text(num_of_parameters[i], rmse_values[i]+0.002, f'{rmse_values[i]:.4f},width={5+i*2}', ha='center', va='bottom', fontsize=10, color='#1f77b4')
    # plt.text(num_of_parameters[i], rmse_values[i]+0.002, txt, ha='center', va='bottom', fontsize=8)

# Add horizontal line at y=0.018490976
plt.axhline(y=0.02, color='blue', linestyle='--', alpha=0.8)

# Plot eDCNN-zeropadding results
plt.plot(eDCNN_zeropadding_depth_num_of_parameters, eDCNN_zeropadding_depth_RMSE, color='#2ca02c', linestyle='-', marker='o', label='eDCNN-zeropadding(s=11, bias=0.01) with different depth')
plt.fill_between(eDCNN_zeropadding_depth_num_of_parameters, eDCNN_zeropadding_depth_RMSE, color='#2ca02c', alpha=0.3)
for i, txt in enumerate(eDCNN_zeropadding_depth_legends):
    plt.text(eDCNN_zeropadding_depth_num_of_parameters[i], eDCNN_zeropadding_depth_RMSE[i]+0.002, f'{eDCNN_zeropadding_depth_RMSE[i]:.4f},L={4+i*2}', ha='center', va='bottom', fontsize=10,color='#2ca02c')
    # plt.text(eDCNN_zeropadding_depth_num_of_parameters[i], eDCNN_zeropadding_depth_RMSE[i]+0.002, txt, ha='center', va='bottom', fontsize=8)


# Plot eDCNN-maxpooling results
plt.plot(eDCNN_depth_num_of_parameters, eDCNN_depth_RMSE, color='#d62728', linestyle='-', marker='o', label='eDCNN-maxpooling(s=11, bias=0.01) with different depth')
plt.fill_between(eDCNN_depth_num_of_parameters, eDCNN_depth_RMSE, color='#d62728', alpha=0.3)
for i, txt in enumerate(eDCNN_depth_legends):
    plt.text(eDCNN_depth_num_of_parameters[i], eDCNN_depth_RMSE[i]+0.002, f'{eDCNN_depth_RMSE[i]:.4f},L={4+i*2}', ha='center', va='bottom', fontsize=10, color='#d62728')
    # plt.text(eDCNN_depth_num_of_parameters[i], eDCNN_depth_RMSE[i]+0.002, txt, ha='center', va='bottom', fontsize=8)


plt.xlim([30, 480])
plt.ylim([0, 0.11])
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel('Test RMSE', fontsize=12)
# plt.title('DFCN vs eDCNN Model Performance Comparison')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()

plt.savefig('./func1-results/func1-noisefree-RMSEvsParameters.png')
plt.savefig('./func1-results/func1-noisefree-RMSEvsParameters.pdf')

plt.show()
 
 