import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'text.usetex':True,
})

noise_levels = ['$\\delta$=0.01',  '$\\delta$=0.1', '$\\delta$=0.3', '$\\delta$=0.5']

dfcn_rmse =  [0.01827326,  0.03782246, 0.094149977, 0.154162725]
eDCNN_zeropadding_rmse = [0.03017341, 0.04153486, 0.11129766, 0.14819959]
eDCNN_maxpooling_rmse =  [0.01291923, 0.0210867, 0.05128761, 0.10997002]


plt.figure(figsize=(8, 4.5))

x = np.arange(len(noise_levels))
width = 0.25

rects1 = plt.bar(x - width, dfcn_rmse, width, label='DFCN(L=2, width=2d+10), num. param.=451', color='#1f77b4', alpha=1.0)
rects2 = plt.bar(x, eDCNN_zeropadding_rmse, width, label='eDCNN-zeropadding(L=8, s=11, bias=0.01), num. param.=181', color='#2ca02c', alpha=1.0)
rects3 = plt.bar(x + width, eDCNN_maxpooling_rmse, width, label='eDCNN-maxpooling(L=8, s=11, bias=0.01), num. param.=128', color='#d62728', alpha=1.0)

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.xlabel('Noise levels', fontsize=12)
plt.ylabel('Test RMSE', fontsize=12)
plt.xticks(x, noise_levels, fontsize=12)
plt.legend(fontsize=10, loc='upper left')
plt.ylim(0, 0.24)

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

plt.savefig('./func1-results/func1-noisydata-performance-comparison.png')
plt.savefig('./func1-results/func1-noisydata-performance-comparison.pdf')

plt.show()
