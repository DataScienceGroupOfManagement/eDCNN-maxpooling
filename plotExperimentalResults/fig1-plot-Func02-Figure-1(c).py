import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 10,
    # 'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
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

# Data preparation
learning_rates = ['0.003', '0.001', '0.0006']
dfcn_2d10_rmse = [0.010224224, 0.012052028,0.011591077 ] # DFCN(L=2, width=2d+10)
dfcn_d1_rmse = [0.010634283, 0.011549765, 0.011071357 ] # DFCN(L=2, width=d+1)
cdcnn_rmse = [0.04419543,0.03427833, 0.03163435] # cdcnn(L=10,S=5,bias=0.03)
# cdcnnfc_rmse = [0.02518564,0.01960594, 0.01779538] # cdcnn-fc(L=5,s=3,bias=0.03)
cdcnnfc_rmse = [0.0286562, 0.01848916,  0.0167425] # cdcnn-fc(L=5,s=3,bias=0.01)
edcnn_zeropadding_rmse = [ 0.01879013, 0.01382816, 0.00788369] # eDCNN-zero-padding(L=4, S=5, bias=0.03)
edcnn_rmse = [0.0116907, 0.0102743, 0.0092074] # eDCNN-maxpooling(L=4, S=5, bias=0.06)
# edcnn_rmse = [0.01383947,  0.00998918, 0.00961645] # eDCNN-maxpooling(L=4, S=5, bias=0.03)

# Set figure size
plt.figure(figsize=(12, 7))

# Set bar positions
x = np.arange(len(learning_rates))
width = 0.12

# Create bar chart
rects1 = plt.bar(x - 2.5*width, dfcn_2d10_rmse, width, label='DFCN(L=2, width=2d+10)', color='#1f77b4', alpha=1.0)
rects2 = plt.bar(x - 1.5*width, dfcn_d1_rmse, width, label='DFCN(L=2, width=d+1)', color='#4c96d7', alpha=1.0)
rects3 = plt.bar(x - 0.5*width, cdcnn_rmse, width, label='cDCNN(L=10, s=5, bias=0.03)', color='#ff7f0e', alpha=1.0)
rects4 = plt.bar(x + 0.5*width, cdcnnfc_rmse, width, label='cDCNN-fc(L=5, s=3, bias=0.01)', color='#ECE70F', alpha=1.0)
rects5 = plt.bar(x + 1.5*width, edcnn_zeropadding_rmse, width, label='eDCNN-zeropadding(L=4, s=5, bias=0.03)', color='#2ca02c', alpha=1.0) # 新增条形
rects6 = plt.bar(x + 2.5*width, edcnn_rmse, width, label='eDCNN-maxpooling(L=4, s=5, bias=0.06)', color='#d62728', alpha=1.0) # 调整位置

# Add value labels with larger font size
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, rotation=0)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)
add_labels(rects5)
add_labels(rects6)

# Add title and labels (in English)
# plt.title('Prediction Performance of Different Network Structures with Various Learning Rates', fontsize=14)
plt.xlabel('Learning Rate', fontsize=14)
plt.ylabel('Test RMSE', fontsize=14)
plt.xticks(x, learning_rates, fontsize=12)
plt.legend(fontsize=14, loc='upper right')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Set y-axis range to [0, 0.05] to accommodate the new data
plt.ylim(0, 0.06)

# Adjust layout
plt.tight_layout()

plt.savefig('./func2-results/func2-noisefree-performance-comparison.png')
plt.savefig('./func2-results/func2-noisefree-performance-comparison.pdf')

# Show figure
plt.show()


