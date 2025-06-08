import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 10,
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

# Data preparation
learning_rates = ['0.003', '0.001', '0.0006']
dfcn_d1_rmse = [ 0.056275799,  0.083842251,  0.098341231]
dfcn_2d10_rmse = [0.018490976,  0.025772519, 0.025071278 ]
edcnn_zeropadding_rmse = [ 0.05178661,0.02855303, 0.02559973] # eDCNN-zeropadding(L=8,S=11,b=0.01)
edcnn_rmse = [ 0.016177114, 0.013556986, 0.01272163]

# Set figure size
plt.figure(figsize=(10, 6))
# Set bar positions
x = np.arange(len(learning_rates))
width = 0.18  # Adjusted bar width for 4 bars
# Create bar chart dfcn_2d10_rmse
rects1 = plt.bar(x - 1.5*width, dfcn_d1_rmse, width, label='DFCN(L=2, width=d+1)', color='#4c96d7', alpha=0.8)
rects2 = plt.bar(x - 0.5*width, dfcn_2d10_rmse, width, label='DFCN(L=2, width=2d+10)', color='#1f77b4', alpha=1.0)
rects3 = plt.bar(x + 0.5*width, edcnn_zeropadding_rmse, width, label='eDCNN-zeropadding(L=8, s=11, bias=0.01)', color='#2ca02c', alpha=1.0) # Added new bar set
rects4 = plt.bar(x + 1.5*width, edcnn_rmse, width, label='eDCNN-maxpooling(L=8, s=11, bias=0.01)', color='#d62728', alpha=1.0)

# Add value labels with larger font size
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10, rotation=0)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3) # Add labels for the new bar set
add_labels(rects4)

# Add title and labels (in English)
# plt.title('Prediction Performance of Different Network Structures with Various Learning Rates', fontsize=14)
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Test RMSE', fontsize=12)
plt.xticks(x, learning_rates, fontsize=12)
plt.legend(fontsize=12, loc='upper left')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.3)
# Set y-axis range to [0, 0.02]
plt.ylim(0, 0.14)  # Adjusted y-axis range to accommodate higher values
# Adjust layout
plt.tight_layout()

plt.savefig('./func1-results/func1-noisefree-Performance-comparison.png')
plt.savefig('./func1-results/func1-noisefree-Performance-comparison.pdf')

# Show figure
plt.show()

