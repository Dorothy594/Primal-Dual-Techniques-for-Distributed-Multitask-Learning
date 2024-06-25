import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'ci_weight_11.xlsx'
df = pd.read_excel(file_path)

# Extract the columns as two separate lists
weights_col1 = df[0]
weights_col2 = df[1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(weights_col1, weights_col2, alpha=0.7, edgecolors='w', s=100)
plt.scatter([2, -2], [2, 2], color='red', marker='x', s=100)
plt.title('Scatter Plot of Node Weights')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(np.arange(-2, 2.1, 0.5))
plt.yticks(np.arange(1.5, 2.5 + 0.5, 0.5))

plt.show()
plt.close()
