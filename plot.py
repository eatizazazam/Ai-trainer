import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("landmarks_kv7.csv")

# Extract one frame
frame = df.iloc[0]  # first frame

# Convert landmarks to numpy array
points = np.array([[frame[f'x_{i}'], frame[f'y_{i}']] for i in range(33)])

# Plot the pose skeleton (2D)
plt.scatter(points[:, 0], -points[:, 1])  # inverted y-axis for visualization
plt.title("Pose Landmarks (Frame 0)")
plt.show()
