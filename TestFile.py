import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data structure based on your screenshot
data = {
    'fName': [
        'File1', 'File1', 'File2', 'File2',
        'File3', 'File3', 'File4', 'File4',
        'File5', 'File5', 'File6', 'File6'
    ],
    'label': [
        'FrameTime', 'DisplayedTime', 'FrameTime', 'DisplayedTime',
        'FrameTime', 'DisplayedTime', 'FrameTime', 'DisplayedTime',
        'FrameTime', 'DisplayedTime', 'FrameTime', 'DisplayedTime'
    ],
    0: [1.3, 16.1, 10.2, 16.1, 2.1, 16.3, 1.2, 16.2, 9.2, 16.0, 1.0, 16.2],
    0.01: [6.2, 16.6, 27.0, 16.6, 6.9, 16.5, 6.9, 16.5, 25.7, 16.6, 6.9, 16.6],
    # ... (other percentile columns as shown in your image)
    1: [911.9, 984.4, 117.7, 984.4, 1674.4, 1701.7, 1658.3, 1534.9, 224.5, 183.5, 1477.2, 1901.8]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Get unique files
files = df['fName'].unique()

# Define colors for each percentile column
colors = plt.cm.viridis([0, 0.5, 1])
print(np.linspace(0, 1, len(df.columns) - 2))

# Create the plot
fig, ax = plt.subplots(figsize=(10, len(files) * 1.2))

# Plot each file's FrameTime and DisplayedTime as stacked bars
for i, file in enumerate(files):
    frame_time_data = df[(df['fName'] == file) & (df['label'] == 'FrameTime')].iloc[0, 2:]
    displayed_time_data = df[(df['fName'] == file) & (df['label'] == 'DisplayedTime')].iloc[0, 2:]

    # Plot FrameTime
    ax.barh(
        i * 2, frame_time_data, color=colors, edgecolor='black',
        left=frame_time_data.cumsum() - frame_time_data,
        label=f'{file} FrameTime' if i == 0 else ""
    )

    # Plot DisplayedTime
    ax.barh(
        i * 2 + 1, displayed_time_data, color=colors, edgecolor='black',
        left=displayed_time_data.cumsum() - displayed_time_data,
        label=f'{file} DisplayedTime' if i == 0 else ""
    )

# Customizing the chart
ax.set_yticks([i * 2 + 0.5 for i in range(len(files))])
ax.set_yticklabels(files)
ax.set_xlabel("Values")
ax.set_title("FrameTime and DisplayedTime Percentiles by File")
ax.legend(loc='upper right')

plt.show()