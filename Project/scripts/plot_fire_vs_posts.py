import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load fire detection data
print("Loading fire detection data...")
fire_data = pd.read_csv('../data/nasa/combined_fire_detections.csv')
fire_data['datetime'] = pd.to_datetime(fire_data['datetime'], utc=True)
fire_data['hour'] = fire_data['datetime'].dt.floor('h')

# Aggregate fire intensity by hour (using FRP - Fire Radiative Power as intensity measure)
fire_by_hour = fire_data.groupby('hour').agg({
    'frp': 'sum',  # Total fire radiative power
    'brightness': 'count'  # Number of detections
}).reset_index()
fire_by_hour.columns = ['hour', 'total_frp', 'detection_count']

# Load Bluesky posts data
print("Loading Bluesky posts data...")
posts = pd.read_json('../data/bluesky/ca_fire_20250101_20250207.jsonl', lines=True)
posts['indexed_at'] = pd.to_datetime(posts['indexed_at'], utc=True)
posts['hour'] = posts['indexed_at'].dt.floor('h')

# Aggregate posts by hour
posts_by_hour = posts.groupby('hour').size().reset_index(name='post_count')

# Merge the two datasets on hour
print("Merging datasets...")
merged = pd.merge(fire_by_hour, posts_by_hour, on='hour', how='outer').fillna(0)
merged = merged.sort_values('hour')

# Create the visualization
print("Creating visualization...")
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot fire intensity (FRP) on left y-axis
color1 = 'firebrick'
ax1.set_xlabel('Time (Hour)', fontsize=12)
ax1.set_ylabel('Fire Radiative Power (MW)', color=color1, fontsize=12)
line1 = ax1.plot(merged['hour'], merged['total_frp'], color=color1, linewidth=2,
                 label='Fire Intensity (FRP)', marker='o', markersize=4)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, linestyle='--')

# Create second y-axis for Bluesky posts
ax2 = ax1.twinx()
color2 = 'steelblue'
ax2.set_ylabel('Number of Bluesky Posts', color=color2, fontsize=12)
line2 = ax2.plot(merged['hour'], merged['post_count'], color=color2, linewidth=2,
                 label='Bluesky Posts', marker='s', markersize=4)
ax2.tick_params(axis='y', labelcolor=color2)

# Format x-axis to show dates nicely
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add title
plt.title('Fire Intensity vs. Bluesky Posts Over Time (Hourly)', fontsize=14, fontweight='bold', pad=20)

# Create combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)

# Tight layout
fig.tight_layout()

# Save the figure
output_file = '../figures/fire_vs_posts_hourly.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

# Also display summary statistics
print("\n=== Summary Statistics ===")
print(f"Time range: {merged['hour'].min()} to {merged['hour'].max()}")
print(f"Total fire detections: {fire_by_hour['detection_count'].sum():,.0f}")
print(f"Total FRP: {fire_by_hour['total_frp'].sum():,.2f} MW")
print(f"Total Bluesky posts: {posts_by_hour['post_count'].sum():,.0f}")
print(f"Peak fire hour: {merged.loc[merged['total_frp'].idxmax(), 'hour']}")
print(f"Peak posts hour: {merged.loc[merged['post_count'].idxmax(), 'hour']}")

# Show the plot
plt.show()

print("\nDone!")
