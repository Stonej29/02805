import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime, timedelta
import contextily as ctx
from pyproj import Transformer

# Read the combined fire data
print("Loading fire detection data...")
df = pd.read_csv('combined_fire_detections.csv')

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter for California fires (approximate bounding box)
# California roughly: 32-42°N, -124 to -114°W
df = df[(df['latitude'] >= 32) & (df['latitude'] <= 42) &
        (df['longitude'] >= -125) & (df['longitude'] <= -114)]

print(f"Total fire detections in California: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Sort by datetime
df = df.sort_values('datetime')

# Group by day for the animation (showing daily fire activity)
df['date'] = df['datetime'].dt.date
daily_groups = df.groupby('date')

print(f"Number of days: {len(daily_groups)}")

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Set up the map boundaries
west, east = -125, -114
south, north = 32, 42

ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('California Wildfire Detections - January to March 2025', fontsize=14, fontweight='bold')

# Add basemap (black and white)
print("Adding background map...")
try:
    # Convert lat/lon to Web Mercator for contextily
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    west_merc, south_merc = transformer.transform(west, south)
    east_merc, north_merc = transformer.transform(east, north)

    # Add the basemap in grayscale
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                    attribution=False, alpha=0.7)
    print("Background map added successfully")
except Exception as e:
    print(f"Could not add basemap: {e}")
    print("Continuing with grid background...")
    ax.grid(True, alpha=0.3, linestyle='--')

# Custom colormap: yellow -> orange -> red -> dark red (based on intensity)
colors = ['#FFFF00', '#FFA500', '#FF4500', '#8B0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('fire', colors, N=n_bins)

# Initialize scatter plot
scatter = ax.scatter([], [], c=[], cmap=cmap, s=[], alpha=0.6, edgecolors='black', linewidth=0.5)

# Add colorbar for FRP (Fire Radiative Power)
cbar = plt.colorbar(scatter, ax=ax, label='Fire Radiative Power (MW)')

# Date text
date_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Stats text
stats_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Keep fires visible for N days (fade effect)
FADE_DAYS = 3
fire_history = []

def init():
    """Initialize animation"""
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_array(np.array([]))
    scatter.set_sizes(np.array([]))
    date_text.set_text('')
    stats_text.set_text('')
    return scatter, date_text, stats_text

def animate(frame):
    """Update function for each frame"""
    global fire_history

    dates = sorted(daily_groups.groups.keys())
    current_date = dates[frame]

    # Get current day's fires
    current_fires = daily_groups.get_group(current_date)

    # Add current fires to history with timestamp
    for _, fire in current_fires.iterrows():
        fire_history.append({
            'lon': fire['longitude'],
            'lat': fire['latitude'],
            'frp': fire['frp'],
            'date': current_date,
            'brightness': fire['brightness']
        })

    # Remove fires older than FADE_DAYS
    cutoff_date = current_date - timedelta(days=FADE_DAYS)
    fire_history = [f for f in fire_history if f['date'] >= cutoff_date]

    # Prepare data for plotting
    if fire_history:
        lons = [f['lon'] for f in fire_history]
        lats = [f['lat'] for f in fire_history]
        frps = [f['frp'] for f in fire_history]

        # Size based on FRP (min 20, max 500)
        sizes = [min(max(20, frp * 2), 500) for frp in frps]

        # Set positions, colors, and sizes
        scatter.set_offsets(np.c_[lons, lats])
        scatter.set_array(np.array(frps))
        scatter.set_sizes(np.array(sizes))

        # Set color limits
        scatter.set_clim(0, 100)  # FRP range for colormap
    else:
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        scatter.set_sizes(np.array([]))

    # Update date text
    date_text.set_text(f'Date: {current_date}')

    # Update stats
    num_fires_today = len(current_fires)
    num_fires_visible = len(fire_history)
    avg_frp = np.mean(frps) if frps else 0
    max_frp = max(frps) if frps else 0

    stats_text.set_text(
        f'New today: {num_fires_today}\n'
        f'Visible: {num_fires_visible}\n'
        f'Avg FRP: {avg_frp:.1f} MW\n'
        f'Max FRP: {max_frp:.1f} MW'
    )

    return scatter, date_text, stats_text

# Create animation
print("\nCreating animation...")
dates = sorted(daily_groups.groups.keys())
anim = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(dates), interval=200,
                              blit=True, repeat=True)

# Save as MP4 (requires ffmpeg) or GIF
output_file = 'california_wildfires_animation.gif'
print(f"Saving animation to {output_file}...")
print("This may take a few minutes...")

try:
    # Try to save as GIF
    anim.save(output_file, writer='pillow', fps=5, dpi=100)
    print(f"✓ Animation saved as: {output_file}")
except Exception as e:
    print(f"Error saving as GIF: {e}")
    try:
        # Try to save as MP4
        output_file = 'california_wildfires_animation.mp4'
        anim.save(output_file, writer='ffmpeg', fps=5, dpi=100)
        print(f"✓ Animation saved as: {output_file}")
    except Exception as e2:
        print(f"Error saving as MP4: {e2}")
        print("Showing animation instead (close window when done)...")
        plt.show()

print("\nAnimation details:")
print(f"  - Total frames: {len(dates)} days")
print(f"  - Fires fade after {FADE_DAYS} days")
print(f"  - Color indicates fire intensity (FRP)")
print(f"  - Size indicates fire intensity")
print(f"  - Yellow = low intensity, Red/Dark red = high intensity")
