import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import re

# Output directory for when working on server
# OUTPUT_DIR = "/home/soft/snasri/scripts/Outputs/plots"

# Output directory for when working on local
OUTPUT_DIR = "/Users/samanasri/Desktop/Heliophysics/Data/Plots"

def plot_heatmap(filename):
    print(f"Reading {filename}...")

    # Read text file
    try:
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            skiprows=1,
            names=["ID", "RA", "DEC", "Brightness", "Time"],
            usecols=["ID", "RA", "DEC", "Brightness"]
        )
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    elong = np.sqrt(df["RA"]**2 + df["DEC"]**2)
    elong = np.clip(elong, 1e-6, None)
    df["Brightness"] = df["Brightness"] * (elong ** 2.3)
    df["Brightness"] = df["Brightness"] * (0.42 / 90.0)
    df["Brightness"] = df["Brightness"] - np.median(df["Brightness"])   # Subtract background levels

    # To determine vmin/vmax percentiles
    # print("Brightness min/max:", df["Brightness"].min(), df["Brightness"].max())
    # print("Brightness percentiles (5,50,95):", np.percentile(df["Brightness"], [5, 50, 95]))

    # Filter for visualization
    # We create a clipped version for plotting so stars don't wash out the image.
    # Typical K-Corona is 10-100 S10. Stars are >1000.
    # Set vmin=0 to ignore negative subtraction artifacts.
    # Set vmax=150 to highlight the solar wind structure.
    vmin = -50
    vmax = 50

    print(f"Plotting {len(df)} points...")
    print(f"Color scale clipped to: {vmin} - {vmax} S10 units (to hide stars)")

    plt.figure(figsize=(12, 8))

    # Use a scatter plot which acts like a heatmap for dense data points
    # where s=15 determines the size of each pixel point. Adjust if gaps appear.
    sc = plt.scatter(
        df["RA"],
        df["DEC"],
        c=df["Brightness"],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        s=4,
        marker="s"
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Brightness (S10 Units)")

    plt.title(f"PUNCH Data Visualization: {filename}")
    plt.xlabel("Right Ascension (RA) [deg]")
    plt.ylabel("Declination (DEC) [deg]")

    # Astronomy plots usually flip RA so East is to the left,
    # but we will keep it standard for now. Uncomment next line to flip:
    # plt.gca().invert_xaxis()

    plt.grid(True, linestyle="--", alpha=0.5)

    # --- SAVE FIGURE INSTEAD OF SHOWING ---
    m = re.search(r"PUNCH_(\d{8})", filename)
    date_tag = m.group(1) if m else "UNKNOWN_DATE"

    day_dir = os.path.join(OUTPUT_DIR, f"PUNCHPLOTS_{date_tag}")
    os.makedirs(day_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filename))[0]
    output_png = os.path.join(day_dir, base + ".jpg")

    plt.savefig(output_png, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()

    print(f"Saved plot to {output_png}")

if __name__ == "__main__":
    # Allow running from command line: python plot_punch.py data.txt
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "punch_output.txt"  # Default filename

    plot_heatmap(input_file)