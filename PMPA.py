import PMPV
import datetime
import os
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def low(array_low, l_bins):
    # Calculate the average of the lowest x% of the data.
    col_names = array_low.columns.values
    low_out = pd.DataFrame(0.0, columns=col_names, index=l_bins)

    for col in array_low:
        for bi in l_bins:
            q = array_low[col].quantile(1 - bi)
            try:
                low_out.loc[bi, col] = array_low.loc[array_low[col] >= q, col].mean()
            except TypeError:
                pass

    return low_out


def delta(array_delta):
    last_i = len(array_delta)-2
    l_arr = array_delta.loc[:last_i]
    u_arr = array_delta.loc[1:]

    return l_arr.sub(u_arr.values).abs()


def frequency(array_dist, step, bins, percentage=True):
    # Calculate the frequency of values within specified bins.
    # Define bin edges for faster processing
    bin_edges = np.arange(0, (bins + 1) * step, step)

    # Clip values at the maximum bin edge (overflow)
    array_dist_clipped = array_dist.clip(upper=bin_edges[-1])

    # Digitize the values in the DataFrame (binning)
    binned = np.digitize(array_dist_clipped.values, bin_edges, right=False) - 1

    # Initialize the output DataFrame
    dist_out = pd.DataFrame(0, index=bin_edges[:], columns=array_dist.columns)

    # Count occurrences in each bin for each column
    for col_idx, col_name in enumerate(array_dist.columns):
        counts = np.bincount(binned[:, col_idx], minlength=bins + 1)
        dist_out[col_name] = counts[:bins + 1]

    # Convert counts to percentages if needed
    if percentage:
        dist_out = dist_out.div(dist_out.sum(), axis=1)

    return dist_out


def summary_dict():

    return 0


def file_opener():
    # Select CSV file/s.
    root = tk.Tk()
    root.withdraw()

    # Set the path.
    file_paths = filedialog.askopenfilenames()

    # Count files.
    files_qty = len(file_paths)

    return file_paths, files_qty


def file_reader(file_path_str):
    # Read a CSV file.
    return pd.read_csv(file_path_str)


def output_path():
    # Define output path.
    opath = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        f"Output_{datetime.datetime.now(datetime.UTC).strftime('%d-%m-%y_T%H%M%S')}.csv"
    )

    return opath


# Main script execution starts here.
if __name__ == '__main__':

    # Get the config json file.
    with open('config.json', "r") as file:
        config = json.load(file)

    # Select files.
    file_range = file_opener()

    # Set output path.
    output_path = output_path()

    # Set output DataFrame.
    output_array = pd.DataFrame()

    # Process each selected file
    for idx, file in enumerate(file_range[0]):

        # Initialize data from the source file.
        data_collected = file_reader(file)

        # Create times_list variable to replace constant data_collected referencing.
        times_list = data_collected[["FrameTime", "DisplayedTime"]]

        # Data means are used in a few places so instead of recalculating them, here's the variable.
        data_means = data_collected[
            [
                "FrameTime",
                "DisplayedTime",
                "CPUUtilization",
                "CPUFrequency",
                "GPUUtilization",
                "GPUFrequency",
                "GPUMemorySizeUsed"
            ]
        ].mean()

        # Create output dictionary.
        summary_data = {
            "fileName": os.path.basename(file),
            "presentMode": data_collected["PresentMode"].iloc[-1],
            "api": data_collected["PresentRuntime"].iloc[-1],
            "record": len(data_collected.index),
            # The source "CPUStartTime" column is in seconds while all the calculations are in milliseconds.
            # Thence, CPUStartTime needs to be converted to milliseconds (sec x 1000).
            "tTime": data_collected["CPUStartTime"].iloc[-1]*1000,
            "mean": data_means.to_dict(),
            "waitTime": data_collected[["CPUWait", "GPUWait"]].sum().to_dict(),
            "utilStat":
                data_collected[["CPUUtilization", "GPUUtilization"]].quantile(config["util_quantiles"]).to_dict(),
            "lows": low(times_list, l_bins=config["low_n%"]).to_dict(),
            "percentiles": times_list.quantile(config["frame_quantiles"]).to_dict(),
            "mad": times_list.sub(data_means[["FrameTime", "DisplayedTime"]]).abs().mean().to_dict(),
            "std": times_list.std().to_dict(),
            "skew": times_list.skew().to_dict(),
            "kurt": times_list.kurt().to_dict(),
            "delta_dist":
                frequency(delta(times_list), config["delta_bins"]["step"], config["delta_bins"]["bins"]).to_dict(),
            "dist":
                frequency(times_list, config["dist_bins"]["step"], config["dist_bins"]["bins"]).to_dict()
        }

        # Append the summarized data to the output DataFrame.
        output_array = pd.concat([output_array, pd.DataFrame([summary_data])])
        # Display progress.
        progress = int((idx + 1) / file_range[1] * 100)
        print(f"\rProcessing Data ... Progress: {progress}%", end="")

    # Check if file exists.
    file_exists = os.path.isfile(output_path)

    # Write to the output file.
    output_array.to_csv(output_path, mode='a', index=False, header=not file_exists)

    # print(pd.DataFrame().from_records(pd.DataFrame().from_records(output_array["dist"])["FrameTime"]))
    PMPV.from_data(output_array)

    # Info when completed.
    print(f"\rProcessing complete. Output file created on the Desktop.", end="")
