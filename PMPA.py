import os
import numpy as np
import pandas as pd


def calculate_low_percentiles(array, percentiles):
    """Calculate the mean of the lowest percentiles in each column."""
    low_means = pd.DataFrame(index=percentiles, columns=array.columns)

    for col in array:
        for p in percentiles:
            threshold = array[col].quantile(1 - p)
            low_means.loc[p, col] = array[col][array[col] >= threshold].mean()

    return low_means


def calculate_deltas(array):
    """Calculate absolute difference between consecutive rows."""
    return array.iloc[:-1].sub(array.iloc[1:].values).abs()


def calculate_frequency(array, step, bins, as_percentage=True):
    """Calculate frequency distribution within specified bins."""
    bin_edges = np.arange(0, (bins + 1) * step, step)
    clipped_array = array.clip(upper=bin_edges[-1])
    binned_values = np.digitize(clipped_array.values, bin_edges) - 1

    frequency_counts = pd.DataFrame(0, index=bin_edges, columns=array.columns)
    for idx, col in enumerate(array.columns):
        counts = np.bincount(binned_values[:, idx], minlength=bins + 1)
        frequency_counts[col] = counts[:bins + 1]

    if as_percentage:
        frequency_counts = frequency_counts.div(frequency_counts.sum(), axis=1)

    return frequency_counts


def read_file(file_path):
    """Read CSV file into DataFrame."""
    return pd.read_csv(file_path)


def process_files(config, file_paths, num_files):
    """Process each selected file and summarize data according to config."""
    output_dict = {}

    for idx, file_path in enumerate(file_paths):
        data = read_file(file_path)
        time_data = data[config["times_cols"]]
        data_means = data[config["data_means"]].mean()

        output_dict[os.path.splitext(os.path.basename(file_path))[0]] = {
            "presentMode": data["PresentMode"].iloc[-1],
            "api": data["PresentRuntime"].iloc[-1],
            "record": len(data),
            "tTime": data["CPUStartTime"].iloc[-1] * 1000,
            "mean": data_means.to_dict(),
            "waitTime": data[["CPUWait", "GPUWait"]].sum().to_dict(),
            "utilStat": data[["CPUUtilization", "GPUUtilization"]].quantile(config["util_quantiles"]).to_dict(),
            "lows": calculate_low_percentiles(time_data, config["low_n%"]).to_dict(),
            "percentiles": time_data.quantile(list(map(float, config['frame_quantiles'].keys()))).to_dict(),
            "mad": time_data.sub(data_means[["FrameTime", "DisplayedTime"]]).abs().mean().to_dict(),
            "std": time_data.std().to_dict(),
            "skew": time_data.skew().to_dict(),
            "kurt": time_data.kurt().to_dict(),
            "delta_dist": calculate_frequency(
                calculate_deltas(time_data), config["delta_bins"]["step"], config["delta_bins"]["bins"]
            ).to_dict(),
            "dist": calculate_frequency(
                time_data, config["dist_bins"]["step"], config["dist_bins"]["bins"]
            ).to_dict()
        }

        progress = int((idx + 1) / num_files * 100)
        print(f"\rProcessing Data ... Progress: {progress}%", end="")

    return pd.DataFrame.from_dict(output_dict, orient='index')


def main(file_paths, num_files, config):

    output_df = process_files(config, file_paths, num_files)

    return output_df
