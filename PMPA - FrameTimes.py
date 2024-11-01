import datetime
import os
import csv
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def array_manipulator(times_list, GPU_Time_list, times_list_sorted, first_time, data_size):
    """
    This function takes lists of times and calculates various metrics.

    Columns:
    0 = running sum of times_list
    1 = absolute difference between consecutive times_list values
    2 = times_list_sorted values
    3 = running sum of times_list_sorted
    4 = difference between times_list and GPU_Time_list
    5 = absolute difference of column 4
    """

    data_summary = [[0] * 6 for _ in range(data_size)]

    data_summary[0] = [
        first_time,
        0,
        times_list_sorted[0],
        times_list_sorted[0],
        times_list[0] - GPU_Time_list[0],
        abs(0)
    ]

    for row in range(1, data_size):
        data_summary[row][0] = times_list[row] + data_summary[row - 1][0]
        data_summary[row][1] = abs(times_list[row] - times_list[row - 1])
        data_summary[row][2] = times_list_sorted[row]
        data_summary[row][3] = times_list_sorted[row] + data_summary[row - 1][3]
        data_summary[row][4] = times_list[row] - GPU_Time_list[row]
        data_summary[row][5] = abs(data_summary[row][4])

    return data_summary


def low_ranges(extended_times_list, data_size, low_bins_list):
    """
    Calculate the average of the lowest x% of the data.
    """
    bins_qty = len(low_bins_list)
    low_result = [0] * bins_qty
    for bin_val in range(0, bins_qty):
        index = int(round((data_size - 1) * low_bins_list[bin_val], 0))
        low_result[bin_val] = extended_times_list[index][3] / (index + 2)
    return low_result


def percentiles(extended_times_list, data_size, perc_bins_list):
    """
    Calculate exact percentiles from the sorted data.
    """
    bins_qty = len(perc_bins_list)
    perc_result = [0] * bins_qty
    for bin_val in range(bins_qty):
        index = int(round((data_size - 1) * perc_bins_list[bin_val], 0))
        perc_result[bin_val] = extended_times_list[index][2]
    return perc_result


def frequency(extended_times_list, freq_bins_list, col, percentage):
    """
    Calculate the frequency of values within specified bins.
    """
    bins_list_len = len(freq_bins_list)
    frequency_arr = [0.0] * (bins_list_len + 1)
    value = (1 / len(extended_times_list)) if percentage else 1.0

    for row in extended_times_list:
        for index, freq_bin in enumerate(freq_bins_list):
            if row[col] <= freq_bin:
                frequency_arr[index] += value
                break
        else:
            frequency_arr[bins_list_len] += value

    return frequency_arr


def data_stats(extended_times_list):
    """
    Calculate various statistical metrics of the given data.
    """
    data_size = len(extended_times_list)
    average = extended_times_list[data_size - 1][0] / data_size

    ave_dev, var, var_temp, kurt, skew = [0] * 5
    step_a, step_b = int(data_size * 0.001), int(data_size * 0.0025)
    step_c, step_d = int(data_size * 0.9975), int(data_size * 0.999)
    data_size_inner = step_c - step_b + 1

    for row in range(step_a, step_b):
        diff = extended_times_list[row][2] - average
        ave_dev += abs(diff)
        var += diff ** 2

    for row in range(step_b + 1, step_c):
        diff = extended_times_list[row][2] - average
        ave_dev += abs(diff)
        var_temp += diff ** 2
        skew += diff ** 3
        kurt += diff ** 4

    for row in range(step_c + 1, step_d):
        diff = extended_times_list[row][2] - average
        ave_dev += abs(diff)
        var += diff ** 2

    ave_dev /= data_size
    var = (var + var_temp) / (data_size - 1)
    var_temp /= data_size_inner
    std_dev = var ** 0.5
    skew /= data_size_inner
    kurt /= data_size_inner
    skew /= var_temp ** 1.5
    kurt /= var_temp ** 2
    kurt -= 3
    cv = std_dev / average

    return [average, ave_dev, std_dev, skew, kurt, cv]


def file_opener(file_path_str):
    """
    Open and read a CSV file.
    """
    return pd.read_csv(file_path_str)


def data_summarizer(times_list, CPU_list, GPU_list, Runtime, PresentMode, percentiles_bins_list,
                    low_bins_list, delta_bins_list, distribution_bins_list, file_name, CPU_Wait,
                    GPU_Wait, GPU_Time, VRAM_used, GPU_Clock, CPU_Clock):
    """
    Summarize data by calculating various statistics and metrics.
    """
    data_size = len(times_list)

    extended_times = array_manipulator(
        times_list,
        GPU_Time,
        times_list.sort_values(ignore_index=True, ascending=False),
        times_list.iloc[0],
        data_size
    )

    CPU_Wait_Sum = CPU_Wait.sum()
    GPU_Wait_Sum = GPU_Wait.sum()
    CGt_diff = sum(extended_times[:][4]) / data_size
    CGt_diff_abs = sum(extended_times[:][5]) / data_size
    total_time = extended_times[-1][0]
    VRAM_Avg = VRAM_used.mean()
    CPU_Clock_Avg = CPU_Clock.mean()
    GPU_Clock_Avg = GPU_Clock.mean()

    stats_times = data_stats(extended_times)
    low_times = low_ranges(extended_times, data_size, low_bins_list)
    percentiles_times = percentiles(extended_times, data_size, percentiles_bins_list)
    delta_times = frequency(extended_times, delta_bins_list, 1, True)
    distribution_times = frequency(extended_times, distribution_bins_list, 2, True)

    CPU_avg, CPU_min, CPU_med, CPU_max = "N/A", "N/A", "N/A", "N/A"
    GPU_avg, GPU_min, GPU_med, GPU_max = "N/A", "N/A", "N/A", "N/A"

    try:
        CPU_avg = CPU_list.mean()
        CPU_sorted = CPU_list.sort_values(ignore_index=True)
        CPU_min = CPU_sorted.iloc[0]
        CPU_med = CPU_sorted.iloc[data_size // 2 - 1]
        CPU_max = CPU_sorted.iloc[-1]
    except:
        pass

    try:
        GPU_avg = GPU_list.mean()
        GPU_sorted = GPU_list.sort_values(ignore_index=True)
        GPU_min = GPU_sorted.iloc[0]
        GPU_med = GPU_sorted.iloc[data_size // 2 - 1]
        GPU_max = GPU_sorted.iloc[-1]
    except:
        pass

    output_arr = [
        os.path.basename(file_name), PresentMode.iloc[0], Runtime.iloc[0], data_size, total_time,
        CPU_avg, CPU_med, CPU_min, CPU_max, CPU_Wait_Sum, CPU_Clock_Avg,
        GPU_avg, GPU_med, GPU_min, GPU_max, GPU_Wait_Sum, GPU_Clock_Avg,
        CGt_diff, CGt_diff_abs, VRAM_Avg, *stats_times, *low_times, *percentiles_times,
        *delta_times, *distribution_times
    ]

    return output_arr


# Main script execution starts here
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames()
    files_qty = len(file_paths)

    output_path = os.path.join(os.path.expanduser('~'), 'Documents', f"Output_{datetime.datetime.now(datetime.UTC).strftime('%d-%m-%y_T%H%M%S')}.csv")

    percentile_bins = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]
    low_bins = [0.001, 0.01]
    delta_bins = [2, 4, 8, 12]
    distribution_bins = [x * 0.5 for x in range(134)]

    header = [
        "FileName", "Present Mode", "API", "Records", "Total Time",
        "CPU avg", "CPU median", "CPU min", "CPU max", "CPU wait sum", "CPU Clock avg",
        "GPU avg", "GPU med", "GPU min", "GPU max", "GPU wait sum", "GPU Clock avg",
        "CPUt-GPUt", "CPUt-GPUt abs", "VRAM avg", "Average", "Ave_dev", "Std_dev", "Skew", "Kurt", "CV"
    ] + low_bins + percentile_bins + delta_bins + [">"] + distribution_bins + [">"]

    # Create the output file and write the header
    pd.DataFrame([header]).to_csv(output_path, mode='w', index=False, header=False)

    # Process each selected file
    for idx, file in enumerate(file_paths):
        data_collected = file_opener(file)

        # Summarize the data
        summarized_presents = data_summarizer(
            data_collected["FrameTime"],
            data_collected["CPUUtilization"],
            data_collected["GPUUtilization"],
            data_collected["PresentRuntime"],
            data_collected["PresentMode"],
            percentile_bins,
            low_bins,
            delta_bins,
            distribution_bins,
            file,
            data_collected["CPUWait"],
            data_collected["GPUWait"],
            data_collected["GPUTime"],
            data_collected["GPUMemorySizeUsed"],
            data_collected["GPUFrequency"],
            data_collected["CPUFrequency"]
        )

        # Append the summarized data to the output file
        pd.DataFrame([summarized_presents]).to_csv(output_path, mode='a', index=False, header=False)

        # Display progress
        progress = int((idx + 1) / files_qty * 100)
        print(f"\rProgress: {progress}% - Processing file: {os.path.basename(file)}", end="")

    print("\nProcessing complete. Output file created on the Desktop.")
