import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import json
import Main


def load_json(file_path) -> (dict, str):
    """Load data from the JSON file"""
    try:
        with open(file_path) as file:
            return json.load(file), None
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        return None, str(e)


def matrix_heatmap(data, cmap, vmin, vmax):
    fig = plt.figure()
    ax = sns.heatmap(data, annot=True, linewidth=0.5, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, fontsize=8, rotation_mode='anchor', ha='right')
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index)
    return ax


def main():
    file_paths, num_files = Main.open_files()
    dt_df, ft_df, mad_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for file_path in file_paths:
        data, status = load_json(file_path)
        column = os.path.splitext(os.path.basename(file_path))[0]
        temp_dt_df = pd.DataFrame.from_dict(data["mean"], orient='index')[["DisplayedTime"]]
        temp_ft_df = pd.DataFrame.from_dict(data["mean"], orient='index')[["FrameTime"]]
        temp_mad_df = pd.DataFrame.from_dict(data["mad"], orient='index')[["FrameTime"]]
        temp_dt_df.columns = [column]
        temp_ft_df.columns = [column]
        temp_mad_df.columns = [column]

        dt_df = pd.merge(dt_df, temp_dt_df, left_index=True, right_index=True, how='outer')
        ft_df = pd.merge(ft_df, temp_ft_df, left_index=True, right_index=True, how='outer')
        mad_df = pd.merge(mad_df, temp_mad_df, left_index=True, right_index=True, how='outer')

    try:
        dt_df.sort_values(by=['_NATIVE'], inplace=True)
        ft_df.sort_values(by=['_NATIVE'], inplace=True)
        mad_df.sort_values(by=['_NATIVE'], inplace=True)
    except Exception as e:
        print(e)

    dt_plot = matrix_heatmap(dt_df, 'turbo', vmin=8, vmax=30)
    ft_plot = matrix_heatmap(ft_df, 'turbo', vmin=8, vmax=30)
    mad_plot = matrix_heatmap(mad_df, 'viridis', vmin=0, vmax=4)
    plt.subplots_adjust(top=0.99, bottom=0.13, left=0.055, right=1)
    plt.show()


if __name__ == '__main__':
    main()
