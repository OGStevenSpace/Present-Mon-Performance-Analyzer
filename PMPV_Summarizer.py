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


def matrix_heatmap(data, cmap):
    fig = plt.figure()
    ax = sns.heatmap(data, annot=True, linewidth=0.5, cmap=cmap)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    ax.set_yticklabels(data.index)
    return ax


def main():
    file_paths, num_files = Main.open_files()
    std_df, mad_df = pd.DataFrame(), pd.DataFrame()
    for file_path in file_paths:
        data, status = load_json(file_path)
        column = os.path.splitext(os.path.basename(file_path))[0]
        std_df[column] = pd.DataFrame.from_dict(data["std"], orient='index')["DisplayedTime"]
        mad_df[column] = pd.DataFrame.from_dict(data["mad"], orient='index')["DisplayedTime"]
    std_plot = matrix_heatmap(std_df, 'viridis')
    mad_plot = matrix_heatmap(mad_df, 'magma')
    plt.show()



if __name__ == '__main__':
    main()
