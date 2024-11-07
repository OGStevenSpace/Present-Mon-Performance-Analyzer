import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import json


def from_data(array, shape=True):

    if shape:
        output = reshape(array)
    else:
        output = array

    main(output)


def reshape(array):
    out_list = []
    for col in array:
        temp_list = []
        for r_idx, row in enumerate(array[col]):
            if isinstance(row, dict):
                parsed_data = parse_array(row)
                temp_list.insert(r_idx, parsed_data)
            else:
                temp_list.insert(r_idx, row)
        out_list.append(temp_list)

    output = pd.DataFrame(out_list)

    output = output.rename(columns=output.loc[0, :]).drop(0).reset_index(drop=True)
    return output


def parse_array(value):

    df = pd.DataFrame(value.values(), index=list(value.keys())).transpose()
    return df


def dist_plot(gs, fig, data_list, step=0.05, y_tick=2):
    rows = len(data_list.keys())
    cols = 2
    top = rows * 0.05 - 0.05
    axes = [fig.add_subplot(gs[0, j]) for j in range(cols)]

    for i_k, k in enumerate(data_list.keys()):
        for i_d, dist in enumerate(data_list[k]):

            flag_vis_y = i_d == 0
            x = list(data_list[k][dist])
            pad = [top - 0.05 * i_k] * len(x)
            x = [sum(x) for x in zip(x, pad)]
            y = list(data_list[k][dist].keys())
            ticks_x = list(range(0, len(x)//y_tick, y_tick))
            ticks_y = [y for y in [step * n for n in range(rows)]]
            axes[i_d].fill_between(y, x, 0)
            axes[i_d].fill_between(y, pad, 0, color='white', ec='0.8')
            axes[i_d].get_xaxis().set_visible(True)
            axes[i_d].get_yaxis().set_visible(flag_vis_y)
            axes[i_d].set_xlim([0.0, 67])
            axes[i_d].set_ylim([0.0, top + 3 * step])
            axes[i_d].set_xticks(ticks_x)
            axes[i_d].set_yticks(ticks=ticks_y, labels=reversed(list(data_list.keys())))
            axes[i_d].tick_params(axis='both', which='major', labelsize=6)

    fig.tight_layout(pad=-0.5)
    fig.subplots_adjust(left=0.1, right=0.99)


def bar_dist_plot(gs, fig, data_list, color):

    key_list = data_list.keys()
    rows = len(key_list)
    ext_key_list = [item for item in key_list for _ in range(2)]
    dist_df = pd.DataFrame()
    ax = fig.add_subplot(gs[0, 2])

    p = -(rows / 4)
    hline_size = 0.05 + 3 ** p
    hline_size = -1 / hline_size
    hline_size += 21

    for key in key_list:
        lab_col = data_list[key].transpose()
        dist_df = pd.concat([dist_df, lab_col])

    for i, col in reversed(list(enumerate(dist_df.keys()))):
        if i > 0:
            dist_df.iloc[:, i] = dist_df.iloc[:, i].subtract(dist_df.iloc[:, i-1])

    dist_df = dist_df.reset_index()
    dist_df['fName'] = pd.DataFrame(ext_key_list)
    # dist_df.sort_values(by=[0.5, 'fName'],inplace=True)

    for i, file in enumerate(reversed(key_list)):

        frame_time_data = dist_df[(dist_df['fName'] == file) & (dist_df['index'] == 'FrameTime')].iloc[0, 1:-1]
        displayed_time_data = dist_df[(dist_df['fName'] == file) & (dist_df['index'] == 'DisplayedTime')].iloc[0, 1:-1]

        ax.hlines(y=i*2-0.5, xmin=0, xmax=67, colors='0.8', lw=1)

        # Plot FrameTime
        ax.barh(
            i * 2, frame_time_data, color=color,
            left=frame_time_data.cumsum() - frame_time_data,
            height=0.4
        )

        ax.vlines(x=0, ymin=i*2-0.40, ymax=i*2+0.40, colors='royalblue', lw=4)
        ax.vlines(x=frame_time_data[:5].sum(), ymin=i*2-0.25, ymax=i*2+0.25, colors='red')

        # Plot DisplayedTime
        ax.barh(
            i * 2 + 1, displayed_time_data, color=color,
            left=displayed_time_data.cumsum() - displayed_time_data,
            height=0.4
        )

        ax.vlines(x=0, ymin=i*2+0.60, ymax=i*2+1.40, colors='orange', lw=4)
        ax.vlines(x=displayed_time_data[:5].sum(), ymin=i*2+0.75, ymax=i*2+1.25, colors='red')

    # Customizing the chart
    ax.get_yaxis().set_visible(False)
    ax.set_ylim([-0.5, len(ext_key_list) + 3.5])
    ax.set_xlim([0.0, 67.00])
    ax.tick_params(axis='both', which='major', labelsize=6)
    return 0


def main(array):

    # Get the config json file.
    with open('config.json', "r") as file:
        config = json.load(file)

    fig = plt.figure(figsize=(10, 6))
    gs = grid_spec.GridSpec(1, 3, figure=fig)

    c_mean = array.iloc[3, :].div(array.iloc[2, :])
    array = array.transpose()
    array['c_mean'] = c_mean
    array = array.sort_values(by='c_mean').transpose()

    dist_plot(gs, fig, array.iloc[14])
    bar_dist_plot(gs, fig, array.iloc[8], color=config["frame_quantiles_colors"])

    plt.show()
    return 0
