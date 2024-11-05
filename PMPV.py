import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk


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
            axes[i_d].set_yticks(ticks=ticks_y, labels=list(data_list.keys()))
            axes[i_d].tick_params(axis='both', which='major', labelsize=6)

    fig.tight_layout(pad=-0.5)
    fig.subplots_adjust(left=0.1, right=0.99)


def bar_dist_plot(gs, fig, data_list):

    key_list = data_list.keys()
    ext_key_list = [item for item in key_list for _ in range(2)]
    dist_df = pd.DataFrame()
    ax = fig.add_subplot(gs[0, 2])
    color = [
        'white',
        'lightgrey',
        'lightsteelblue',
        'cornflowerblue',
        'green',
        'green',
        'cornflowerblue',
        'lightsteelblue',
        'lightgrey'
    ]
    for key in key_list:
        lab_col = data_list[key].transpose()
        dist_df = pd.concat([dist_df, lab_col])

    for i, col in reversed(list(enumerate(dist_df.keys()))):
        if i > 0:
            dist_df.iloc[:, i] = dist_df.iloc[:, i].subtract(dist_df.iloc[:, i-1])

    dist_df = dist_df.reset_index()
    dist_df['fName'] = pd.DataFrame(ext_key_list)

    for i, file in enumerate(key_list):
        frame_time_data = dist_df[(dist_df['fName'] == file) & (dist_df['index'] == 'FrameTime')].iloc[0, 1:-1]
        displayed_time_data = dist_df[(dist_df['fName'] == file) & (dist_df['index'] == 'DisplayedTime')].iloc[0, 1:-1]
        print(frame_time_data)
        print(displayed_time_data)
        # Plot FrameTime
        ax.barh(
            i * 2, frame_time_data, color=color,
            left=frame_time_data.cumsum() - frame_time_data,
            label=f'{file} FrameTime' if i == 0 else ""
        )

        # Plot DisplayedTime
        ax.barh(
            i * 2 + 1, displayed_time_data, color=color,
            left=displayed_time_data.cumsum() - displayed_time_data,
            label=f'{file} DisplayedTime' if i == 0 else ""
        )

    # Customizing the chart
    ax.set_yticks([i * 2 + 0.5 for i in range(len(key_list))])
    ax.set_yticklabels(key_list)
    ax.set_xlabel("Values")
    ax.set_xlim([0.0, 67])
    ax.set_title("FrameTime and DisplayedTime Percentiles by File")
    ax.legend(loc='upper right')

    return 0


def on_closing(root):
    root.quit()
    root.destroy()


def main(array):

    fig = plt.figure(figsize=(10, 6))
    gs = grid_spec.GridSpec(1, 3, figure=fig)

    dist_plot(gs, fig, array.iloc[14])
    bar_dist_plot(gs, fig, array.iloc[8])

    plt.show()

    '''root = tk.Tk()
    root.title('Plots')

    distribution = FigureCanvasTkAgg(fig, master=root)
    distribution.draw()

    toolbar = NavigationToolbar2Tk(distribution, root)
    toolbar.update()

    distribution.get_tk_widget().pack()

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()'''
    return 0
