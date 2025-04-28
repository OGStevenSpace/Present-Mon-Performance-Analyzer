import Main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec


class Figure:
    def __init__(self, title=None, x=16, y=9, nrows=1, ncols=1, height_ratios=None, width_ratios=None,
                 left=0.10, right=0.98, top=0.90, bottom=0.10, wspace=0.05, hspace=0.1):
        self.fig = (plt.figure(figsize=(16, 9)))
        self.fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
        self.fig.canvas.manager.set_window_title(title=title)
        self.gs = grid_spec.GridSpec(nrows=nrows, ncols=ncols, figure=self.fig,
                                     height_ratios=height_ratios, width_ratios=width_ratios)
        self.axis = [self.fig.add_subplot(self.gs[r, c]) for r in range(nrows) for c in range(ncols)]

    def set_axis_x(self, icol, ticks=None, label_size=0, show_axis=True, top=None):
        if ticks is None:
            ticks = [0, 1]
        (ax := self.axis[icol]).get_xaxis().set_visible(show_axis)
        ax.set_xlim([min(ticks), top if top else max(ticks)])
        ax.set_xticks(ticks)
        ax.tick_params(axis='x', labelsize=label_size)

    def set_axis_y(self, icol, ticks=None, data_keys=None, label_size=0, show_axis=True, bot_offset=0, top_offset=0):
        if ticks is None:
            ticks = [0, 1]
        (ax := self.axis[icol]).get_yaxis().set_visible(show_axis)
        lower = min(ticks) - bot_offset
        upper = max(ticks) + top_offset
        if lower == upper:
            eps = np.finfo(float).eps  # Smallest positive float delta
            upper += eps
        ax.set_ylim([lower, upper])
        ax.set_yticks(ticks, data_keys if show_axis else [])
        ax.tick_params(axis='y', labelsize=label_size)

    def set_axis_title(self, icol, title):
        self.axis[icol].set_title(title)

    def set_vlines(self, icol, *lines):
        ax = self.axis[icol]
        if not lines:
            print("set_vlines warning: Missing 'lines' arg.")
        else:
            for i, line in enumerate(lines):
                try:
                    color, x_line, linestyle = line
                    ax.vlines(x_line, ymin=-0.5, ymax=len(ax.get_yticks()) - 0.5,
                              colors=color, linewidth=0.5, linestyle=linestyle)
                except ValueError as e:
                    print(f"{e}: The 'lines' arg tuple [i={i}] does not have 3 correct components: "
                          f"(color[str], x[num], linestyle[str]).")

    def set_legend(self, icol, handles, labels, bbox_to_anchor):
        self.axis[icol].legend(
            handles=handles,
            labels=labels,
            ncol=len(labels),
            handletextpad=0.5,
            handlelength=1.0,
            columnspacing=-0.5,
            fontsize=7,
            loc='lower center',
            bbox_to_anchor=bbox_to_anchor
        )


def main():
    file_path, file_count = Main.open_files()
    if not file_count:
        "no files selected. Closing"
        return 0

    fig_height = int(max(6.0, file_count * 0.5))
    raw_plot = Figure(title='Raw Performance Log', y=fig_height, nrows=file_count)
    for i, file in enumerate(file_path):
        with open(file) as log:
            print(f"log {file}")
            data = pd.read_csv(log)
            raw_plot.axis[i].plot(data["CPUStartTime"], data["DisplayedTime"], linewidth=0.5)
            raw_plot.axis[i].set_ylim(0, 66)
            raw_plot.axis[i].set_yticks([16.5, 33, 66])
            raw_plot.axis[i].tick_params(axis='both', which='major', labelsize=6)
            raw_plot.axis[i].grid(visible=True, axis="y")

    raw_plot.fig.subplots_adjust(hspace=0)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
