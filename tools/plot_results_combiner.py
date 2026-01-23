import argparse
import os

import csv
import matplotlib.pyplot as plt
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser(description='Combine results from different runs')
    parser.add_argument('-ids', '--ids', nargs="*", type=str, help='ID of all the runs, seperated by commas')
    parser.add_argument('--file', default="eval", type=str, help='File to combine from each run')
    parser.add_argument('--input_dir', default="../plots/", nargs="*", type=str, help='Directory containing the results')
    parser.add_argument('--output_dir', default="../combined_plots/", type=str, help='Directory to save the combined results')
    parser.add_argument("--min", action="store_true", help="Whether to take the minimum value or not")
    parser.add_argument("--names", nargs="*", type=str,  default=None, help="Names of the runs (for the legend)")
    parser.add_argument("--use-min", action="store_true", help="Whether to use the minimum value or not")
    parser.add_argument("--combine", nargs="*", type=int, default=None, help="Combine the runs")
    parser.add_argument("--combine_names", nargs="*", type=str, default=None, help="Combine the runs, with name")
    parser.add_argument("--remove_plots", type=int, default=1, help="Remove plots")
    parser.add_argument("--ymin", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--ymax", type=float, default=None, help="Maximum y-axis value")
    parser.add_argument("--figsize", nargs=2, type=float, default=None, help="Figure size: width height")
    parser.add_argument("--save_name", type=str, default=None, help="Custom output filename (without extension)")

    return parser.parse_args()


def getCsv(input_dir, ids, file):
    csv_lis = []
    for id in ids:
        file_now = os.path.join(input_dir, f"train_{id}/{file}.csv")
        if not os.path.exists(file_now):
            raise FileNotFoundError(f"Expected CSV not found: {file_now}")
        with open(file_now, 'r') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        if len(rows) == 0:
            raise ValueError(f"Empty CSV: {file_now}")

        # Detect header row (non-numeric first row)
        header_row = False
        try:
            [float(x) for x in rows[0]]
        except Exception:
            header_row = True

        # If header exists and first column is 'Epoch', we'll drop that column
        drop_first_col = False
        if header_row:
            first = rows[0][0].strip().lower()
            if first.startswith('epoch'):
                drop_first_col = True

        # Filter and convert numeric rows
        numeric_rows = []
        for r in rows:
            # skip empty rows
            if len(r) == 0:
                continue
            try:
                if drop_first_col:
                    nums = [float(x) for x in r[1:]]
                else:
                    nums = [float(x) for x in r]
            except ValueError:
                # skip rows that contain non-numeric entries
                continue
            numeric_rows.append(nums)

        if len(numeric_rows) == 0:
            raise ValueError(f"No numeric rows found in CSV: {file_now}")

        arr = np.array(numeric_rows, dtype=np.float64)

        # If no header but first column looks like epoch indices (0..N-1), drop it
        if not drop_first_col and arr.shape[1] >= 2:
            first_col = arr[:, 0]
            # check if first column equals integers 0..N-1
            if np.allclose(first_col, np.arange(arr.shape[0])):
                arr = arr[:, 1:]

        # If the CSV is epochs x metrics (rows > cols), transpose to components x epochs
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T

        csv_lis.append(arr)
    return csv_lis


def cropLis(csv_lis):
    # csv_lis is a list of 2D numpy arrays (rows x cols). We want to crop columns (epochs)
    # to the minimum number of columns available across the provided arrays.
    min_cols = min([x.shape[1] for x in csv_lis])
    # Also ensure each array has at least one row
    for i in range(len(csv_lis)):
        if csv_lis[i].ndim != 2 or csv_lis[i].shape[0] == 0:
            raise ValueError("Each CSV must contain at least one numeric row")
        csv_lis[i] = csv_lis[i][:, :min_cols]
    # Return a 3D numpy array: (num_files, rows, cols)
    return np.array(csv_lis, dtype=np.float64)

# csv_lis: numpy of csv files, which are lists of components
def formatScores(csv_lis):
    error = []
    mean = []

    for i in range(len(csv_lis[0])):
        error.append([])
        mean.append([])
    
    for i in range(len(csv_lis[0])):
        for j in range(len(csv_lis[0][0])):
            error[i].append(np.std(csv_lis[:, i, j]))
            mean[i].append(np.mean(csv_lis[:, i, j]))
    return np.array(error), np.array(mean)


# Plot graph with error range shaded
def plot(error, mean, names, out, types = ["Test", "Train"], title="", remove_plots = 1, y_lim=None):
    plt.rcParams["font.size"] = 15
    colors = ['#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf']

    pair_counter = 0
    if hasattr(plot, "figsize") and plot.figsize is not None:
        plt.figure(figsize=plot.figsize)
    else:
        plt.figure(figsize=(6,6))
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for i in range(len(mean)):
        # Plot solid line (Test) with legend, dashed line (Train) without legend
        range_list = list(range(len(mean[i][0])))
        new_mean = mean[i][0]
        new_mean_2 = mean[i][1] if mean[i].shape[0] > 1 else None
        if remove_plots > 1:
            range_list = list(range(0, len(mean[i][0]), remove_plots))
            new_mean = [mean[i][0][k] for k in range(0, len(mean[i][0]), remove_plots)]
            new_error = [error[i][0][k] for k in range(0, len(mean[i][0]), remove_plots)]
            if new_mean_2 is not None:
                new_mean_2 = [mean[i][1][k] for k in range(0, len(mean[i][1]), remove_plots)]
                new_error_2 = [error[i][1][k] for k in range(0, len(error[i][1]), remove_plots)]
        else:
            new_error = error[i][0]
            if new_mean_2 is not None:
                new_error_2 = error[i][1]
        # Add one epoch with the copy of the last epoch
        range_list = range_list + [range_list[-1]+1]
        new_mean = list(new_mean) + [new_mean[-1]]
        new_error = list(new_error) + [new_error[-1]]
        if new_mean_2 is not None:
            new_mean_2 = list(new_mean_2) + [new_mean_2[-1]]
            new_error_2 = list(new_error_2) + [new_error_2[-1]]
        color = colors[pair_counter % len(colors)]
        # Solid line (Test) with legend
        plt.plot(range_list, new_mean, label=names[i], linewidth=2, color=color)
        plt.fill_between(range_list, np.array(new_mean)-np.array(new_error), np.array(new_mean)+np.array(new_error), 
                         alpha=0.2, edgecolor=color, facecolor=color, linewidth=1, antialiased=True)
        # Dashed line (Train) without legend
        if new_mean_2 is not None:
            plt.plot(range_list, new_mean_2, linewidth=1, linestyle='dashed', color=color)
            plt.fill_between(range_list, np.array(new_mean_2)-np.array(new_error_2), np.array(new_mean_2)+np.array(new_error_2), 
                             alpha=0.2, edgecolor=color, facecolor=color, linewidth=1, antialiased=True)
        # Track min/max for tight axis
        this_min_y = min(new_mean + (new_mean_2 if new_mean_2 is not None else []))
        this_max_y = max(new_mean + (new_mean_2 if new_mean_2 is not None else []))
        min_x = range_list[0] if min_x is None else min(min_x, range_list[0])
        max_x = range_list[-1] if max_x is None else max(max_x, range_list[-1])
        min_y = this_min_y if min_y is None else min(min_y, this_min_y)
        max_y = this_max_y if max_y is None else max(max_y, this_max_y)
        pair_counter += 1

    # Place legend below the plot so it doesn't overlap the graph
    plt.legend(ncol=2, prop={'size': 16})#, loc='upper center', bbox_to_anchor=(0.67, 0.93))
    # plt.legend(ncol=4, prop={'size': 12}, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.minorticks_on()
    plt.xlabel("Epochs", fontsize=20)
    plt.xlim(min_x, max_x)
    if y_lim is not None:
        plt.ylim(*y_lim)
    else:
        plt.ylim(min_y, max_y)
    plt.ylabel(title, fontsize=20)
    plt.grid(axis='y', linestyle='-', linewidth='0.5', which='major')
    plt.grid(axis='y', linestyle='--', which='minor', linewidth='0.3')
    plt.tight_layout()
    plt.show()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    args = getArgs()
    input_dir = args.input_dir

    # Tuple of error and mean
    csv_lis_per = [formatScores(cropLis(getCsv(input_dir[i], id.split(","), args.file))) for i, id in enumerate(args.ids)]

    # Put combined outputs in project_root/plots/combined/<ids_join>
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    combined_root = os.path.join(project_root, "plots", "combined")
    if args.save_name:
        output_folder = os.path.join(combined_root, args.save_name)
    else:
        ids_join = "-".join(list(map(str, args.ids)))
        output_folder = os.path.join(combined_root, ids_join)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    errors = np.array([x[0] for x in csv_lis_per])
    means = np.array([x[1] for x in csv_lis_per])
    print(means.shape)
    half_len = errors.shape[1]//2
    if args.file == "loss":
        title = "Loss"
        default_y_lim = (0, 0.5)
    else:
        title = "Satisfied Weight Ratio"
        default_y_lim = (0.5, 0.85)

    # Handle custom y-axis limits
    y_lim = None
    if args.ymin is not None and args.ymax is not None:
        y_lim = (args.ymin, args.ymax)
    else:
        y_lim = default_y_lim

    # Handle custom figsize
    if args.figsize is not None:
        plot.figsize = tuple(args.figsize)
    else:
        plot.figsize = None

    # Handle custom save_name
    def get_outfile(i):
        if args.save_name:
            return os.path.join(output_folder, f"{args.save_name}_{i}.png")
        else:
            return os.path.join(output_folder, f"{args.file}_{i}.png")

    for i in range(half_len):
        plot(errors[:, [i+half_len, i]], means[:, [i+half_len, i]], args.names, get_outfile(i), title=title, remove_plots=args.remove_plots, y_lim=y_lim)

    if args.combine is not None:
        combine_inds = list(map(int, args.combine))
        comp_count = means.shape[1]
        if max(combine_inds) >= comp_count or min(combine_inds) < 0:
            print(f"ERROR: combine indices {combine_inds} out of range for available components per input ({comp_count}).")
            print(f"Available per-input component indices: 0..{comp_count-1}")
            print("If you intended to refer to flattened indices across all inputs, please adjust or run without --combine.")
            raise SystemExit(1)
        if args.save_name:
            combine_outfile = os.path.join(output_folder, f"{args.save_name}_combine.png")
        else:
            combine_outfile = os.path.join(output_folder, f"{args.file}_combine.png")
        plot(errors[:, combine_inds], means[:, combine_inds], args.names, combine_outfile, title=title, types=args.combine_names, remove_plots=args.remove_plots, y_lim=y_lim)