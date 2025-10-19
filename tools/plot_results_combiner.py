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

    return parser.parse_args()


def getCsv(input_dir, ids, file):
    csv_lis = []
    for id in ids:
        file_now = os.path.join(input_dir, f"train_{id}/{file}.csv")
        with open(file_now, 'r') as f:
            csv_reader = csv.reader(f)
            csv_lis.append(np.array(list(csv_reader)))
    return csv_lis

def cropLis(csv_lis):
    min_val = min([len(x[0]) for x in csv_lis])
    for i in range(len(csv_lis)):
        csv_lis[i] = csv_lis[i][:min_val]
    return np.array(csv_lis).astype(np.float64)

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

    total = 0
    plt.figure(figsize=(6,6))
    for i in range(len(mean)):
        for j in range(len(mean[i])//2):
            range_list = list(range(len(mean[i][0])))
            new_mean = mean[i][j]
            new_mean_2 = mean[i][j + len(mean[i])//2]
            if remove_plots > 1:
                range_list = list(range(0, len(mean[i][0]), remove_plots))

                new_mean = []
                new_mean_2 = []
                new_error = []
                new_error_2 = []
                for k in range(0, len(mean[i][0]), remove_plots):
                    # new_mean.append(np.mean(mean[i][j][k: k + remove_plots]))
                    # new_mean_2.append(np.mean(mean[i][j + len(mean[i])//2][k: k + remove_plots]))
                    # new_error.append(np.mean(error[i][j][k: k + remove_plots]))
                    # new_error_2.append(np.mean(error[i][j + len(mean[i])//2][k: k + remove_plots]))
                    new_mean.append(mean[i][j][k])
                    new_mean_2.append(mean[i][j + len(mean[i])//2][k])
                    new_error.append(error[i][j][k])
                    new_error_2.append(error[i][j + len(mean[i])//2][k])
            plt.plot(range_list, new_mean, label=names[i] + f" ({types[j]})", linewidth=2)
            plt.fill_between(range_list, np.array(new_mean)-np.array(new_error), np.array(new_mean)+np.array(new_error), 
                             alpha=0.2, edgecolor=colors[total], facecolor=colors[total],linewidth=1, antialiased=True)
            total += 1
            plt.plot(range_list, new_mean_2, label=names[i] + f" ({types[j+len(mean[i])//2]})", linewidth=1, linestyle='dashed')
            
            plt.fill_between(range_list, np.array(new_mean_2)-np.array(new_error_2), np.array(new_mean_2)+np.array(new_error_2), 
                             alpha=0.2, edgecolor=colors[total], facecolor=colors[total],linewidth=1, antialiased=True)
            total += 1
            # plt.fill_between(mean[i][1], mean[i][1]-error[i][1], mean[i][1]+error[i][1], 
            #                  alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=1, linestyle='dashdot', antialiased=True)
    
    #plt.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4, prop={'size': 8})
    # plt.legend(bbox_to_anchor=(0, 1.05, 0.5, 0.2), loc="upper left", mode="expand", borderaxespad=0, ncol=2, prop={'size': 8})
    plt.legend(ncol=2, prop={'size': 12})
    # plt.tick_params(axis='y', which='minor', bottom=False)
    plt.minorticks_on()
    plt.xlabel("Epochs", fontsize=20)
    # plt.xlim(0, len(mean[0][0]))
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.ylabel(title, fontsize=20)
    plt.grid(axis='y', linestyle='-', linewidth='0.5', which='major')
    plt.grid(axis='y', linestyle='--', which='minor', linewidth='0.3')
    plt.show()
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    args = getArgs()
    input_dir = args.input_dir

    # Tuple of error and mean
    csv_lis_per = [formatScores(cropLis(getCsv(input_dir[i], id.split(","), args.file))) for i, id in enumerate(args.ids)]

    output_dir = args.output_dir
    ids_join = "-".join(list(map(str, args.ids)))
    output_folder = os.path.join(output_dir, f"{ids_join}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    errors = np.array([x[0] for x in csv_lis_per])
    means = np.array([x[1] for x in csv_lis_per])
    print(means.shape)
    half_len = errors.shape[1]//2
    if args.file == "loss":
        title = "Loss"
        y_lim = (0, 0.5)
    else:
        title = "Evaluation Score"
        y_lim = (0.5, 0.85)
    for i in range(half_len):
        # if i == 1:
        #     plt.ylim(0.8, 0.85)

        plot(errors[:, [i+half_len, i]], means[:, [i+half_len, i]], args.names, os.path.join(output_folder, f"{args.file}_{i}.png"), title=title, remove_plots=args.remove_plots, y_lim=y_lim)

    if args.combine is not None:
        plot(errors[:, args.combine], means[:, args.combine], args.names, os.path.join(output_folder, f"{args.file}_combine.png"), title=title, types=args.combine_names, remove_plots=args.remove_plots, y_lim=y_lim)