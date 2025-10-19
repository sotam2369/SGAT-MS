import csv
import os

import argparse
import numpy as np
import re

def getArgs():
    parser = argparse.ArgumentParser(description="Combine results from different solvers")
    parser.add_argument("dir", type=str, help="Directory of the solver output files")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--files", type=str, nargs="+", help="Solver files to combine")
    parser.add_argument("--years", type=str, nargs="+", help="Years to combine")
    parser.add_argument("--names", type=str, nargs="+", help="Names of the solvers")
    parser.add_argument("--best_cost", type=str, help="Best cost file")
    parser.add_argument("--show_scores", action="store_true", help="Show scores")
    parser.add_argument("--proposed_method", type=int, nargs="+", default=[2], help="Where to put dashed line")
    parser.add_argument("--remove-ones", action="store_true", help="Remove ones from the scores")
    parser.add_argument("--show-timeouts", action="store_true", help="Show timeouts")
    parser.add_argument("--show-all", action="store_true", help="Show all years in one plot")
    parser.add_argument("--show-error", action="store_true", help="Show error in the latex table")
    parser.add_argument("--add-unfound", action="store_true", help="Add unfound to the scores")
    return parser.parse_args()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def combine_results(dir, files, year, names, add_unfound=True):
    csv_list = []
    for file_single in files:
        csv_list_temp = []
        for file in file_single.split(","):
            with open(os.path.join(dir, file.replace("year", year))) as f:
                reader = csv.reader(f)
                csv_list_temp.append(list(reader))
        csv_list.append(csv_list_temp)

    final_list_names = csv_list[0][0][-1]
    removed = None
    if not add_unfound and csv_list[0][0][0][-1] == '-1':
        removed = final_list_names.pop(-1)
    final_list_names = list(set(final_list_names))

    for i in range(len(files)):
        for j in range(len(csv_list[i])):
            for key in csv_list[i][j][-1]:
                if key not in final_list_names:
                    final_list_names.append(key)

    if not add_unfound and removed is not None:
        print("Removing", removed)
        final_list_names.append(removed)

    final_csv_list = [[[] for _ in range(len(csv_list[i]))] for i in range(len(files))]
    for i in range(len(files)):
        for j in range(len(csv_list[i])):
            for key in list(final_list_names):
                if key in csv_list[i][j][-1]:
                    final_csv_list[i][j].append(csv_list[i][j][0][csv_list[i][j][-1].index(key)])
                else:
                    if add_unfound:
                        print(key, "not in", names[i])
                        final_csv_list[i][j].append(-1)
                    else:
                        final_list_names.remove(key)


    return final_csv_list, final_list_names


def getScores(final_csv_list, final_list_names, year, best_cost_file, scores_all, scores_per_set_all, remove_ones=False):
    set_names = [["scpc"], ["maxcut"], ["clq"], ["t", "pm"], ["s2v"], ["dimacs"], ["s3v"], ["ramsey"], ["spi"]]#, ["ran"]]
    best_cost_dict = {}
    # remove_list = [[], []]
    with open(best_cost_file.replace("year", year), "r") as f:
        reader = csv.reader(f)
        best_cost = list(reader)
    
    for line in best_cost:
        best_cost_dict[line[0].split("/")[-1].replace(".gz","")] = line[1]
    
    scores = [[[] for _ in range(len(final_csv_list[i]))] for i in range(len(final_csv_list))]
    scores_per_set = [[{"-".join(x): [] for x in set_names} for _ in range(len(final_csv_list[i]))] for i in range(len(final_csv_list))]
    if scores_all is None:
        scores_all = [[[] for _ in range(len(final_csv_list[i]))] for i in range(len(final_csv_list))]
        scores_per_set_all = [[{"-".join(x): [] for x in set_names} for _ in range(len(final_csv_list[i]))] for i in range(len(final_csv_list))]
    
    for i in range(len(final_csv_list[0][0])):
        true_name = None
        for key in best_cost_dict.keys():
            if final_list_names[i] in key:
                true_name = key
        if true_name is None:
            all_list = []
            for x in final_csv_list:
                for y in x:
                    if float(y[i]) > 0:
                        all_list.append(float(y[i]))
            best_cost_dict[final_list_names[i]] = min(all_list)
            true_name = final_list_names[i]
        
        # Replace all numbers with N
        current_name = final_list_names[i]
        current_name = re.sub(r'\d+', '[N]', current_name)
        set_name_list = re.split(r'[-\d]+', current_name.split(".")[0])
        set_name = None
        for set in set_names:
            if all([x in set_name_list for x in set]):
                set_name = "-".join(set)
                break
        if set_name is None:
            # set_name = final_list_names[i].split(".")[0].split("-")[0].split("_")[0]
            set_name = set_name_list[0]
            set_names.append([set_name])
            # print("Set name not found", set_name)
            for j, x in enumerate(final_csv_list):
                for k, y in enumerate(x):
                    if set_name not in scores_per_set_all[j][k].keys():
                        scores_per_set_all[j][k][set_name] = []
                    scores_per_set[j][k][set_name] = []
        
        # Check if all the same
        all_same = True
        res_same = 1
        for y in final_csv_list:
            for x in y:
                res_now = 0
                if float(x[i]) >= 0:
                    res_now = (float(best_cost_dict[true_name])+1)/(float(x[i])+1)
                if res_same != res_now:
                    all_same = False
                    break
        if all_same and remove_ones:
            print("Removing", true_name)
            # remove_list[0].append(final_csv_list[0][0][i])
            # remove_list[1].append(final_list_names[i])
            continue

        for j, y in enumerate(final_csv_list):
            for k, x in enumerate(y):
                res_now = 0
                if float(x[i]) >= 0:
                    res_now = min((float(best_cost_dict[true_name])+1)/(float(x[i])+1), 1)
                
                scores_per_set[j][k][set_name].append(res_now)
                scores_per_set_all[j][k][set_name].append(res_now)
                scores[j][k].append(res_now)
                scores_all[j][k].append(res_now)
    # os.makedirs("../solver_output/remove", exist_ok=True)
    # with open(f"../solver_output/remove/{year}.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(remove_list)

    # Clump up sets in scores_per_set and scores_per_set_all that have only one problem as others
    # for i in range(len(scores_per_set)):
    #     for j in range(len(scores_per_set[i])):
    #         for key in list(scores_per_set[i][j].keys()):
    #             if len(scores_per_set[i][j][key]) == 1:
    #                 if "others" in scores_per_set[i][j].keys():
    #                     scores_per_set[i][j]["others"] += scores_per_set[i][j][key]
    #                 else:
    #                     scores_per_set[i][j]["others"] = scores_per_set[i][j][key]
    #                 scores_per_set[i][j].pop(key)
    #         for key in list(scores_per_set_all[i][j].keys()):
    #             if len(scores_per_set_all[i][j][key]) == 1:
    #                 if "others" in scores_per_set_all[i][j].keys():
    #                     scores_per_set_all[i][j]["others"] += scores_per_set_all[i][j][key]
    #                 else:
    #                     scores_per_set_all[i][j]["others"] = scores_per_set_all[i][j][key]
    #                 scores_per_set_all[i][j].pop(key)
    return scores, scores_per_set, scores_all, scores_per_set_all


def getMeanAndError(scores, scores_per_set):
    max_len = max([len(x) for x in scores])
    for i in range(len(scores)):
        scores[i] += [list(scores[i][0])]*(max_len-len(scores[i]))
    scores = np.array(scores)
    mean = np.mean(scores, axis=1).tolist()
    error = np.std(scores, axis=1).tolist()

    scores_per_set_mean = [{} for _ in range(len(scores_per_set))]
    scores_per_set_error = [{} for _ in range(len(scores_per_set))]
    for j in range(len(scores_per_set)):
        for key in scores_per_set[j][0].keys():
            scores_per_set_mean[j][key] = np.mean(np.array([scores_per_set[j][i][key] for i in range(len(scores_per_set[j]))]), axis=0).tolist()
            scores_per_set_error[j][key] = np.std(np.array([scores_per_set[j][i][key] for i in range(len(scores_per_set[j]))]), axis=0).tolist()

    return mean, error, scores_per_set_mean, scores_per_set_error

def plotResults(scores, scores_per_set, year, output_dir, names, show_scores, plt_scatter=False):
    import matplotlib.pyplot as plt
    plt.rcParams["font.size"] = 20

    marker_array = ["o", "x",  "^", "s", "+", "4"]
    linewidth = 1.5
    plt.figure(figsize=(10,6))
    for i, name in enumerate(names):
        if show_scores:
            plt.plot(list(range(len(scores[i]))), sorted(scores[i]), marker=marker_array[i], label=f"{name} ({sum(scores[i])/len(scores[i]):.4f})", mfc='none', linewidth=linewidth)
        else:
            plt.plot(list(range(len(scores[i]))), sorted(scores[i]), marker=marker_array[i], label=f"{name}", mfc='none', linewidth=linewidth)
    
    if not os.path.exists(os.path.join(output_dir, f"full")):
        os.mkdir(os.path.join(output_dir, f"full"))
    plt.xlabel("#Benchmarks")
    plt.ylabel("Score")
    plt.title(f"Incomplete Scores ({year})")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = sorted(range(len(handles)), key=lambda i: 1-sum(scores[i])/len(scores[i]))
    plt.legend([handles[i] for i in order], [labels[i] for i in order]) 
    plt.savefig(os.path.join(output_dir, f"full/{year}.png"))
    plt.close()
    plt.clf()
    if plt_scatter:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(scores[0], scores[1], marker="x")
        plt.plot([0, 1], [0, 1], color="black", linestyle="--")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        plt.savefig(os.path.join(output_dir, f"full/{year}_scatter.png"))
        plt.close()

    for key in scores_per_set[0].keys():
        if scores_per_set[0][key] == [] or len(scores_per_set[0][key]) == len(scores[0]):
            continue
        if not os.path.exists(os.path.join(output_dir, key)):
            os.mkdir(os.path.join(output_dir, key))
        for i in range(len(names)):
            if show_scores:
                plt.plot(list(range(len(scores_per_set[i][key]))), sorted(scores_per_set[i][key]), marker=marker_array[i], label=f"{names[i]} ({sum(scores_per_set[i][key])/len(scores_per_set[i][key]):.4f})", mfc='none', linewidth=linewidth)
            else:
                plt.plot(list(range(len(scores_per_set[i][key]))), sorted(scores_per_set[i][key]), marker=marker_array[i], label=f"{names[i]}", mfc='none', linewidth=linewidth)
        plt.xlabel("#Benchmarks")
        plt.ylabel("Score")
        plt.title(f"{key} ({year})")
        handles, labels = plt.gca().get_legend_handles_labels() 
        order = sorted(range(len(handles)), key=lambda i: 1-sum(scores_per_set[i][key])/len(scores_per_set[i][key]))
        plt.legend([handles[i] for i in order], [labels[i] for i in order]) 
        # Fit labels in the image
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}/{year}.png"))
        plt.close()
        plt.clf()



# Output scores with print
def outputScores(scores, scores_per_set, year, names):
    keys = scores_per_set[0].keys()
    print(f"Year: {year}")
    print(f"Order: {', '.join(names)}")
    print("Full average:", end=" ")
    max_score = max([sum(x)/len(x) for x in scores])
    for i in range(len(names)-1):
        if sum(scores[i])/len(scores[i]) == max_score:
            print(f"{bcolors.OKCYAN}{sum(scores[i])/len(scores[i]):.4f}{bcolors.ENDC}", end=", ")
        else:
            print(f"{sum(scores[i])/len(scores[i]):.4f}", end=", ")
    if sum(scores[-1])/len(scores[-1]) == max_score:
        print(f"{bcolors.OKCYAN}{sum(scores[-1])/len(scores[-1]):.4f}{bcolors.ENDC}")
    else:
        print(f"{sum(scores[-1])/len(scores[-1]):.4f}")

    print("Timeouts:", end=" ")
    for i in range(len(names)-1):
        print(f"{scores[i].count(0)}", end=", ")
    print(f"{scores[-1].count(0)} ({len(scores[-1])})")
    
    for key in keys:
        if scores_per_set[0][key] == [] or len(scores_per_set[0][key]) == len(scores[0]):
            continue
        print(f"{key} average:", end=" ")
        max_score = max([sum(scores_per_set[i][key])/len(scores_per_set[i][key]) for i in range(len(names))])
        for i in range(len(names)-1):
            if sum(scores_per_set[i][key])/len(scores_per_set[i][key]) >= max_score:
                print(f"{bcolors.OKCYAN}{sum(scores_per_set[i][key])/len(scores_per_set[i][key]):.4f}{bcolors.ENDC}", end=", ")
            else:
                print(f"{sum(scores_per_set[i][key])/len(scores_per_set[i][key]):.4f}", end=", ")
        if sum(scores_per_set[-1][key])/len(scores_per_set[-1][key]) >= max_score:
            print(f"{bcolors.OKCYAN}{sum(scores_per_set[-1][key])/len(scores_per_set[-1][key]):.4f}{bcolors.ENDC}")
        else:
            print(f"{sum(scores_per_set[-1][key])/len(scores_per_set[-1][key]):.4f}")

        print(f"{key} timeouts:", end=" ")
        for i in range(len(names)-1):
            print(f"{scores_per_set[i][key].count(0)}", end=", ")
        print(f"{scores_per_set[-1][key].count(0)} ({len(scores_per_set[-1][key])})")

def outputLatexTable(scores_per_year, names, proposed_method=2, show_timeouts=False, error=None, error_per_set=None):
    # Find best name for each year
    best_names = {}
    for year in scores_per_year.keys():
        best_name = None
        best_score = 0
        for i in range(len(names)):
            score = sum(scores_per_year[year][i])/len(scores_per_year[year][i])
            if score > best_score:
                best_score = score
                best_name = str(i)
        best_names[year] = best_name
    best_names_timeout = {}
    for year in scores_per_year.keys():
        best_name = None
        best_score = 1000
        for i in range(len(names)):
            score = scores_per_year[year][i].count(0)
            if score < best_score:
                best_score = score
                best_name = str(i)
            elif score == best_score:
                best_name += f",{i}"
        if len(best_name.split(",")) == len(names):
            best_name = "None"
        best_names_timeout[year] = best_name
    
    print()
    print("Latex Table")
    for i in range(len(names)):
        print(f"{names[i]} & ", end="")
        for year in list(scores_per_year.keys())[:-1]:
            if error is None:
                if best_names[year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[year][i])/len(scores_per_year[year][i]):.4f}}} & ", end="")
                else:
                    print(f"{sum(scores_per_year[year][i])/len(scores_per_year[year][i]):.4f} & ", end="")
                if str(i) in best_names_timeout[year].split(",") and show_timeouts:
                    print(f"\\textbf{{{scores_per_year[year][i].count(0)}/{len(scores_per_year[year][i])}}}", end=" & ")
                elif show_timeouts:
                    print(f"{scores_per_year[year][i].count(0)}/{len(scores_per_year[year][i])} & ", end="")
            else:
                if best_names[year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[year][i])/len(scores_per_year[year][i]):.4f}}} $\pm$ {error[year][i]:.2f}& ", end="")
                else:
                    print(f"{sum(scores_per_year[year][i])/len(scores_per_year[year][i]):.4f} $\pm$ {error[year][i]:.2f}& ", end="")
                if str(i) in best_names_timeout[year].split(",") and show_timeouts:
                    print(f"\\textbf{{{scores_per_year[year][i].count(0)}/{len(scores_per_year[year][i])}}}", end=" & ")
                elif show_timeouts:
                    print(f"{scores_per_year[year][i].count(0)}/{len(scores_per_year[year][i])} & ", end="")

        
        final_year = list(scores_per_year.keys())[-1]
        if error is None:
            if show_timeouts:
                if best_names[final_year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f}}} & ", end="")
                else:
                    print(f"{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f} & ", end="")
                if str(i) in best_names_timeout[final_year].split(",") and show_timeouts:
                    print(f"\\textbf{{{scores_per_year[final_year][i].count(0)}/{len(scores_per_year[final_year][i])}}} \\\\")
                else:
                    print(f"{scores_per_year[final_year][i].count(0)}/{len(scores_per_year[final_year][i])} \\\\")
            else:
                if best_names[final_year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f}}} \\\\")
                else:
                    print(f"{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f} \\\\")
        else:
            if show_timeouts:
                if best_names[final_year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f}}} $\pm$ {error[final_year][i]:.2f}& ", end="")
                else:
                    print(f"{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f} $\pm$ {error[final_year][i]:.2f}& ", end="")
                if str(i) in best_names_timeout[final_year].split(",") and show_timeouts:
                    print(f"\\textbf{{{scores_per_year[final_year][i].count(0)}/{len(scores_per_year[final_year][i])}}} \\\\")
                else:
                    print(f"{scores_per_year[final_year][i].count(0)}/{len(scores_per_year[final_year][i])} \\\\")
            else:
                if best_names[final_year] == str(i):
                    print(f"\\textbf{{{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f}}} $\pm$ {error[final_year][i]:.2f} \\\\")
                else:
                    print(f"{sum(scores_per_year[final_year][i])/len(scores_per_year[final_year][i]):.4f} $\pm$ {error[final_year][i]:.2f} \\\\")
        if i in proposed_method:
            print("\\hdashline")

if __name__ == "__main__":
    args = getArgs()

    dir = args.dir
    output_dir = args.output_dir
    multi = len(args.files) > 2

    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.mkdir(output_dir)

    scores_all = None
    scores_per_set_all = None
    scores_per_year = {}
    errors_per_year = {}

    for year in args.years:
        final_csv_list, final_list_names = combine_results(dir, args.files, year, args.names, add_unfound=args.add_unfound)
        scores, scores_per_set, scores_all, scores_per_set_all  = getScores(final_csv_list, final_list_names, year, args.best_cost, scores_all, scores_per_set_all, remove_ones=args.remove_ones)
        scores, error, scores_per_set, _ = getMeanAndError(scores, scores_per_set)
        plotResults(scores, scores_per_set, year, output_dir, args.names, args.show_scores)
        print(f"Year {year} done")
        outputScores(scores, scores_per_set, year, args.names)
        print()
        scores_per_year[year] = scores
        errors_per_year[year] = np.mean(np.array(error), axis=1).tolist()
    
    scores_all, error_all, scores_per_set_all, _ = getMeanAndError(scores_all, scores_per_set_all)
    plotResults(scores_all, scores_per_set_all, "all", output_dir, args.names, args.show_scores, plt_scatter=not multi)
    outputScores(scores_all, scores_per_set_all, "all", args.names)
    if args.show_all:
        scores_per_year["all"] = scores_all
        errors_per_year["all"] = np.mean(np.array(error_all), axis=1).tolist()
    if not args.show_error:
        errors_per_year = None
    outputLatexTable(scores_per_year, args.names, args.proposed_method, args.show_timeouts, error=errors_per_year)