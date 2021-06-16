


import numpy as np
from numpy.lib.npyio import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from excel_handler import eval_df

path_rand = 'rand_03eps.xlsx'
path_new = 'new_03eps.xlsx'
# path_rand = 'rand_05eps.xlsx'
# path_new = 'new_05eps.xlsx'
# path_rand = 'rand_07eps.xlsx'
# path_new = 'new_07eps.xlsx'

rand_progress = pd.read_excel(path_rand, sheet_name='progress', header=None)
new_progress = pd.read_excel(path_new, sheet_name='progress', header=None)

rand_mutation_stats = pd.read_excel(path_rand, sheet_name='mutation_stats', header=None)
new_mutation_stats = pd.read_excel(path_new, sheet_name='mutation_stats', header=None)

rand_mutation_stats = eval_df(pd.read_excel(path_rand, sheet_name='mutation_stats', header=None)).values
new_mutation_stats = eval_df(pd.read_excel(path_new, sheet_name='mutation_stats', header=None)).values

def summarize_mutation_stats(mutation_stats):
    res = np.zeros(rand_mutation_stats.shape)
    for i in range(mutation_stats.shape[0]):
        for j in range(mutation_stats.shape[1]):
            res[i,j] = np.mean(mutation_stats[i,j])
    return res


def append_mutation_stats(mutation_stats):
    res = {}
    for i in range(mutation_stats.shape[0]):
        for j in range(mutation_stats.shape[1]):
            if j not in res:
                res[j] = []
            res[j]+=mutation_stats[i,j]
    return res



def create_mutation_distribution_compare(generation_list):
    fig, axs = plt.subplots(2, len(generation_list))
    for i,g in enumerate(generation_list):
        naive = rand_mutation_append[g]
        new = new_mutation_append[g]
        lim_min = min(np.min(naive), np.min(new))
        lim_max = max(np.max(naive), np.max(new))

        axs[0, i].hist(naive,bins=100)
        axs[0, i].set_title('Gen: {}'.format(g+1))
        mean = np.mean(naive)
        median = np.median(naive)
        std = np.std(naive)
        axs[0, i].axvline(mean, color='lime', linestyle='dashed', linewidth=5)
        axs[0, i].axvline(median, color='m', linestyle='dashed', linewidth=5)
        mean_patch = mpatches.Patch(color='lime', label='Mean: {}'.format(mean))
        std_patch = mpatches.Patch(color=None, label='Std: {}'.format(std))
        median_patch = mpatches.Patch(color='m', label='Median: {}'.format(median))
        axs[0, i].legend(handles=[mean_patch, median_patch, std_patch], frameon=True)
        axs[0, i].set_xlim(lim_min, lim_max)

        axs[1, i].hist(new, bins=100)
        axs[1, i].set_title('Gen: {}'.format(g+1))
        mean = np.mean(new)
        median = np.median(new)
        std = np.std(new)
        axs[1, i].axvline(mean, color='lime', linestyle='dashed', linewidth=5)
        axs[1, i].axvline(median, color='m', linestyle='dashed', linewidth=5)
        mean_patch = mpatches.Patch(color='lime', label='Mean: {}'.format(mean))
        std_patch = mpatches.Patch(color=None, label='Std: {}'.format(std))
        median_patch = mpatches.Patch(color='m', label='Median: {}'.format(median))
        axs[1, i].legend(handles=[mean_patch, median_patch, std_patch], frameon=True)
        axs[1, i].set_xlim(lim_min, lim_max)
    fig.supylabel('New Algorithm                  Naive Algorithm', fontsize=25)
    plt.show()



def plot_all_runs(rand_progress, new_progress, title='All Runs', ylabel='Route Distance'):
    plt.plot(rand_progress.T, color='r')
    plt.plot([],[], 'r', label='Naive')
    plt.plot(new_progress.T, color='b')
    plt.plot([],[], 'b', label='New')
    plt.legend(prop={'size': 20})
    plt.title(title, fontsize=30)
    plt.xlabel('Generations', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()

def plot_mean_runs(rand_progress, new_progress, title='Mean Runs', ylabel='Route Distance'):
    plt.plot(rand_progress.mean(axis=0), color='r', marker='o',
        linestyle='dashed',linewidth=2, markersize=12, label='Naive')
    plt.plot(new_progress.mean(axis=0), color='b', marker='*',
        linestyle='dashed',linewidth=2, markersize=12, label='New')
    plt.legend(prop={'size': 20})
    plt.title(title, fontsize=30)
    plt.xlabel('Generations', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()




rand_mutation_stats_mean_per_generation = summarize_mutation_stats(rand_mutation_stats)
new_mutation_stats_mean_per_generation = summarize_mutation_stats(new_mutation_stats)

rand_mutation_append = append_mutation_stats(rand_mutation_stats)
new_mutation_append = append_mutation_stats(new_mutation_stats)


plot_all_runs(rand_progress, new_progress)   
plot_mean_runs(rand_progress, new_progress)   

plot_all_runs(rand_mutation_stats_mean_per_generation, new_mutation_stats_mean_per_generation, ylabel='Change In Route Distance')
plot_mean_runs(rand_mutation_stats_mean_per_generation, new_mutation_stats_mean_per_generation, ylabel='Change In Route Distance')

generation_list = [0,4,9,14]
create_mutation_distribution_compare(generation_list)



