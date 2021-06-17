


import numpy as np
from numpy.lib.npyio import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy

from excel_handler import eval_df
from mlxtend.evaluate import permutation_test

path_rand = 'rand_03eps.xlsx'
path_new = 'new_03eps.xlsx'
# path_rand = 'rand_05eps.xlsx'
# path_new = 'new_05eps.xlsx'
# path_rand = 'rand_07eps.xlsx'
# path_new = 'new_07eps.xlsx'

rand_progress = pd.read_excel(path_rand, sheet_name='progress', header=None)
new_progress = pd.read_excel(path_new, sheet_name='progress', header=None)

rand_mutation_stats = eval_df(pd.read_excel(path_rand, sheet_name='mutation_stats', header=None)).values
new_mutation_stats = eval_df(pd.read_excel(path_new, sheet_name='mutation_stats', header=None)).values



def p_value_one_sided(treatment, control):
    p_value = permutation_test(treatment, control,
                            method='approximate',
                            num_rounds=10000,
                            seed=42,
                            func=lambda x, y: np.mean(y) - np.mean(x))
    return p_value

def calculate_all_p_values(generations):
    p_values = np.zeros(len(generations))
    for i,g in enumerate(generation_list):
        naive = rand_mutation_append[g]
        new = new_mutation_append[g]
        p_value = p_value_one_sided(new, naive)
        p_values[i] = p_value
    return p_values

def harmonic_mean_p_value(generations, p_values):
    G = len(generation_list)
    general_p_value = 1/((1/(G*p_values[generation_list])).sum())*np.log(G)*np.e
    return general_p_value

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
    fig, axs = plt.subplots(len(generation_list)+1//2, 2, sharex=True, sharey=True)
    
    for i,g in enumerate(generation_list):
        naive = rand_mutation_append[g]
        new = new_mutation_append[g]
        lim_min = min(np.min(naive), np.min(new))
        lim_max = max(np.max(naive), np.max(new))

        axs[i, 0].hist(naive,bins=100)
        axs[i, 0].set_title('Gen: {}'.format(g+1))
        mean = np.mean(naive)
        median = np.median(naive)
        std = np.std(naive)
       
        axs[i, 0].axvline(mean, color='lime', linestyle='dashed', linewidth=5)
        axs[i, 0].axvline(median, color='m', linestyle='dashed', linewidth=5)
        mean_patch = mpatches.Patch(color='lime', label='Mean: {}'.format(mean))
        std_patch = mpatches.Patch(color=None, label='Std: {}'.format(std))
        median_patch = mpatches.Patch(color='m', label='Median: {}'.format(median))
        axs[i, 0].legend(handles=[mean_patch, median_patch, std_patch], frameon=True)
        axs[i, 0].set_xlim(lim_min, lim_max)

        axs[i, 1].hist(new, bins=100)
        axs[i, 1].set_title('Gen: {}'.format(g+1))
        mean = np.mean(new)
        median = np.median(new)
        std = np.std(new)
        p_value = p_values[g]

        axs[i, 1].axvline(mean, color='lime', linestyle='dashed', linewidth=5)
        axs[i, 1].axvline(median, color='m', linestyle='dashed', linewidth=5)
        mean_patch = mpatches.Patch(color='lime', label='Mean: {}'.format(mean))
        std_patch = mpatches.Patch(color=None, label='Std: {}'.format(std))
        median_patch = mpatches.Patch(color='m', label='Median: {}'.format(median))
        p_value_patch = mpatches.Patch(color='b', label='p_value: {}'.format(p_value))
        axs[i, 1].legend(handles=[mean_patch, median_patch, std_patch, p_value_patch], frameon=True)
        axs[i, 1].set_xlim(lim_min, lim_max)

    fig.text(0.5, 0.04, 'fitness difference', ha='center', fontsize=16)
    fig.text(0.07, 0.5, 'mutations', va='center', rotation='vertical', fontsize=16)
    fig.supylabel('New Algorithm                  Naive Algorithm', fontsize=25)
    plt.show()


def plot_all_runs(rand_progress, new_progress, title='All Runs', ylabel='Route Distance'):
    plt.plot(rand_progress.T, color='r')
    plt.plot([],[], 'r', label='Baseline')
    plt.plot(new_progress.T, color='b')
    plt.plot([],[], 'b', label='MULANN')
    plt.legend(prop={'size': 20})
    plt.title(title, fontsize=30)
    plt.xlabel('Generations', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()

def plot_mean_runs(rand_progress, new_progress, title='Best Individual (average for all runs)', ylabel='Route Distance'):
    plt.plot(rand_progress.mean(axis=0), color='r', marker='o',
        linestyle='dashed',linewidth=2, markersize=12, label='Baseline')
    plt.plot(new_progress.mean(axis=0), color='b', marker='*',
        linestyle='dashed',linewidth=2, markersize=12, label='MULANN')
    plt.legend(prop={'size': 20})
    plt.title(title, fontsize=30)
    plt.xlabel('Generations', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()

def plot_all_runs_final_result(rand_progress, new_progress):
    progress = list(sorted(zip(rand_progress.max(axis=1).tolist(), new_progress.max(axis=1).tolist()), key=lambda x: x[0]+x[1]))
    plt.plot([x[0] for x in progress], color='r', linestyle='None', marker='o', markersize=12, label='Baseline')
    plt.plot([x[1] for x in progress], color='b', linestyle='None', marker='*', markersize=12, label='MULANN')
    plt.legend(prop={'size': 20})
    plt.title('Best Individual per Experiment', fontsize=30)
    plt.xlabel('Experiment', fontsize=22)
    plt.ylabel('Route Distance', fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.show()
    


rand_mutation_stats_mean_per_generation = summarize_mutation_stats(rand_mutation_stats)
new_mutation_stats_mean_per_generation = summarize_mutation_stats(new_mutation_stats)

rand_mutation_append = append_mutation_stats(rand_mutation_stats)
new_mutation_append = append_mutation_stats(new_mutation_stats)


# plot_all_runs(rand_progress, new_progress)   
# plot_mean_runs(rand_progress, new_progress)   

print("Best individual of rand:")
print(rand_progress.max(axis=1))
print("Best individual of new:")
print(new_progress.max(axis=1))


_, paired_t_test = scipy.stats.ttest_rel(new_progress.max(axis=1), rand_progress.max(axis=1), alternative='less') 
print(f"p value for paired t-test is {paired_t_test}")

plot_all_runs_final_result(rand_progress, new_progress)

plot_all_runs(rand_mutation_stats_mean_per_generation, new_mutation_stats_mean_per_generation, ylabel='Change In Route Distance')
plot_mean_runs(rand_mutation_stats_mean_per_generation, new_mutation_stats_mean_per_generation, ylabel='Change In Route Distance')

# generation_list = range(15)
generation_list = [0,4,9,14]
# p_values = calculate_all_p_values(generation_list)

p_values = np.array([2.20977902e-01, 7.98220178e-01, 2.85471453e-01, 1.25887411e-01,
 3.94960504e-02, 1.02889711e-01, 2.67973203e-02, 9.99900010e-05,
 1.65883412e-01, 3.03269673e-01, 1.79982002e-03, 1.79982002e-02,
 2.01679832e-01, 5.07749225e-01, 1.19988001e-03])

create_mutation_distribution_compare(generation_list)

# print(p_values)
print(harmonic_mean_p_value(generation_list, p_values))


