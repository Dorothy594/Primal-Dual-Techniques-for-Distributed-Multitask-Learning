import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing

from SBM_experiment_two_cluster import get_sbm_2blocks_data, algorithm_1, get_scores

num_cores = multiprocessing.cpu_count()
print(num_cores)
num_tries = 5

PENALTY_FUNCS = ['norm1', 'norm2', 'mocha']

lambda_lasso = 0.01

# K = 3000
K = 2000
# sampling_ratio = 0.6

# pouts = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
pouts = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
sampling_ratios = [0.2, 0.4, 0.6]
colors = ['steelblue', 'darkorange', 'green']


def fun(penalty_func, sampling_ratio, pout):
    B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(pin=0.5, pout=pout,
                                                                  is_torch_model=False)
    E, N = B.shape

    samplingset = random.sample([i for i in range(N)], k=int(sampling_ratio * N))

    _, predicted_w = algorithm_1(K, B, weight_vec, datapoints, true_labels, samplingset,
                                 lambda_lasso, penalty_func)

    alg1_score, _, _ = get_scores(datapoints, predicted_w, samplingset)
    return sampling_ratio, pout, alg1_score


for penalty_func in PENALTY_FUNCS:
    print('penalty_func:', penalty_func)

    results = Parallel(n_jobs=num_cores)(delayed(fun)(penalty_func, sampling_ratio, pout)
                                         for sampling_ratio in sampling_ratios
                                         for pout in pouts for i in range(num_tries))

    pout_mses = defaultdict(list)

    for sr, pout, alg1_score in results:
        pout_mses[(sr, pout)].append(alg1_score['total'])

    for i, sr in enumerate(sampling_ratios):
        MSEs_mean = {}
        MSEs_std = {}
        for pout in pouts:
            MSEs_mean[pout] = np.mean(pout_mses[(sr, pout)])
            MSEs_std[pout] = np.std(pout_mses[(sr, pout)])

        plt.errorbar(list(MSEs_mean.keys()), list(MSEs_mean.values()), yerr=list(MSEs_std.values()),
                     ecolor=colors[i], capsize=3,
                     label='M=' + str(sr), c=colors[i])

        print('M:', sr)
        print(' MSEs_mean:', MSEs_mean)
        print(' MSEs_std:', MSEs_std)

    plt.xlabel('p_out')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.title('Penalty function : %s' % penalty_func)
    plt.show()
    plt.close()
    plt.savefig('two cluster penalty function.png')