"""Script to run simulations of single-step selection experiments."""
import os

import numpy as np
import torch

# pylint: disable=no-member, invalid-name

# parameters: initial population
N_tot_initial = 1e11
# pre- and post-selection sampling
N_sampleds = [(1e8, 1e8)]
# number of variants
num_var = 20**4
# number of experiments
num_exp = 100

# initial populations and selection probabilities for each variant
N_initial_per_var = N_tot_initial * np.random.dirichlet([1 for _ in range(num_var)], num_exp)
N_initial_per_var = np.array([[int(val) for val in vallist] for vallist in N_initial_per_var])
N_tot_initial = np.sum(N_initial_per_var, axis=1)
p_sel = np.random.uniform(0, 1, (num_var,))

N_sampled_initial_per_var_all = []
N_sampled_sel_per_var_all = []
N_sampled_initial_array = []
N_sampled_array = []
for N_sampled, N_sampled_initial in N_sampleds:
    for i in range(num_exp):
        N_sampled_initial_per_var = np.random.multinomial(
            N_sampled_initial, N_initial_per_var[i, :] / N_tot_initial[i]
        )
        N_sampled_initial_per_var_all += [N_sampled_initial_per_var]
        N_sel_per_var = np.array(
            [np.random.binomial(init, sel) for init, sel in zip(N_initial_per_var[i, :], p_sel)]
        )
        N_sampled_sel_per_var = np.random.multinomial(
            N_sampled, N_sel_per_var / np.sum(N_sel_per_var)
        )
        N_sampled_sel_per_var_all += [N_sampled_sel_per_var]
        N_sampled_array += [np.tile(N_sampled, (num_var,))]
        N_sampled_initial_array += [np.tile(N_sampled_initial, (num_var,))]

N_sampled_initial_per_var_all = np.array(N_sampled_initial_per_var_all)
N_sampled_sel_per_var_all = np.array(N_sampled_sel_per_var_all)
N_sampled_initial_array = np.array(N_sampled_initial_array)
N_sampled_array = np.array(N_sampled_array)

selection_data = np.vstack(
    (
        np.tile(np.arange(num_var), (1, num_exp * len(N_sampleds), 1)),
        N_sampled_initial_per_var_all[np.newaxis, :],
        N_sampled_sel_per_var_all[np.newaxis, :],
        N_sampled_initial_array[np.newaxis, :],
        N_sampled_array[np.newaxis, :],
    )
)
selection_data = torch.tensor(selection_data, dtype=torch.int64)
selection_data = selection_data.permute(1, 2, 0)

data_dir = "Data/Selection_Simulations/"
torch.save(selection_data, os.path.join(data_dir, "selection_data.pt"))
np.save(os.path.join(data_dir, "p_sel.npy"), p_sel)
