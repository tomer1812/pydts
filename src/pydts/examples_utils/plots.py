from matplotlib import pyplot as plt
import numpy as np


def plot_first_model_coefs(models, times, expanded_train_df, n_cov=5):
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    ax = axes[0]
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    ax.scatter(times, models[1].params[:len(times)].values, label='J=1 (Pred)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue')
    ax.scatter(times, models[2].params[:len(times)].values, label='J=2 (Pred)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=16)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=20)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    ax2.hist(expanded_train_df['X'], color='r', alpha=0.3, bins=times)
    ax2.set_ylabel('N patients', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.bar(np.arange(1, n_cov+1), models[1].params[-n_cov:], label='J=1 (Pred)', width=0.3, alpha=0.4, color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), models[2].params[-n_cov:], color='tab:green', label='J=2 (Pred)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()


def plot_second_model_coefs(alpha_df, beta_models, times, n_cov=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    tmp_ajt = alpha_df[alpha_df['J'] == 1]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=1 (Pred)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue' )
    tmp_ajt = alpha_df[alpha_df['J'] == 2]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=2 (Pred)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=16)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=20)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    ax2.bar(alpha_df.groupby('X')['n_jt'].sum().index, alpha_df.groupby('X')['n_jt'].sum().values, color='r', alpha=0.3)
    ax2.set_ylabel('N patients', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.bar(np.arange(1, n_cov+1), beta_models[1].params_.values, label='J=1 (Pred)', width=0.3, alpha=0.4,
           color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), beta_models[2].params_.values, color='tab:green', label='J=2 (Pred)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()
    ax.legend(loc='upper center', fontsize=14)
    fig.tight_layout()
