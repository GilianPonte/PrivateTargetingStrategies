import numpy as np
import pandas as pd

def protect_CATEs(percent, CATE, CATE_estimates, n, epsilons):
    top = int(n * percent)
    selection_true = np.zeros(n)
    selection_tau = np.zeros(n)
    indices_tau = np.argsort(CATE_estimates)[::-1][:top]
    selection_tau[indices_tau] = 1
    if len(CATE) > 0:
        indices_true = np.argsort(CATE)[::-1][:top]
        selection_true[indices_true] = 1

    collection = pd.DataFrame({'customer': np.arange(1, n+1)})
    for epsilon in epsilons:
        print("running epsilon: "+ str(epsilon))
        protected_selection = protect_selection(epsilon, selection_tau, top)
        collection[f'epsilon_{epsilon:.2f}'.replace('.', '')] = protected_selection

    collection['random'] = np.random.choice([0, 1], size=n, replace=True, p=[1-percent, percent])
    collection['percentage'] = percent
    collection['selection_true'] = selection_true
    collection['selection_tau'] = selection_tau
    if len(CATE) > 0:
        collection['tau'] = CATE
    return collection

def protect_selection(epsilon, selection, top):
    P = np.zeros((2, 2))
    exp_eps = np.exp(epsilon)
    P[np.diag_indices_from(P)] = exp_eps / (2 - 1 + exp_eps)
    P[P == 0] = 1 / (2 - 1 + exp_eps)
    responses = np.zeros(len(selection))
    for i in range(len(selection)):
        responses[i] = np.random.choice([0, 1], p=P[int(selection[i]), :])
    protected_selection = np.zeros(len(selection))
    index_0 = np.where(responses == 0)[0]
    index_1 = np.where(responses == 1)[0]
    if top > len(index_1):
        protected_selection[np.random.choice(index_1, len(index_1), replace=False)] = 1
        protected_selection[np.random.choice(index_0, top - len(index_1), replace=False)] = 1
    else:
        protected_selection[np.random.choice(index_1, top, replace=False)] = 1
    return protected_selection

def policy_profit(data, bootstrap=False):
    # Selecting required columns and performing pivot_longer equivalent operation
    if bootstrap:
        data_filtered = data[['tau', 'selection_true', 'selection_tau', 'epsilon_005', 'epsilon_050',
                              'epsilon_100', 'epsilon_300', 'epsilon_500', 'random', 'percent', 'bootstrap']]
    else:
        data_filtered = data[['tau', 'selection_true', 'selection_tau', 'epsilon_005', 'epsilon_050',
                              'epsilon_100', 'epsilon_300', 'epsilon_500', 'random', 'percent']]

    # Melting the DataFrame to long format
    data_long = data_filtered.melt(id_vars=['tau', 'percent'] + (['bootstrap'] if bootstrap else []),
                                   value_vars=['selection_true', 'selection_tau', 'epsilon_005', 'epsilon_050',
                                               'epsilon_100', 'epsilon_300', 'epsilon_500', 'random'],
                                   var_name='name', value_name='value')

    # Grouping and summarizing to calculate profits
    group_keys = ['percent', 'name'] + (['bootstrap'] if bootstrap else [])
    profit = data_long.groupby(group_keys).apply(
        lambda df: pd.Series({
            'profit': (df['tau'] * df['value']).sum()
        })
    ).reset_index()

    # Calculating mean profit, lower and upper quantiles if bootstrap is True
    if bootstrap:
        summary = profit.groupby(['percent', 'name']).agg({
            'profit': ['mean', lambda x: np.quantile(x, 0.025), lambda x: np.quantile(x, 0.975)]
        })
        summary.columns = ['mean_profit', 'lower', 'upper']  # Renaming the columns
        return summary.reset_index()

    return profit

def policy_overlap(data, bootstrap=False):
    if bootstrap:
        # Selecting the necessary columns, filtering, and computing overlaps
        cols = ['customer', 'selection_true', 'selection_tau', 'epsilon_005', 'epsilon_050',
                                               'epsilon_100', 'epsilon_300', 'epsilon_500', 'random', 'percent', 'bootstrap']
        data = data.loc[(data['percent'] > 0) & (data['percent'] < 1), cols]
        grouped = data.groupby(['percent', 'bootstrap'])
    else:
        # Selecting the necessary columns, filtering, and computing overlaps
        cols = ['customer', 'selection_true', 'selection_tau', 'epsilon_005', 'epsilon_050',
                                               'epsilon_100', 'epsilon_300', 'epsilon_500', 'random', 'percent']
        data = data.loc[(data['percent'] > 0) & (data['percent'] < 1), cols]
        grouped = data.groupby('percent')

    def compute_overlap(group):
        results = {}
        true_values = group['selection_true'].values
        for col in ['random', 'epsilon_005', 'epsilon_050',
                                               'epsilon_100', 'epsilon_300', 'epsilon_500']:
            pred_values = group[col].values
            # Creating a contingency table and calculating overlap
            contingency_table = pd.crosstab(true_values, pred_values)
            try:
                overlap = contingency_table.loc[1, 1] / true_values.sum()
            except KeyError:
                overlap = 0
            results[f'overlap_{col}'] = overlap
        return pd.Series(results)

    # Apply the compute_overlap function
    overlaps = grouped.apply(compute_overlap).reset_index()

    if bootstrap:
        # If bootstrap is True, pivot and calculate confidence intervals
        overlaps = overlaps.melt(id_vars=['percent', 'bootstrap'], var_name='method', value_name='overlap')
        summary = overlaps.groupby(['percent', 'method']).agg(
            mean_overlap=('overlap', 'mean'),
            lower=('overlap', lambda x: np.quantile(x, 0.025)),
            upper=('overlap', lambda x: np.quantile(x, 0.975))
        ).reset_index()
    else:
        summary = overlaps

    return summary


def bootstrap_strat_2(bootstraps, CATE, CATE_estimates, percentage=np.arange(0.05, 0.95, 0.05), epsilons=[0.05, 0.5, 1, 3, 5], seed = 1):
    np.random.seed(seed)
    seeds = np.random.choice(range(1, 1000000), size=bootstraps, replace=False)
    bootstrap_results = pd.DataFrame()
    bootstrap_results_profit = pd.DataFrame()
    bootstrap_results_overlap = pd.DataFrame()
    for b in range(bootstraps):
        print(seeds[b])
        print("running bootstrap: " + str(b))
        np.random.seed(seeds[b])
        percentage_collection = pd.DataFrame()
        for percent in percentage:
            collection = protect_CATEs(percent, CATE_estimates, CATE_estimates, len(CATE_estimates), epsilons)
            collection['percent'] = percent
            percentage_collection = pd.concat([percentage_collection, collection], ignore_index=True)
        percentage_collection['bootstrap'] = b
        profit = policy_profit(percentage_collection)
        profit['bootstrap'] = b
        overlap = policy_overlap(percentage_collection)
        overlap['bootstrap'] = b
        bootstrap_results_profit = pd.concat([bootstrap_results_profit, profit], ignore_index=True)
        bootstrap_results_overlap = pd.concat([bootstrap_results_overlap, overlap], ignore_index=True)
    return bootstrap_results_profit, bootstrap_results_overlap
