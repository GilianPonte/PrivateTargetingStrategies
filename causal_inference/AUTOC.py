import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def TOC(phi, data):
    # Calculate the threshold based on the quantile
    threshold = data['tau'].quantile(1 - phi)
    filtered_data = data[data['tau'] >= threshold]
    
    if len(filtered_data) < 1:
        return 0  # Return a neutral value if no data above the threshold
    
    # Calculate the mean differences
    mean_treated = filtered_data[filtered_data['w'] == 1]['y'].mean()
    mean_control = filtered_data[filtered_data['w'] == 0]['y'].mean()
    overall_mean_diff = data[data['w'] == 1]['y'].mean() - data[data['w'] == 0]['y'].mean()
    
    toc_value = mean_treated - mean_control - overall_mean_diff
    return toc_value

def monte_carlo_integration(func, lower, upper, data, samples):
    random_samples = np.random.uniform(lower, upper, samples)
    function_values = np.array([func(phi, data) for phi in random_samples])
    return function_values.mean() * (upper - lower)

def calculate_and_plot_TOC_mc(data, samples):
    phi_values = np.random.uniform(0.000001, 1, samples)
    toc_values = np.array([TOC(phi, data) for phi in phi_values])
    
    # Create a dataframe for plotting
    toc_data = pd.DataFrame({'phi': phi_values, 'toc': toc_values})
    
    # Estimate AUTOC using Monte Carlo integration
    AUTOC_value = monte_carlo_integration(TOC, 0, 1, data, samples)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='phi', y='toc', data=toc_data, color='blue', alpha=0.4)
    plt.title('Monte Carlo Integration of TOC Curve')
    plt.xlabel('Quantile (phi)')
    plt.ylabel('TOC Value')
    plt.axhline(y=toc_values.min(), color='red', linestyle='--')
    plt.text(0.5, toc_values.min(), f'AUTOC: {round(AUTOC_value, 3)}', color='red')
    
    plt.show()
    return AUTOC_value
