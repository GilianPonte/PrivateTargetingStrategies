import numpy as np
import pandas as pd

def data_simulation(n):
    # Generate random covariates
    covariate_1 = np.random.normal(size=n)
    covariate_2 = np.random.normal(size=n)
    covariate_3 = np.random.normal(size=n)
    covariate_4 = np.random.normal(size=n)
    covariate_5 = np.random.normal(size=n)
    covariate_6 = np.random.normal(size=n)

    # Create the design matrix
    x = np.column_stack((covariate_1, covariate_2, covariate_3, covariate_4, covariate_5, covariate_6))

    # Generate treatment indicator
    p = 0.5
    w = np.random.binomial(1, p, size=n)

    # Generate outcomes and treatment effects
    m = np.maximum(0, x[:, 0] + x[:, 1], x[:, 2]) + np.maximum(0, x[:, 3] + x[:, 4])
    tau = x[:, 0] + np.log(1 + np.exp(x[:, 1])) ** 2
    mu1 = m + tau / 2
    mu0 = m - tau / 2
    y = w * mu1 + (1 - w) * mu0 + 0.5 * np.random.normal(size=n)

    # Create a Pandas DataFrame
    df = pd.DataFrame({'covariate_1': covariate_1,
                       'covariate_2': covariate_2,
                       'covariate_3': covariate_3,
                       'covariate_4': covariate_4,
                       'covariate_5': covariate_5,
                       'covariate_6': covariate_6,
                       'w': w,
                       'm': m,
                       'tau': tau,
                       'mu1': mu1,
                       'mu0': mu0,
                       'y': y})
    

    return df
