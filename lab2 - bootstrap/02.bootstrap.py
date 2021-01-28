import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/albanda/CE888/master/lab2%20-%20bootstrap/customers.csv')
data = df.values.T[1]


# Checking the notes from the lecture, create here your own bootstrap function:
# 1. Sample from the input array x to create an array of samples of shape (n_bootstraps, sample_size)
# Hint: Check the function random.choice() on Numpy
# 2. Calculate and save the mean of the array (this is "data_mean" that is returned by the function)
# 3. Calculate the mean from each bootstrap (i.e., row) and store it.
# (This should be an array of n_bootstraps values)
# 4. Calculate the lower and upper bounds for a 95% CI (hint: check the percentile function on Numpy)
# 5. Return data_mean, and the lower and upper bounds of your interval
def bootstrap_mean(x, sample_size, n_bootstraps):
    ## bootstrap and save the means of samples
    samples_mean = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(x, size=sample_size, replace=True)
        samples_mean.append(np.mean(sample))

    ## calculate the quantiles
    data_mean = np.mean(samples_mean)
    lower, upper = np.percentile(samples_mean, [2.5, 97.5])

    return data_mean, lower, upper


# Call your bootstrap function and plot the results

boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean(data, data.shape[0], i)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot = pd.DataFrame(boots, columns=['Bootstrap Iterations', 'Mean', "Value"])
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 100000)
plt.show()

def bootstrap_mean_ci(sample, sample_size, n_bootstraps, ci):
    ## bootstrap and save the means of samples
    samples_mean = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(sample, size=sample_size, replace=True)
        samples_mean.append(np.mean(sample))

    ## calculate the quantiles
    data_mean = np.mean(samples_mean)
    lower, upper = np.percentile(samples_mean, [50-ci/2, 50+ci/2])

    return data_mean, lower, upper


boots = []
for i in range(100, 50000, 1000):
    boot = bootstrap_mean_ci(data, data.shape[0], i, 80)
    boots.append([i, boot[0], "mean"])
    boots.append([i, boot[1], "lower"])
    boots.append([i, boot[2], "upper"])

df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

sns_plot.axes[0, 0].set_ylim(0,)
sns_plot.axes[0, 0].set_xlim(0, 100000)
plt.show()

#sns_plot.savefig("bootstrap_confidence_80.pdf", bbox_inches='tight')


# Load and visualise the vehicles dataset
# To load the dataset: https://neptune.ai/blog/google-colab-dealing-with-files (check section "Load individual files directly from GitHub")

# Note that the current and new fleets are in different columns and have different lengths, so bear this in mind when you're plotting.
# You can create separate scatterplots for the two fleets, as you would with the histograms, 
# or plot them both in one plot (but not one against the other).
df = pd.read_csv('https://github.com/albanda/CE888/raw/master/lab2%20-%20bootstrap/vehicles.csv')
print(df.columns)

df['idx'] = df.index

## two types
df_scatter = pd.DataFrame(np.concatenate((df.iloc[:,[0, 2]].values, df.iloc[:,[1, 2]].values), axis=0), columns=["fleet", "idx"])
df_scatter_type = pd.DataFrame(np.append(1 * np.ones(df.shape[0]), 2 * np.ones(df.shape[0])), columns=["type"])
df_scatter = pd.concat([df_scatter, df_scatter_type], axis=1)

## scatter
sns.lmplot(x="idx", y="fleet", hue="type", data=df_scatter, fit_reg = False)
plt.show()

## hist
sns.histplot(x="fleet", hue="type", data=df_scatter, kde=False)
plt.show()

# Note: you can add more cells as needed to organise your code and your plots

df_current = df.iloc[:, 0]
df_new = df.iloc[:, 1]
df_new = df_new[~np.isnan(df_new)]
mean_current = np.mean(df_current)
mean_new = np.mean(df_new)
print("mean of current : {}\nmean of new : {}\n".format(mean_current, mean_new))

boot = bootstrap_mean_ci(df_current, df.shape[0], 2000, 95)
print(boot[1], boot[2])
boot = bootstrap_mean_ci(df_new, df.shape[0], 2000, 95)
print(boot[1], boot[2])

# Create your own function for a permutation test here (you will need it for the lab quiz!):
def permut_test(sample1, sample2, n_permutations):
    """
    sample1: 1D array
    sample2: 1D array (note that the size of the two arrays can be different)
    n_permutations: number of permutations to calculate the p-value
    """

    obs = np.mean(sample2) - np.mean(sample1)
    num = 0
    for _ in range(n_permutations):
        s1 = np.random.choice(sample1, size=len(sample1), replace=True)
        s2 = np.random.choice(sample2, size=len(sample2), replace=True)
        if ((np.mean(s2) - np.mean(s1)) - obs) * np.sign(obs) > 0:
            num += 1
    pvalue = num / n_permutations

    return pvalue

permut_test(df_current, df_current, 10000)
