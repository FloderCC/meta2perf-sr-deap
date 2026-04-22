# results based on dataset features


import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def plot_regression_results(y_true, y_pred, title, target='', set_name='train'):
    """
    Plots true vs. predicted values for regression results, with R² and MAE metrics.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values
    - title: str, plot title

    Saves the figure as 'regression_plot.pdf' in high-resolution vector format.
    """
    # Compute metrics
    # r2 = r2_score(y_true, y_pred)
    # mae = mean_absolute_error(y_true, y_pred)

    # print(min(y_pred))

    # Create plot
    plt.figure(figsize=(5, 5))
    plt.scatter(
        y_true,
        y_pred,
        s=16,
        alpha=0.3,
        edgecolors='none',
        label='Predictions',
        # color='#054f82'
        color='#054f82'
    )
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
        linestyle='--',
        color='#c42525',
        label='Ideal Fit (y = x)')

    # limit the x and y axis to the range of the data
    x_margin = 0.03 * (max(y_true) - min(y_true))
    y_margin = 0.03 * (max(y_pred) - min(y_pred))  # Similarly for y_pred if needed

    plt.xlim(min(y_true) - x_margin, max(y_true) + x_margin)
    plt.ylim(min(y_pred)- y_margin, max(y_pred) + y_margin)
    # plt.ylim(0, 1)

    # Labels and title
    plt.xlabel(f'True {target} Values', fontsize=12)
    plt.ylabel(f'Predicted {target} Values', fontsize=12)
    # plt.title(title, fontsize=14)

    # Annotate metrics
    # plt.text(0.05, 0.95,
    #          f'$R^2$: {r2:.2f}\nMAE: {mae:.2f}',
    #          transform=plt.gca().transAxes,
    #          fontsize=10,
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Final touches
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig(f'plots/pred_vs_true_stage2.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
    # print(f'y = {y}')
    if isinstance(b, str):
        bins = np.histogram_bin_edges(y, bins=b)
        # remove the last index (end point)
        bins = bins[:-1]
    elif isinstance(b, int):
        bins = np.linspace(min(y), max(y), num=b, endpoint=False)
    else:
        raise Exception(f'Undefined bins {b}')

    # print(f'Bins: {bins}')
    groups = np.digitize(y, bins)
    # print(f'Group: {groups}')
    return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings("ignore")

df_meta = pd.read_csv('./results/meta_dataset.csv')
# df_meta = pd.read_csv('./results/meta_dataset_v2.csv')
df_meta = df_meta.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
target_column = 'MCC'

# now for the filtered version
df_meta = df_meta[df_meta['MCC'] > 0]

# # Get columns names with NaN values
# cols_with_nan = df_meta.columns[df_meta.isnull().any()].tolist()
# print(f"Columns with NaN values: {cols_with_nan}")
# # Remove columns with NaN values
# df_meta = df_meta.drop(columns=cols_with_nan)

# Defining the regression score
def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))

# split into train and test sets
X = df_meta.iloc[:, :-1]  # Features
y = df_meta.iloc[:, -1]  # Target variable
# Apply the function
X_train, X_test, y_train, y_test = train_test_split_regression(X, y, test_size=0.2, b='auto')
# Recombine into train/test DataFrames
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train[target_column] = y_train
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test[target_column] = y_test

# /// Model test begin \\\ #

# --- Symbolic regression expression ---

import numpy as np

def predict(df): # pasar al latex
    df = df.copy()

    # Rename columns to x0, x1, ...
    original_cols = list(df.columns)
    for i, col in enumerate(original_cols):
        df = df.rename(columns={col: f'x{i}'})
        print(f'x{i}', '->', col, 'min: ', df[f'x{i}'].min(), 'max: ', df[f'x{i}'].max())

    X0  = df['x0']
    X1  = df['x1']
    X2  = df['x2']
    X3  = df['x3']
    X4  = df['x4']
    X5  = df['x5']
    X6  = df['x6']
    X7  = df['x7']
    X8  = df['x8']
    X9  = df['x9']
    X10 = df['x10']
    X11 = df['x11']
    X12 = df['x12']
    X13 = df['x13']
    X14 = df['x14']
    X15 = df['x15']
    X17 = df['x17']
    X18 = df['x18']

    c1 = 12.988635321657412
    c2 = 74.290676027194
    c3 = 69.5722268035895
    c4 = 50.641811862548366
    c5 = 93.1968146134748

    # -------------------------
    # A
    # exact from the original inline equation
    # -------------------------
    A = (
            (
                    (X13 ** 2) / (X2 * (c1 - X10))
                    + c1 / (X2 ** 2)
                    + np.log(
                (X11 / X2) * (
                        np.exp(2 * (np.exp(X0) - np.exp(X6 / 2)) ** 3 - X1)
                        + X1 ** 6 * ((X3 * X12 * X7) / c2 + np.sqrt(X9))
                        + X7 * X4 ** 4
                )
                + X13 ** 12
            )
                    + X9
            ) / c3
    )

    # -------------------------
    # B
    # -------------------------
    B = X11 / np.sqrt(
        X3 + (X5 * X0 ** 27 * X2) / (X12 * (((X5 - X6) ** 2) + X9))
    )

    # -------------------------
    # C
    # -------------------------
    C = np.exp(
        (
                np.exp(np.exp(np.sqrt(np.sqrt(X4))) / c3)
                / np.log(
            (X8 / X2) / ((X6 - X12) ** 8)
            + np.exp(
                0.5 * (
                        (X3 * np.exp(-(c3 ** 2))) / (X1 - X0)
                        + X14 ** 3
                        + X13
                )
            )
        )
        )
        / np.log(
            np.exp((2 * X12) / c2)
            + np.log(X0 ** 2 + X18) / (X9 + X12)
            + (X13 + X18) ** 3
            + (X12 / (X2 + X18)) ** 2
        )
    )

    # -------------------------
    # D
    # -------------------------
    D = np.exp(
        (X1 / X4) * np.exp(
            (X15 * np.exp(X17))
            / np.log(
                np.exp(X14 / (X4 ** 2))
                + np.exp(
                    np.exp(X6 ** 2 / (X4 + X7 / (X2 + X18)))
                    / (X15 * X0 ** 3 * (np.sqrt(X7) + X10))
                )
                + (np.exp(X5 - X11) + X5) / c2
                + np.sqrt(
                    (X1 * np.sqrt(X3) * np.exp(3 * np.sqrt(X18)))
                    / (c1 * X11 * X2)
                )
            )
        )
    )

    # -------------------------
    # E
    # -------------------------
    E = np.exp(
        0.5 * (
                X11 / X12
                + (
                        X1 ** 2 / (
                        np.log(
                            X2
                            + np.exp(
                                ((X1 + X11) ** 3)
                                / (X11 ** 3 * (X9 + X7) * np.log(X0 * X3))
                            )
                            + np.exp(
                                3 * np.abs(
                                    (c4 / X4)
                                    + X5
                                    - X11 * np.sqrt((X13 ** 2) / X2)
                                )
                            )
                        )
                        + X8 / np.exp(np.exp(X0))
                )
                )
                + X0
                - np.log(
            np.sqrt(X18)
            + (X0 - X7) ** 3
            + np.exp(
                0.5 * (
                        np.log(
                            np.exp(X14 ** 3)
                            + np.exp((c5 - X11 * X13) / X4)
                            + (X18 * (X11 ** 2)) ** 3
                        ) / X12
                        + X11 / np.log(X12 ** 3)
                        + X5 / np.log(X3)
                        - (
                                ((X4 - X15 ** 4) ** 2) / (X0 * X2)
                                + X9
                        )
                )
            )
        )
        )
    )

    # -------------------------
    # Final prediction
    # -------------------------
    yhat = np.sqrt(
        np.exp(A) / np.log(B + C + D + E)
    )

    return yhat

# --- Rename columns to match expression variable names ---
# Converts 'nr_attr' -> 'nrattr', etc.
df_train.columns = [col.lower().replace('_', '') for col in df_train.columns]
df_test.columns = [col.lower().replace('_', '') for col in df_test.columns]

# inferencing the train dataset
y_pred_train = predict(df_train)



# \\\ Model test end/// #

y_train = df_train.iloc[:, -1].values  # Target variable

# Evaluating training set
train_r2 = r2_score(y_train, y_pred_train)
train_mape = smape_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)

n = len(y_train)  # Total samples
k = df_train.shape[1] - 1  # Number of predictors
train_adj_r2 = 1 - (1 - train_r2) * ((n - 1) / (n - k - 1))

# inferencing the test dataset
y_pred_test = predict(df_test)

y_test = df_test.iloc[:, -1].values  # Target variable
test_r2 = r2_score(y_test, y_pred_test)
test_mape = smape_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

n = len(y_test)
test_adj_r2 = 1 - (1 - test_r2) * ((n - 1) / (n - k - 1))

# Logging results
print(f"Final results:")
print(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
print(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")

# plot_regression_results(y_train, y_pred_train, 'Regression Predictions vs. True Values (Train Set)', 'MCC', 'train')
plot_regression_results(y_test, y_pred_test, 'Regression Predictions vs. True Values (Test Set)', 'MCC', 'test')

# Train dataset (20668 rows): R^2: 0.378, Adjusted R^2: 0.377, sMAPE: 0.512, MAE: 0.206
# Test dataset (5167 rows): R^2: 0.361, Adjusted R^2: 0.36, sMAPE: 0.515, MAE: 0.209

# exporting a csv with the real and predicted values
df_test['y_pred'] = y_pred_test
df_test['y_true'] = y_test

#remove the columns that are not y_pred or y_true
df_test = df_test[['y_true', 'y_pred']]

df_test.to_csv('plots/regression_results_test.csv', index=False)

# doing the same with train
df_train['y_pred'] = y_pred_train
df_train['y_true'] = y_train
#remove the columns that are not y_pred or y_true
df_train = df_train[['y_true', 'y_pred']]
df_train.to_csv('plots/regression_results_train.csv', index=False)