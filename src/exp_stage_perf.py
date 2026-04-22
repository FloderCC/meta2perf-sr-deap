"""
Script to find a symbolic regression model for MCC inference
using DEAP Genetic Programming (GP)
"""

import logging
import random
import warnings
import math
import operator
import numpy as np
import pandas as pd
import multiprocessing
import operator as op
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, gp

# ============================================================
# CONFIG — CONFIGURATION 2
# ============================================================
random_seed = 42

MAX_NODES = 400
MAX_HEIGHT = 30

max_seconds = None
max_iterations = 50000

target_pop_size = 10000
init_pop_size = 10000
init_pop_method = "oblesa"

tourn_size = 200
cxpb = 0.8
mutpb = 0.8

constants_range = (-100.0, 100.0)

BAD_FITNESS = 1e30
verbose = True

binary_operator_names = ['add', 'sub', 'mul', 'div']
unary_operator_names = ['identity', 'sq2', 'sq3', 'exp', 'log', 'sqrt']

# ============================================================
# LOGGING
# ============================================================
class GreenStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(f"\033[92m{msg}\033[0m\n")
            self.flush()
        except Exception:
            self.handleError(record)

file_handler = logging.FileHandler(
    "logs/exp_stage_perf.txt", mode="w"
)
console_handler = GreenStreamHandler()

def console_filter(record):
    return not getattr(record, "file_only", False)

console_handler.addFilter(console_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
    handlers=[file_handler, console_handler],
)

# ============================================================
# HELPERS
# ============================================================
def train_test_split_regression(X, y, test_size=0.2, b="auto", random_state=42):
    bins = np.histogram_bin_edges(y, bins=b)[:-1]
    groups = np.digitize(y, bins)
    return train_test_split(
        X, y, test_size=test_size, stratify=groups, random_state=random_state
    )

def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))

def is_feasible(ind):
    return len(ind) <= MAX_NODES and ind.height <= MAX_HEIGHT

# ============================================================
# PRIMITIVES
# ============================================================
def div(a, b): return a / b
def identity(a): return a
def sqrt(a): return math.sqrt(a)
def log(a): return math.log(a)
def exp(a): return math.exp(a)
def sq2(a): return a * a
def sq3(a): return a * a * a
def rand_const(): return random.uniform(*constants_range)

# ============================================================
# DATA
# ============================================================
np.random.seed(random_seed)
random.seed(random_seed)
warnings.filterwarnings("ignore")

df = pd.read_csv("./results/meta_dataset.csv")
df = df.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
df = df[df["MCC"] > 0]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split_regression(
    X.values, y.values, test_size=0.2
)

df_train = pd.DataFrame(X_train, columns=X.columns)
df_train["MCC"] = y_train
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test["MCC"] = y_test

X_TRAIN = df_train.iloc[:, :-1].values
Y_TRAIN = df_train.iloc[:, -1].values

X_TEST = df_test.iloc[:, :-1].values
Y_TEST = df_test.iloc[:, -1].values

# ============================================================
# GP SETUP
# ============================================================
def ensure_deap_creators():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

ensure_deap_creators()

pset = gp.PrimitiveSet("MAIN", X.shape[1])
for i in range(X.shape[1]):
    pset.renameArguments(**{f"ARG{i}": f"X{i}"})

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(identity, 1)
pset.addPrimitive(sq2, 1)
pset.addPrimitive(sq3, 1)
pset.addPrimitive(exp, 1)
pset.addPrimitive(log, 1)
pset.addPrimitive(sqrt, 1)
pset.addEphemeralConstant("rand", rand_const)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)

def huber_np(delta, r):
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return 0.5 * quad**2 + delta * lin

def eval_symbreg(ind):
    try:
        func = toolbox.compile(ind)
        preds = np.array([func(*row) for row in X_TRAIN], dtype=float)
        if not np.all(np.isfinite(preds)):
            return (BAD_FITNESS,)
    except Exception:
        return (BAD_FITNESS,)

    loss = np.mean(huber_np(1.0, preds - Y_TRAIN))
    return (loss,)

def selTournamentFeasible(individuals, k):
    feas = [i for i in individuals if is_feasible(i)]
    if not feas:
        feas = individuals
    return tools.selTournament(feas, k, tournsize=tourn_size)

toolbox.register("evaluate", eval_symbreg)
toolbox.register("select", selTournamentFeasible)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(len, MAX_NODES))
toolbox.decorate("mutate", gp.staticLimit(len, MAX_NODES))
toolbox.decorate("mate", gp.staticLimit(op.attrgetter("height"), MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(op.attrgetter("height"), MAX_HEIGHT))

# ============================================================
# MAIN
# ============================================================
def main():
    if init_pop_size > 0:
        # defining the fitness function for initialization
        def fitness_function_for_deap_str(individual_as_str):
            # ensure creators also exist in worker processes
            ensure_deap_creators()
            ind = creator.Individual(gp.PrimitiveTree.from_string(individual_as_str, pset))
            return eval_symbreg(ind)[0]

        from src.utils.ga_init_ext import create_pop_for_deap

        eq_strings, seed_expr_strings = create_pop_for_deap(pop_size=init_pop_size,
                                                            n_vars=df_train.shape[1] - 1,
                                                            const_min=constants_range[0],
                                                            const_max=constants_range[1],
                                                            method=init_pop_method,
                                                            fitness_function_for_deap_str=fitness_function_for_deap_str,
                                                            binary_operators=binary_operator_names,
                                                            unary_operators=unary_operator_names
                                                            )
        # printing the seeded expressions
        logging.info("Seeded initial population expressions:")
        for eq_string in seed_expr_strings:
            logging.info(eq_string)
    else:
        seed_expr_strings = []

    # build initial population
    def is_valid_tree(ind):
        # checks if all subtrees can be accessed without IndexError
        try:
            for i in range(len(ind)):
                _ = ind.searchSubtree(i)
            return True
        except IndexError:
            return False

    initial_population = []
    for expr_str in seed_expr_strings:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(expr_str, pset))
            if is_valid_tree(ind) and is_feasible(ind):
                initial_population.append(ind)
            else:
                logging.warning(f"Discarding invalid or infeasible seed: {expr_str}")
        except Exception as e:
            logging.warning(f"Could not parse seed '{expr_str}': {e}")

    while len(initial_population) < target_pop_size:
        rand_expr = gp.genHalfAndHalf(pset=pset, min_=1, max_=3)
        ind = creator.Individual(rand_expr)
        # only keep feasible individuals in the initial population
        if is_feasible(ind):
            initial_population.append(ind)

    population = initial_population

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    hof = tools.HallOfFame(5)
    pbar = tqdm(total=max_iterations, desc="GP evolution (Config 2)")

    for gen in range(max_iterations):
        invalid = [i for i in population if not i.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        hof.update(population)

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m in offspring:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values

        population[:] = offspring
        pbar.update(1)

    pbar.close()
    pool.close()
    pool.join()

    best = min(hof, key=lambda i: i.fitness.values[0])
    func = toolbox.compile(best)

    def predict(X):
        return np.array([func(*x) for x in X], dtype=float)

    ytr = predict(X_TRAIN)
    yte = predict(df_test.iloc[:, :-1].values)

    train_r2 = r2_score(Y_TRAIN, ytr)
    train_mape = smape_score(Y_TRAIN, ytr)
    train_mae = mean_absolute_error(Y_TRAIN, ytr)
    n_train = len(Y_TRAIN)
    k = df_train.shape[1] - 1
    train_adj_r2 = 1 - (1 - train_r2) * ((n_train - 1) / (n_train - k - 1))

    logging.info(
        f"Train dataset ({len(Y_TRAIN)} rows): "
        f"R^2: {round(train_r2, 3)}, "
        f"Adjusted R^2: {round(train_adj_r2, 3)}, "
        f"sMAPE: {round(train_mape, 3)}, "
        f"MAE: {round(train_mae, 3)}"
    )

    test_r2 = r2_score(Y_TEST, yte)
    test_mape = smape_score(Y_TEST, yte)
    test_mae = mean_absolute_error(Y_TEST, yte)
    n_test = len(Y_TEST)
    k = df_test.shape[1] - 1
    test_adj_r2 = 1 - (1 - test_r2) * ((n_test - 1) / (n_test - k - 1))

    logging.info(
        f"Test dataset ({len(Y_TEST)} rows): "
        f"R^2: {round(test_r2, 3)}, "
        f"Adjusted R^2: {round(test_adj_r2, 3)}, "
        f"sMAPE: {round(test_mape, 3)}, "
        f"MAE: {round(test_mae, 3)}"
    )

    logging.info(f"Best expression: {best}")

    best_ind = best
    logging.info(f"Best individual size: {len(best_ind)} nodes, height={best_ind.height}")
    best_fit = eval_symbreg(best_ind)[0]
    logging.info(f"Best individual (train MSE): {best_ind} -> {best_fit}")

if __name__ == "__main__":
    main()
