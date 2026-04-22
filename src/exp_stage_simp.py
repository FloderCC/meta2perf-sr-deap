"""
Script to find a symbolic regression model for MCC inference
using DEAP Genetic Programming (GP)
CONFIGURATION 2: high-risk / high-reward probe
"""

import logging
import random
import warnings
import math
import operator
from collections import Counter

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

# for the first evolution runs, we want to keep the search space small to get quick feedback on whether the setup is promising
MAX_NODES = 405
MAX_HEIGHT = 42

# for the second evolution runs, adjusted to accept the solutions of the first runs and explore more complex models
max_depth = MAX_HEIGHT + 1 # in the current init data, Max height is 42, Max length is 796
max_tokens = 450

max_iterations = 50

target_pop_size = init_pop_size = 100

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
# PARSIMONY — DISABLED (IMPORTANT)
# ============================================================
TOTAL_VARS = 19

LAMBDA_VARS = LAMBDA_LEN = LAMBDA_HEIGHT = LAMBDA_EXP = LAMBDA_LOG = LAMBDA_SQRT = LAMBDA_SQ2 = LAMBDA_SQ3 = 0

# LAMBDA_VARS = 5e-4
# LAMBDA_LEN = 1e-4
LAMBDA_HEIGHT = 1e-3#
#

# LAMBDA_OPS = 1e-4
LAMBDA_EXP = 5e-2#
LAMBDA_LOG = 10e-2#
LAMBDA_SQRT = 5e-2#
LAMBDA_SQ2 = 10e-4#
LAMBDA_SQ3 = 10e-4#

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

console_handler = GreenStreamHandler()

def console_filter(record):
    return not getattr(record, "file_only", False)

console_handler.addFilter(console_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
    handlers=[console_handler],
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
    if len(ind) > MAX_NODES:
        logging.warning(f"Individual discarded for exceeding max nodes: {len(ind)} > {MAX_NODES}")
        return False
    if ind.height > MAX_HEIGHT:
        logging.warning(f"Individual discarded for exceeding max height: {ind.height} > {MAX_HEIGHT}")
        return False
    return True

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

    # Base accuracy objective
    huber_loss = np.mean(huber_np(1.0, preds - Y_TRAIN))

    # Distinct variables used
    used_vars = {
        node.name
        for node in ind
        if isinstance(node, gp.Terminal)
        and hasattr(node, "name")
        and node.name.startswith("ARG")
    }
    num_used_vars = len(used_vars)

    # Expensive operators
    num_logs = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "log"
    )
    num_exps = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "exp"
    )
    num_sqrts = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sqrt"
    )
    num_sq2 = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sq2"
    )
    num_sq3 = sum(
        1 for node in ind
        if isinstance(node, gp.Primitive) and node.name == "sq3"
    )

    # Total parsimony penalty
    penalty = (
        LAMBDA_VARS * num_used_vars
        + LAMBDA_LEN * len(ind)
        + LAMBDA_HEIGHT * ind.height
        + LAMBDA_EXP * num_exps
        + LAMBDA_LOG * num_logs
        + LAMBDA_SQRT * num_sqrts
        + LAMBDA_SQ2 * num_sq2
        + LAMBDA_SQ3 * num_sq3
    )

    # ensuring huber_loss >= than 0.01277170689388824
    TARGET_HUBER = 0.01177170689388824
    LAMBDA_THRESHOLD = 100.0  # tune this

    # Penalize individuals whose Huber loss is above the target
    threshold_penalty = 0.0
    if huber_loss > TARGET_HUBER:
        threshold_penalty = LAMBDA_THRESHOLD * (huber_loss - TARGET_HUBER)

    return (huber_loss + penalty + threshold_penalty,)

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

        logging.info("Creating seeded initial population using method: %s", init_pop_method)

        logging.info(" - Creating the oblesa population seeds...")

        # seeding the best results from previous optimization runs, to explore the neighborhood of these solutions and find better ones
        oblesa_population = [
            # deap 0.743, 403 nodes, height=29, 19v (Atari)
            "sqrt(div(exp(div(add(div(div(sq2(X13),X2),sub(12.988635321657412,X10)),add(div(12.988635321657412,sq2(X2)),add(log(add(div(add(div(sq2(exp(sq3(sub(exp(X0),sqrt(exp(X6)))))),exp(X1)),add(mul(sq3(sq2(X1)),add(mul(mul(X3,sq3(X18)),X7),sqrt(X9))),sq2(mul(sqrt(X7),sq2(X4))))),div(X2,X11)),sq3(sq2(sq2(X13))))),X9))),69.5722268035895)),log(add(add(add(div(X11,sqrt(add(X3,mul(mul(X5,sq3(sq3(sq3(X0)))),div(div(X2,add(sq2(sub(X5,X6)),X9)),X12))))),exp(div(div(exp(div(exp(sqrt(sqrt(X4))),69.5722268035895)),log(add(div(div(X8,X2),sq2(sq2(sq2(sub(X6,X12))))),sqrt(exp(add(div(exp(sub(sqrt(exp(X14)),sq2(add(sq2(log(sq2(X0))),add(X13,div(X9,X1)))))),sub(X1,sqrt(X5))),add(sq3(X14),X13))))))),log(add(sq2(exp(div(X12,74.290676027194))),add(add(div(log(sq2(X0)),add(X9,X12)),sq3(add(X13,X18))),sq2(div(sq3(X12),mul(sq3(log(div(exp(sq3(X13)),X11))),X2))))))))),exp(div(exp(mul(div(exp(X17),log(add(add(add(exp(div(add(X14,X13),mul(mul(X0,log(add(X6,sq3(sq2(sq2(div(div(X2,add(div(exp(sq2(X6)),X4),div(12.988635321657412,sq3(X18)))),53.56315638711436))))))),X4))),exp(div(exp(div(sq2(X6),add(X4,div(log(X3),sq3(X18))))),mul(mul(X15,sq3(X0)),add(sqrt(X7),X10))))),div(add(exp(sub(X5,X11)),add(log(div(exp(X9),X3)),mul(sqrt(X7),-43.525539787537525))),74.290676027194)),sqrt(div(sq3(exp(sqrt(X18))),mul(div(12.988635321657412,sqrt(add(X3,sq3(X6)))),mul(div(X11,sqrt(X3)),X2))))))),X15)),div(X4,X1)))),sqrt(exp(add(div(X11,X12),sub(add(div(sq2(X1),add(log(add(add(X2,exp(div(sq3(add(X1,X11)),mul(mul(sq3(X11),add(X9,X7)),log(mul(X0,X3)))))),sq3(exp(sqrt(sq2(sub(add(div(50.641811862548366,X4),X5),mul(X11,sqrt(div(sq2(X13),X2)))))))))),div(X8,exp(exp(X0))))),X0),log(add(add(sqrt(X18),sq3(sub(X0,X7))),sqrt(exp(add(div(log(add(add(exp(sq3(X14)),exp(div(sub(93.1968146134748,mul(X11,X13)),X4))),sq3(mul(X18,sq2(X11))))),add(sq2(X15),div(sqrt(div(X6,sq3(sq3(X0)))),mul(X15,exp(div(X10,X12)))))),add(div(X11,log(sq3(X12))),sub(div(X5,log(X3)),add(div(sq2(sub(X4,sq2(sq2(X15)))),mul(X0,X2)),X9))))))))))))))))",
            # deap 0.706, 402 nodes, height=26, 14v (Ultron)
            "exp(div(sq2(log(add(mul(mul(add(X0, div(exp(sub(X16, sqrt(X18))), exp(sq3(X14)))), X15), add(sq2(mul(add(add(div(sq2(log(add(add(add(add(add(add(add(sq3(sq3(sqrt(sq3(sqrt(X5))))), sq3(div(sq2(exp(sqrt(X11))), exp(X5)))), exp(sqrt(div(sq2(-29.82262832759703), X2)))), sq3(exp(sub(div(X4, X1), sqrt(X2))))), -29.89765080088243), sq2(div(X3, sub(log(sq2(sq3(sq3(X0)))), div(sq2(27.46734532160076), X2))))), -82.38944297987814), div(X2, X4)))), -82.38944297987814), X16), add(add(div(sq2(log(add(mul(X16, add(add(add(add(-86.7853000381759, sq3(div(sq2(-35.726733713674875), X2))), X3), sq3(exp(sq2(log(X11))))), sq2(add(X3, div(mul(X3, X11), sq3(sub(log(55.94919114088265), X4))))))), div(X2, sq2(sq3(X11)))))), -82.38944297987814), sqrt(X11)), add(add(div(sq2(log(add(add(add(sq2(div(div(X3, sq3(X0)), sq3(sq3(X0)))), sq3(exp(sqrt(sq3(X0))))), sq2(div(X3, add(X11, -1.3991216727321216)))), sq3(X5)))), -82.38944297987814), sqrt(X11)), X11))), mul(add(log(36.97338688989558), X0), X17))), sq2(add(add(div(sq2(log(add(mul(X2, add(add(mul(mul(X16, div(X7, X11)), sq3(sq3(X0))), mul(X2, div(mul(sqrt(mul(X5, sq3(X7))), exp(sq2(log(X11)))), exp(sq3(X18))))), exp(-35.726733713674875))), sq2(add(sq3(sub(sq2(2.000179614175096), X4)), mul(sq3(X4), X7)))))), -86.7853000381759), sqrt(div(X11, sqrt(X3)))), add(add(div(sq2(log(add(mul(sq2(sq3(X18)), add(add(sq3(div(exp(sq2(log(X11))), exp(sq3(X18)))), sq3(div(exp(sq2(log(X11))), X2))), sq2(add(69.97988533419635, X3)))), exp(sub(exp(X14), sq3(add(X0, div(log(38.06533499866654), sub(10.799516381383697, X5))))))))), -82.38944297987814), log(X11)), add(add(div(sq2(log(add(sq2(div(X3, sub(sqrt(68.46135391761126), sq3(X5)))), mul(X18, add(add(sq2(div(sq2(sq3(X11)), sub(div(sq3(div(X1, X11)), X7), div(X2, sq2(exp(sqrt(X11))))))), exp(sub(X11, div(X2, X13)))), sq3(sq3(X11))))))), -82.38944297987814), log(add(add(exp(sub(X1, div(X15, exp(sub(10.799516381383697, X5))))), exp(div(X15, log(X2)))), exp(sub(log(sq3(exp(X18))), sq3(X0)))))), X11)))))), add(sq2(log(add(add(add(exp(sub(mul(X7, 14.827578477170249), sq3(X0))), exp(sub(sub(add(X1, 10.799516381383697), div(sq2(sub(X11, sq3(sq3(add(X0, X15))))), exp(sub(X1, log(X2))))), exp(sub(sq3(log(sq3(log(X4)))), sq2(X2)))))), exp(div(X4, sq3(log(sq3(log(X4))))))), exp(sub(X1, sq3(log(sq3(log(X4))))))))), sq2(X17))))), -82.38944297987814))",
            # deap 0.701, 404 nodes, height=30
            "exp(div(-39.01410102844238, div(mul(sqrt(sub(sub(X12, mul(X16, X0)), -15.293491333575517)), div(mul(sqrt(sub(div(div(mul(div(mul(X10, X10), X2), sq3(X17)), log(sqrt(add(exp(X5), exp(X18))))), X2), -15.293491333575517)), div(mul(sqrt(sub(div(sub(sq3(X13), sub(sub(sub(log(X3), X5), sub(mul(add(X4, mul(X7, X11)), X18), X1)), sub(sqrt(mul(X9, exp(sub(log(add(X8, mul(X8, X5))), div(mul(X10, div(mul(log(X3), sub(div(mul(sq3(exp(X13)), sq3(X1)), X2), -15.293491333575517)), sub(identity(div(X1, add(X7, mul(mul(sq3(mul(sq3(X7), 93.99339561937131)), sq3(X8)), sq3(X13))))), -15.293491333575517))), sq3(exp(X13))))))), sub(X16, X5)))), sqrt(X11)), -15.293491333575517)), div(mul(sqrt(sub(div(sub(div(mul(X10, div(mul(sub(div(mul(sq3(X13), mul(sub(add(X3, X9), X11), sqrt(sqrt(X9)))), X1), -15.293491333575517), sub(sqrt(mul(X18, X3)), -15.293491333575517)), div(X11, X0))), div(X11, sqrt(add(X5, sqrt(X18))))), sub(sq3(X15), sq2(X6))), X2), -15.293491333575517)), div(mul(sqrt(sub(div(sub(sqrt(sq3(sq3(X14))), sub(exp(X15), X5)), sqrt(X11)), -15.293491333575517)), div(mul(sqrt(sub(div(sub(sq3(X13), sub(sub(mul(X0, sqrt(X1)), sq3(X18)), div(sub(sub(X12, X4), sub(sq3(X6), sq3(add(X0, X17)))), X2))), sqrt(sub(sub(sqrt(mul(sq3(exp(X5)), sq2(mul(sq3(X7), X2)))), -15.293491333575517), log(div(X8, 10.41783661058524))))), -15.293491333575517)), div(mul(sqrt(sub(div(sub(div(mul(X13, mul(mul(sq3(sq3(sq3(X0))), mul(X17, X3)), X9)), sqrt(X2)), -15.293491333575517), X2), -15.293491333575517)), div(mul(sqrt(sub(div(mul(X10, X10), X2), -15.293491333575517)), 59.62879719122009), sqrt(sub(div(mul(X10, sub(sub(add(add(X12, exp(X17)), exp(X17)), sub(sqrt(mul(X15, sqrt(X8))), X5)), -15.293491333575517)), X2), -15.293491333575517)))), sqrt(sub(sqrt(mul(sq3(X15), mul(X15, mul(X15, mul(sq2(X15), mul(X15, mul(X15, mul(X15, X10)))))))), -15.293491333575517)))), sqrt(sub(sqrt(mul(mul(X17, sub(sub(mul(div(mul(10.41783661058524, X10), sqrt(X3)), 10.41783661058524), mul(X12, X9)), -39.01410102844238)), mul(X15, sub(sqrt(sqrt(mul(X2, sq3(sq2(X0))))), mul(X0, -4.540613602662674))))), -15.293491333575517)))), sqrt(sub(div(sq3(sq3(sqrt(X14))), add(X4, mul(sq2(mul(sq2(X7), X2)), X2))), -15.293491333575517)))), sqrt(sub(div(X12, add(log(add(sqrt(X4), sq2(X5))), mul(sq2(X7), mul(X18, X3)))), -15.293491333575517)))), sqrt(sub(sqrt(mul(X12, mul(identity(mul(X9, X7)), mul(35.143684266931956, X9)))), -15.293491333575517)))), sqrt(sub(div(sq3(sq2(X17)), sqrt(X8)), -15.293491333575517)))), sqrt(X1))))",
            # deap 0.709, 403 nodes, height=28, 16v
            "exp(div(add(add(sq2(log(X1)), div(add(-14.843658750339884, add(sq3(log(add(div(sq3(log(add(div(57.063579852299625, X12), X11))), div(57.063579852299625, add(div(sq3(log(add(div(60.44738444112048, X12), X11))), div(57.063579852299625, add(sq3(log(add(add(div(sq2(-14.843658750339884), div(mul(X0, sq2(log(sq2(log(X1))))), div(add(div(div(60.44738444112048, exp(mul(X13, 60.44738444112048))), sq3(log(sqrt(X12)))), X12), div(exp(sq2(X0)), add(sq3(mul(X1, X7)), X15))))), exp(sq2(div(X6, X2)))), sq3(log(X2))))), div(sq3(log(add(sq3(log(sq3(X0))), add(sq3(log(X2)), mul(X11, sq3(sq3(X0))))))), div(exp(sq2(X6)), add(div(sq3(log(add(div(mul(mul(X18, mul(X1, sq3(X0))), X8), X2), X4))), div(57.063579852299625, mul(exp(sq2(X0)), 60.44738444112048))), sq2(sq2(identity(sq2(X6)))))))))), sq3(log(add(div(sq3(log(add(div(sq3(X10), div(div(sq3(X11), X2), X2)), exp(X13)))), div(identity(div(div(X2, div(div(60.44738444112048, X11), div(sq2(X11), sq3(log(sqrt(X3)))))), sqrt(X3))), add(sq3(X0), mul(X18, sq3(X6))))), div(sq3(sq3(X0)), 9.01898900076246))))))), sub(sq3(X0), X11)))), sq3(log(log(sqrt(sqrt(sqrt(sqrt(sqrt(log(add(exp(div(div(sq3(9.01898900076246), X2), -14.843658750339884)), X4)))))))))))), -14.843658750339884)), add(sq3(X15), add(div(X4, -14.843658750339884), add(div(60.44738444112048, add(div(sq3(log(sq3(exp(X18)))), 60.44738444112048), sq3(log(add(div(sq3(log(add(add(div(add(div(X3, sq2(X1)), div(X1, div(div(sq3(X0), div(sq3(log(sq3(log(add(X2, log(X4)))))), add(X3, X10))), X1))), div(div(X2, div(add(X11, sq2(X12)), div(sq2(sq2(log(mul(X1, X0)))), div(sq3(log(add(X4, log(X2)))), div(sq2(exp(exp(X17))), sqrt(X3)))))), exp(X18))), add(div(div(X4, add(exp(mul(X13, X6)), sqrt(exp(X17)))), sqrt(X3)), exp(X13))), X11))), div(X11, add(exp(div(exp(X17), -14.843658750339884)), sq3(sq3(div(sq3(div(9.01898900076246, X4)), add(X4, X12))))))), add(sq3(log(exp(sqrt(X14)))), sq3(X13))))))), add(div(add(add(sub(log(X2), mul(X1, sq3(X0))), div(sq3(log(add(mul(X15, add(div(57.063579852299625, X12), X11)), X11))), div(60.44738444112048, add(sq3(log(add(div(sq3(log(add(sqrt(X8), X3))), sq2(sq3(exp(X15)))), div(add(sqrt(X3), X10), div(X2, div(add(X10, X14), div(sq2(X11), 7.255765247918504))))))), sq2(log(mul(div(X8, exp(X17)), sq3(X0)))))))), log(X2)), -14.843658750339884), sq3(log(add(div(57.063579852299625, X12), X11)))))))), -82.38944297987814))",
        ]

        logging.info(f" - Generating the initial population seeds using the provided {len(oblesa_population)} seeds.")

        eq_strings, seed_expr_strings = create_pop_for_deap(
                                                            pop_size=init_pop_size,
                                                            n_vars=df_train.shape[1] - 1,
                                                            const_min=constants_range[0],
                                                            const_max=constants_range[1],
                                                            method=init_pop_method,
                                                            fitness_function_for_deap_str=fitness_function_for_deap_str,
                                                            binary_operators=binary_operator_names,
                                                            unary_operators=unary_operator_names,
                                                            max_depth=max_depth,
                                                            max_tokens=max_tokens,
                                                            oblesa_population=oblesa_population
                                                            )
        # printing the seeded expressions
        logging.info(f"------ Seeded initial population expressions. Total seeds: {len(seed_expr_strings)} ------")
        for eq_string in seed_expr_strings:
            logging.info(eq_string)

        # logging a breakline for better readability
        logging.info("-"*80)
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

    max_len = 0
    max_height = 0

    for expr_str in seed_expr_strings:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(expr_str, pset))
            if is_valid_tree(ind) and is_feasible(ind):
                initial_population.append(ind)

                # update max_len and max_height
                if len(ind) > max_len:
                    max_len = len(ind)
                if ind.height > max_height:
                    max_height = ind.height
            else:
                raise ValueError(f"Invalid or infeasible seed: {expr_str}")
        except Exception as e:
            raise ValueError(f"Error processing seed expression: {expr_str}. Error: {e}")

    print(f"Generated initial population with {len(initial_population)} valid seeds. Max length: {max_len}, Max height: {max_height}") # Max length: 796, Max height: 42

    if len(initial_population) != target_pop_size:
        # rise a error if the number of valid seeds is less than the target population size
        raise ValueError(f"Number of valid seeds ({len(initial_population)}) does not match target population size ({target_pop_size}). Adjust the seeds or the target population size.")

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
                try:
                    toolbox.mutate(m)
                    del m.fitness.values
                except Exception as e:
                    logging.warning(f"Mutation failed for individual {m}. Error: {e}")
                    # remove the individual from the offspring to avoid issues in the next generation
                    offspring.remove(m)

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

    def assess_individual(ind) -> list:
        # return for a given individual a list of:
        # [train_r2, test_r2, train_huber_loss, #nodes, height, count_vars, #log, #exp, #sqrt, #sq2, #sq3]

        # getting the performance metrics
        try:
            func_ind = toolbox.compile(ind)

            def predict_ind(X):
                return np.array([func_ind(*x) for x in X], dtype=float)

            ytr = predict_ind(X_TRAIN)
            yte = predict_ind(X_TEST)

            if not np.all(np.isfinite(ytr)) or not np.all(np.isfinite(yte)):
                train_r2 = test_r2 = train_huber_loss = None
            else:
                train_r2 = r2_score(Y_TRAIN, ytr)
                test_r2 = r2_score(Y_TEST, yte)
                train_huber_loss = float(np.mean(huber_np(1.0, ytr - Y_TRAIN)))

        except Exception:
            train_r2 = test_r2 = train_huber_loss = None

        # getting the complexity metrics
        num_nodes = len(ind)
        height = ind.height

        var_counts = Counter(
            node.name
            for node in ind
            if isinstance(node, gp.Terminal)
            and hasattr(node, "name")
            and node.name.startswith("ARG")
        )
        count_vars = ";".join(
            f"{count}"
            for _, count in sorted(var_counts.items(), key=lambda x: int(x[0][3:]))
        )

        num_logs = sum(1 for node in ind if isinstance(node, gp.Primitive) and node.name == "log")
        num_exps = sum(1 for node in ind if isinstance(node, gp.Primitive) and node.name == "exp")
        num_sqrts = sum(1 for node in ind if isinstance(node, gp.Primitive) and node.name == "sqrt")
        num_sq2 = sum(1 for node in ind if isinstance(node, gp.Primitive) and node.name == "sq2")
        num_sq3 = sum(1 for node in ind if isinstance(node, gp.Primitive) and node.name == "sq3")

        return [
            train_r2,
            test_r2,
            train_huber_loss,
            num_nodes,
            height,
            count_vars,
            num_logs,
            num_exps,
            num_sqrts,
            num_sq2,
            num_sq3,
        ]

    # saving to a csv the performance and complexity metrics of all individuals in the hall of fame
    hof_data = []
    for ind in hof:
        ind_metrics = assess_individual(ind)
        hof_data.append(ind_metrics + [str(ind)])

    hof_df = pd.DataFrame(
        hof_data,
        columns=[
            "train_r2",
            "test_r2",
            "train_huber_loss",
            "num_nodes",
            "height",
            "count_vars",
            "num_logs",
            "num_exps",
            "num_sqrts",
            "num_sq2",
            "num_sq3",
            "expression",
        ],
    )
    # sorting by test_r2 in descending order
    hof_df = hof_df.sort_values(by="test_r2", ascending=False)
    hof_df.to_csv("results/exp_stage_sim_hall_of_fame.csv", index=False)

    # saving to a csv the performance and complexity metrics of all individuals in the last population
    population_data = []
    for ind in population:
        ind_metrics = assess_individual(ind)
        population_data.append(ind_metrics + [str(ind)])

    population_df = pd.DataFrame(
        population_data,
        columns=[
            "train_r2",
            "test_r2",
            "train_huber_loss",
            "num_nodes",
            "height",
            "count_vars",
            "num_logs",
            "num_exps",
            "num_sqrts",
            "num_sq2",
            "num_sq3",
            "expression",
        ],
    )
    # sorting by test_r2 in descending order
    population_df = population_df.sort_values(by="test_r2", ascending=False)
    population_df.to_csv("results_exp_stage_create_inf_model_deap_v2_2_best_models_oblesa3_simp_all_final_population.csv", index=False)


if __name__ == "__main__":
    main()
