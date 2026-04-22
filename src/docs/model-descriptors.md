# Model descriptors and complexity

The following notation is adopted throughout the document:

- **n**: number of training samples  
- **m**: number of input features  
- **c**: number of output classes  

We report three complementary views:

1. approximate number of processing units per model (architectural scale),
2. training and inference complexities (Big-O),
3. other qualitative descriptors (regularization, robustness, representational capacity),
4. and finally the **encoding scheme** used in the meta-dataset.

---

## 1. Approximate number of processing units

**Table 1 – Approximate number of processing units (`proc_units`) per model.**

| Model | Formula | Explanation |
|-------|---------|-------------|
| LR | `c × (m + 1)` | Linear classifiers with one weight vector and one bias per class. |
| Ridge | `c × (m + 1)` | Linear classifiers with one weight vector and one bias per class. |
| SGD | `c × (m + 1)` | Linear classifiers with one weight vector and one bias per class. |
| Perceptron | `c × (m + 1)` | Linear classifiers with one weight vector and one bias per class. |
| LinearSVC | `c × (m + 1)` | Linear classifiers with one weight vector and one bias per class. |
| DT | `2^{max_depth} - 1` | Each node performs a split decision. |
| ET | `n_estimators × (2^{max_depth} - 1)` | Tree ensembles; each estimator follows the same structure as DT. |
| RF | `n_estimators × (2^{max_depth} - 1)` | Tree ensembles; each estimator follows the same structure as DT. |
| ETs | `n_estimators × (2^{max_depth} - 1)` | Tree ensembles; each estimator follows the same structure as DT. |
| AB | `n_estimators × 3` | Each weak learner (typically DT with `max_depth = 1`) is an independent unit. |
| HGB | `max_iter × (2^{max_depth} - 1)` | Each iteration adds one tree to the ensemble. |
| Bagging | `n_estimators × 2^{160}` | Each estimator runs independently; default DT(`max_depth = 160`). |
| GaussianNB | `c × m × 2` | Learns mean and variance per feature and class. |
| BernoulliNB | `c × (m + 1)` | Learns one probability per feature and class. |
| MLP | `Σ hidden_layer_sizes` | Each neuron is a processing unit. |
| DNN | `Σ hidden_layer_sizes` | Each neuron is a processing unit. |

---

## 2. Training and inference complexities

**Table 2 – Training and inference complexities of the evaluated models regarding `n` (samples) and `m` (features).**

| Model | `train_complexity` | `inf_complexity` |
|-------|--------------------|------------------|
| LR | `O(n × m)` | `O(m)` |
| Ridge | `O(n × m²)` | `O(m)` |
| SGD | `O(n × m)` | `O(m)` |
| Perceptron | `O(n × m)` | `O(m)` |
| LinearSVC | `O(n × m)` | `O(m)` |
| DT | `O(n × m × log m)` | `O(log m)` |
| ET | `O(n × m × log m)` | `O(log m)` |
| RF | `O(n × m × log m)` | `O(log m)` |
| ETs | `O(n × m × log m)` | `O(log m)` |
| AB | `O(n × m)` | `O(m)` |
| HGB | `O(m)` | `O(log m)` |
| Bagging | `O(n × m × log m)` | `O(log m)` |
| GaussianNB | `O(n × m)` | `O(m)` |
| BernoulliNB | `O(n × m)` | `O(m)` |
| MLP | `O(n × m)` | `O(n × m)` |
| DNN | `O(n × m)` | `O(n × m)` |


---

## 3. Other model descriptors

**Table 3 – Other model descriptors.**

| Model | `model_type` | `regularization` | `rob_outliers` | `repr_capacity` |
|-------|---------------|------------------|----------------|-----------------|
| LR | Linear | L1, L2, or ElasticNet (depends on `penalty`) | No | Low |
| Ridge | Linear | L2 (if `alpha ≠ 0.1`), None (if `alpha = 0.1`) | No | Low |
| SGD | Linear | L2 (default) | Medium | Medium |
| Perceptron | Linear | L1, L2, or ElasticNet (depends on `penalty`) | Medium | Medium |
| LinearSVC | Linear | L1, L2, or ElasticNet (depends on `penalty`) | No | Medium |
| DT | Tree-based | None | Yes | Medium |
| ET | Tree-based | None | Yes | Medium |
| RF | Ensemble | None | Yes | High |
| ETs | Ensemble | None | Yes | High |
| AB | Ensemble | None | No | Medium |
| HGB | Ensemble | L2 (if `l2_regularization ≠ 0.0`), None (if `l2_regularization = 0.0`) | Yes | High |
| Bagging | Ensemble | None | Yes | Medium |
| GaussianNB | Probabilistic | None | No | Low |
| BernoulliNB | Probabilistic | L2 (if `alpha ≠ 0.1`), None (if `alpha = 0.1`) | No | Low |
| MLP | Neural Network | L2 (if `alpha ≠ 0.0001`), None (if `alpha = 0.0001`) | No | Very High |
| DNN | Neural Network | L2 (if `alpha ≠ 0.0001`), None (if `alpha = 0.0001`) | No | Very High |

---

## 4. Encoding scheme

For consistency and reproducibility, all qualitative descriptors above were **numerically encoded** in the meta-dataset as follows.

### 4.1 Model type

- Linear → **0**  
- Probabilistic → **1**  
- Tree-based → **2**  
- Ensemble → **3**  
- Neural Network → **4**

### 4.2 Training complexity (function of n and m)

- `O(m)` → **0**  
- `O(n × m)` → **1**  
- `O(n × m × log(m))` → **2**  
- `O(n × m²)` → **3**

### 4.3 Inference complexity (function of n and m)

- `O(log(m))` → **0**  
- `O(m)` → **1**  
- `O(n × m)` → **2**

### 4.4 Regularization strength

- None → **0**  
- Laplace → **1**  
- L1 → **2**  
- L2 → **3**  
- ElasticNet → **4**

### 4.5 Robustness to outliers

- No → **0**  
- Medium → **1**  
- Yes → **2**

### 4.6 Representational capacity

- Low → **0**  
- Medium → **1**  
- High → **2**  
- Very High → **3**
