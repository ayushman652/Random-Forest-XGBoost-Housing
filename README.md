# 🌲 Random Forest & XGBoost — California Housing Prediction

<p align="center">

  <b>A machine learning regression project comparing Random Forest and XGBoost on the California Housing dataset.</b>

</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">

  <img src="https://img.shields.io/badge/Scikit--learn-1.7.2-orange?logo=scikit-learn" alt="Scikit-learn">

  <img src="https://img.shields.io/badge/XGBoost-3.2.0-green" alt="XGBoost">

  <img src="https://img.shields.io/badge/ML-Regression-purple" alt="Machine Learning">

</p>

---

## 📌 Overview

This project compares two powerful tree-based ensemble learning algorithms for a regression problem:

* 🌲 **Random Forest Regressor**

* ⚡ **XGBoost Regressor**

Both models are trained on the **California Housing dataset** and evaluated using:

* Mean Squared Error (MSE)

* R² score

* Training time

* Prediction time

* Actual vs. predicted visualizations

The main purpose of this project is not simply to train two models, but to understand **how ensemble learning works, how Random Forest and XGBoost differ internally, and how their performance compares under the same experimental setup**.

---

# 🎯 Objectives

This project demonstrates:

* Ensemble learning for regression

* Decision-tree-based prediction

* Random Forest bagging

* Bootstrap sampling

* Random feature selection

* Variance reduction

* XGBoost gradient boosting

* Additive boosting models

* Gradients and Hessians

* Second-order optimization

* L2 regularization

* Tree complexity control

* Model evaluation

* Training and inference time comparison

* Modular machine-learning project architecture

---

# 📊 Dataset

## California Housing Dataset

The project uses the **California Housing dataset**.

The dataset contains:

* **20,640 observations**

* **8 numerical predictive features**

* **1 continuous target**

* **No missing attribute values**

The dataset was derived from the **1990 U.S. Census**, with observations representing California census block groups.

The target represents the **median house value for a district in units of $100,000**.

### Official sources

* [Scikit-learn California Housing documentation]$https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

* [Scikit-learn Real World Datasets]\(https://scikit-learn.org/stable/datasets/real_world.html)

* [Original StatLib dataset]\(https://lib.stat.cmu.edu/datasets/houses.zip)

---

## 🧾 Dataset Features

\| Feature      | Description                              |

\| ------------ | ---------------------------------------- |

\| `MedInc`     | Median income in the block group         |

\| `HouseAge`   | Median house age                         |

\| `AveRooms`   | Average number of rooms per household    |

\| `AveBedrms`  | Average number of bedrooms per household |

\| `Population` | Block-group population                   |

\| `AveOccup`   | Average number of household members      |

\| `Latitude`   | Geographic latitude                      |

\| `Longitude`  | Geographic longitude                     |

\| `Target`     | Median house value in $100,000 units     |

For example:

```text

Target = 3.00

```

corresponds to approximately:

```text

3 × $100,000 = $300,000

```

> **Important:** This dataset is historical and is based on 1990 census data. Therefore, this project should be considered a machine-learning learning/benchmark project rather than a modern real-estate valuation system.

---

# 🧠 Why Ensemble Learning?

A single decision tree can model nonlinear relationships extremely well, but a sufficiently complex tree can become sensitive to the training data.

This is the **variance** problem.

Ensemble learning addresses this by combining multiple models.

The two algorithms in this project use fundamentally different ensemble strategies:

```text

                    Ensemble Learning

                           │

             ┌─────────────┴─────────────┐

             │                           │

       Random Forest                  XGBoost

             │                           │

          Bagging                   Boosting

             │                           │

     Independent trees        Sequential trees

             │                           │

          Average             Correct errors

```

---

# 🌲 Random Forest

## What is Random Forest?

Random Forest is an ensemble of decision trees.

Instead of relying on one decision tree, it trains many trees and combines their predictions.

For regression:

```text

Tree 1 ──┐

Tree 2 ──┤

Tree 3 ──┤

Tree 4 ──┼──→ Average → Final Prediction

   ...   ┤

Tree 100 ┘

```

The central idea is:

> **Build many diverse trees and aggregate their predictions to obtain a more stable model.**

---

## 🔀 How Random Forest Creates Diversity

Random Forest primarily introduces diversity through two mechanisms.

### 1. Bootstrap Sampling

Each tree is trained using a bootstrap sample of the training data.

Bootstrap sampling means sampling **with replacement**.

```text

Original Training Data

          │

          ├── Bootstrap Sample → Tree 1

          ├── Bootstrap Sample → Tree 2

          ├── Bootstrap Sample → Tree 3

          ├── Bootstrap Sample → Tree 4

          └── ...

```

Consequently, different trees see different versions of the training data.

---

### 2. Random Feature Selection

At tree splits, only a subset of available features is considered.

This prevents every tree from repeatedly relying on exactly the same strongest features.

The result is a collection of less-correlated trees.

```text

Different samples

        +

Different feature subsets

        ↓

Diverse trees

        ↓

Aggregation

        ↓

Reduced variance

```

---

# 📐 Random Forest Mathematics

Suppose the forest contains `B` trees.

Each tree produces:

$$
\hat f_1(x),\hat f_2(x),...,\hat f_B(x)
$$

For regression, the final prediction is:

$$
\boxed{

\hat f_{RF}(x)

=

\frac{1}{B}

\sum_{b=1}^{B}

\hat f_b(x)

}
$$

where:

* \\(B\$ = number of trees

* \$\hat f_b(x)\$ = prediction from tree \$b\$

* \$\hat f_{RF}(x)\$ = final Random Forest prediction

Averaging helps reduce the effect of individual-tree errors.

---

# 🌳 Regression Trees Inside Random Forest

Each Random Forest tree is still a decision tree.

For regression, a common split criterion is Mean Squared Error:

$$
MSE=

\frac{1}{n}

\sum_{i=1}^{n}

(y_i-\bar y)^2
$$

where:

* \$n\$ = number of observations

* \$y_i\$ = actual target

* \$\bar y\$ = mean target in the node

A candidate split divides observations into left and right nodes.

The weighted split impurity is:

$$
MSE_{split}

=

\frac{n_L}{n}MSE_L

+

\frac{n_R}{n}MSE_R
$$

The tree prefers splits that reduce impurity.

---

# ⚙️ Random Forest Configuration

This project uses:

```python

RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

```

### Important parameters

\| Parameter           | Purpose                            |

\| ------------------- | ---------------------------------- |

\| `n_estimators`      | Number of trees                    |

\| `max_depth`         | Maximum tree depth                 |

\| `max_features`      | Features considered at splits      |

\| `min_samples_split` | Minimum samples required to split  |

\| `min_samples_leaf`  | Minimum samples in a leaf          |

\| `bootstrap`         | Whether bootstrap samples are used |

---

# ⚡ XGBoost

## What is XGBoost?

**XGBoost** stands for **Extreme Gradient Boosting**.

It is an optimized implementation of gradient-boosted decision trees.

Official documentation:

[XGBoost Documentation]$https://xgboost.readthedocs.io/)

Unlike Random Forest, XGBoost does not primarily build independent trees and average them.

Instead:

> **Trees are added sequentially, with every new tree attempting to improve the existing ensemble.**

---

# 🔄 How XGBoost Works

The simplified process is:

```text

Initial prediction

       ↓

Calculate loss

       ↓

Calculate gradient + Hessian

       ↓

Build a new tree

       ↓

Update predictions

       ↓

Calculate remaining error

       ↓

Build another tree

       ↓

Repeat

```

This process continues for a specified number of boosting rounds.

---

# 📐 XGBoost Additive Model

At boosting iteration \\(t\$:

$$
\boxed{

\hat y_i^{(t)}

=

\hat y_i^{(t-1)}

+

f_t(x_i)

}
$$

where:

* \$\hat y_i^{(t-1)}\$ = previous prediction

* \$f_t(x_i)\$ = contribution from the new tree

* \$\hat y_i^{(t)}\$ = updated prediction

After \$K\$ trees:

$$
\boxed{

\hat y(x)

=

\hat y^{(0)}

+

\sum_{k=1}^{K}f_k(x)

}
$$

The final prediction is therefore the accumulated contribution of all trees.

---

# 🎯 XGBoost Objective Function

XGBoost minimizes an objective containing two components:

$$
\boxed{

Objective = Loss + Regularization

}
$$

More formally:

$$
Obj

=

\sum_{i=1}^{n}

L(y_i,\hat y_i)

+

\sum_{k=1}^{K}

\Omega(f_k)
$$

where:

* \$L\$ = training loss

* \$y_i\$ = actual target

* \$\hat y_i\$ = predicted target

* \$f_k\$ = tree \$k\$

* \$\Omega(f_k)\$ = complexity penalty

The loss measures prediction error.

The regularization term discourages unnecessarily complex models.

---

# 🧮 Gradients and Hessians

XGBoost uses first- and second-order information about the loss.

## Gradient

$$
g_i

=

\frac{\partial L(y_i,\hat y_i)}

{\partial\hat y_i}
$$

The gradient tells the model the direction in which the prediction should move to reduce the loss.

## Hessian

$$
h_i

=

\frac{\partial^2 L(y_i,\hat y_i)}

{\partial\hat y_i^2}
$$

The Hessian describes the curvature of the loss.

---

## Squared-Error Example

For squared-error loss:

$$
L(y,\hat y)

=

\frac{1}{2}(y-\hat y)^2
$$

the gradient becomes:

$$
g_i=\hat y_i-y_i
$$

and the Hessian becomes:

$$
h_i=1
$$

This explains the common intuition that boosting "corrects residuals."

However, technically, XGBoost uses **gradients and Hessians**, rather than simply fitting raw residuals.

---

# 🔬 Second-Order Taylor Approximation

XGBoost uses a second-order approximation of the loss:

$$
Obj^{(t)}

\approx

\sum_i

\left[

g_i f_t(x_i)

+

\frac{1}{2}

h_i f_t(x_i)^2

\right]

+

\Omega(f_t)
$$

This allows XGBoost to optimize the next tree using both gradient and curvature information.

---

# 🛡️ XGBoost Regularization

One of the important characteristics of XGBoost is that model complexity is explicitly included in the objective.

The tree complexity term is:

$$
\boxed{

\Omega(f)

=

\gamma T

+

\frac{1}{2}

\lambda

\sum_{j=1}^{T}w_j^2

}
$$

where:

* \$T\$ = number of leaves

* \$w_j\$ = value/score of leaf \$j\$

* \$\lambda\$ = L2 regularization strength

* \$\gamma\$ = penalty associated with adding leaves

---

# 🧱 L2 Regularization

The L2 component is:

$$
\frac{1}{2}

\lambda

\sum_{j=1}^{T}w_j^2
$$

The purpose is to discourage excessively large leaf values.

As \$\lambda\$ increases:

```text

Higher λ

   ↓

Stronger penalty

   ↓

More conservative leaf values

   ↓

Lower model complexity

```

In XGBoost, this regularization applies to **tree leaf weights/scores**.

---

# 🍃 Optimal Leaf Weight

For a leaf \$j\$, define:

$$
G_j=\sum_{i\in I_j}g_i
$$

and:

$$
H_j=\sum_{i\in I_j}h_i
$$

The optimal leaf weight for a fixed tree structure is:

$$
\boxed{

w_j^*

=

-\frac{G_j}{H_j+\lambda}

}
$$

This equation demonstrates the interaction between:

* Gradient information

* Hessian information

* L2 regularization

---

# ✂️ XGBoost Split Gain

XGBoost also evaluates whether a proposed split improves the objective enough to justify the additional complexity.

A simplified gain expression is:

$$
Gain

=

\frac{1}{2}

\left[

\frac{G_L^2}{H_L+\lambda}

+

\frac{G_R^2}{H_R+\lambda}

-

\frac{(G_L+G_R)^2}

{H_L+H_R+\lambda}

\right]

-

\gamma
$$

where:

* \$G_L,G_R\$ = gradient sums

* \$H_L,H_R\$ = Hessian sums

* \$\lambda\$ = L2 regularization

* \$\gamma\$ = split penalty

A split is useful only when its improvement justifies the additional complexity.

---

# ⚙️ XGBoost Configuration

This project uses:

```python

XGBRegressor(

    n_estimators=100,

    random_state=42

)

```

Important XGBoost parameters include:

\| Parameter          | Purpose                            |

\| ------------------ | ---------------------------------- |

\| `n_estimators`     | Number of boosting trees           |

\| `learning_rate`    | Contribution of each tree          |

\| `max_depth`        | Maximum tree depth                 |

\| `min_child_weight` | Minimum child weight               |

\| `subsample`        | Fraction of rows sampled           |

\| `colsample_bytree` | Fraction of features sampled       |

\| `gamma`            | Minimum loss reduction for a split |

\| `reg_lambda`       | L2 regularization                  |

\| `reg_alpha`        | L1 regularization                  |

---

# 🆚 Random Forest vs XGBoost

\| Aspect              | Random Forest                               | XGBoost                                 |

\| ------------------- | ------------------------------------------- | --------------------------------------- |

\| Ensemble strategy   | Bagging                                     | Boosting                                |

\| Tree relationship   | Mostly independent                          | Sequential                              |

\| Main idea           | Reduce variance                             | Iteratively reduce loss                 |

\| Training            | Trees can be trained independently          | Trees depend on previous predictions    |

\| Prediction          | Average tree predictions                    | Sum tree contributions                  |

\| Sampling            | Bootstrap observations + feature randomness | Optional row/feature subsampling        |

\| Error correction    | Indirect through aggregation                | Directly improves previous model        |

\| Regularization      | Tree/ensemble parameters                    | Explicit regularization + tree controls |

\| Main characteristic | Stability                                   | Iterative optimization                  |

### Mental model

```text

RANDOM FOREST

────────────────────────

Training Data

     │

     ├── Bootstrap → Tree 1 ──┐

     ├── Bootstrap → Tree 2 ──┤

     ├── Bootstrap → Tree 3 ──┤

     │          ...            ├──→ Average

     └── Bootstrap → Tree 100 ─┘



XGBOOST

────────────────────────

Initial Prediction

        │

        ↓

      Tree 1

        │

        ↓

Updated Prediction

        │

        ↓

      Tree 2

        │

        ↓

Updated Prediction

        │

        ↓

      Tree 3

        │

       ...

        ↓

Final Prediction

```

---

# 🔬 Why No Feature Scaling?

This project intentionally does **not** use `StandardScaler`.

Random Forest and XGBoost are tree-based algorithms.

Their decisions are based on threshold splits such as:

```text

MedInc < 4.5

```

rather than distances between observations.

Scaling changes the numerical values of the features and thresholds, but does not fundamentally change the ordering used by threshold-based tree splits.

Therefore, feature scaling is unnecessary for this project.

---

# 🏗️ Project Architecture

```text

08-Random-Forest-XGBoost-Housing/

│

├── main.py

│

├── src/

│   ├── __init__.py

│   ├── config.py

│   ├── data_loader.py

│   ├── preprocessing.py

│   ├── trainer.py

│   ├── evaluator.py

│   └── visualizer.py

│

├── outputs/

│   ├── random_forest_actual_vs_predicted.png

│   └── xgboost_actual_vs_predicted.png

│

├── requirements.txt

├── .gitignore

└── README.md

```

### Module Responsibilities

\| Module             | Responsibility                                       |

\| ------------------ | ---------------------------------------------------- |

\| `config.py`        | Paths and model configuration                        |

\| `data_loader.py`   | Dataset loading                                      |

\| `preprocessing.py` | Feature/target separation and train/test split       |

\| `trainer.py`       | Model training and training-time measurement         |

\| `evaluator.py`     | Predictions, MSE, R² and prediction-time measurement |

\| `visualizer.py`    | Actual-vs-predicted plots                            |

\| `main.py`          | End-to-end orchestration                             |

---

# 🔄 Project Workflow

```text

California Housing Dataset

          │

          ↓

     Load Dataset

          │

          ↓

   Basic Inspection

          │

          ↓

      Separate X/y

          │

          ↓

    Train/Test Split

       80% / 20%

          │

     ┌────┴────┐

     ↓         ↓

Random Forest XGBoost

     ↓         ↓

Predictions  Predictions

     └────┬────┘

          ↓

    Model Evaluation

          │

     ┌────┼────┐

     ↓    ↓    ↓

    MSE   R²  Timing

          │

          ↓

Actual vs Predicted

```

---

# 📈 Evaluation Metrics

## Mean Squared Error

$$
MSE=

\frac{1}{n}

\sum_{i=1}^{n}

(y_i-\hat y_i)^2
$$

Lower MSE indicates smaller squared prediction errors.

Because errors are squared, large errors receive greater weight.

---

## R² Score

$$
R^2=

1-

\frac{

\sum_i(y_i-\hat y_i)^2

}{

\sum_i(y_i-\bar y)^2

}
$$

Interpretation:

* `R² = 1` → perfect predictions

* `R² = 0` → equivalent to predicting the test-set mean

* `R² < 0` → worse than the mean baseline

---

# ⏱️ Computational Performance

The project measures both training and prediction time.

### Training time

Measured around:

```python

model.fit(X_train, y_train)

```

### Prediction time

Measured around:

```python

model.predict(X_test)

```

This allows the comparison to consider not only predictive performance but also computational cost.

---

# 🧪 Experimental Configuration

Both models use the same train/test split:

```python

test_size=0.2

random_state=42

```

### Random Forest

```python

RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

```

### XGBoost

```python

XGBRegressor(

    n_estimators=100,

    random_state=42

)

```

Using the same split and evaluation procedure makes the comparison consistent.

---

# 📊 Results

Results from this project's run:

\| Metric          | Random Forest |  XGBoost |

\| --------------- | ------------: | -------: |

\| Training Time   |     10.4109 s | 2.4220 s |

\| Prediction Time |      0.1886 s | 0.0060 s |

\| MSE             |        0.2554 |   0.2226 |

\| R²              |        0.8051 |   0.8301 |

For this particular train/test split and parameter configuration:

* XGBoost produced a lower MSE.

* XGBoost produced a higher R².

* XGBoost had lower measured training time.

* XGBoost had lower measured prediction time.

These results describe **this experiment** and should not be interpreted as a universal claim that XGBoost will always outperform Random Forest.

---

# 📉 Visualizations

The project generates Actual vs Predicted plots for both models.

The ideal prediction line is:

$$
y=x
$$

Interpretation:

* Points close to the line → predictions close to actual values

* Points above the line → overprediction

* Points below the line → underprediction

* Larger distance from the line → larger prediction error

The plots also include `±1` standard deviation reference lines based on the test-target distribution.

These lines are **visual reference lines only**. They are not confidence intervals, prediction intervals, or model error bounds.

---

# ⚙️ Installation

## 1. Clone the repository

```bash

git clone https://github.com/ayushman652/Random-Forest-XGBoost-Housing.git

cd Random-Forest-XGBoost-Housing

```

## 2. Activate the environment

```powershell

.\\.venv\Scripts\Activate.ps1

```

## 3. Install dependencies

```powershell

python -m pip install -r requirements.txt

```

Pinned dependencies:

```text

numpy==2.2.0

pandas==2.3.3

matplotlib==3.10.9

scikit-learn==1.7.2

xgboost==3.2.0

```

---

# 📥 Dataset Setup

The project expects the dataset in the shared workspace:

```text

AI-Engineering-workspace/

│

├── datasets/

│   └── housing/

│       └── housing.csv

│

└── 08-Random-Forest-XGBoost-Housing/

```

The CSV should contain the California Housing features and a target column named:

```text

Target

```

If generating the dataset using scikit-learn:

```python

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

df = housing.frame

df = df.rename(columns={"MedHouseVal": "Target"})

df.to_csv("housing.csv", index=False)

```

---

# ▶️ Running the Project

From the workspace root:

```powershell

python .\08-Random-Forest-XGBoost-Housing\main.py

```

Or from inside the project:

```powershell

python main.py

```

Example output:

```text

Dataset shape: (20640, 9)

Missing values: 0

Training samples: 16512

Testing samples: 4128

Random Forest

Training time: 10.4109 seconds

Prediction time: 0.1886 seconds

MSE: 0.2554

R²: 0.8051

XGBoost

Training time: 2.4220 seconds

Prediction time: 0.0060 seconds

MSE: 0.2226

R²: 0.8301

```

---

# 🧠 Key Concepts Learned

### Machine Learning

* Supervised learning

* Regression

* Train/test splitting

* Model evaluation

* Bias-variance tradeoff

### Decision Trees

* Recursive splitting

* Regression-tree impurity

* Tree depth

* Leaf predictions

### Random Forest

* Ensemble learning

* Bagging

* Bootstrap sampling

* Random feature selection

* Variance reduction

* Prediction aggregation

### XGBoost

* Gradient boosting

* Additive models

* Loss minimization

* Gradients

* Hessians

* Second-order Taylor approximation

* L1/L2 regularization

* Tree complexity

* Split gain

* Learning rate

* Boosting rounds

### Engineering

* Modular Python

* Configuration management

* Type hints

* Docstrings

* Reproducible experiments

* Runtime measurement

* Visualization

* Git/GitHub organization

---

# 💼 Real-World Applications

Tree-based ensemble models are widely useful for structured/tabular data.

### Finance

* Credit risk

* Fraud detection

* Default prediction

* Risk scoring

### Retail

* Demand forecasting

* Customer behavior prediction

* Customer segmentation

### Healthcare

* Risk prediction

* Patient outcome prediction

* Medical tabular prediction

### Industry

* Predictive maintenance

* Failure prediction

* Quality control

### Real Estate

* Property valuation

* Price estimation

* Market analysis

---

# ⚠️ Limitations

## Dataset limitations

The California Housing dataset is historical and based on 1990 census data.

Therefore:

* It is not a current housing-market dataset.

* Economic conditions have changed.

* Housing markets differ across locations and time.

* The model should not be treated as a production property valuation system.

## Model limitations

Both Random Forest and XGBoost can overfit when model complexity is not properly controlled.

Performance depends on:

* Dataset quality

* Feature quality

* Hyperparameters

* Train/test split

* Data distribution

* Evaluation methodology

The reported metrics represent one experimental configuration and are not guaranteed future performance.

---

# 🎓 Interview Questions

## Random Forest

### 1. What is Random Forest?

An ensemble of decision trees trained using randomized observations and feature selection, whose predictions are aggregated.

### 2. Why does Random Forest reduce variance?

Averaging predictions from diverse trees reduces the influence of individual-tree fluctuations.

### 3. What is bootstrap sampling?

Sampling observations with replacement to create different training sets for individual trees.

### 4. Why randomly select features?

To reduce correlation between trees and increase ensemble diversity.

### 5. Does Random Forest require feature scaling?

Generally no, because tree-based splits are threshold-based.

---

## XGBoost

### 6. What is XGBoost?

An optimized gradient boosting framework that builds trees sequentially to minimize an objective function.

### 7. Random Forest vs XGBoost?

Random Forest primarily uses bagging and independently trained trees, while XGBoost uses sequential boosting.

### 8. What is a gradient?

The first derivative of the loss with respect to the current prediction.

### 9. What is a Hessian?

The second derivative of the loss with respect to the prediction.

### 10. Why does XGBoost use regularization?

To control model complexity and reduce overfitting.

### 11. What does `learning_rate` do?

It controls how strongly each new tree contributes to the final model.

### 12. What does `n_estimators` mean?

The number of trees/boosting rounds used by the model.

---

# 📚 References

### California Housing

* [Scikit-learn — California Housing Dataset]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

* [Scikit-learn — Real World Datasets]\(https://scikit-learn.org/stable/datasets/real_world.html)

* [StatLib — California Housing Dataset]\(https://lib.stat.cmu.edu/datasets/houses.zip)

### XGBoost

* [XGBoost Documentation]\(https://xgboost.readthedocs.io/)

* [Introduction to Boosted Trees]\(https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

* [XGBoost Parameters]\(https://xgboost.readthedocs.io/en/latest/parameter.html)

* [XGBoost Parameter Tuning]\(https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)

---

# 👨‍💻 Author

**Ayushman Singh**

Computer Science Engineering Student | AI & Machine Learning

GitHub: [@ayushman652]\(https://github.com/ayushman652)

---

<p align="center">

  <b>Part of my AI Engineering learning journey — building ML projects from theory to modular implementation.</b>

</p>