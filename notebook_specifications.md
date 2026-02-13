
# Credit Default Classification: A Financial Professional's Workflow for Risk Assessment

## Case Study: Optimizing Loan Risk Management at CapitalFlow Bank

**Persona:** Alex Chen, Senior Credit Risk Analyst at CapitalFlow Bank

**Organization:** CapitalFlow Bank, a mid-sized financial institution specializing in consumer lending.

**Scenario:** Alex is tasked with evaluating and improving CapitalFlow Bank's credit default prediction models. The current models are outdated, leading to suboptimal loan origination decisions, inaccurate risk-based pricing, and potential issues with regulatory capital requirements. Alex needs to develop and compare two models—a traditional Logistic Regression scorecard and a more flexible Random Forest classifier—to estimate the Probability of Default (PD) for new loan applicants. The goal is to provide well-calibrated PD estimates and robust insights into default drivers, explicitly accounting for critical financial considerations like class imbalance and asymmetric misclassification costs. This analysis will directly inform the bank's lending strategy, balance sheet provisioning, and compliance with regulatory frameworks like Basel III/IV and IFRS 9.

---

### **1. Setting the Stage: Assessing Credit Risk for 'CapitalFlow Bank'**

Alex, a Senior Credit Risk Analyst at CapitalFlow Bank, begins a crucial project to enhance the bank's loan default prediction capabilities. Accurate Probability of Default (PD) models are paramount for CapitalFlow Bank, directly influencing which loans are approved, the interest rates charged, and the capital reserves held against potential losses. This project isn't just about prediction accuracy; it's about making financially sound, compliant, and transparent lending decisions. Alex will navigate challenges like class imbalance (defaults are rare events) and the asymmetric costs of misclassifying a good loan versus missing a true default.

```python
# Install required libraries
!pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Import required dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE # Used for handling class imbalance

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styles for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 # Adjust for higher resolution plots
```

---

### **2. Unpacking Loan Data: From Raw Records to Actionable Features**

Alex's first task is to transform raw LendingClub loan data into a clean, structured dataset suitable for modeling. This involves filtering irrelevant entries, defining the binary target variable (default or not), selecting pertinent features, and preprocessing them. A critical step is recognizing and addressing the inherent class imbalance in credit data, where defaults are much less frequent than fully paid loans. Furthermore, Alex must be vigilant about potential data leakage, particularly concerning features like `int_rate` which can inadvertently reveal future information.

The target variable, $y_i$, for each loan $i$, is defined as:
$$ y_i = \begin{cases} 1 & \text{if loan } i \text{ defaulted (charged off or defaulted)} \\ 0 & \text{if loan } i \text{ was fully repaid} \end{cases} $$

```python
# Load the dataset
try:
    df = pd.read_csv('lending_club_loans.csv')
except FileNotFoundError:
    print("Error: 'lending_club_loans.csv' not found. Please ensure the dataset is in the correct directory.")
    exit() # Exit if the file is not found

# Filter to completed loans only (remove current/in-grace period)
completed_statuses = ['Fully Paid', 'Charged Off', 'Default']
df = df[df['loan_status'].isin(completed_statuses)].copy()

# Create binary target variable: 1 for default/charged off, 0 for fully paid
df['default'] = (df['loan_status'] != 'Fully Paid').astype(int)

print(f"Initial default rate: {df['default'].mean():.2%}")
print(f"Initial dataset size: {len(df):,} loans")

# Select features based on domain knowledge and available data
# Note: 'int_rate' is included for this exercise to illustrate leakage concepts,
# though in a real-world scenario, its inclusion would require careful consideration
# due to its nature as an outcome of a prior risk assessment.
feature_cols = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'fico_range_low', 'fico_range_high', 'open_acc', 'revol_bal', 'revol_util',
    'total_acc', 'delinq_2yrs', 'pub_rec', 'emp_length', 'home_ownership',
    'purpose', 'term'
]

# Ensure all selected feature columns exist in the DataFrame
existing_feature_cols = [col for col in feature_cols if col in df.columns]
missing_feature_cols = [col for col in feature_cols if col not in df.columns]

if missing_feature_cols:
    print(f"\nWarning: The following feature columns were not found and will be skipped: {missing_feature_cols}")
feature_cols = existing_feature_cols

df_filtered = df[feature_cols + ['default']].copy()

# Feature Engineering: Use average FICO score
if 'fico_range_low' in df_filtered.columns and 'fico_range_high' in df_filtered.columns:
    df_filtered['fico_score'] = (df_filtered['fico_range_low'] + df_filtered['fico_range_high']) / 2
    df_filtered = df_filtered.drop(columns=['fico_range_low', 'fico_range_high'])
    feature_cols.remove('fico_range_low')
    feature_cols.remove('fico_range_high')
    feature_cols.append('fico_score')


# Preprocess 'term' by mapping ' 36 months' to 0 and ' 60 months' to 1
# Remove leading/trailing spaces from 'term' values
df_filtered['term'] = df_filtered['term'].str.strip()
df_filtered['term'] = df_filtered['term'].map({'36 months': 0, '60 months': 1})

# Handle missing values for 'emp_length' and 'revol_util' using median imputation
# For 'emp_length', first convert to numerical (e.g., years)
emp_length_map = {
    '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10
}
# Apply map, then convert to numeric, coercing errors to NaN
df_filtered['emp_length'] = df_filtered['emp_length'].map(emp_length_map).fillna(np.nan)
df_filtered['emp_length'].fillna(df_filtered['emp_length'].median(), inplace=True)

df_filtered['revol_util'].fillna(df_filtered['revol_util'].median(), inplace=True)
df_filtered['dti'].fillna(df_filtered['dti'].median(), inplace=True) # DTI can also have NAs

# One-hot encode categorical features: 'home_ownership', 'purpose'
categorical_features = ['home_ownership', 'purpose']
df_filtered = pd.get_dummies(df_filtered, columns=categorical_features, drop_first=True, dtype=int)

# Separate features (X) and target (y)
X = df_filtered.drop('default', axis=1)
y = df_filtered['default']

# Align X columns after one-hot encoding, feature_cols needs to be updated.
# This makes sure that the scaled X_test has the same columns as X_train after fitting the scaler.
final_feature_cols = X.columns.tolist()

# Split the data into training and testing sets (75/25 split) using stratified sampling
# Stratified sampling ensures the proportion of default cases is similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining set default rate: {y_train.mean():.2%}")
print(f"Testing set default rate: {y_test.mean():.2%}")
print(f"Training set size: {len(X_train):,}")
print(f"Testing set size: {len(X_test):,}")

# Scale numerical features for Logistic Regression
# Random Forest is not sensitive to feature scaling, but Logistic Regression is.
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames with column names for interpretability later
X_train_sc_df = pd.DataFrame(X_train_sc, columns=final_feature_cols, index=X_train.index)
X_test_sc_df = pd.DataFrame(X_test_sc, columns=final_feature_cols, index=X_test.index)
```

### **3. Building the Traditional Scorecard: Logistic Regression for PD Estimation**

Alex starts with Logistic Regression, a robust and widely accepted model in finance due to its interpretability and ease of regulatory compliance. The model estimates the probability of default for each loan applicant. The coefficients of the Logistic Regression model can be directly translated into odds ratios, offering a transparent understanding of how each feature influences the likelihood of default. Given the class imbalance, Alex implements `class_weight='balanced'` to ensure the model doesn't simply favor the majority (non-default) class.

The Logistic Regression model uses a sigmoid function to map a linear combination of features to a probability:
$$ P(y_i = 1 | \mathbf{x}_i) = \sigma(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i)}} $$
The log-odds, or logit, interpretation is:
$$ \log\left(\frac{P(\text{default})}{1 - P(\text{default})}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p $$
For each feature $j$, the odds ratio ($\text{OR}_j$) is derived from its coefficient $\beta_j$:
$$ \text{OR}_j = e^{\beta_j} $$
An odds ratio of 1.15, for example, means that for a one-unit increase in $x_j$ (holding other features constant), the odds of default are multiplied by 1.15, indicating a 15% increase in default odds.

The coefficients ($\mathbf{\beta}$) are estimated by maximizing the likelihood function, which is equivalent to minimizing the binary cross-entropy (or log loss):
$$ \mathcal{L}(\mathbf{\beta}) = \sum_{i=1}^N [y_i \log P(y_i=1|\mathbf{x}_i) + (1-y_i) \log(1 - P(y_i=1|\mathbf{x}_i))] $$

```python
# Initialize and train Logistic Regression model with balanced class weights
# 'class_weight='balanced'' automatically adjusts weights inversely proportional
# to class frequencies, giving more importance to the minority class (defaults).
log_model = LogisticRegression(
    penalty='l2', # L2 regularization to prevent overfitting
    C=1.0,        # Inverse of regularization strength; smaller values specify stronger regularization
    class_weight='balanced', # Crucial for imbalanced datasets
    solver='lbfgs', # Algorithm to use in the optimization problem
    max_iter=1000,  # Maximum number of iterations for the solver to converge
    random_state=42
)

# Fit the model on the scaled training data
log_model.fit(X_train_sc, y_train)

# Predict probabilities of default on the scaled test set
y_prob_log = log_model.predict_proba(X_test_sc)[:, 1]

# Extract coefficients and calculate odds ratios for interpretation
coefficients = log_model.coef_[0]
odds_ratios = np.exp(coefficients)

# Create a DataFrame for better presentation of coefficients and odds ratios
odds_ratio_df = pd.DataFrame({
    'Feature': final_feature_cols,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios,
    'Direction': ['Risk+' if c > 0 else 'Risk-' for c in coefficients]
}).sort_values('Odds Ratio', ascending=False)

print("Logistic Regression Odds Ratio Table:")
print(odds_ratio_df.round(4).to_markdown(index=False))
```

### **4. Enhancing Prediction with Machine Learning: Random Forest for Complex Risk Patterns**

While Logistic Regression provides excellent interpretability, Alex wants to explore if a more complex model like Random Forest can capture non-linear relationships and intricate feature interactions that might improve predictive power. Random Forests are ensemble models that build multiple decision trees and average their predictions. To further tackle the class imbalance, Alex uses SMOTE (Synthetic Minority Over-sampling Technique) on the training data, generating synthetic samples of the minority class. Hyperparameter tuning with `GridSearchCV` ensures the Random Forest model is optimized for performance.

A Random Forest constructs $B$ decision trees. Each tree is trained on a bootstrap sample of the data, considering a random subset of $m \approx \sqrt{p}$ features at each split (where $p$ is the total number of features). The ensemble prediction for the probability of default given features $\mathbf{x}$ is:
$$ \hat{P}(\text{default} | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{P}_b(\text{default} | \mathbf{x}) $$
where $\hat{P}_b(\text{default} | \mathbf{x})$ is the probability prediction from tree $b$.

When `class_weight='balanced'` is used in tree-based models, the loss function (often Gini impurity or entropy) is modified such that the samples of the minority class (defaults) are given higher weight during tree construction. For a weighted loss function, the contribution of each sample $i$ to the total loss $\mathcal{L}_w$ is adjusted by its class weight $w_{y_i}$:
$$ \mathcal{L}_w(\mathbf{\beta}) = \sum_{i=1}^N w_{y_i} [y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)] $$
where $w_1 = N / (2 \cdot N_{\text{default}})$ and $w_0 = N / (2 \cdot N_{\text{non-default}})$. For a dataset with 5% defaults, $w_1 \approx 10$ and $w_0 \approx 0.526$, effectively making each default sample contribute significantly more to the loss, forcing the model to "pay attention" to the minority class.

```python
# Apply SMOTE to the training data to address class imbalance
# SMOTE generates synthetic samples for the minority class.
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

print(f"After SMOTE: {sum(y_train_sm==1)} defaults, {sum(y_train_sm==0)} non-defaults")

# Define hyperparameter grid for Random Forest using GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 500], # Number of trees in the forest
    'max_depth': [5, 8, 12, None],  # Maximum depth of the tree
    'min_samples_leaf': [20, 50, 100], # Minimum number of samples required to be at a leaf node
    'class_weight': ['balanced', 'balanced_subsample'] # Balanced class weights
}

# Initialize GridSearchCV for Random Forest
# Scoring is set to 'roc_auc' because AUC is a robust metric for imbalanced data.
rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5, # 5-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1, # Use all available CPU cores
    verbose=0
)

# Fit GridSearchCV on SMOTE-resampled training data
# Note: Scaling is not necessary for tree-based models, but we use the scaled X_train_sm
# to ensure consistency with LR or if other scaled features were created.
rf_grid_search.fit(X_train_sm, y_train_sm)

# Get the best Random Forest estimator
best_rf = rf_grid_search.best_estimator_

# Predict probabilities of default on the scaled test set using the best RF model
y_prob_rf = best_rf.predict_proba(X_test_sc)[:, 1]

print(f"\nBest Random Forest Parameters: {rf_grid_search.best_params_}")
print(f"Best Random Forest CV AUC: {rf_grid_search.best_score_:.4f}")
```

---

### **5. Benchmarking Credit Models: Beyond Accuracy with Financial Metrics**

Alex knows that "accuracy" alone is a misleading metric for imbalanced credit default data. A model predicting "no default" for all loans could achieve high accuracy but would miss all actual defaults, which is financially catastrophic. Instead, Alex focuses on credit-appropriate metrics: ROC-AUC, Precision-Recall curves, and the Kolmogorov-Smirnov (KS) statistic, which are crucial for assessing a model's ability to discriminate between good and bad loans, especially for the rare default events.

*   **ROC-AUC (Area Under the Receiver Operating Characteristic):** Measures the model's discrimination power. Probabilistic interpretation: the chance that a randomly selected default case is scored higher than a randomly selected non-default.
    $$ \text{AUC} = P(\hat{P}_{\text{default}} > \hat{P}_{\text{non-default}}) $$
    An AUC of 0.5 is random, >0.70 is acceptable, >0.80 is strong for consumer credit.
*   **Gini Coefficient:** An industry-standard alternative to AUC, directly related by:
    $$ \text{Gini} = 2 \times \text{AUC} - 1 $$
*   **Precision-Recall Curve:** Particularly informative for imbalanced datasets, showing the trade-off between precision and recall at various thresholds.
*   **Kolmogorov-Smirnov (KS) Statistic:** Measures the maximum separation between the cumulative distribution functions (CDFs) of predicted probabilities for default and non-default classes.
    $$ \text{KS} = \max_t |F_1(t) - F_0(t)| $$
    where $F_1(t)$ is the CDF of predicted probabilities for defaults and $F_0(t)$ is the CDF for non-defaults. A KS > 0.30 is generally considered acceptable.

```python
# Helper function to calculate common metrics
def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2 * roc_auc - 1

    # KS Statistic calculation
    fpr, tpr, thresholds_ks = roc_curve(y_true, y_prob)
    ks_stat = np.max(tpr - fpr)

    return {
        'AUC': roc_auc,
        'Gini': gini,
        'KS Statistic': ks_stat,
        'F1-score': f1,
        'Precision': precision,
        'Recall': recall
    }

# Create a figure with three subplots for ROC, Precision-Recall, and KS curves
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
plt.suptitle('Model Performance Curves for Credit Default Prediction', fontsize=16)

# --- ROC Curve ---
for name, y_prob, color in [
    ('Logistic Regression', y_prob_log, 'blue'),
    ('Random Forest', y_prob_rf, 'red')
]:
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc_val:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)


# --- Precision-Recall Curve ---
for name, y_prob, color in [
    ('Logistic Regression', y_prob_log, 'blue'),
    ('Random Forest', y_prob_rf, 'red')
]:
    prec, rec, thresholds_pr = precision_recall_curve(y_test, y_prob)
    avg_prec_score = average_precision_score(y_test, y_prob)
    axes[1].plot(rec, prec, color=color, label=f'{name} (AP = {avg_prec_score:.3f})')

# A dashed line for random baseline (proportion of positive class)
# The no-skill line for a precision-recall curve is the fraction of positive samples
no_skill = y_test.sum() / len(y_test)
axes[1].plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.5, label=f'No-skill (AP = {no_skill:.3f})')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].set_ylim([-0.05, 1.05])
axes[1].grid(True, linestyle='--', alpha=0.6)


# --- Kolmogorov-Smirnov (KS) Plot ---
for name, y_prob, color in [
    ('Logistic Regression', y_prob_log, 'blue'),
    ('Random Forest', y_prob_rf, 'red')
]:
    fpr_ks, tpr_ks, thresholds_ks = roc_curve(y_test, y_prob)
    ks_statistic = np.max(tpr_ks - fpr_ks)
    ks_threshold_idx = np.argmax(tpr_ks - fpr_ks)
    ks_threshold = thresholds_ks[ks_threshold_idx]

    axes[2].plot(thresholds_ks, tpr_ks - fpr_ks, color=color,
                 label=f'{name} (KS = {ks_statistic:.3f} at Thresh={ks_threshold:.2f})')
    axes[2].axvline(x=ks_threshold, color=color, linestyle=':', alpha=0.6)

axes[2].set_xlabel('Probability Threshold')
axes[2].set_ylabel('TPR - FPR')
axes[2].set_title('Kolmogorov-Smirnov (KS) Plot')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.savefig('model_performance_curves.png', dpi=300)
plt.show()

# Generate and print a summary table of key metrics at a default threshold of 0.5
metrics_log_default_th = calculate_metrics(y_test, y_prob_log)
metrics_rf_default_th = calculate_metrics(y_test, y_prob_rf)

metrics_df = pd.DataFrame({
    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
    'Logistic Regression (Th=0.5)': [f"{metrics_log_default_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']],
    'Random Forest (Th=0.5)': [f"{metrics_rf_default_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']]
})

print("\nModel Comparison Table (at default threshold 0.5):")
print(metrics_df.to_markdown(index=False))
```

### **6. Optimizing Lending Decisions: A Cost-Sensitive Approach to Threshold Selection**

This is where Alex translates model probabilities into tangible financial decisions for CapitalFlow Bank. The cost of a False Negative (approving a loan that defaults) is significantly higher than a False Positive (rejecting a loan that would have been repaid). Alex defines these asymmetric costs, calculates the Expected Total Cost across various probability thresholds, and identifies the optimal threshold that minimizes overall financial losses. This cost-sensitive approach is vital for maximizing bank profitability and managing risk effectively.

The Expected Total Cost for a given threshold is calculated as:
$$ C_{\text{total}} = \text{FN} \times (\text{LGD} \times \text{EAD}) + \text{FP} \times C_{\text{opportunity}} $$
Where:
*   $\text{FN}$ = Number of False Negatives (actual defaults predicted as non-defaults)
*   $\text{FP}$ = Number of False Positives (actual non-defaults predicted as defaults)
*   $\text{LGD}$ = Loss Given Default (percentage of exposure lost if default occurs). Let's assume 60% of the loan amount.
*   $\text{EAD}$ = Exposure at Default (the outstanding loan amount at the time of default). For simplicity, we assume this is the `avg_loan_amount`.
*   $C_{\text{opportunity}}$ = Opportunity cost of declining a good loan (e.g., lost interest margin). Let's assume 3% of the loan amount.

**Cost Assumptions for CapitalFlow Bank:**
*   Average Loan Amount (`avg_loan_amount`): \$15,000
*   Loss Given Default (`LGD`): 60%
*   Opportunity Cost Rate (`opportunity_cost_rate`): 3% (lost margin on a good loan)

Therefore:
*   `cost_fn` (Cost of a missed default) = `avg_loan_amount` $\times$ `LGD` = \$15,000 $\times$ 0.60 = \$9,000
*   `cost_fp` (Cost of declining a good loan) = `avg_loan_amount` $\times$ `opportunity_cost_rate` = \$15,000 $\times$ 0.03 = \$450

```python
# Define financial cost assumptions for CapitalFlow Bank
avg_loan_amount = 15000  # Average loan amount in USD
loss_given_default = 0.60  # LGD: 60% loss on default
opportunity_cost_rate = 0.03 # 3% lost margin from declining a good loan

# Calculate the financial impact of misclassifications
cost_fn = avg_loan_amount * loss_given_default
cost_fp = avg_loan_amount * opportunity_cost_rate

print(f"Cost of a False Negative (missed default): ${cost_fn:,.2f}")
print(f"Cost of a False Positive (declined good loan): ${cost_fp:,.2f}")

# Generate a range of probability thresholds to evaluate
thresholds = np.arange(0.05, 0.95, 0.01) # From 5% to 94% in 1% increments

# Lists to store total costs for each model at each threshold
costs_log = []
costs_rf = []

# Evaluate Expected Total Cost for Logistic Regression
for t in thresholds:
    y_pred = (y_prob_log >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    costs_log.append(total_cost)

# Evaluate Expected Total Cost for Random Forest
for t in thresholds:
    y_pred = (y_prob_rf >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    costs_rf.append(total_cost)

# Plotting the Cost Curves
plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs_log, 'b-', label='Logistic Regression')
plt.plot(thresholds, costs_rf, 'r-', label='Random Forest')

# Find optimal thresholds (minimum cost)
opt_t_log = thresholds[np.argmin(costs_log)]
min_cost_log = np.min(costs_log)
opt_t_rf = thresholds[np.argmin(costs_rf)]
min_cost_rf = np.min(costs_rf)

plt.axvline(x=opt_t_log, color='blue', linestyle='--', alpha=0.7, label=f'Optimal LR Th: {opt_t_log:.2f} (Cost: ${min_cost_log:,.0f})')
plt.axvline(x=opt_t_rf, color='red', linestyle='--', alpha=0.7, label=f'Optimal RF Th: {opt_t_rf:.2f} (Cost: ${min_cost_rf:,.0f})')

plt.xlabel('Classification Threshold')
plt.ylabel('Total Expected Cost ($)')
plt.title('Cost-Sensitive Threshold Analysis for CapitalFlow Bank')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('cost_threshold_analysis.png', dpi=300)
plt.show()

print(f"\nOptimal threshold for Logistic Regression: {opt_t_log:.2f} (Min Cost: ${min_cost_log:,.0f})")
print(f"Optimal threshold for Random Forest: {opt_t_rf:.2f} (Min Cost: ${min_cost_rf:,.0f})")

# --- Generate Financial Confusion Matrices at Optimal Thresholds ---
def plot_financial_confusion_matrix(y_true, y_prob, threshold, cost_fn, cost_fp, model_name, ax):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate dollar impacts
    fn_cost_dollars = fn * cost_fn
    fp_cost_dollars = fp * cost_fp

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Default', 'Predicted Default'],
                yticklabels=['Actual Non-Default', 'Actual Default'], ax=ax,
                annot_kws={"size": 14})
    ax.set_title(f'{model_name} Confusion Matrix (Th={threshold:.2f})\n', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    # Annotate with financial costs
    ax.text(1.5, 0.5, f'FP: {fp} (${fp_cost_dollars:,.0f})',
            horizontalalignment='center', verticalalignment='center', color='darkred', fontsize=10)
    ax.text(0.5, 1.5, f'FN: {fn} (${fn_cost_dollars:,.0f})',
            horizontalalignment='center', verticalalignment='center', color='darkred', fontsize=10)
    ax.text(0.5, 0.5, f'TN: {tn}', horizontalalignment='center', verticalalignment='center', fontsize=10)
    ax.text(1.5, 1.5, f'TP: {tp}', horizontalalignment='center', verticalalignment='center', fontsize=10)


fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle('Financial Impact of Model Decisions at Optimal Thresholds', fontsize=16)

# Plot for Logistic Regression
plot_financial_confusion_matrix(y_test, y_prob_log, opt_t_log, cost_fn, cost_fp, 'Logistic Regression', axes_cm[0])

# Plot for Random Forest
plot_financial_confusion_matrix(y_test, y_prob_rf, opt_t_rf, cost_fn, cost_fp, 'Random Forest', axes_cm[1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('financial_confusion_matrices.png', dpi=300)
plt.show()

# Final model comparison table with metrics at optimal thresholds
metrics_log_opt_th = calculate_metrics(y_test, y_prob_log, threshold=opt_t_log)
metrics_rf_opt_th = calculate_metrics(y_test, y_prob_rf, threshold=opt_t_rf)

metrics_df_optimal = pd.DataFrame({
    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
    f'Logistic Regression (Th={opt_t_log:.2f})': [f"{metrics_log_opt_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']],
    f'Random Forest (Th={opt_t_rf:.2f})': [f"{metrics_rf_opt_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']]
})

print("\nModel Comparison Table (at optimal thresholds):")
print(metrics_df_optimal.to_markdown(index=False))
```

### **7. Deconstructing Default Drivers: Comparing Feature Insights**

To gain deeper insights into credit risk, Alex needs to understand which factors are most influential in predicting default. This involves extracting and comparing feature importance from both the Logistic Regression and Random Forest models. While Logistic Regression provides clear coefficient magnitudes (and odds ratios), Random Forest uses Gini impurity to rank features. This comparison helps CapitalFlow Bank understand different perspectives on risk drivers and informs targeted interventions. It also highlights the trade-off between model interpretability and predictive power, a key consideration for regulatory transparency.

```python
# --- Logistic Regression Feature Importance (Coefficient Magnitudes) ---
# Use absolute values of coefficients for ranking importance regardless of direction
log_importance_df = pd.DataFrame({
    'Feature': final_feature_cols,
    'LogReg |Coef|': np.abs(log_model.coef_[0])
}).sort_values('LogReg |Coef|', ascending=False)

# --- Random Forest Feature Importance (Gini Importance) ---
rf_importance_df = pd.DataFrame({
    'Feature': final_feature_cols,
    'RF Importance': best_rf.feature_importances_
}).sort_values('RF Importance', ascending=False)

# Merge and display side-by-side comparison
combined_importance_df = log_importance_df.merge(rf_importance_df, on='Feature', how='outer')
combined_importance_df['LogReg Rank'] = combined_importance_df['LogReg |Coef|'].rank(ascending=False).astype(int)
combined_importance_df['RF Rank'] = combined_importance_df['RF Importance'].rank(ascending=False).astype(int)

print("Feature Importance Comparison:")
print(combined_importance_df[['Feature', 'LogReg Rank', 'RF Rank', 'LogReg |Coef|', 'RF Importance']].sort_values('RF Rank').round(4).to_markdown(index=False))

# --- Visualize Feature Importance Comparison ---
fig_fi, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.suptitle('Feature Importance Comparison', fontsize=16)

top_n = 15 # Display top N features

# Logistic Regression Feature Importance
log_top = log_importance_df.head(top_n)
sns.barplot(x='LogReg |Coef|', y='Feature', data=log_top, ax=ax1, palette='viridis')
ax1.set_title(f'Logistic Regression: Top {top_n} |Coefficient|')
ax1.set_xlabel('Absolute Coefficient Value (Standardized)')
ax1.set_ylabel('')
ax1.invert_yaxis() # Display highest importance at the top

# Random Forest Feature Importance
rf_top = rf_importance_df.head(top_n)
sns.barplot(x='RF Importance', y='Feature', data=rf_top, ax=ax2, palette='magma')
ax2.set_title(f'Random Forest: Top {top_n} Gini Importance')
ax2.set_xlabel('Gini Importance')
ax2.set_ylabel('')
ax2.invert_yaxis() # Display highest importance at the top

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_importance_credit.png', dpi=300)
plt.show()

# --- Predicted Probability Distributions ---
plt.figure(figsize=(14, 6))
sns.histplot(y_prob_log[y_test == 0], color='blue', label='LR: Non-Default (Actual)', kde=True, stat='density', alpha=0.5)
sns.histplot(y_prob_log[y_test == 1], color='red', label='LR: Default (Actual)', kde=True, stat='density', alpha=0.5)
plt.title('Predicted Probability Distributions (Logistic Regression)')
plt.xlabel('Predicted Probability of Default (PD)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('predicted_pd_distributions_lr.png', dpi=300)
plt.show()

plt.figure(figsize=(14, 6))
sns.histplot(y_prob_rf[y_test == 0], color='blue', label='RF: Non-Default (Actual)', kde=True, stat='density', alpha=0.5)
sns.histplot(y_prob_rf[y_test == 1], color='red', label='RF: Default (Actual)', kde=True, stat='density', alpha=0.5)
plt.title('Predicted Probability Distributions (Random Forest)')
plt.xlabel('Predicted Probability of Default (PD)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('predicted_pd_distributions_rf.png', dpi=300)
plt.show()
```

### **8. Ensuring Reliable Probabilities: Calibrating PD Estimates**

For CapitalFlow Bank, it's not enough for a model to simply rank borrowers by risk; the predicted probabilities of default (PDs) must also be well-calibrated. This means that if a model predicts a 10% PD, approximately 10% of those borrowers should actually default. Well-calibrated PDs are essential for accurate risk-based pricing, regulatory provisioning (e.g., IFRS 9 Expected Credit Loss calculations), and portfolio stress testing. Alex uses calibration curves to assess how closely the predicted PDs align with observed default rates.

A calibration curve (also known as a reliability diagram) plots the mean predicted probability for a given bin against the fraction of positive outcomes (observed default rate) in that bin. A perfectly calibrated model would have its points fall along the diagonal line $y=x$.

```python
# --- Calibration Curves ---
def plot_calibration_curve(y_true, y_prob, model_name, ax, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name, color='blue' if 'Logistic' in model_name else 'red')
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability (PD)")
    ax.set_ylabel("Fraction of Positives (Observed Default Rate)")
    ax.set_title(f'Calibration Curve: {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim([-0.05, 1.05])

fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle('Model Calibration Curves', fontsize=16)

# Plot for Logistic Regression
plot_calibration_curve(y_test, y_prob_log, 'Logistic Regression', axes_cal[0])

# Plot for Random Forest
# For Random Forest, it's common to apply isotonic regression for post-hoc calibration
# to improve the reliability of its probabilities.
# Here, we'll plot the uncalibrated RF for comparison, but note that a production model
# might use CalibratedClassifierCV.
plot_calibration_curve(y_test, y_prob_rf, 'Random Forest (Uncalibrated)', axes_cal[1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('calibration_curves.png', dpi=300)
plt.show()

# To demonstrate calibration: Post-hoc calibration for Random Forest
calibrated_rf = CalibratedClassifierCV(best_rf, method='isotonic', cv=5)
calibrated_rf.fit(X_train_sm, y_train_sm) # Fit on resampled training data
y_prob_rf_calibrated = calibrated_rf.predict_proba(X_test_sc)[:, 1]

plt.figure(figsize=(7, 6))
plot_calibration_curve(y_test, y_prob_rf_calibrated, 'Random Forest (Calibrated)', plt.gca())
plt.suptitle('Random Forest Calibration After Isotonic Regression', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('calibration_curve_rf_calibrated.png', dpi=300)
plt.show()

# Calculate Brier score to quantify calibration loss
brier_log = brier_score_loss(y_test, y_prob_log)
brier_rf = brier_score_loss(y_test, y_prob_rf)
brier_rf_calibrated = brier_score_loss(y_test, y_prob_rf_calibrated)

print(f"\nBrier Score (Logistic Regression): {brier_log:.4f}")
print(f"Brier Score (Random Forest, Uncalibrated): {brier_rf:.4f}")
print(f"Brier Score (Random Forest, Calibrated): {brier_rf_calibrated:.4f}")
print("\nA lower Brier score indicates better calibration.")
```
