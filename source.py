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
from imblearn.over_sampling import SMOTE

import warnings
import os

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Global plotting style settings (can be configured in app.py or within specific plotting functions)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- Helper Function for Dummy Data Generation ---
def _generate_dummy_data(n=1000, random_state=42):
    """
    Generates a semi-synthetic LendingClub-like dataset for demonstration
    purposes if the actual data file is not found.

    Args:
        n (int): Number of rows (loans) to generate.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with synthetic loan data.
    """
    rng = np.random.default_rng(random_state)

    # --- 1) Generate borrower + loan characteristics with realistic-ish distributions ---
    emp_length_levels = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                        '6 years', '7 years', '8 years', '9 years', '10+ years']
    emp_length = rng.choice(
        emp_length_levels, size=n,
        p=[0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10, 0.10, 0.10, 0.10, 0.14]
    )

    home_ownership = rng.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=n, p=[0.44, 0.46, 0.08, 0.02])

    purpose = rng.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'car'],
        size=n,
        p=[0.55, 0.22, 0.10, 0.08, 0.05]
    )

    term = rng.choice([' 36 months', ' 60 months'], size=n, p=[0.70, 0.30])
    term_num = np.where(np.char.strip(term) == '60 months', 60, 36)

    annual_inc = rng.lognormal(mean=np.log(65000), sigma=0.55, size=n)
    annual_inc = np.clip(annual_inc, 20000, 250000).round(0).astype(int)

    fico = rng.normal(loc=690, scale=35, size=n)
    fico = np.clip(fico, 600, 820).round(0).astype(int)
    fico_range_low = (fico - rng.integers(0, 6, size=n)).clip(600, 820)
    fico_range_high = (fico_range_low + 4).clip(604, 824)

    revol_util = (rng.beta(2.2, 2.8, size=n) * 100)
    revol_util = np.clip(revol_util + (680 - fico) * 0.10, 0, 100).round(2)

    revol_bal = (annual_inc * (revol_util / 100) * rng.uniform(0.05, 0.25, size=n))
    revol_bal = np.clip(revol_bal, 0, 120000).round(0).astype(int)

    open_acc = rng.integers(2, 25, size=n)
    total_acc = (open_acc + rng.integers(3, 35, size=n)).clip(5, 80)

    base_delinq = rng.poisson(lam=0.15, size=n)
    delinq_2yrs = (base_delinq + (fico < 660).astype(int) * rng.poisson(0.6, size=n)).clip(0, 10)

    base_pub = rng.poisson(lam=0.05, size=n)
    pub_rec = (base_pub + (fico < 650).astype(int) * rng.poisson(0.25, size=n)).clip(0, 5)

    dti = rng.normal(loc=16, scale=7, size=n)
    dti += (revol_util / 100) * 8
    dti += (annual_inc < 45000).astype(int) * rng.uniform(2, 6, size=n)
    dti = np.clip(dti, 0, 45).round(2)

    purpose_boost = np.select(
        [
            purpose == 'debt_consolidation',
            purpose == 'home_improvement',
            purpose == 'credit_card',
            purpose == 'car',
        ],
        [2500, 3500, 1500, 1000],
        default=0
    )
    loan_amnt = (annual_inc * rng.uniform(0.08, 0.22, size=n) + purpose_boost)
    loan_amnt = np.clip(loan_amnt, 1000, 40000)
    loan_amnt = (np.round(loan_amnt / 100) * 100).astype(int)

    int_rate = 7.5 \
        + (700 - fico) * 0.06 \
        + (dti - 15) * 0.08 \
        + (term_num == 60) * 1.4 \
        + delinq_2yrs * 0.35 \
        + pub_rec * 0.60 \
        + rng.normal(0, 1.2, size=n)
    int_rate = np.clip(int_rate, 5.0, 32.0).round(2)

    installment = (loan_amnt * (0.02 + int_rate / 100 / 12) / (term_num / 36))
    installment = np.clip(installment, 30, 1600).round(2)

    # --- 2) Create a "risk score" and generate loan_status from it ---
    risk = (
        (700 - fico) / 60
        + (dti / 30)
        + (revol_util / 100) * 0.9
        + delinq_2yrs * 0.25
        + pub_rec * 0.35
        + (term_num == 60) * 0.35
        + (int_rate - 10) / 12
    )

    intercept = -3.2
    pd_default = 1 / (1 + np.exp(-(intercept + 1.15 * risk)))
    pd_default = np.clip(pd_default, 0.002, 0.60)

    u = rng.uniform(0, 1, size=n)

    loan_status = np.where(u < pd_default * 0.20, 'Default',
                  np.where(u < pd_default, 'Charged Off', 'Fully Paid'))

    df = pd.DataFrame({
        'loan_status': loan_status,
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_range_low': fico_range_low,
        'fico_range_high': fico_range_high,
        'open_acc': open_acc,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'delinq_2yrs': delinq_2yrs,
        'pub_rec': pub_rec,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'term': term
    })
    print("Generated semi-synthetic dataset (fallback):")
    print(df['loan_status'].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    print("Default rate (non Fully Paid):", (df['loan_status'] != 'Fully Paid').mean().round(4))
    return df

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath='lending_club_loans.csv'):
    """
    Loads the dataset, handles missing files by generating dummy data,
    filters relevant statuses, and performs initial feature engineering and cleaning.

    Args:
        filepath (str): Path to the lending club loans CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for modeling.
        list: List of final feature columns after preprocessing.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}")
    except FileNotFoundError:
        df = _generate_dummy_data()

    completed_statuses = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(completed_statuses)].copy()
    df['default'] = (df['loan_status'] != 'Fully Paid').astype(int)

    print(f"\nInitial default rate: {df['default'].mean():.2%}")
    print(f"Initial dataset size: {len(df):,} loans")

    feature_cols = [
        'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'open_acc', 'revol_bal', 'revol_util',
        'total_acc', 'delinq_2yrs', 'pub_rec', 'emp_length', 'home_ownership',
        'purpose', 'term'
    ]

    # Filter out missing feature columns if the dummy data or real data doesn't have them all
    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    missing_feature_cols = [col for col in feature_cols if col not in df.columns]

    if missing_feature_cols:
        print(f"\nWarning: The following feature columns were not found and will be skipped: {missing_feature_cols}")
    
    df_processed = df[existing_feature_cols + ['default']].copy()
    current_feature_cols = existing_feature_cols[:] # Create a mutable copy

    # Feature Engineering for FICO score
    if 'fico_range_low' in df_processed.columns and 'fico_range_high' in df_processed.columns:
        df_processed['fico_score'] = (df_processed['fico_range_low'] + df_processed['fico_range_high']) / 2
        df_processed = df_processed.drop(columns=['fico_range_low', 'fico_range_high'])
        if 'fico_range_low' in current_feature_cols: current_feature_cols.remove('fico_range_low')
        if 'fico_range_high' in current_feature_cols: current_feature_cols.remove('fico_range_high')
        current_feature_cols.append('fico_score')
    
    # Process 'term' column
    if 'term' in df_processed.columns:
        df_processed['term'] = df_processed['term'].astype(str).str.strip()
        df_processed['term'] = df_processed['term'].map({'36 months': 0, '60 months': 1})
        # Fill any remaining NaNs after mapping, e.g., if there were unexpected string values
        if df_processed['term'].isnull().any():
            df_processed['term'].fillna(df_processed['term'].median(), inplace=True)

    # Process 'emp_length' column
    emp_length_map = {
        '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
        '10+ years': 10
    }
    if 'emp_length' in df_processed.columns:
        df_processed['emp_length'] = df_processed['emp_length'].map(emp_length_map).fillna(np.nan)
        df_processed['emp_length'].fillna(df_processed['emp_length'].median(), inplace=True)

    # Handle missing values for numerical features
    numerical_cols = ['revol_util', 'dti', 'annual_inc', 'installment', 'loan_amnt', 
                      'int_rate', 'open_acc', 'revol_bal', 'total_acc', 'delinq_2yrs', 'pub_rec']
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # One-hot encoding for categorical features
    categorical_features = [col for col in ['home_ownership', 'purpose'] if col in df_processed.columns]
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True, dtype=int)

    final_feature_cols = df_processed.drop('default', axis=1).columns.tolist()
    
    return df_processed, final_feature_cols

# --- Data Splitting and Scaling ---
def split_and_scale_data(df_processed, final_feature_cols, test_size=0.25, random_state=42):
    """
    Splits data into training and testing sets, and scales numerical features using StandardScaler.

    Args:
        df_processed (pd.DataFrame): Preprocessed DataFrame.
        final_feature_cols (list): List of final feature columns.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, X_train_scaled_df, X_test_scaled_df)
               X_train, X_test are original (unscaled) for reference if needed.
               X_train_scaled_df, X_test_scaled_df are scaled DataFrames.
               scaler is the fitted StandardScaler object.
    """
    X = df_processed.drop('default', axis=1)
    y = df_processed['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set default rate: {y_train.mean():.2%}")
    print(f"Testing set default rate: {y_test.mean():.2%}")
    print(f"Training set size: {len(X_train):,}")
    print(f"Testing set size: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_train_sc_df = pd.DataFrame(X_train_sc, columns=final_feature_cols, index=X_train.index)
    X_test_sc_df = pd.DataFrame(X_test_sc, columns=final_feature_cols, index=X_test.index)
    
    return X_train, X_test, y_train, y_test, scaler, X_train_sc_df, X_test_sc_df

# --- SMOTE Application ---
def apply_smote(X_train_sc, y_train, random_state=42, k_neighbors=1):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to the training data
    if the minority class is large enough for the specified k_neighbors.

    Args:
        X_train_sc (np.array or pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target variable.
        random_state (int): Random state for reproducibility.
        k_neighbors (int): Number of nearest neighbors to use for SMOTE.

    Returns:
        tuple: (X_train_sm, y_train_sm) - Resampled training features and target.
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

    minority_class_label = y_train.value_counts().idxmin()
    minority_class_count = y_train.value_counts().min()

    if minority_class_count <= smote.k_neighbors:
        print(f"Warning: SMOTE k_neighbors ({smote.k_neighbors}) is too large for minority class size ({minority_class_count}). Skipping SMOTE.")
        X_train_sm, y_train_sm = X_train_sc, y_train
    else:
        X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

    print(f"After (potential) SMOTE: {sum(y_train_sm==1)} defaults, {sum(y_train_sm==0)} non-defaults")
    return X_train_sm, y_train_sm

# --- Model Training ---
def train_logistic_regression(X_train_sc, y_train, final_feature_cols, random_state=42):
    """
    Trains a Logistic Regression model and prints its odds ratios.

    Args:
        X_train_sc (np.array or pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target variable.
        final_feature_cols (list): List of feature names for odds ratio output.
        random_state (int): Random state for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegression: Trained Logistic Regression model.
    """
    log_model = LogisticRegression(
        penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=random_state
    )
    log_model.fit(X_train_sc, y_train)

    coefficients = log_model.coef_[0]
    odds_ratios = np.exp(coefficients)

    odds_ratio_df = pd.DataFrame({
        'Feature': final_feature_cols,
        'Coefficient': coefficients,
        'Odds Ratio': odds_ratios,
        'Direction': ['Risk+' if c > 0 else 'Risk-' for c in coefficients]
    }).sort_values('Odds Ratio', ascending=False)

    print("\nLogistic Regression Odds Ratio Table:")
    print(odds_ratio_df.round(4).to_string(index=False))
    
    return log_model

def train_random_forest(X_train_sm, y_train_sm, random_state=42, n_jobs=1):
    """
    Trains a Random Forest model using GridSearchCV for hyperparameter tuning.

    Args:
        X_train_sm (np.array or pd.DataFrame): SMOTE-resampled training features.
        y_train_sm (pd.Series): SMOTE-resampled training target variable.
        random_state (int): Random state for reproducibility.
        n_jobs (int): Number of jobs to run in parallel for GridSearchCV (-1 means use all processors).

    Returns:
        sklearn.ensemble.RandomForestClassifier: Best Random Forest estimator found by GridSearchCV.
    """
    # Adjusted param_grid_rf for potentially small datasets (like the dummy data)
    param_grid_rf = {
        'n_estimators': [10, 20], # Reduced for faster testing with small data
        'max_depth': [2, 3],      # Reduced for faster testing, suitable for very few samples
        'min_samples_leaf': [1, 2], # min_samples_leaf must be <= total samples / cv folds
        'class_weight': ['balanced']
    }
    
    # Calculate appropriate cv value: min(5, minority_class_count - 1)
    # The smallest class count in y_train_sm will be used to determine max cv folds
    min_samples_for_cv = y_train_sm.value_counts().min()
    cv_folds = min(5, max(2, min_samples_for_cv - 1)) # At least 2, at most 5, and less than minority size

    rf_grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid_rf,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=0
    )

    rf_grid_search.fit(X_train_sm, y_train_sm)
    best_rf = rf_grid_search.best_estimator_

    print(f"\nBest Random Forest Parameters: {rf_grid_search.best_params_}")
    print(f"Best Random Forest CV AUC: {rf_grid_search.best_score_:.4f}")
    
    return best_rf

# --- Model Evaluation Metrics ---
def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculates various classification metrics for a given model's predictions.

    Args:
        y_true (pd.Series): True labels.
        y_prob (np.array): Predicted probabilities for the positive class.
        threshold (float): Classification threshold to convert probabilities to binary predictions.

    Returns:
        dict: A dictionary of calculated metrics including AUC, Gini, KS Statistic, F1-score, Precision, and Recall.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2 * roc_auc - 1

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

# --- Plotting Functions ---
def plot_performance_curves(y_test, y_prob_log, y_prob_rf, save_path='model_performance_curves.png'):
    """
    Generates and saves a figure containing ROC, Precision-Recall, and KS curves
    for Logistic Regression and Random Forest models.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_prob_log (np.array): Predicted probabilities from Logistic Regression.
        y_prob_rf (np.array): Predicted probabilities from Random Forest.
        save_path (str): Full path to save the generated plot image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plt.suptitle('Model Performance Curves for Credit Default Prediction', fontsize=16)

    # ROC Curve
    for name, y_prob, color in [
        ('Logistic Regression', y_prob_log, 'blue'),
        ('Random Forest', y_prob_rf, 'red')
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc_val:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Precision-Recall Curve
    for name, y_prob, color in [
        ('Logistic Regression', y_prob_log, 'blue'),
        ('Random Forest', y_prob_rf, 'red')
    ]:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        avg_prec_score = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, label=f'{name} (AP = {avg_prec_score:.3f})')
    no_skill = y_test.sum() / len(y_test)
    axes[1].plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.5, label=f'No-skill (AP = {no_skill:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Kolmogorov-Smirnov (KS) Plot
    for name, y_prob, color in [
        ('Logistic Regression', y_prob_log, 'blue'),
        ('Random Forest', y_prob_rf, 'red')
    ]:
        fpr_ks, tpr_ks, thresholds_ks = roc_curve(y_test, y_prob)
        ks_statistic = np.max(tpr_ks - fpr_ks)
        ks_threshold_idx = np.argmax(tpr_ks - fpr_ks)
        ks_threshold = thresholds_ks[ks_threshold_idx]

        axes[2].plot(
            thresholds_ks,
            tpr_ks - fpr_ks,
            color=color,
            label=f'{name} (KS = {ks_statistic:.3f} at Thresh={ks_threshold:.2f})'
        )
        axes[2].axvline(x=ks_threshold, color=color, linestyle=':', alpha=0.6)
    axes[2].set_xlabel('Probability Threshold')
    axes[2].set_ylabel('TPR - FPR')
    axes[2].set_title('Kolmogorov-Smirnov (KS) Plot')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Performance curves saved to {save_path}")


def plot_cost_analysis(y_test, y_prob_log, y_prob_rf, avg_loan_amount, loss_given_default, opportunity_cost_rate, save_path='cost_threshold_analysis.png'):
    """
    Performs a cost-sensitive threshold analysis, plots total expected cost versus threshold,
    and identifies optimal thresholds for each model.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_prob_log (np.array): Predicted probabilities from Logistic Regression.
        y_prob_rf (np.array): Predicted probabilities from Random Forest.
        avg_loan_amount (float): Average loan amount, used in cost calculations.
        loss_given_default (float): Proportion of loan amount lost if a default occurs (for FN cost).
        opportunity_cost_rate (float): Rate representing the opportunity cost of rejecting a good loan (for FP cost).
        save_path (str): Full path to save the generated plot image.

    Returns:
        tuple: (opt_t_log, min_cost_log, opt_t_rf, min_cost_rf)
               Optimal thresholds and their corresponding minimum costs for LR and RF.
    """
    cost_fn = avg_loan_amount * loss_given_default
    cost_fp = avg_loan_amount * opportunity_cost_rate

    print(f"\nCost of a False Negative (missed default): ${cost_fn:,.2f}")
    print(f"Cost of a False Positive (declined good loan): ${cost_fp:,.2f}")

    thresholds = np.arange(0.05, 0.95, 0.01)

    costs_log = []
    for t in thresholds:
        y_pred = (y_prob_log >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs_log.append(total_cost)

    costs_rf = []
    for t in thresholds:
        y_pred = (y_prob_rf >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs_rf.append(total_cost)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs_log, 'b-', label='Logistic Regression')
    plt.plot(thresholds, costs_rf, 'r-', label='Random Forest')

    opt_t_log = thresholds[np.argmin(costs_log)]
    min_cost_log = np.min(costs_log)
    opt_t_rf = thresholds[np.argmin(costs_rf)]
    min_cost_rf = np.min(costs_rf)

    plt.axvline(
        x=opt_t_log,
        color='blue',
        linestyle='--',
        alpha=0.7,
        label=f'Optimal LR Th: {opt_t_log:.2f} (Cost: ${min_cost_log:,.0f})'
    )
    plt.axvline(
        x=opt_t_rf,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f'Optimal RF Th: {opt_t_rf:.2f} (Cost: ${min_cost_rf:,.0f})'
    )

    plt.xlabel('Classification Threshold')
    plt.ylabel('Total Expected Cost ($)')
    plt.title('Cost-Sensitive Threshold Analysis for CapitalFlow Bank')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Cost analysis plot saved to {save_path}")

    print(f"\nOptimal threshold for Logistic Regression: {opt_t_log:.2f} (Min Cost: ${min_cost_log:,.0f})")
    print(f"Optimal threshold for Random Forest: {opt_t_rf:.2f} (Min Cost: ${min_cost_rf:,.0f})")
    
    return opt_t_log, min_cost_log, opt_t_rf, min_cost_rf


def _plot_single_financial_confusion_matrix(y_true, y_prob, threshold, cost_fn, cost_fp, model_name, ax):
    """
    Helper function to plot a single confusion matrix with financial impact annotations.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fn_cost_dollars = fn * cost_fn
    fp_cost_dollars = fp * cost_fp

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted Non-Default', 'Predicted Default'],
        yticklabels=['Actual Non-Default', 'Actual Default'],
        ax=ax,
        annot_kws={"size": 14}
    )
    ax.set_title(f'{model_name} Confusion Matrix (Th={threshold:.2f})\n', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    ax.text(
        1.5, 0.5,
        f'FP: {fp} (${fp_cost_dollars:,.0f})',
        horizontalalignment='center',
        verticalalignment='center',
        color='darkred',
        fontsize=10
    )
    ax.text(
        0.5, 1.5,
        f'FN: {fn} (${fn_cost_dollars:,.0f})',
        horizontalalignment='center',
        verticalalignment='center',
        color='darkred',
        fontsize=10
    )
    ax.text(0.5, 0.5, f'TN: {tn}', horizontalalignment='center', verticalalignment='center', fontsize=10)
    ax.text(1.5, 1.5, f'TP: {tp}', horizontalalignment='center', verticalalignment='center', fontsize=10)


def plot_financial_confusion_matrices(y_test, y_prob_log, y_prob_rf, opt_t_log, opt_t_rf, avg_loan_amount, loss_given_default, opportunity_cost_rate, save_path='financial_confusion_matrices.png'):
    """
    Generates and saves a figure showing financial confusion matrices for Logistic Regression
    and Random Forest at their respective optimal thresholds.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_prob_log (np.array): Predicted probabilities from Logistic Regression.
        y_prob_rf (np.array): Predicted probabilities from Random Forest.
        opt_t_log (float): Optimal threshold for Logistic Regression.
        opt_t_rf (float): Optimal threshold for Random Forest.
        avg_loan_amount (float): Average loan amount, used in cost calculations.
        loss_given_default (float): Proportion of loan amount lost if a default occurs (for FN cost).
        opportunity_cost_rate (float): Rate representing the opportunity cost of rejecting a good loan (for FP cost).
        save_path (str): Full path to save the generated plot image.
    """
    cost_fn = avg_loan_amount * loss_given_default
    cost_fp = avg_loan_amount * opportunity_cost_rate

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle('Financial Impact of Model Decisions at Optimal Thresholds', fontsize=16)

    _plot_single_financial_confusion_matrix(
        y_test, y_prob_log, opt_t_log, cost_fn, cost_fp, 'Logistic Regression', axes_cm[0]
    )
    _plot_single_financial_confusion_matrix(
        y_test, y_prob_rf, opt_t_rf, cost_fn, cost_fp, 'Random Forest', axes_cm[1]
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig_cm)
    print(f"Financial confusion matrices saved to {save_path}")


def plot_feature_importance(log_model, best_rf, final_feature_cols, save_path='feature_importance_credit.png'):
    """
    Plots and prints feature importances (or absolute coefficients for LR) for
    Logistic Regression and Random Forest models.

    Args:
        log_model (sklearn.linear_model.LogisticRegression): Trained Logistic Regression model.
        best_rf (sklearn.ensemble.RandomForestClassifier): Trained Random Forest model.
        final_feature_cols (list): List of final feature columns.
        save_path (str): Full path to save the generated plot image.
    """
    log_importance_df = pd.DataFrame({
        'Feature': final_feature_cols,
        'LogReg |Coef|': np.abs(log_model.coef_[0])
    }).sort_values('LogReg |Coef|', ascending=False)

    rf_importance_df = pd.DataFrame({
        'Feature': final_feature_cols,
        'RF Importance': best_rf.feature_importances_
    }).sort_values('RF Importance', ascending=False)

    combined_importance_df = log_importance_df.merge(rf_importance_df, on='Feature', how='outer')
    combined_importance_df['LogReg Rank'] = combined_importance_df['LogReg |Coef|'].rank(ascending=False).astype(int)
    combined_importance_df['RF Rank'] = combined_importance_df['RF Importance'].rank(ascending=False).astype(int)

    print("\nFeature Importance Comparison:")
    print(
        combined_importance_df[
            ['Feature', 'LogReg Rank', 'RF Rank', 'LogReg |Coef|', 'RF Importance']
        ].sort_values('RF Rank').round(4).to_markdown(index=False)
    )

    fig_fi, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.suptitle('Feature Importance Comparison', fontsize=16)

    top_n = min(15, len(final_feature_cols))

    log_top = log_importance_df.head(top_n)
    sns.barplot(x='LogReg |Coef|', y='Feature', data=log_top, ax=ax1, palette='viridis')
    ax1.set_title(f'Logistic Regression: Top {top_n} |Coefficient|')
    ax1.set_xlabel('Absolute Coefficient Value (Standardized)')
    ax1.set_ylabel('')
    ax1.invert_yaxis()

    rf_top = rf_importance_df.head(top_n)
    sns.barplot(x='RF Importance', y='Feature', data=rf_top, ax=ax2, palette='magma')
    ax2.set_title(f'Random Forest: Top {top_n} Gini Importance')
    ax2.set_xlabel('Gini Importance')
    ax2.set_ylabel('')
    ax2.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig_fi)
    print(f"Feature importance plot saved to {save_path}")


def plot_predicted_pd_distributions(y_test, y_prob_log, y_prob_rf, save_path_log='predicted_pd_distributions_lr.png', save_path_rf='predicted_pd_distributions_rf.png'):
    """
    Generates and saves histograms of predicted probabilities of default (PD) for
    actual default and non-default loans, for both Logistic Regression and Random Forest.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_prob_log (np.array): Predicted probabilities from Logistic Regression.
        y_prob_rf (np.array): Predicted probabilities from Random Forest.
        save_path_log (str): Full path to save the LR distribution plot.
        save_path_rf (str): Full path to save the RF distribution plot.
    """
    fig_lr = plt.figure(figsize=(14, 6))
    sns.histplot(y_prob_log[y_test == 0], color='blue', label='LR: Non-Default (Actual)', kde=True, stat='density', alpha=0.5)
    sns.histplot(y_prob_log[y_test == 1], color='red', label='LR: Default (Actual)', kde=True, stat='density', alpha=0.5)
    plt.title('Predicted Probability Distributions (Logistic Regression)')
    plt.xlabel('Predicted Probability of Default (PD)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path_log, dpi=300)
    plt.close(fig_lr)
    print(f"Predicted PD distributions for LR saved to {save_path_log}")

    fig_rf = plt.figure(figsize=(14, 6))
    sns.histplot(y_prob_rf[y_test == 0], color='blue', label='RF: Non-Default (Actual)', kde=True, stat='density', alpha=0.5)
    sns.histplot(y_prob_rf[y_test == 1], color='red', label='RF: Default (Actual)', kde=True, stat='density', alpha=0.5)
    plt.title('Predicted Probability Distributions (Random Forest)')
    plt.xlabel('Predicted Probability of Default (PD)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path_rf, dpi=300)
    plt.close(fig_rf)
    print(f"Predicted PD distributions for RF saved to {save_path_rf}")


def _plot_single_calibration_curve(y_true, y_prob, model_name, ax, n_bins=10):
    """
    Helper function to plot a single calibration curve (reliability diagram).
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=model_name,
        color='blue' if 'Logistic' in model_name else 'red'
    )
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability (PD)")
    ax.set_ylabel("Fraction of Positives (Observed Default Rate)")
    ax.set_title(f'Calibration Curve: {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim([-0.05, 1.05])


def plot_all_calibration_curves(y_test, y_prob_log, y_prob_rf, y_prob_rf_calibrated=None, save_path='calibration_curves.png', save_path_calibrated_rf='calibration_curve_rf_calibrated.png'):
    """
    Generates and saves calibration curves for Logistic Regression, uncalibrated Random Forest,
    and (optionally) calibrated Random Forest.

    Args:
        y_test (pd.Series): True labels for the test set.
        y_prob_log (np.array): Predicted probabilities from Logistic Regression.
        y_prob_rf (np.array): Predicted probabilities from Random Forest (uncalibrated).
        y_prob_rf_calibrated (np.array, optional): Predicted probabilities from calibrated Random Forest. Defaults to None.
        save_path (str): Full path to save the combined LR/RF uncalibrated plot.
        save_path_calibrated_rf (str): Full path to save the calibrated RF plot.
    """
    fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle('Model Calibration Curves', fontsize=16)

    _plot_single_calibration_curve(y_test, y_prob_log, 'Logistic Regression', axes_cal[0])
    _plot_single_calibration_curve(y_test, y_prob_rf, 'Random Forest (Uncalibrated)', axes_cal[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig_cal)
    print(f"Calibration curves saved to {save_path}")

    if y_prob_rf_calibrated is not None:
        fig_cal_rf = plt.figure(figsize=(7, 6))
        _plot_single_calibration_curve(y_test, y_prob_rf_calibrated, 'Random Forest (Calibrated)', plt.gca())
        plt.suptitle('Random Forest Calibration After Isotonic Regression', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path_calibrated_rf, dpi=300)
        plt.close(fig_cal_rf)
        print(f"Calibrated RF curve saved to {save_path_calibrated_rf}")


# --- Main Pipeline Function ---
def run_model_pipeline(filepath='lending_club_loans.csv', random_state=42, output_dir='./model_outputs'):
    """
    Executes the entire credit default prediction model pipeline, including data loading,
    preprocessing, model training, evaluation, and plotting.

    Args:
        filepath (str): Path to the lending club loans CSV file.
        random_state (int): Random state for reproducibility across data splitting and models.
        output_dir (str): Directory where all generated plots and outputs will be saved.

    Returns:
        dict: A dictionary containing trained models, scaler, feature names, test data,
              predicted probabilities, optimal thresholds, and key evaluation metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all outputs to directory: {os.path.abspath(output_dir)}")

    # 1. Load and Preprocess Data
    df_processed, final_feature_cols = load_and_preprocess_data(filepath)

    # 2. Split and Scale Data
    # X_train, X_test (unscaled) are not used after this point, so we can ignore them.
    _, _, y_train, y_test, scaler, X_train_sc_df, X_test_sc_df = split_and_scale_data(
        df_processed, final_feature_cols, random_state=random_state
    )
    # Convert scaled DataFrames to NumPy arrays for SMOTE and model training, as models
    # typically expect NumPy arrays or DataFrames with consistent column order.
    X_train_sc = X_train_sc_df.values 
    X_test_sc = X_test_sc_df.values

    # 3. Apply SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train_sc, y_train, random_state=random_state)

    # 4. Train Models
    log_model = train_logistic_regression(X_train_sc, y_train, final_feature_cols, random_state=random_state)
    best_rf = train_random_forest(X_train_sm, y_train_sm, random_state=random_state)

    # 5. Get Predictions
    y_prob_log = log_model.predict_proba(X_test_sc)[:, 1]
    y_prob_rf = best_rf.predict_proba(X_test_sc)[:, 1]

    # 6. Evaluate Models & Generate Plots

    # Performance Curves
    plot_performance_curves(y_test, y_prob_log, y_prob_rf, save_path=os.path.join(output_dir, 'model_performance_curves.png'))

    # Model Comparison Table (at default threshold 0.5)
    metrics_log_default_th = calculate_metrics(y_test, y_prob_log)
    metrics_rf_default_th = calculate_metrics(y_test, y_prob_rf)

    metrics_df = pd.DataFrame({
        'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
        'Logistic Regression (Th=0.5)': [f"{metrics_log_default_th[m]:.3f}" for m in metrics_log_default_th.keys()],
        'Random Forest (Th=0.5)': [f"{metrics_rf_default_th[m]:.3f}" for m in metrics_rf_default_th.keys()]
    })
    print("\nModel Comparison Table (at default threshold 0.5):")
    print(metrics_df.to_markdown(index=False))

    # Cost-Sensitive Threshold Analysis
    avg_loan_amount = 15000
    loss_given_default = 0.60
    opportunity_cost_rate = 0.03

    opt_t_log, min_cost_log, opt_t_rf, min_cost_rf = plot_cost_analysis(
        y_test, y_prob_log, y_prob_rf, avg_loan_amount, loss_given_default, opportunity_cost_rate,
        save_path=os.path.join(output_dir, 'cost_threshold_analysis.png')
    )

    # Financial Confusion Matrices
    plot_financial_confusion_matrices(
        y_test, y_prob_log, y_prob_rf, opt_t_log, opt_t_rf, avg_loan_amount, loss_given_default, opportunity_cost_rate,
        save_path=os.path.join(output_dir, 'financial_confusion_matrices.png')
    )

    # Model Comparison Table (at optimal thresholds)
    metrics_log_opt_th = calculate_metrics(y_test, y_prob_log, threshold=opt_t_log)
    metrics_rf_opt_th = calculate_metrics(y_test, y_prob_rf, threshold=opt_t_rf)

    metrics_df_optimal = pd.DataFrame({
        'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
        f'Logistic Regression (Th={opt_t_log:.2f})': [f"{metrics_log_opt_th[m]:.3f}" for m in metrics_log_opt_th.keys()],
        f'Random Forest (Th={opt_t_rf:.2f})': [f"{metrics_rf_opt_th[m]:.3f}" for m in metrics_rf_opt_th.keys()]
    })
    print("\nModel Comparison Table (at optimal thresholds):")
    print(metrics_df_optimal.to_markdown(index=False))

    # Feature Importance
    plot_feature_importance(log_model, best_rf, final_feature_cols, save_path=os.path.join(output_dir, 'feature_importance_credit.png'))

    # Predicted PD Distributions
    plot_predicted_pd_distributions(
        y_test, y_prob_log, y_prob_rf,
        save_path_log=os.path.join(output_dir, 'predicted_pd_distributions_lr.png'),
        save_path_rf=os.path.join(output_dir, 'predicted_pd_distributions_rf.png')
    )

    # Calibration
    calibrated_rf_model = None
    y_prob_rf_calibrated = None
    # Ensure both classes are present in y_train_sm for CalibratedClassifierCV
    if len(np.unique(y_train_sm)) > 1 and y_train_sm.value_counts().min() > 1: # Need at least 2 samples in the minority class to form a fold for cv=2
        # Use a sensible number of CV folds for calibration, considering dataset size.
        # Max cv can't exceed minority class count. min(3, max_possible_folds)
        calibration_cv_folds = min(3, y_train_sm.value_counts().min())
        calibrated_rf_model = CalibratedClassifierCV(best_rf, method='isotonic', cv=calibration_cv_folds)
        calibrated_rf_model.fit(X_train_sm, y_train_sm)
        y_prob_rf_calibrated = calibrated_rf_model.predict_proba(X_test_sc)[:, 1]
    else:
        print("Skipping Random Forest calibration: Not enough classes or samples in training data for Cross-Validation calibration.")

    plot_all_calibration_curves(
        y_test, y_prob_log, y_prob_rf, y_prob_rf_calibrated,
        save_path=os.path.join(output_dir, 'calibration_curves.png'),
        save_path_calibrated_rf=os.path.join(output_dir, 'calibration_curve_rf_calibrated.png')
    )

    brier_log = brier_score_loss(y_test, y_prob_log)
    brier_rf = brier_score_loss(y_test, y_prob_rf)
    
    print(f"\nBrier Score (Logistic Regression): {brier_log:.4f}")
    print(f"Brier Score (Random Forest, Uncalibrated): {brier_rf:.4f}")
    brier_rf_calibrated = None
    if y_prob_rf_calibrated is not None:
        brier_rf_calibrated = brier_score_loss(y_test, y_prob_rf_calibrated)
        print(f"Brier Score (Random Forest, Calibrated): {brier_rf_calibrated:.4f}")
    print("\nA lower Brier score indicates better calibration.")

    return {
        'log_model': log_model,
        'random_forest_model': best_rf,
        'calibrated_rf_model': calibrated_rf_model,
        'scaler': scaler,
        'final_feature_cols': final_feature_cols,
        'X_test_scaled': X_test_sc, # Return scaled X_test for potential external predictions
        'y_test': y_test,
        'y_prob_log': y_prob_log,
        'y_prob_rf': y_prob_rf,
        'y_prob_rf_calibrated': y_prob_rf_calibrated,
        'optimal_threshold_log': opt_t_log,
        'min_cost_log': min_cost_log,
        'optimal_threshold_rf': opt_t_rf,
        'min_cost_rf': min_cost_rf,
        'metrics_df_default_th': metrics_df,
        'metrics_df_optimal_th': metrics_df_optimal,
        'brier_score_log': brier_log,
        'brier_score_rf': brier_rf,
        'brier_score_rf_calibrated': brier_rf_calibrated
    }


if __name__ == '__main__':
    # This block ensures the pipeline runs when the script is executed directly
    # but not when imported as a module in app.py
    print("Executing model pipeline as main script...")
    results = run_model_pipeline(filepath='lending_club_loans.csv', output_dir='./model_outputs')
    print("\nModel pipeline execution complete. Results dictionary returned.")
    # Example of accessing results:
    # print(f"Logistic Regression AUC: {results['metrics_df_default_th'].loc[results['metrics_df_default_th']['Metric'] == 'AUC', 'Logistic Regression (Th=0.5)'].iloc[0]}")
    # print(f"Optimal RF Threshold: {results['optimal_threshold_rf']:.2f}")
