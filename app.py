import streamlit as st
from source import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(page_title="QuLab: Lab 5: Credit Default Classification", layout="wide")

# Sidebar Header
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 5: Credit Default Classification")
st.divider()

# --- Session State Initialization ---

# General page state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Data & Preprocessing state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_data = None
    st.session_state.X_train_sc = None
    st.session_state.X_test_sc = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.final_feature_cols = None

# Logistic Regression Model state
if 'log_model_trained' not in st.session_state:
    st.session_state.log_model_trained = False
    st.session_state.log_model = None
    st.session_state.y_prob_log = None
    st.session_state.odds_ratio_df = None

# Random Forest Model state
if 'rf_model_trained' not in st.session_state:
    st.session_state.rf_model_trained = False
    st.session_state.best_rf = None
    st.session_state.y_prob_rf = None

# Model Comparison & Evaluation state
if 'metrics_calculated_default_th' not in st.session_state:
    st.session_state.metrics_calculated_default_th = False
    st.session_state.metrics_df_default_th = None 

# Cost-Sensitive Analysis state
if 'cost_analysis_performed' not in st.session_state:
    st.session_state.cost_analysis_performed = False
    st.session_state.opt_t_log = None
    st.session_state.min_cost_log = None
    st.session_state.opt_t_rf = None
    st.session_state.min_cost_rf = None
    st.session_state.cost_fn = None
    st.session_state.cost_fp = None
    st.session_state.metrics_df_optimal_th = None
    st.session_state.thresholds_costs_data = None

# Feature Importance state
if 'feature_importance_analyzed' not in st.session_state:
    st.session_state.feature_importance_analyzed = False
    st.session_state.combined_importance_df = None
    st.session_state.log_importance_df = None
    st.session_state.rf_importance_df = None

# PD Calibration state
if 'pd_calibration_performed' not in st.session_state:
    st.session_state.pd_calibration_performed = False
    st.session_state.y_prob_rf_calibrated = None
    st.session_state.brier_log = None
    st.session_state.brier_rf = None
    st.session_state.brier_rf_calibrated = None

# --- Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Home",
        "1. Data Preparation & Imbalance Handling",
        "2. Logistic Regression Model",
        "3. Random Forest Model",
        "4. Model Comparison & Evaluation",
        "5. Cost-Sensitive Analysis & Optimization",
        "6. Feature Importance",
        "7. PD Calibration"
    ],
    index=0 if st.session_state.page == "Home" else [
        "Home",
        "1. Data Preparation & Imbalance Handling",
        "2. Logistic Regression Model",
        "3. Random Forest Model",
        "4. Model Comparison & Evaluation",
        "5. Cost-Sensitive Analysis & Optimization",
        "6. Feature Importance",
        "7. PD Calibration"
    ].index(st.session_state.page)
)

if page_selection != st.session_state.page:
    st.session_state.page = page_selection
    st.rerun()

# --- Page Rendering Logic ---

if st.session_state.page == "Home":
    st.title("Credit Default Classification: A Financial Professional's Workflow")
    st.markdown(f"## Case Study: Optimizing Loan Risk Management at CapitalFlow Bank")
    st.markdown(f"**Persona:** Alex Chen, Senior Credit Risk Analyst at CapitalFlow Bank")
    st.markdown(f"**Organization:** CapitalFlow Bank, a mid-sized financial institution specializing in consumer lending.")
    st.markdown(f"**Scenario:** Alex is tasked with evaluating and improving CapitalFlow Bank's credit default prediction models. The current models are outdated, leading to suboptimal loan origination decisions, inaccurate risk-based pricing, and potential issues with regulatory capital requirements. Alex needs to develop and compare two models—a traditional Logistic Regression scorecard and a more flexible Random Forest classifier—to estimate the Probability of Default (PD) for new loan applicants. The goal is to provide well-calibrated PD estimates and robust insights into default drivers, explicitly accounting for critical financial considerations like class imbalance and asymmetric misclassification costs. This analysis will directly inform the bank's lending strategy, balance sheet provisioning, and compliance with regulatory frameworks like Basel III/IV and IFRS 9.")
    st.markdown(f"---")
    st.markdown(f"### **1. Setting the Stage: Assessing Credit Risk for 'CapitalFlow Bank'**")
    st.markdown(f"Alex, a Senior Credit Risk Analyst at CapitalFlow Bank, begins a crucial project to enhance the bank's loan default prediction capabilities. Accurate Probability of Default (PD) models are paramount for CapitalFlow Bank, directly influencing which loans are approved, the interest rates charged, and the capital reserves held against potential losses. This project isn't just about prediction accuracy; it's about making financially sound, compliant, and transparent lending decisions. Alex will navigate challenges like class imbalance (defaults are rare events) and the asymmetric costs of misclassifying a good loan versus missing a true default.")

elif st.session_state.page == "1. Data Preparation & Imbalance Handling":
    st.title("1. Data Preparation & Imbalance Handling")
    st.markdown(f"### **2. Unpacking Loan Data: From Raw Records to Actionable Features**")
    st.markdown(f"Alex's first task is to transform raw LendingClub loan data into a clean, structured dataset suitable for modeling. This involves filtering irrelevant entries, defining the binary target variable (default or not), selecting pertinent features, and preprocessing them. A critical step is recognizing and addressing the inherent class imbalance in credit data, where defaults are much less frequent than fully paid loans. Furthermore, Alex must be vigilant about potential data leakage, particularly concerning features like `int_rate` which can inadvertently reveal future information.")
    st.markdown(r"The target variable, $y_i$, for each loan $i$, is defined as:")
    st.markdown(r"$$ y_i = \begin{cases} 1 & \text{if loan } i \text{ defaulted (charged off or defaulted)} \\ 0 & \text{if loan } i \text{ was fully repaid} \end{cases} $$")
    st.markdown(r"where $y_i$ is the binary target variable for loan $i$.")

    st.warning("⚠️ **Practitioner Warning: Data Leakage Risk**\n"
               "The feature `int_rate` is set by LendingClub based on their own risk assessment. "
               "Including it as a predictor creates information leakage—the rate already encodes the platform's view of default risk. "
               "For this exercise, we include it with a discussion note, as it teaches the leakage concept. "
               "In production, consider excluding it or treating it as a target for a secondary model.")

    if st.button("Load and Preprocess Data"):
        with st.spinner("Loading and preprocessing data..."):
            X_train_sc, X_test_sc, y_train, y_test, final_feature_cols, df_processed = load_and_preprocess_data()
            st.session_state.df_data = df_processed
            st.session_state.X_train_sc = X_train_sc
            st.session_state.X_test_sc = X_test_sc
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.final_feature_cols = final_feature_cols
            st.session_state.data_loaded = True
            st.success("Data loaded and preprocessed!")
            st.markdown(f"Initial default rate: `{st.session_state.df_data['default'].mean():.2%}`")
            st.markdown(f"Initial dataset size: `{len(st.session_state.df_data):,}` loans")
            st.markdown(f"Training set default rate: `{st.session_state.y_train.mean():.2%}`")
            st.markdown(f"Testing set default rate: `{st.session_state.y_test.mean():.2%}`")
            st.markdown(f"Training set size: `{len(st.session_state.X_train_sc):,}`")
            st.markdown(f"Testing set size: `{len(st.session_state.X_test_sc):,}`")

    if st.session_state.data_loaded:
        st.subheader("Processed Data Head (Sample)")
        st.dataframe(st.session_state.df_data.head())

elif st.session_state.page == "2. Logistic Regression Model":
    st.title("2. Logistic Regression Model")
    st.markdown(f"### **3. Building the Traditional Scorecard: Logistic Regression for PD Estimation**")
    st.markdown(f"Alex starts with Logistic Regression, a robust and widely accepted model in finance due to its interpretability and ease of regulatory compliance. The model estimates the probability of default for each loan applicant. The coefficients of the Logistic Regression model can be directly translated into odds ratios, offering a transparent understanding of how each feature influences the likelihood of default. Given the class imbalance, Alex implements `class_weight='balanced'` to ensure the model doesn't simply favor the majority (non-default) class.")

    st.markdown(r"The Logistic Regression model uses a sigmoid function to map a linear combination of features to a probability:")
    st.markdown(r"$$ P(y_i = 1 | \mathbf{x}_i) = \sigma(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i)}} $$")
    st.markdown(r"where $P(y_i = 1 | \mathbf{x}_i)$ is the probability of default for loan $i$, $\mathbf{x}_i$ is the vector of features for loan $i$, $\beta_0$ is the intercept, and $\mathbf{\beta}^T$ is the vector of coefficients.")

    st.markdown(r"The log-odds, or logit, interpretation is:")
    st.markdown(r"$$ \log\left(\frac{P(\text{default})}{1 - P(\text{default})}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p $$")
    st.markdown(r"where $P(\text{default})$ is the probability of default, and $x_j$ are the individual features.")

    st.markdown(r"For each feature $j$, the odds ratio ($\text{OR}_j$) is derived from its coefficient $\beta_j$:")
    st.markdown(r"$$ \text{OR}_j = e^{\beta_j} $$")
    st.markdown(r"where $\text{OR}_j$ is the odds ratio for feature $j$, and $\beta_j$ is its coefficient. An odds ratio of 1.15, for example, means that for a one-unit increase in $x_j$ (holding other features constant), the odds of default are multiplied by 1.15, indicating a 15% increase in default odds.")

    st.markdown(r"The coefficients ($\mathbf{\beta}$) are estimated by maximizing the likelihood function, which is equivalent to minimizing the binary cross-entropy (or log loss):")
    st.markdown(r"$$ \mathcal{L}(\mathbf{\beta}) = \sum_{i=1}^N [y_i \log P(y_i=1|\mathbf{x}_i) + (1-y_i) \log(1 - P(y_i=1|\mathbf{x}_i))] $$")
    st.markdown(r"where $\mathcal{L}(\mathbf{\beta})$ is the likelihood function, $y_i$ is the actual outcome for loan $i$, and $P(y_i=1|\mathbf{x}_i)$ is the predicted probability of default.")

    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first on the 'Data Preparation' page.")
    else:
        if st.button("Train Logistic Regression Model"):
            with st.spinner("Training Logistic Regression model..."):
                log_model, y_prob_log, odds_ratio_df = train_logistic_regression(
                    st.session_state.X_train_sc,
                    st.session_state.y_train,
                    st.session_state.final_feature_cols
                )
                st.session_state.log_model = log_model
                st.session_state.y_prob_log = y_prob_log
                st.session_state.odds_ratio_df = odds_ratio_df
                st.session_state.log_model_trained = True
                st.success("Logistic Regression model trained!")

        if st.session_state.log_model_trained:
            st.subheader("Logistic Regression Odds Ratio Table")
            st.dataframe(st.session_state.odds_ratio_df.round(4))

elif st.session_state.page == "3. Random Forest Model":
    st.title("3. Random Forest Model")
    st.markdown(f"### **4. Enhancing Prediction with Machine Learning: Random Forest for Complex Risk Patterns**")
    st.markdown(f"While Logistic Regression provides excellent interpretability, Alex wants to explore if a more complex model like Random Forest can capture non-linear relationships and intricate feature interactions that might improve predictive power. Random Forests are ensemble models that build multiple decision trees and average their predictions. To further tackle the class imbalance, Alex uses SMOTE (Synthetic Minority Over-sampling Technique) on the training data, generating synthetic samples of the minority class. Hyperparameter tuning with `GridSearchCV` ensures the Random Forest model is optimized for performance.")

    st.markdown(r"A Random Forest constructs $B$ decision trees. Each tree is trained on a bootstrap sample of the data, considering a random subset of $m \approx \sqrt{p}$ features at each split (where $p$ is the total number of features). The ensemble prediction for the probability of default given features $\mathbf{x}$ is:")
    st.markdown(r"$$ \hat{P}(\text{default} | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{P}_b(\text{default} | \mathbf{x}) $$")
    st.markdown(r"where $\hat{P}(\text{default} | \mathbf{x})$ is the ensemble predicted probability, $B$ is the number of trees, and $\hat{P}_b(\text{default} | \mathbf{x})$ is the prediction from tree $b$.")

    st.markdown(r"When `class_weight='balanced'` is used in tree-based models, the loss function (often Gini impurity or entropy) is modified such that the samples of the minority class (defaults) are given higher weight during tree construction. For a weighted loss function, the contribution of each sample $i$ to the total loss $\mathcal{L}_w$ is adjusted by its class weight $w_{y_i}$:")
    st.markdown(r"$$ \mathcal{L}_w(\mathbf{\beta}) = \sum_{i=1}^N w_{y_i} [y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)] $$")
    st.markdown(r"where $\mathcal{L}_w(\mathbf{\beta})$ is the weighted loss function, $w_{y_i}$ is the class weight for sample $i$, $y_i$ is the actual outcome, and $\hat{p}_i$ is the predicted probability.")
    st.markdown(r"For a dataset with 5% defaults, $w_1 \approx 10$ and $w_0 \approx 0.526$, effectively making each default sample contribute significantly more to the loss, forcing the model to 'pay attention' to the minority class.")

    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first on the 'Data Preparation' page.")
    elif not st.session_state.log_model_trained:
        st.warning("Please train the Logistic Regression model first.")
    else:
        if st.button("Train Random Forest Model (with SMOTE & GridSearchCV)"):
            with st.spinner("Training Random Forest model with SMOTE and GridSearchCV (this may take a while)..."):
                best_rf, y_prob_rf = train_random_forest(
                    st.session_state.X_train_sc,
                    st.session_state.y_train,
                    st.session_state.X_test_sc,
                    st.session_state.final_feature_cols
                )
                st.session_state.best_rf = best_rf
                st.session_state.y_prob_rf = y_prob_rf
                st.session_state.rf_model_trained = True
                st.success("Random Forest model trained!")
                
                # Display best parameters
                try:
                    st.markdown(f"Best Random Forest Parameters: `{best_rf.get_params()}`")
                except:
                    st.markdown("Best Random Forest Model trained.")

        if st.session_state.rf_model_trained:
            st.info("Random Forest model ready for comparison!")

elif st.session_state.page == "4. Model Comparison & Evaluation":
    st.title("4. Model Comparison & Evaluation")
    st.markdown(f"### **5. Benchmarking Credit Models: Beyond Accuracy with Financial Metrics**")
    st.markdown(f"Alex knows that 'accuracy' alone is a misleading metric for imbalanced credit default data. A model predicting 'no default' for all loans could achieve high accuracy but would miss all actual defaults, which is financially catastrophic. Instead, Alex focuses on credit-appropriate metrics: ROC-AUC, Precision-Recall curves, and the Kolmogorov-Smirnov (KS) statistic, which are crucial for assessing a model's ability to discriminate between good and bad loans, especially for the rare default events.")

    st.warning("⚠️ **Practitioner Warning: Accuracy is NOT a Primary Metric**\n"
               "With a high non-default rate, a trivial 'predict all non-default' model achieves high accuracy but catches zero defaults. "
               "Reporting accuracy in a credit scoring context signals a fundamental misunderstanding. "
               "Always lead with AUC, KS, and the confusion matrix at a decision-relevant threshold.")

    st.markdown(r"*   **ROC-AUC (Area Under the Receiver Operating Characteristic):** Measures the model's discrimination power. Probabilistic interpretation: the chance that a randomly selected default case is scored higher than a randomly selected non-default.")
    st.markdown(r"$$ \text{AUC} = P(\hat{P}_{\text{default}} > \hat{P}_{\text{non-default}}) $$")
    st.markdown(r"where $\text{AUC}$ is the Area Under the Curve, and $\hat{P}_{\text{default}}$ and $\hat{P}_{\text{non-default}}$ are the predicted probabilities for default and non-default cases respectively. An AUC of 0.5 is random, >0.70 is acceptable, >0.80 is strong for consumer credit.")
    st.markdown(r"*   **Gini Coefficient:** An industry-standard alternative to AUC, directly related by:")
    st.markdown(r"$$ \text{Gini} = 2 \times \text{AUC} - 1 $$")
    st.markdown(r"where $\text{Gini}$ is the Gini coefficient and $\text{AUC}$ is the Area Under the Curve.")
    st.markdown(r"*   **Precision-Recall Curve:** Particularly informative for imbalanced datasets, showing the trade-off between precision and recall at various thresholds.")
    st.markdown(r"*   **Kolmogorov-Smirnov (KS) Statistic:** Measures the maximum separation between the cumulative distribution functions (CDFs) of predicted probabilities for default and non-default classes.")
    st.markdown(r"$$ \text{KS} = \max_t |F_1(t) - F_0(t)| $$")
    st.markdown(r"where $\text{KS}$ is the Kolmogorov-Smirnov statistic, $F_1(t)$ is the CDF of predicted probabilities for defaults, and $F_0(t)$ is the CDF for non-defaults. A KS > 0.30 is generally considered acceptable.")

    if not st.session_state.rf_model_trained:
        st.warning("Please ensure both Logistic Regression and Random Forest models are trained first.")
    else:
        if st.button("Calculate Performance Metrics (Default Threshold 0.5)"):
            with st.spinner("Calculating metrics..."):
                metrics_log = calculate_model_metrics(st.session_state.y_test, st.session_state.y_prob_log, threshold=0.5)
                metrics_rf = calculate_model_metrics(st.session_state.y_test, st.session_state.y_prob_rf, threshold=0.5)

                metrics_df = pd.DataFrame({
                    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
                    'Logistic Regression (Th=0.5)': [f"{metrics_log[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']],
                    'Random Forest (Th=0.5)': [f"{metrics_rf[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']]
                })
                st.session_state.metrics_df_default_th = metrics_df
                st.session_state.metrics_calculated_default_th = True
                st.success("Metrics calculated!")

        if st.session_state.metrics_calculated_default_th:
            st.subheader("Model Comparison Table (at default threshold 0.5)")
            st.dataframe(st.session_state.metrics_df_default_th.round(3))

            st.subheader("Model Performance Curves (ROC, Precision-Recall, KS)")
            with st.spinner("Generating plots..."):
                fig_curves = plot_roc_pr_ks(st.session_state.y_test, st.session_state.y_prob_log, st.session_state.y_prob_rf)
                st.pyplot(fig_curves)
                plt.close(fig_curves)

elif st.session_state.page == "5. Cost-Sensitive Analysis & Optimization":
    st.title("5. Cost-Sensitive Analysis & Optimization")
    st.markdown(f"### **6. Optimizing Lending Decisions: A Cost-Sensitive Approach to Threshold Selection**")
    st.markdown(f"This is where Alex translates model probabilities into tangible financial decisions for CapitalFlow Bank. The cost of a False Negative (approving a loan that defaults) is significantly higher than a False Positive (rejecting a loan that would have been repaid). Alex defines these asymmetric costs, calculates the Expected Total Cost across various probability thresholds, and identifies the optimal threshold that minimizes overall financial losses. This cost-sensitive approach is vital for maximizing bank profitability and managing risk effectively.")

    st.markdown(r"The Expected Total Cost for a given threshold is calculated as:")
    st.markdown(r"$$ C_{\text{total}} = \text{FN} \times (\text{LGD} \times \text{EAD}) + \text{FP} \times C_{\text{opportunity}} $$")
    st.markdown(r"Where:")
    st.markdown(r"*   $\text{FN}$ = Number of False Negatives (actual defaults predicted as non-defaults)")
    st.markdown(r"*   $\text{FP}$ = Number of False Positives (actual non-defaults predicted as defaults)")
    st.markdown(r"*   $\text{LGD}$ = Loss Given Default (percentage of exposure lost if default occurs).")
    st.markdown(r"*   $\text{EAD}$ = Exposure at Default (the outstanding loan amount at the time of default). For simplicity, we assume this is the `avg_loan_amount`.")
    st.markdown(r"*   $C_{\text{opportunity}}$ = Opportunity cost of declining a good loan (e.g., lost interest margin).")

    st.markdown(f"**Cost Assumptions for CapitalFlow Bank (defaults can be adjusted using sliders below):**")
    st.markdown(f"*   Average Loan Amount (`avg_loan_amount`): $15,000")
    st.markdown(f"*   Loss Given Default (`LGD`): 60%")
    st.markdown(f"*   Opportunity Cost Rate (`opportunity_cost_rate`): 3% (lost margin on a good loan)")
    st.markdown(f"Therefore:")
    st.markdown(f"*   `cost_fn` (Cost of a missed default) = `avg_loan_amount` $\times$ `LGD` = $15,000 $\times$ 0.60 = $9,000")
    st.markdown(f"*   `cost_fp` (Cost of declining a good loan) = `avg_loan_amount` $\times$ `opportunity_cost_rate` = $15,000 $\times$ 0.03 = $450")

    st.info("💡 **Key Insight: Optimal Threshold is Rarely 0.5**\n"
            "In credit scoring, the cost of a missed default (FN: $9,000) far exceeds the cost of a declined good loan (FP: $450). "
            "The optimal threshold is typically much lower (e.g., 0.15-0.25). This 20:1 asymmetry in costs drives this shift—a critical concept for credit professionals.")

    if not st.session_state.rf_model_trained:
        st.warning("Please ensure both Logistic Regression and Random Forest models are trained first.")
    else:
        st.sidebar.subheader("Adjust Cost Assumptions")
        avg_loan_amount_input = st.sidebar.number_input("Average Loan Amount ($)", value=15000, min_value=1000, step=1000)
        lgd_input = st.sidebar.slider("Loss Given Default (LGD)", value=0.60, min_value=0.01, max_value=1.00, step=0.01)
        opp_cost_rate_input = st.sidebar.slider("Opportunity Cost Rate (%)", value=0.03, min_value=0.01, max_value=0.10, step=0.01)

        if st.button("Perform Cost-Sensitive Optimization"):
            with st.spinner("Calculating optimal thresholds and costs..."):
                opt_t_log, min_cost_log, opt_t_rf, min_cost_rf, cost_fn, cost_fp, thresholds_costs = perform_cost_sensitive_analysis(
                    st.session_state.y_test,
                    st.session_state.y_prob_log,
                    st.session_state.y_prob_rf,
                    avg_loan_amount_input,
                    lgd_input,
                    opp_cost_rate_input
                )
                st.session_state.opt_t_log = opt_t_log
                st.session_state.min_cost_log = min_cost_log
                st.session_state.opt_t_rf = opt_t_rf
                st.session_state.min_cost_rf = min_cost_rf
                st.session_state.cost_fn = cost_fn
                st.session_state.cost_fp = cost_fp
                st.session_state.thresholds_costs_data = thresholds_costs

                # Calculate metrics at optimal thresholds for display
                metrics_log_opt_th = calculate_model_metrics(st.session_state.y_test, st.session_state.y_prob_log, threshold=opt_t_log)
                metrics_rf_opt_th = calculate_model_metrics(st.session_state.y_test, st.session_state.y_prob_rf, threshold=opt_t_rf)

                metrics_df_optimal = pd.DataFrame({
                    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
                    f'Logistic Regression (Th={opt_t_log:.2f})': [f"{metrics_log_opt_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']],
                    f'Random Forest (Th={opt_t_rf:.2f})': [f"{metrics_rf_opt_th[m]:.3f}" for m in ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall']]
                })
                st.session_state.metrics_df_optimal_th = metrics_df_optimal
                st.session_state.cost_analysis_performed = True
                st.success("Cost-sensitive analysis complete!")

        if st.session_state.cost_analysis_performed:
            st.subheader("Optimal Classification Thresholds")
            st.markdown(f"Optimal threshold for Logistic Regression: `{st.session_state.opt_t_log:.2f}` (Min Cost: `${st.session_state.min_cost_log:,.0f}`)")
            st.markdown(f"Optimal threshold for Random Forest: `{st.session_state.opt_t_rf:.2f}` (Min Cost: `${st.session_state.min_cost_rf:,.0f}`)")

            st.subheader("Cost Curve Analysis")
            with st.spinner("Generating cost curve..."):
                thresholds, costs_log, costs_rf = st.session_state.thresholds_costs_data
                fig_cost = plot_cost_curve(
                    thresholds, costs_log, costs_rf,
                    st.session_state.opt_t_log, st.session_state.min_cost_log,
                    st.session_state.opt_t_rf, st.session_state.min_cost_rf
                )
                st.pyplot(fig_cost)
                plt.close(fig_cost)

            st.subheader("Financial Impact of Model Decisions (Confusion Matrices)")
            col1, col2 = st.columns(2)
            with col1:
                with st.spinner("Generating LR confusion matrix..."):
                    fig_cm_lr = plot_financial_confusion_matrix(
                        st.session_state.y_test, st.session_state.y_prob_log, st.session_state.opt_t_log,
                        st.session_state.cost_fn, st.session_state.cost_fp, 'Logistic Regression'
                    )
                    st.pyplot(fig_cm_lr)
                    plt.close(fig_cm_lr)
            with col2:
                with st.spinner("Generating RF confusion matrix..."):
                    fig_cm_rf = plot_financial_confusion_matrix(
                        st.session_state.y_test, st.session_state.y_prob_rf, st.session_state.opt_t_rf,
                        st.session_state.cost_fn, st.session_state.cost_fp, 'Random Forest'
                    )
                    st.pyplot(fig_cm_rf)
                    plt.close(fig_cm_rf)

            st.subheader("Model Comparison Table (at Optimal Thresholds)")
            st.dataframe(st.session_state.metrics_df_optimal_th.round(3))

elif st.session_state.page == "6. Feature Importance":
    st.title("6. Feature Importance")
    st.markdown(f"### **7. Deconstructing Default Drivers: Comparing Feature Insights**")
    st.markdown(f"To gain deeper insights into credit risk, Alex needs to understand which factors are most influential in predicting default. This involves extracting and comparing feature importance from both the Logistic Regression and Random Forest models. While Logistic Regression provides clear coefficient magnitudes (and odds ratios), Random Forest uses Gini impurity to rank features. This comparison helps CapitalFlow Bank understand different perspectives on risk drivers and informs targeted interventions. It also highlights the trade-off between model interpretability and predictive power, a key consideration for regulatory transparency.")

    st.info("📚 **Discussion Point: Explainability-Accuracy Trade-off**\n"
            "Logistic Regression is highly transparent (coefficients directly explain impact), which is often a regulatory requirement. "
            "Random Forests, while potentially more accurate, are less inherently interpretable and may require additional XAI tools (like SHAP/LIME) for individual-level explanations.")

    st.info("📊 **Discussion Point: AUC Gap Narrows with Feature Engineering**\n"
            "Much of the Random Forest's advantage comes from automatically capturing non-linearities and interactions. "
            "However, if an analyst manually engineers interaction terms and non-linear transforms for the Logistic Regression model, "
            "the AUC gap often shrinks, demonstrating the value of domain expertise in feature engineering.")

    if not st.session_state.rf_model_trained:
        st.warning("Please ensure both Logistic Regression and Random Forest models are trained first.")
    else:
        if st.button("Analyze Feature Importance"):
            with st.spinner("Analyzing feature importance..."):
                combined_importance_df, log_importance_df, rf_importance_df = analyze_feature_importance(
                    st.session_state.log_model,
                    st.session_state.best_rf,
                    st.session_state.final_feature_cols
                )
                st.session_state.combined_importance_df = combined_importance_df
                st.session_state.log_importance_df = log_importance_df
                st.session_state.rf_importance_df = rf_importance_df
                st.session_state.feature_importance_analyzed = True
                st.success("Feature importance analyzed!")

        if st.session_state.feature_importance_analyzed:
            st.subheader("Feature Importance Comparison Table")
            st.dataframe(st.session_state.combined_importance_df[['Feature', 'LogReg Rank', 'RF Rank', 'LogReg |Coef|', 'RF Importance']].round(4))

            st.subheader("Feature Importance Bar Charts")
            with st.spinner("Generating feature importance bar charts..."):
                fig_fi = plot_feature_importance_bars(
                    st.session_state.log_importance_df,
                    st.session_state.rf_importance_df,
                    st.session_state.final_feature_cols
                )
                st.pyplot(fig_fi)
                plt.close(fig_fi)

            st.subheader("Predicted Probability of Default (PD) Distributions")
            with st.spinner("Generating PD distributions..."):
                fig_pd_lr, fig_pd_rf = plot_predicted_pd_distributions(
                    st.session_state.y_test,
                    st.session_state.y_prob_log,
                    st.session_state.y_prob_rf
                )
                st.pyplot(fig_pd_lr)
                plt.close(fig_pd_lr)
                st.pyplot(fig_pd_rf)
                plt.close(fig_pd_rf)

elif st.session_state.page == "7. PD Calibration":
    st.title("7. PD Calibration")
    st.markdown(f"### **8. Ensuring Reliable Probabilities: Calibrating PD Estimates**")
    st.markdown(f"For CapitalFlow Bank, it's not enough for a model to simply rank borrowers by risk; the predicted probabilities of default (PDs) must also be well-calibrated. This means that if a model predicts a 10% PD, approximately 10% of those borrowers should actually default. Well-calibrated PDs are essential for accurate risk-based pricing, regulatory provisioning (e.g., IFRS 9 Expected Credit Loss calculations), and portfolio stress testing. Alex uses calibration curves to assess how closely the predicted PDs align with observed default rates.")

    st.markdown(r"A calibration curve (also known as a reliability diagram) plots the mean predicted probability for a given bin against the fraction of positive outcomes (observed default rate) in that bin. A perfectly calibrated model would have its points fall along the diagonal line $y=x$.")

    st.info("💰 **Discussion Point: Why Probability Calibration Matters**\n"
            "AUC measures ranking quality, not calibration. For accurate risk-based pricing (rate based on PD) and provisioning (ECL = PD × LGD × EAD), well-calibrated probabilities are essential. Isotonic regression or Platt scaling can be applied to improve calibration if needed.")

    if not st.session_state.rf_model_trained:
        st.warning("Please ensure both Logistic Regression and Random Forest models are trained first.")
    else:
        if st.button("Assess and Calibrate PDs"):
            with st.spinner("Performing PD calibration and Brier score calculation..."):
                y_prob_rf_calibrated, brier_log, brier_rf, brier_rf_calibrated = perform_pd_calibration(
                    st.session_state.y_test,
                    st.session_state.y_prob_log,
                    st.session_state.y_prob_rf,
                    st.session_state.best_rf,
                    st.session_state.X_train_sc,
                    st.session_state.y_train,
                    st.session_state.X_test_sc
                )
                st.session_state.y_prob_rf_calibrated = y_prob_rf_calibrated
                st.session_state.brier_log = brier_log
                st.session_state.brier_rf = brier_rf
                st.session_state.brier_rf_calibrated = brier_rf_calibrated
                st.session_state.pd_calibration_performed = True
                st.success("PD calibration and Brier scores calculated!")

        if st.session_state.pd_calibration_performed:
            st.subheader("Brier Scores")
            st.markdown(f"Brier Score (Logistic Regression): `{st.session_state.brier_log:.4f}`")
            st.markdown(f"Brier Score (Random Forest, Uncalibrated): `{st.session_state.brier_rf:.4f}`")
            st.markdown(f"Brier Score (Random Forest, Calibrated): `{st.session_state.brier_rf_calibrated:.4f}`")
            st.markdown(f"A lower Brier score indicates better calibration.")

            st.subheader("Calibration Curves")
            with st.spinner("Generating calibration curves..."):
                fig_cal_uncal, fig_cal_cal = plot_calibration_curves(
                    st.session_state.y_test,
                    st.session_state.y_prob_log,
                    st.session_state.y_prob_rf,
                    st.session_state.y_prob_rf_calibrated
                )
                st.pyplot(fig_cal_uncal)
                plt.close(fig_cal_uncal)
                st.pyplot(fig_cal_cal)
                plt.close(fig_cal_cal)
