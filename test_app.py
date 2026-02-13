
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np

# --- Dummy Data and Mock Objects for Session State ---
# These are used to pre-populate session state for tests that
# depend on previous steps (e.g., data loaded, models trained)
# and to simulate the data structures the app expects.

# Dummy data for general data & preprocessing state
DUMMY_DF_DATA = pd.DataFrame({
    'feature_0': np.random.rand(10),
    'feature_1': np.random.rand(10),
    'default': np.random.randint(0, 2, 10)
})
DUMMY_X_TRAIN_SC = pd.DataFrame(np.random.rand(8, 2), columns=['feature_0', 'feature_1'])
DUMMY_X_TEST_SC = pd.DataFrame(np.random.rand(2, 2), columns=['feature_0', 'feature_1'])
DUMMY_Y_TRAIN = pd.Series(np.random.randint(0, 2, 8))
DUMMY_Y_TEST = pd.Series(np.random.randint(0, 2, 2))
DUMMY_FINAL_FEATURE_COLS = ['feature_0', 'feature_1']

# Mock model classes (simplified for testing session state persistence)
class MockLogisticRegressionModel:
    def predict_proba(self, X):
        return np.random.rand(len(X), 2)
    @property
    def coef_(self):
        return np.random.rand(len(DUMMY_FINAL_FEATURE_COLS))
    @property
    def intercept_(self):
        return np.array([0.5])

class MockRandomForestModel:
    def predict_proba(self, X):
        return np.random.rand(len(X), 2)
    def get_params(self):
        return {'n_estimators': 100, 'max_depth': 10}

# Dummy model objects
DUMMY_LOG_MODEL = MockLogisticRegressionModel()
DUMMY_RF_MODEL = MockRandomForestModel()

# Dummy data for Logistic Regression Model state
DUMMY_Y_PROB_LOG = np.random.rand(len(DUMMY_Y_TEST))
DUMMY_ODDS_RATIO_DF = pd.DataFrame({
    'Feature': DUMMY_FINAL_FEATURE_COLS,
    'Coefficient': np.random.rand(len(DUMMY_FINAL_FEATURE_COLS)),
    'Odds Ratio': np.random.rand(len(DUMMY_FINAL_FEATURE_COLS)) + 1
})

# Dummy data for Random Forest Model state
DUMMY_Y_PROB_RF = np.random.rand(len(DUMMY_Y_TEST))

# Dummy data for Model Comparison & Evaluation state
DUMMY_METRICS_DF_DEFAULT_TH = pd.DataFrame({
    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
    'Logistic Regression (Th=0.5)': ['0.750', '0.500', '0.400', '0.600', '0.550', '0.650'],
    'Random Forest (Th=0.5)': ['0.800', '0.600', '0.500', '0.700', '0.650', '0.750']
})

# Dummy data for Cost-Sensitive Analysis state
DUMMY_OPT_T_LOG = 0.15
DUMMY_MIN_COST_LOG = 150000.0
DUMMY_OPT_T_RF = 0.12
DUMMY_MIN_COST_RF = 120000.0
DUMMY_COST_FN = 9000
DUMMY_COST_FP = 450
DUMMY_THRESHOLDS_COSTS_DATA = (
    np.linspace(0, 1, 100), # thresholds
    np.random.rand(100) * 100000, # costs_log
    np.random.rand(100) * 90000 # costs_rf
)
DUMMY_METRICS_DF_OPTIMAL_TH = pd.DataFrame({
    'Metric': ['AUC', 'Gini', 'KS Statistic', 'F1-score', 'Precision', 'Recall'],
    f'Logistic Regression (Th={DUMMY_OPT_T_LOG:.2f})': ['0.760', '0.520', '0.420', '0.620', '0.570', '0.670'],
    f'Random Forest (Th={DUMMY_OPT_T_RF:.2f})': ['0.810', '0.610', '0.510', '0.710', '0.660', '0.760']
})


# Dummy data for Feature Importance state
DUMMY_COMBINED_IMPORTANCE_DF = pd.DataFrame({
    'Feature': DUMMY_FINAL_FEATURE_COLS,
    'LogReg Rank': [1, 2],
    'RF Rank': [2, 1],
    'LogReg |Coef|': np.random.rand(2),
    'RF Importance': np.random.rand(2)
})
DUMMY_LOG_IMPORTANCE_DF = DUMMY_COMBINED_IMPORTANCE_DF[['Feature', 'LogReg |Coef|']].rename(columns={'LogReg |Coef|': 'Importance'})
DUMMY_RF_IMPORTANCE_DF = DUMMY_COMBINED_IMPORTANCE_DF[['Feature', 'RF Importance']].rename(columns={'RF Importance': 'Importance'})

# Dummy data for PD Calibration state
DUMMY_Y_PROB_RF_CALIBRATED = np.random.rand(len(DUMMY_Y_TEST))
DUMMY_BRIER_LOG = 0.15
DUMMY_BRIER_RF = 0.12
DUMMY_BRIER_RF_CALIBRATED = 0.10


# --- Test Functions ---

def test_home_page_navigation_and_content():
    """Tests if the Home page loads correctly and displays expected content."""
    at = AppTest.from_file("app.py").run()
    assert at.sidebar.selectbox[0].value == "Home"
    assert "Credit Default Classification: A Financial Professional's Workflow" in at.title[0].value
    assert "Optimizing Loan Risk Management at CapitalFlow Bank" in at.markdown[1].value


def test_data_preparation_page_initial_state():
    """Tests the initial state of the Data Preparation page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data Preparation & Imbalance Handling").run()
    assert "1. Data Preparation & Imbalance Handling" in at.title[0].value
    assert at.warning[0].value == "⚠️ **Practitioner Warning: Data Leakage Risk**\nThe feature `int_rate` is set by LendingClub based on their own risk assessment. Including it as a predictor creates information leakage—the rate already encodes the platform's view of default risk. For this exercise, we include it with a discussion note, as it teaches the leakage concept. In production, consider excluding it or treating it as a target for a secondary model."
    assert at.button[0].label == "Load and Preprocess Data"
    assert not at.session_state["data_loaded"]


def test_data_preparation_load_data():
    """Tests loading and preprocessing data on the Data Preparation page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("1. Data Preparation & Imbalance Handling").run()
    at.button[0].click().run() # Click 'Load and Preprocess Data'
    assert at.success[0].value == "Data loaded and preprocessed!"
    assert at.session_state["data_loaded"] is True
    assert at.dataframe[0].exists # Check if dataframe is displayed
    assert "Initial default rate:" in at.markdown[4].value


def test_logistic_regression_page_initial_state():
    """Tests the initial state of the Logistic Regression Model page without data loaded."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("2. Logistic Regression Model").run()
    assert "2. Logistic Regression Model" in at.title[0].value
    assert at.warning[0].value == "Please load and preprocess data first on the 'Data Preparation' page."
    assert at.button[0].disabled is True # Button should be disabled if data not loaded


def test_logistic_regression_train_model():
    """Tests training the Logistic Regression model."""
    at = AppTest.from_file("app.py")
    # Simulate data loaded state
    at.session_state["data_loaded"] = True
    at.session_state["X_train_sc"] = DUMMY_X_TRAIN_SC
    at.session_state["y_train"] = DUMMY_Y_TRAIN
    at.session_state["final_feature_cols"] = DUMMY_FINAL_FEATURE_COLS
    at.run()
    at.sidebar.selectbox[0].set_value("2. Logistic Regression Model").run()
    at.button[0].click().run() # Click 'Train Logistic Regression Model'
    assert at.success[0].value == "Logistic Regression model trained!"
    assert at.session_state["log_model_trained"] is True
    assert at.session_state["log_model"] is not None
    assert at.dataframe[0].exists # Check if odds ratio table is displayed


def test_random_forest_page_initial_state():
    """Tests the initial state of the Random Forest Model page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("3. Random Forest Model").run()
    assert "3. Random Forest Model" in at.title[0].value
    assert at.warning[0].value == "Please load and preprocess data first on the 'Data Preparation' page."


def test_random_forest_train_model():
    """Tests training the Random Forest model."""
    at = AppTest.from_file("app.py")
    # Simulate data and LR model trained state
    at.session_state["data_loaded"] = True
    at.session_state["X_train_sc"] = DUMMY_X_TRAIN_SC
    at.session_state["y_train"] = DUMMY_Y_TRAIN
    at.session_state["X_test_sc"] = DUMMY_X_TEST_SC
    at.session_state["final_feature_cols"] = DUMMY_FINAL_FEATURE_COLS
    at.session_state["log_model_trained"] = True # RF depends on LR being trained
    at.run()
    at.sidebar.selectbox[0].set_value("3. Random Forest Model").run()
    at.button[0].click().run() # Click 'Train Random Forest Model'
    assert at.success[0].value == "Random Forest model trained!"
    assert at.session_state["rf_model_trained"] is True
    assert at.session_state["best_rf"] is not None
    assert "Best Random Forest Parameters" in at.markdown[6].value


def test_model_comparison_page_initial_state():
    """Tests the initial state of the Model Comparison page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("4. Model Comparison & Evaluation").run()
    assert "4. Model Comparison & Evaluation" in at.title[0].value
    assert at.warning[0].value == "Please ensure both Logistic Regression and Random Forest models are trained first."


def test_model_comparison_calculate_metrics():
    """Tests calculating performance metrics on the Model Comparison page."""
    at = AppTest.from_file("app.py")
    # Simulate all necessary states
    at.session_state["data_loaded"] = True
    at.session_state["log_model_trained"] = True
    at.session_state["rf_model_trained"] = True
    at.session_state["y_test"] = DUMMY_Y_TEST
    at.session_state["y_prob_log"] = DUMMY_Y_PROB_LOG
    at.session_state["y_prob_rf"] = DUMMY_Y_PROB_RF
    at.run()
    at.sidebar.selectbox[0].set_value("4. Model Comparison & Evaluation").run()
    at.button[0].click().run() # Click 'Calculate Performance Metrics'
    assert at.success[0].value == "Metrics calculated!"
    assert at.session_state["metrics_calculated_default_th"] is True
    assert at.dataframe[0].exists # Check if model comparison table is displayed
    assert at.pyplot[0].exists # Check if performance curves plot is displayed


def test_cost_sensitive_analysis_page_initial_state():
    """Tests the initial state of the Cost-Sensitive Analysis page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("5. Cost-Sensitive Analysis & Optimization").run()
    assert "5. Cost-Sensitive Analysis & Optimization" in at.title[0].value
    assert at.warning[0].value == "Please ensure both Logistic Regression and Random Forest models are trained first."


def test_cost_sensitive_analysis_perform_optimization():
    """Tests performing cost-sensitive optimization."""
    at = AppTest.from_file("app.py")
    # Simulate all necessary states
    at.session_state["data_loaded"] = True
    at.session_state["log_model_trained"] = True
    at.session_state["rf_model_trained"] = True
    at.session_state["y_test"] = DUMMY_Y_TEST
    at.session_state["y_prob_log"] = DUMMY_Y_PROB_LOG
    at.session_state["y_prob_rf"] = DUMMY_Y_PROB_RF
    at.run()
    at.sidebar.selectbox[0].set_value("5. Cost-Sensitive Analysis & Optimization").run()
    # Interact with sliders if desired, then click button
    at.sidebar.number_input[0].set_value(20000).run()
    at.sidebar.slider[0].set_value(0.70).run()
    at.sidebar.slider[1].set_value(0.05).run()
    at.button[0].click().run() # Click 'Perform Cost-Sensitive Optimization'
    assert at.success[0].value == "Cost-sensitive analysis complete!"
    assert at.session_state["cost_analysis_performed"] is True
    assert "Optimal threshold for Logistic Regression:" in at.markdown[4].value
    assert "Optimal threshold for Random Forest:" in at.markdown[5].value
    assert at.pyplot[0].exists # Check for cost curve plot
    assert at.pyplot[1].exists # Check for LR confusion matrix plot
    assert at.pyplot[2].exists # Check for RF confusion matrix plot
    assert at.dataframe[0].exists # Check for optimal thresholds metrics table


def test_feature_importance_page_initial_state():
    """Tests the initial state of the Feature Importance page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("6. Feature Importance").run()
    assert "6. Feature Importance" in at.title[0].value
    assert at.warning[0].value == "Please ensure both Logistic Regression and Random Forest models are trained first."


def test_feature_importance_analyze():
    """Tests analyzing feature importance."""
    at = AppTest.from_file("app.py")
    # Simulate all necessary states
    at.session_state["data_loaded"] = True
    at.session_state["log_model_trained"] = True
    at.session_state["rf_model_trained"] = True
    at.session_state["log_model"] = DUMMY_LOG_MODEL
    at.session_state["best_rf"] = DUMMY_RF_MODEL
    at.session_state["final_feature_cols"] = DUMMY_FINAL_FEATURE_COLS
    at.session_state["y_test"] = DUMMY_Y_TEST
    at.session_state["y_prob_log"] = DUMMY_Y_PROB_LOG
    at.session_state["y_prob_rf"] = DUMMY_Y_PROB_RF
    at.run()
    at.sidebar.selectbox[0].set_value("6. Feature Importance").run()
    at.button[0].click().run() # Click 'Analyze Feature Importance'
    assert at.success[0].value == "Feature importance analyzed!"
    assert at.session_state["feature_importance_analyzed"] is True
    assert at.dataframe[0].exists # Check for importance comparison table
    assert at.pyplot[0].exists # Check for feature importance bar charts
    assert at.pyplot[1].exists # Check for PD distributions (LR)
    assert at.pyplot[2].exists # Check for PD distributions (RF)


def test_pd_calibration_page_initial_state():
    """Tests the initial state of the PD Calibration page."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("7. PD Calibration").run()
    assert "7. PD Calibration" in at.title[0].value
    assert at.warning[0].value == "Please ensure both Logistic Regression and Random Forest models are trained first."


def test_pd_calibration_assess_and_calibrate():
    """Tests assessing and calibrating PDs."""
    at = AppTest.from_file("app.py")
    # Simulate all necessary states
    at.session_state["data_loaded"] = True
    at.session_state["log_model_trained"] = True
    at.session_state["rf_model_trained"] = True
    at.session_state["y_test"] = DUMMY_Y_TEST
    at.session_state["y_prob_log"] = DUMMY_Y_PROB_LOG
    at.session_state["y_prob_rf"] = DUMMY_Y_PROB_RF
    at.session_state["best_rf"] = DUMMY_RF_MODEL
    at.session_state["X_train_sc"] = DUMMY_X_TRAIN_SC
    at.session_state["y_train"] = DUMMY_Y_TRAIN
    at.session_state["X_test_sc"] = DUMMY_X_TEST_SC
    at.run()
    at.sidebar.selectbox[0].set_value("7. PD Calibration").run()
    at.button[0].click().run() # Click 'Assess and Calibrate PDs'
    assert at.success[0].value == "PD calibration and Brier scores calculated!"
    assert at.session_state["pd_calibration_performed"] is True
    assert "Brier Score (Logistic Regression):" in at.markdown[4].value
    assert "Brier Score (Random Forest, Uncalibrated):" in at.markdown[5].value
    assert "Brier Score (Random Forest, Calibrated):" in at.markdown[6].value
    assert at.pyplot[0].exists # Check for uncalibrated calibration curve
    assert at.pyplot[1].exists # Check for calibrated calibration curve

