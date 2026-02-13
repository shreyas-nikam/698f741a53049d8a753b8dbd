Here's a comprehensive `README.md` for your Streamlit application lab project, designed for both developers and users, with a focus on its educational and financial context.

---

# 🚀 QuLab: Lab 5: Credit Default Classification - A Financial Professional's Workflow

## Optimizing Loan Risk Management at CapitalFlow Bank

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

This Streamlit application, "QuLab: Lab 5: Credit Default Classification," is an interactive lab project designed for financial professionals, particularly Credit Risk Analysts, to explore, build, and compare credit default prediction models. It simulates a real-world scenario at **CapitalFlow Bank**, where a Senior Credit Risk Analyst, **Alex Chen**, is tasked with enhancing the bank's loan default prediction capabilities.

The project goes beyond mere predictive accuracy, delving into critical financial considerations such as:
*   **Class Imbalance:** Defaults are rare, but highly impactful events.
*   **Asymmetric Misclassification Costs:** The financial cost of approving a bad loan (False Negative) is significantly higher than rejecting a good loan (False Positive).
*   **Regulatory Compliance:** Emphasizing model interpretability, robust evaluation, and PD (Probability of Default) calibration crucial for frameworks like Basel III/IV and IFRS 9.

Users can interactively progress through a typical credit modeling workflow, from data preparation and model training (Logistic Regression and Random Forest) to advanced evaluation techniques, cost-sensitive optimization, feature importance analysis, and PD calibration.

---

## ✨ Features

This application provides a guided, interactive workflow covering the essential stages of credit default modeling:

1.  **Data Preparation & Imbalance Handling:**
    *   Load and preprocess raw loan data.
    *   Define the binary target variable (`default`).
    *   Address class imbalance using techniques like `class_weight='balanced'` and `SMOTE`.
    *   **Practitioner Warning:** Discussion and awareness of data leakage (e.g., `int_rate`).

2.  **Logistic Regression Model:**
    *   Train a traditional, interpretable Logistic Regression model.
    *   Estimate Probability of Default (PD).
    *   Visualize and interpret **Odds Ratios** to understand feature impact on default likelihood.
    *   Mathematical explanations of the Logistic Regression model and likelihood function.

3.  **Random Forest Model:**
    *   Train a more complex ensemble model, Random Forest, to capture non-linear relationships and interactions.
    *   Utilize `SMOTE` and `GridSearchCV` for hyperparameter tuning and robust imbalance handling.
    *   Mathematical explanation of Random Forest and weighted loss functions.

4.  **Model Comparison & Evaluation:**
    *   Evaluate both models using financially relevant metrics beyond simple accuracy:
        *   **ROC-AUC** & **Gini Coefficient** (discrimination power).
        *   **Kolmogorov-Smirnov (KS) Statistic** (separation between default/non-default distributions).
        *   **Precision-Recall Curve** (critical for imbalanced datasets).
    *   **Practitioner Warning:** Emphasizing why accuracy is often a misleading metric in credit scoring.
    *   Interactive plots for visual comparison.

5.  **Cost-Sensitive Analysis & Optimization:**
    *   Define **asymmetric misclassification costs** (Cost of False Negative vs. False Positive).
    *   Interactively adjust key financial parameters (Average Loan Amount, LGD, Opportunity Cost).
    *   Calculate **Expected Total Cost** across a range of probability thresholds.
    *   Identify the **optimal classification threshold** that minimizes overall financial losses.
    *   Visualize **cost curves** and **financial confusion matrices** at optimal thresholds.
    *   **Key Insight:** Demonstrating why the optimal threshold is rarely 0.5 in credit risk.

6.  **Feature Importance:**
    *   Compare feature importance derived from both Logistic Regression (coefficient magnitudes/odds ratios) and Random Forest (Gini importance).
    *   Understand the **explainability-accuracy trade-off** in model selection.
    *   Visualizations of feature importance and predicted PD distributions.

7.  **PD Calibration:**
    *   Assess the calibration of predicted PDs using **Calibration Curves (Reliability Diagrams)**.
    *   Calculate **Brier Scores** to quantify calibration quality.
    *   Apply **Isotonic Regression** to calibrate Random Forest PDs for improved reliability.
    *   **Discussion Point:** Highlight the importance of calibrated probabilities for risk-based pricing, regulatory provisioning (IFRS 9), and portfolio stress testing.

---

## ⚙️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-credit-default-classification.git
    cd quolab-credit-default-classification
    ```
    *(Note: Replace `your-username` with the actual GitHub username/organization if this project is hosted.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Ensure your virtual environment is active.
2.  Run the Streamlit application from the project root directory:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Your web browser should automatically open to the Streamlit app (usually `http://localhost:8501`).

---

## 🚀 Usage

Upon launching the application:

1.  **Navigate:** Use the sidebar on the left to navigate through the different sections of the lab project.
2.  **Follow the Workflow:** Start from "Home" and proceed sequentially through "1. Data Preparation & Imbalance Handling" to "7. PD Calibration".
3.  **Interact:** Click the "Load and Preprocess Data" button and subsequent action buttons (e.g., "Train Logistic Regression Model") to trigger computations and visualizations for each step.
4.  **Adjust Parameters:** On the "Cost-Sensitive Analysis & Optimization" page, use the sliders in the sidebar to modify cost assumptions and observe their impact.
5.  **Explore:** Read the descriptive text, practitioner warnings, and discussion points on each page to gain deeper insights into the financial and machine learning concepts.

---

## 📂 Project Structure

```
quolab-credit-default-classification/
├── streamlit_app.py        # Main Streamlit application file
├── source.py               # Contains helper functions for data processing, model training, and plotting
├── requirements.txt        # List of Python dependencies
├── README.md               # This README file
└── assets/                 # (Optional) Directory for images or other static assets
    └── logo5.jpg           # QuantUniversity logo (referenced in app)
```

The `source.py` file encapsulates the core logic, keeping `streamlit_app.py` clean and focused on the UI and state management.

---

## 💻 Technology Stack

*   **Python 3.x**: The core programming language.
*   **Streamlit**: For creating the interactive web application user interface.
*   **Pandas**: Essential for data manipulation and analysis.
*   **NumPy**: Fundamental package for numerical computation.
*   **Matplotlib**: For static, high-quality plotting and visualization.
*   **Scikit-learn**: Comprehensive library for machine learning models, preprocessing, and evaluation metrics (e.g., `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `roc_curve`, `precision_recall_curve`, `calibration_curve`, `BrierScoreLoss`).
*   **Imbalanced-learn (imblearn)**: For advanced techniques to handle imbalanced datasets, specifically `SMOTE`.

---

## 🤝 Contributing

This project is primarily intended as a lab exercise for educational purposes. Contributions are welcome, especially for:

*   Bug fixes
*   Improvements to existing visualizations
*   Enhancements to the educational content or practitioner insights
*   Refactoring code for better readability or efficiency

To contribute:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable, otherwise state it's for educational use).

---

## 📧 Contact

For questions or feedback, please contact:

*   **QuantUniversity** - [Website](https://www.quantuniversity.com/)
*   **Your Name/Email** - [your.email@example.com](mailto:your.email@example.com) (Optional)

---