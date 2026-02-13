id: 698f741a53049d8a753b8dbd_user_guide
summary: Lab 5: Credit Default Classification User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 5: Credit Default Classification: A Financial Professional's Workflow

## 1. Introduction to Credit Default Classification
Duration: 0:05

Welcome to this codelab, designed to guide you through the critical process of **Credit Default Classification** from the perspective of a financial professional. In the world of finance, accurately predicting whether a borrower will default on a loan is paramount. It directly impacts a bank's profitability, risk management, and compliance with stringent regulatory frameworks like Basel III/IV and IFRS 9.

Imagine you are Alex Chen, a Senior Credit Risk Analyst at CapitalFlow Bank. Your main goal is to improve the bank's loan default prediction models to make smarter lending decisions, price loans more effectively based on risk, and ensure the bank meets its capital requirements. You'll be working with real-world challenges such as **class imbalance** (defaults are rare events) and **asymmetric misclassification costs** (the financial impact of approving a bad loan is far greater than rejecting a good one).

This guide will walk you through building and comparing two key models—a traditional **Logistic Regression scorecard** and a more flexible **Random Forest classifier**—to estimate the Probability of Default (PD) for new loan applicants. You'll gain practical experience in:

*   **Data Preparation:** Transforming raw loan data into actionable features.
*   **Model Building:** Developing both interpretable and complex predictive models.
*   **Performance Evaluation:** Assessing models using financially relevant metrics, not just simple accuracy.
*   **Cost-Sensitive Optimization:** Translating predictions into financially optimal decisions.
*   **Feature Importance:** Understanding the key drivers of credit default.
*   **PD Calibration:** Ensuring the predicted probabilities are reliable for regulatory and business use.

By the end of this codelab, you'll have a comprehensive understanding of the workflow involved in developing and deploying credit default models in a financial institution.

## 2. Preparing and Understanding Your Loan Data
Duration: 0:08

Alex's journey begins with transforming raw LendingClub loan data into a clean, structured dataset ready for analysis. This step is crucial and involves several key considerations for financial data.

### Defining the Target Variable
Our goal is to predict default. The target variable, $y_i$, for each loan $i$, is defined as:

$$ y_i = \begin{cases} 1 & \text{if loan } i \text{ defaulted (charged off or defaulted)} \\ 0 & \text{if loan } i \text{ was fully repaid} \end{cases} $$

Where $y_i$ is a binary variable indicating whether a loan defaulted (1) or was fully repaid (0).

### Addressing Data Leakage
<aside class="negative">
⚠️ <b>Practitioner Warning: Data Leakage Risk</b>
<br>
The feature `int_rate` (interest rate) is often set by the lending platform based on their internal risk assessment. Including it as a predictor can introduce **information leakage**—meaning the model might inadvertently use future or already-known information to predict the past. For this exercise, we include it to illustrate the concept of leakage, but in a real-world production system, you would carefully consider excluding or treating it differently.
</aside>

### Handling Class Imbalance
Credit default datasets are inherently imbalanced; the number of defaulted loans is typically much smaller than the number of fully repaid loans. It's important to be aware of this, as it influences model training and evaluation later.

To begin, navigate to the "1. Data Preparation & Imbalance Handling" page in the application's sidebar.

Click the "Load and Preprocess Data" button.

Once the data is loaded, you will see the initial default rate, the size of the dataset, and how it's split into training and testing sets. You'll also see a sample of the processed data, giving you a glimpse of the features we'll be using.

## 3. Building a Transparent Logistic Regression Model
Duration: 0:10

Now that our data is ready, Alex starts with a fundamental and highly interpretable model: **Logistic Regression**. This model is a cornerstone in financial risk modeling due to its transparency, ease of understanding, and regulatory acceptance. It directly estimates the probability of default (PD) for each loan applicant.

### The Logistic Regression Model
The Logistic Regression model uses a sigmoid function to map a linear combination of features to a probability:

$$ P(y_i = 1 | \mathbf{x}_i) = \sigma(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{x}_i)}} $$

where $P(y_i = 1 | \mathbf{x}_i)$ is the probability of default for loan $i$, $\mathbf{x}_i$ is the vector of features for loan $i$, $\beta_0$ is the intercept, and $\mathbf{\beta}^T$ is the vector of coefficients.

### Interpreting Odds Ratios
A key advantage of Logistic Regression is that its coefficients can be translated into **odds ratios**, which provide a clear, intuitive understanding of how each feature influences the likelihood of default. The log-odds, or logit, interpretation is:

$$ \log\left(\frac{P(\text{default})}{1 - P(\text{default})}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p $$

For each feature $j$, the odds ratio ($\text{OR}_j$) is derived from its coefficient $\beta_j$:

$$ \text{OR}_j = e^{\beta_j} $$

An odds ratio of 1.15, for example, means that for a one-unit increase in $x_j$ (holding other features constant), the odds of default are multiplied by 1.15, indicating a 15% increase in default odds. This direct interpretability is invaluable for regulatory reporting and stakeholder communication.

The coefficients ($\mathbf{\beta}$) are estimated by minimizing the binary cross-entropy (or log loss):

$$ \mathcal{L}(\mathbf{\beta}) = \sum_{i=1}^N [y_i \log P(y_i=1|\mathbf{x}_i) + (1-y_i) \log(1 - P(y_i=1|\mathbf{x}_i))] $$

In our application, we use `class_weight='balanced'` during training to account for the class imbalance, ensuring the model doesn't simply favor the majority (non-default) class.

Navigate to the "2. Logistic Regression Model" page.

Click the "Train Logistic Regression Model" button.

Once training is complete, examine the "Logistic Regression Odds Ratio Table". This table shows you the impact of each feature on the odds of default.

## 4. Leveraging Random Forest for Advanced Risk Detection
Duration: 0:15

While Logistic Regression offers excellent interpretability, Alex wants to explore if a more sophisticated model can capture non-linear relationships and complex feature interactions to enhance predictive power. This leads us to the **Random Forest** classifier.

### How Random Forest Works
Random Forests are powerful **ensemble models** that build multiple decision trees and combine their predictions. Each tree is trained on a slightly different subset of the data (bootstrap sampling), and at each split point, only a random subset of features is considered. This "randomness" helps reduce overfitting and improves the model's generalization capabilities. The final prediction is an average across all the individual trees.

A Random Forest constructs $B$ decision trees. Each tree is trained on a bootstrap sample of the data, considering a random subset of $m \approx \sqrt{p}$ features at each split (where $p$ is the total number of features). The ensemble prediction for the probability of default given features $\mathbf{x}$ is:

$$ \hat{P}(\text{default} | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{P}_b(\text{default} | \mathbf{x}) $$

where $\hat{P}(\text{default} | \mathbf{x})$ is the ensemble predicted probability, $B$ is the number of trees, and $\hat{P}_b(\text{default} | \mathbf{x})$ is the prediction from tree $b$.

### Addressing Imbalance with SMOTE and Class Weights
To further address the class imbalance, we use **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data. SMOTE generates synthetic samples of the minority class (defaults), helping the model learn from a more balanced dataset. Additionally, `class_weight='balanced'` is used within the Random Forest model to assign higher importance to the minority class samples during tree construction.

When `class_weight='balanced'` is used in tree-based models, the loss function (often Gini impurity or entropy) is modified such that the samples of the minority class (defaults) are given higher weight during tree construction. For a weighted loss function, the contribution of each sample $i$ to the total loss $\mathcal{L}_w$ is adjusted by its class weight $w_{y_i}$:

$$ \mathcal{L}_w(\mathbf{\beta}) = \sum_{i=1}^N w_{y_i} [y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i)] $$

For a dataset with 5% defaults, $w_1 \approx 10$ and $w_0 \approx 0.526$, effectively making each default sample contribute significantly more to the loss, forcing the model to 'pay attention' to the minority class.

**Hyperparameter Tuning** with `GridSearchCV` is also employed to find the optimal settings for the Random Forest, ensuring it's finely tuned for performance.

Navigate to the "3. Random Forest Model" page.

Click the "Train Random Forest Model (with SMOTE & GridSearchCV)" button. This process may take a while due to the extensive tuning involved. Once complete, you'll see a confirmation that the model is ready.

## 5. Evaluating Model Performance with Financial Metrics
Duration: 0:12

Alex knows that in credit risk, simply looking at "accuracy" is a trap. A model that predicts "no default" for all loans could achieve very high accuracy in an imbalanced dataset, yet it would fail to identify any actual defaults—a financially catastrophic outcome. Therefore, Alex focuses on **credit-appropriate metrics** that truly assess a model's ability to discriminate between good and bad loans, especially for the rare default events.

<aside class="negative">
⚠️ <b>Practitioner Warning: Accuracy is NOT a Primary Metric</b>
<br>
With a high non-default rate, a trivial 'predict all non-default' model achieves high accuracy but catches zero defaults. Reporting accuracy in a credit scoring context signals a fundamental misunderstanding. Always lead with AUC, KS, and the confusion matrix at a decision-relevant threshold.
</aside>

Here are the key metrics we'll use:

*   **ROC-AUC (Area Under the Receiver Operating Characteristic):** Measures the model's discrimination power. Probabilistic interpretation: the chance that a randomly selected default case is scored higher than a randomly selected non-default.
    $$ \text{AUC} = P(\hat{P}_{\text{default}} > \hat{P}_{\text{non-default}}) $$
    An AUC of 0.5 is random, >0.70 is acceptable, >0.80 is strong for consumer credit.

*   **Gini Coefficient:** An industry-standard alternative to AUC, directly related by:
    $$ \text{Gini} = 2 \times \text{AUC} - 1 $$

*   **Precision-Recall Curve:** Particularly informative for imbalanced datasets, showing the trade-off between precision (of predicted defaults, how many are actual defaults) and recall (of actual defaults, how many are predicted) at various thresholds.

*   **Kolmogorov-Smirnov (KS) Statistic:** Measures the maximum separation between the cumulative distribution functions (CDFs) of predicted probabilities for default and non-default classes.
    $$ \text{KS} = \max_t |F_1(t) - F_0(t)| $$
    where $\text{KS}$ is the Kolmogorov-Smirnov statistic, $F_1(t)$ is the CDF of predicted probabilities for defaults, and $F_0(t)$ is the CDF for non-defaults. A KS > 0.30 is generally considered acceptable.

Navigate to the "4. Model Comparison & Evaluation" page.

Click the "Calculate Performance Metrics (Default Threshold 0.5)" button.

Examine the "Model Comparison Table" which shows the key metrics for both Logistic Regression and Random Forest using a standard threshold of 0.5. You'll also see plots for the ROC, Precision-Recall, and KS curves, visually comparing the models' performance.

## 6. Optimizing Decisions with Cost-Sensitive Analysis
Duration: 0:15

This is where Alex translates model probabilities into tangible financial decisions for CapitalFlow Bank. The true challenge in credit risk is that the cost of misclassifying a loan is **asymmetric**: approving a loan that defaults (**False Negative**) is significantly more expensive than rejecting a loan that would have been repaid (**False Positive**). Alex must define these costs and identify an optimal classification threshold that minimizes overall financial losses.

### Defining Asymmetric Costs
The Expected Total Cost for a given threshold is calculated as:

$$ C_{\text{total}} = \text{FN} \times (\text{LGD} \times \text{EAD}) + \text{FP} \times C_{\text{opportunity}} $$

Where:

*   $\text{FN}$ = Number of False Negatives (actual defaults predicted as non-defaults)
*   $\text{FP}$ = Number of False Positives (actual non-defaults predicted as defaults)
*   $\text{LGD}$ = Loss Given Default (percentage of exposure lost if default occurs).
*   $\text{EAD}$ = Exposure at Default (the outstanding loan amount at the time of default). For simplicity, we assume this is the `avg_loan_amount`.
*   $C_{\text{opportunity}}$ = Opportunity cost of declining a good loan (e.g., lost interest margin).

Based on CapitalFlow Bank's assumptions:

*   Average Loan Amount (`avg_loan_amount`): $15,000
*   Loss Given Default (`LGD`): 60%
*   Opportunity Cost Rate (`opportunity_cost_rate`): 3% (lost margin on a good loan)

Therefore, the costs are:
*   `cost_fn` (Cost of a missed default) = $15,000 \times 0.60 = $9,000
*   `cost_fp` (Cost of declining a good loan) = $15,000 \times 0.03 = $450

<aside class="positive">
💡 <b>Key Insight: Optimal Threshold is Rarely 0.5</b>
<br>
In credit scoring, the cost of a missed default (FN: $9,000) far exceeds the cost of a declined good loan (FP: $450). This 20:1 asymmetry in costs drives the optimal classification threshold to be typically much lower (e.g., 0.15-0.25) than the standard 0.5—a critical concept for credit professionals.
</aside>

Navigate to the "5. Cost-Sensitive Analysis & Optimization" page.

On the left sidebar, you can adjust the "Average Loan Amount", "Loss Given Default (LGD)", and "Opportunity Cost Rate (%)" using the sliders. Experiment with these values to see how they impact the optimal threshold.

Click the "Perform Cost-Sensitive Optimization" button.

Observe the "Optimal Classification Thresholds" for both models, along with their respective minimum costs. The "Cost Curve Analysis" plot visually shows how total cost changes across different probability thresholds. Finally, two "Financial Impact of Model Decisions" plots display confusion matrices for each model at their optimal thresholds, breaking down the financial implications of each type of decision.

## 7. Understanding What Drives Default: Feature Importance
Duration: 0:10

To gain deeper insights into credit risk, Alex needs to understand which factors are most influential in predicting default. This involves extracting and comparing **feature importance** from both the Logistic Regression and Random Forest models. This comparison helps CapitalFlow Bank understand different perspectives on risk drivers and informs targeted interventions.

*   **Logistic Regression Feature Importance:** Derived from the absolute magnitude of the coefficients (or odds ratios). Larger absolute coefficients indicate a stronger influence on the log-odds of default.
*   **Random Forest Feature Importance:** Typically measured by **Gini importance** (also known as mean decrease in impurity), which quantifies how much each feature contributes to the homogeneity of the nodes in the decision trees.

<aside class="positive">
📚 <b>Discussion Point: Explainability-Accuracy Trade-off</b>
<br>
Logistic Regression is highly transparent (coefficients directly explain impact), which is often a regulatory requirement. Random Forests, while potentially more accurate, are less inherently interpretable and may require additional explainable AI (XAI) tools (like SHAP/LIME) for individual-level explanations.
</aside>

<aside class="positive">
📊 <b>Discussion Point: AUC Gap Narrows with Feature Engineering</b>
<br>
Much of the Random Forest's advantage comes from automatically capturing non-linearities and interactions. However, if an analyst manually engineers interaction terms and non-linear transforms for the Logistic Regression model, the AUC gap often shrinks, demonstrating the value of domain expertise in feature engineering.
</aside>

Navigate to the "6. Feature Importance" page.

Click the "Analyze Feature Importance" button.

Examine the "Feature Importance Comparison Table", which ranks features by their importance in both models. The "Feature Importance Bar Charts" provide a visual comparison.

You'll also find "Predicted Probability of Default (PD) Distributions" plots, showing the distribution of predicted probabilities for default and non-default classes for each model. This helps visualize how well each model separates the two classes.

## 8. Calibrating Probabilities for Regulatory Compliance
Duration: 0:07

For CapitalFlow Bank, it's not enough for a model to simply rank borrowers by risk; the predicted probabilities of default (PDs) must also be **well-calibrated**. This means that if a model predicts a 10% PD for a group of loans, approximately 10% of those loans should actually default. Well-calibrated PDs are essential for:

*   **Accurate Risk-Based Pricing:** Setting appropriate interest rates based on a borrower's true risk.
*   **Regulatory Provisioning:** Calculating Expected Credit Losses (ECL) for accounting standards like IFRS 9.
*   **Portfolio Stress Testing:** Assessing the impact of adverse economic scenarios on the loan book.

### Calibration Curves and Brier Score
A **calibration curve** (also known as a reliability diagram) plots the mean predicted probability for a given bin against the fraction of positive outcomes (observed default rate) in that bin. A perfectly calibrated model would have its points fall along the diagonal line $y=x$.

The **Brier score** is a metric that quantifies the accuracy of probabilistic predictions. It measures the mean squared difference between the predicted probabilities and the actual outcomes. A lower Brier score indicates better calibration.

<aside class="positive">
💰 <b>Discussion Point: Why Probability Calibration Matters</b>
<br>
AUC measures ranking quality, not calibration. For accurate risk-based pricing (rate based on PD) and provisioning (ECL = PD × LGD × EAD), well-calibrated probabilities are essential. Techniques like Isotonic Regression or Platt Scaling can be applied to improve calibration if needed.
</aside>

Navigate to the "7. PD Calibration" page.

Click the "Assess and Calibrate PDs" button.

You will see the "Brier Scores" for Logistic Regression, the uncalibrated Random Forest, and a calibrated version of the Random Forest. A lower Brier score signifies better calibration.

The "Calibration Curves" plots visualize how well each model's predicted probabilities align with the observed default rates. The diagonal line represents perfect calibration. You'll observe that while Random Forest might have higher AUC (better ranking), its raw probabilities might not be as well-calibrated as Logistic Regression's. Calibration techniques aim to correct this.
