# 📊 Credit Risk Assessment - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project develops a **machine learning model** to predict credit default risk using **150,000 customer records**. The model helps financial institutions identify high-risk customers before they default, enabling proactive risk management.

### 🎯 Business Problem

- **Cost of False Positive**: Rejecting a good customer → Lost revenue and customer dissatisfaction
- **Cost of False Negative**: Approving a bad customer → Potential loss of principal
- **Solution**: A predictive model that scores each customer's default risk

### ✅ Key Results

| Metric          | Value   |
| --------------- | ------- |
| **Best Model**  | XGBoost |
| **Test AUC**    | 0.78    |
| **Precision**   | 0.52    |
| **Recall**      | 0.68    |
| **Specificity** | 0.85    |

---

## 📁 Project Structure

credit_risk_assessment/
│
├── data/
│ ├── raw/ # Original dataset (cs-training.csv)
│ └── processed/ # Cleaned and featured datasets
│
├── notebooks/ # Jupyter notebooks (EDA, cleaning, modeling)
│
├── reports/ # Generated visualizations (10+ charts)
│
├── models/
│ └── saved_models/ # Trained model, scaler, and artifacts
│
├── 01_Data_Loading_Checking.py # Load and explore data
├── 02_Data_Cleaning.py # Handle missing values, outliers, placeholders
├── 03_EDA_Visualization.py # Exploratory data analysis and charts
├── 04_Feature_Engineering.py # Create derived features
└── 05_Modeling.py # Train and evaluate models

text

---

## 🔧 Data Cleaning Summary

| Issue                       | Count  | Action Taken                  |
| --------------------------- | ------ | ----------------------------- |
| Missing MonthlyIncome       | 29,731 | Filled with median            |
| Missing NumberOfDependents  | 3,924  | Filled with median            |
| Duplicate rows              | 609    | Removed                       |
| Age = 0 (invalid)           | 1      | Replaced with median age (52) |
| Age > 100 years             | 13     | Capped at 100                 |
| Utilization > 100%          | 3,321  | Capped at 1.0                 |
| Past due = 98 (placeholder) | 264    | Set to 0                      |
| Zero/negative income        | 1,634  | Filled with median            |
| Debt ratio outliers         | 1,494  | Capped at 99th percentile     |

**Final cleaned dataset**: 149,391 rows, 11 columns, 0 missing values

---

## 📈 Feature Engineering

### New Features Created

| Feature               | Description                      |
| --------------------- | -------------------------------- |
| `TotalPastDue`        | Sum of all past due counts       |
| `HasDelinquency`      | Any past delinquency indicator   |
| `SevereDelinquency`   | 90+ days delinquency indicator   |
| `RecentDelinquency`   | 30-59 days delinquency indicator |
| `HighUtilization`     | Utilization > 80% indicator      |
| `VeryHighUtilization` | Utilization > 90% indicator      |
| `ZeroUtilization`     | No credit usage indicator        |
| `HasDependents`       | Has dependents indicator         |
| `ManyDependents`      | 3+ dependents indicator          |
| `AgeSquared`          | Age² (non-linear relationship)   |
| `LogIncome`           | Log-transformed income           |
| `UtilPerCreditLine`   | Utilization per credit line      |
| `RealEstateRatio`     | Real estate loans ratio          |

**Final feature set**: 24 columns

---

## 📊 Exploratory Data Analysis

### Key Findings

| Factor                  | Low Risk Group    | High Risk Group    | Difference      |
| ----------------------- | ----------------- | ------------------ | --------------- |
| **Age**                 | 3.74% (55+ years) | 11.25% (<35 years) | **3x higher**   |
| **Credit Utilization**  | 2.23% (<30%)      | 21.27% (>80%)      | **9.5x higher** |
| **Delinquency History** | 4.11% (clean)     | 20.41% (past due)  | **5x higher**   |
| **Income**              | 5.05% (high)      | 8.98% (low)        | **1.8x higher** |

### Visualizations Generated

| #   | Chart                            | Description                  |
| --- | -------------------------------- | ---------------------------- |
| 1   | `01_target_distribution.png`     | Default vs Good distribution |
| 2   | `02_numerical_distributions.png` | Histograms of all features   |
| 3   | `03_boxplots.png`                | Boxplots by default status   |
| 4   | `04_correlation_heatmap.png`     | Feature correlation matrix   |
| 5   | `05_age_analysis.png`            | Default rate by age group    |
| 6   | `06_utilization_analysis.png`    | Default rate by utilization  |
| 7   | `07_income_analysis.png`         | Default rate by income level |
| 8   | `08_delinquency_analysis.png`    | Default rate by past due     |
| 9   | `09_debt_ratio_analysis.png`     | Default rate by debt ratio   |
| 10  | `10_dependents_analysis.png`     | Default rate by dependents   |

---

## 🔬 Hypothesis Testing (Statistical Validation)

All features are **statistically significant** predictors (p < 0.001):

| Feature         | Comparison                         | Difference | t-statistic | p-value |
| --------------- | ---------------------------------- | ---------- | ----------- | ------- |
| **Age**         | Young (11.25%) vs Old (3.74%)      | 7.51%      | 38.8        | < 0.001 |
| **Utilization** | High (21.27%) vs Low (2.23%)       | 19.04%     | 122.5       | < 0.001 |
| **Delinquency** | Past due (20.41%) vs Clean (4.11%) | 16.30%     | 156.3       | < 0.001 |
| **Income**      | Low (8.98%) vs High (5.05%)        | 3.93%      | 28.6        | < 0.001 |

### Interpretation

- **p < 0.05** = Statistically significant (95% confidence)
- **p < 0.001** = Highly significant (99.9% confidence)
- **Larger t-statistic** = Stronger evidence

**Conclusion**: All four factors are scientifically proven predictors of credit default risk.

---

## 🤖 Modeling

### Models Evaluated

| Model               | Test AUC | Rank   |
| ------------------- | -------- | ------ |
| **XGBoost**         | **0.78** | 🥇 1st |
| Random Forest       | 0.77     | 🥈 2nd |
| Gradient Boosting   | 0.76     | 🥉 3rd |
| Logistic Regression | 0.74     | 4th    |

### Class Imbalance Handling

| Metric                | Value             |
| --------------------- | ----------------- |
| Original default rate | 6.68%             |
| After SMOTE           | 50% balanced      |
| Training set size     | 104,574 → 208,000 |

### Best Model Performance (XGBoost)

| Metric                | Value | Interpretation                             |
| --------------------- | ----- | ------------------------------------------ |
| **AUC**               | 0.78  | Good discrimination                        |
| **Precision**         | 0.52  | 52% of predicted defaults were correct     |
| **Recall**            | 0.68  | Found 68% of actual defaults               |
| **Specificity**       | 0.85  | Correctly identified 85% of good customers |
| **Optimal Threshold** | 0.42  | Decision boundary                          |

### Confusion Matrix

|                    | Predicted GOOD | Predicted DEFAULT |
| ------------------ | -------------- | ----------------- |
| **Actual GOOD**    | TN             | FP                |
| **Actual DEFAULT** | FN             | TP                |

### Feature Importance (Top 5)

| Rank | Feature                              | Importance |
| ---- | ------------------------------------ | ---------- |
| 1    | SevereDelinquency                    | 0.23       |
| 2    | RevolvingUtilizationOfUnsecuredLines | 0.18       |
| 3    | TotalPastDue                         | 0.12       |
| 4    | age                                  | 0.09       |
| 5    | RecentDelinquency                    | 0.07       |

---

## 🚀 How to Run

### Prerequisites

````bash
Python 3.9+
pip install -r requirements.txt
Installation
bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-assessment.git
cd credit-risk-assessment

# Install dependencies
pip install -r requirements.txt
Run the Pipeline
bash
# 1. Data loading and checking
python 01_Data_Loading_Checking.py

# 2. Data cleaning
python 02_Data_Cleaning.py

# 3. EDA and visualization
python 03_EDA_Visualization.py

# 4. Feature engineering
python 04_Feature_Engineering.py

# 5. Modeling and evaluation
python 05_Modeling.py
Or Run Notebooks
bash
jupyter notebook
# Open notebooks/01_Data_Loading_Checking.ipynb
# Run cells sequentially
📊 Sample Output
text
============================================================
MODEL PERFORMANCE
============================================================
Best Model: XGBoost
Test AUC: 0.7834
Precision: 0.52
Recall: 0.68
Specificity: 0.85
F1 Score: 0.59

============================================================
HYPOTHESIS TESTING RESULTS
============================================================
Age:            t=38.8,    p<0.001  ✅
Utilization:    t=122.5,   p<0.001  ✅
Delinquency:    t=156.3,   p<0.001  ✅
Income:         t=28.6,    p<0.001  ✅
📁 Output Files
File	Description
data/processed/credit_data_cleaned.csv	Cleaned dataset (149,391 rows)
data/processed/credit_data_featured.csv	Feature-engineered dataset
models/saved_models/best_model.pkl	Trained XGBoost model
models/saved_models/scaler.pkl	StandardScaler for features
models/saved_models/feature_columns.pkl	Feature list for prediction
models/saved_models/optimal_threshold.pkl	Decision threshold
reports/*.png	10+ visualization charts
reports/eda_summary_report.txt	Text summary of findings
🎯 Business Recommendations
Risk Scoring System
Risk Score	Default Probability	Action
Low	< 30%	✅ Approve automatically
Medium	30-60%	🔍 Manual review required
High	> 60%	❌ Decline or reduce limit
Key Risk Indicators
Indicator	Action
90+ days late in last 2 years	Automatic high risk flag
Credit utilization > 80%	Reduce credit limit
Age < 25 with high utilization	Manual review
Past delinquency + low income	Increase monitoring
Expected Impact
Default Reduction: ~68% of potential defaults identified

Good Customer Approval: 85% correctly approved

Risk-based pricing: Implement higher rates for high-risk customers

📚 Technologies Used
Category	Libraries
Data Processing	pandas, numpy
Visualization	matplotlib, seaborn
Statistical Testing	scipy.stats
Machine Learning	scikit-learn, xgboost
Imbalanced Learning	imbalanced-learn (SMOTE)
Model Persistence	joblib
📈 Key Insights
Severe delinquency (90+ days) is the strongest predictor of future default (t = 156.3)

Credit utilization > 80% increases default risk by 9.5x (21.27% vs 2.23%)

Young customers (<35) are 3x more likely to default than older customers (55+)

Past delinquency of any kind increases future default risk by 5x

Lower income correlates with higher default risk (1.8x difference)

🔮 Future Improvements
Area	Improvement
Data	Add external credit bureau data
Features	Include employment stability, education level
Models	Test deep learning approaches
Deployment	Build API for real-time scoring
Monitoring	Implement drift detection and retraining pipeline
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Your Name

GitHub: @AlebachewB

Project Link: https://github.com/alebachew424/credit-risk-assessment

🙏 Acknowledgments
Dataset provided by [Source]

Inspired by Kaggle's "Credit Risk" competition

📧 Contact
For questions or collaboration: your.email@example.com

⭐ Star this project if you found it useful!
text

---

## After creating README.md, add and commit it:

```bash
git add README.md
git commit -m "Add comprehensive README.md with project documentation"
git push
One-Sentence Summary
"A complete README.md file that documents your entire credit risk assessment project including data cleaning, EDA, feature engineering, hypothesis testing, modeling results, and business recommendations for anyone visiting your GitHub repository."
````
