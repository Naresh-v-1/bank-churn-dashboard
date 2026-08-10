# 🏦 Bank Customer Churn Prediction Dashboard

An end-to-end machine learning project that predicts which bank customers are likely to churn, explains *why* using SHAP, and translates those predictions into a business-ready retention strategy — built and deployed as an interactive web dashboard.

**🔗 Live Demo:** [https://bank-churn-dashboard-naresh.streamlit.app/](https://bank-churn-dashboard-naresh.streamlit.app/)
**📁 Repository:** https://github.com/Naresh-v-1/bank-churn-dashboard

---

## 📌 Overview

Banks lose significant revenue every year to customer attrition, and it's far cheaper to retain an existing customer than acquire a new one. This project builds a full pipeline — from raw data to a live, interactive dashboard — that:

1. Predicts which customers are likely to churn
2. Explains *why* each customer is at risk (not just a black-box score)
3. Segments at-risk customers by business value
4. Automatically generates a personalized retention offer per customer
5. Quantifies the expected revenue impact of acting on the predictions

The goal isn't just a model with good accuracy — it's a decision-support tool a retention team could actually use.

---

## 🎯 Business Problem

> "Reduce customer attrition for a bank by identifying high-risk customers early and recommending targeted, cost-effective retention actions — instead of losing customers silently or spending equally on everyone."

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Churn Prediction Model** | Random Forest classifier trained on customer demographic, financial, and engagement data |
| **Class Imbalance Handling** | SMOTE oversampling, since churners are a minority class (~16% of customers) |
| **Model Explainability** | SHAP (SHapley Additive exPlanations) identifies the top factor driving each customer's individual risk |
| **Risk & Value Segmentation** | Customers grouped into High Risk–High Value, High Risk–Low Value, and Low Risk |
| **Automatic Personalized Offers** | Each at-risk customer's top SHAP driver is mapped to a matching retention offer (e.g., low product count → bundled product offer) |
| **What-If Simulator** | Enter a hypothetical customer's details and get a live churn prediction, risk segment, and offer recommendation |
| **ROI Calculator** | Adjustable sliders for retention success rate, revenue yield, and campaign cost — recalculates net revenue protected live |
| **Upload Your Own Data** | Upload a CSV/Excel file of customers and get predictions from the same trained model |
| **Interactive Dashboard** | Built with Streamlit — filterable tables, charts, and downloadable results, no coding required to use |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.85 |
| Recall (Churners) | 0.58 |
| Precision (Churners) | 0.63 |

**Why these metrics matter:** Recall is prioritized over raw accuracy because missing an actual churner (false negative) costs the bank a customer, while a false alarm just costs an unnecessary retention offer — a much cheaper mistake.

---

## 💡 Key Business Insight

Feature importance analysis shows **Age**, **Number of Products**, and **Balance** are the strongest churn signals — suggesting the bank should prioritize deepening product engagement early in the customer lifecycle and paying closer attention to specific age segments, rather than treating all customers the same.

**Estimated impact:** Acting on the ~240 High Risk–High Value customers identified in this dataset, with a conservative 30% retention success rate, protects an estimated **$278,000+ in annual revenue**.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML/Data:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Dataset:** [Bank Customer Churn Prediction (Kaggle)](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

---

## 📁 Project Structure

```
bank-churn-dashboard/
├── app.py                 # Streamlit dashboard (main application)
├── churn_model.py          # Standalone training script (CLI version)
├── bank_churn.csv          # Dataset (not included in repo — see setup below)
├── requirements.txt        # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Setup & Installation (run locally)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naresh-v-1/bank-churn-dashboard.git
   cd bank-churn-dashboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**
   Download the [Bank Customer Churn dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) from Kaggle and place it in the project root as `bank_churn.csv`.

5. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

Or skip all of the above and just use the **[live demo](https://bank-churn-dashboard-naresh.streamlit.app/)**.

---

## 🔍 How the Personalized Offers Work

For each customer flagged as high-risk, SHAP identifies the single feature contributing most to *that specific customer's* predicted churn risk. That driver is then mapped to a matching retention offer — for example:

- Low `NumOfProducts` → Bundled product offer
- Low `IsActiveMember` → Relationship manager outreach
- High `Balance` (dormant) → Premium banking tier invitation

The offer *selection* is automated and personalized per customer via the model's explainability layer. The offer *content* is a curated set of business rules — intentionally not free-generated text, since in a regulated industry like banking, retention offers should be reviewed and compliant, not machine-generated on the fly.

---

## 📈 Future Improvements

- Hyperparameter tuning (GridSearchCV) for further model performance gains
- A/B testing framework to validate the assumed retention success rate against real campaign data
- Integration with a CRM system to trigger retention workflows automatically
- Time-series churn risk tracking (risk score trend per customer over time, not just a snapshot)

---

## 👤 Author

**Naresh V**
Built as a portfolio project demonstrating end-to-end analytics: from raw data to a deployed, business-ready decision-support tool.

---

## 📄 License

This project is for educational and portfolio purposes. Dataset sourced from Kaggle under its respective license.