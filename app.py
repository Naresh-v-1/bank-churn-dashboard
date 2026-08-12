import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import shap

st.set_page_config(page_title="Bank Customer Churn Dashboard", layout="wide")

st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.caption("Predicts at-risk customers and recommends retention actions based on risk & value segments.")

# ---------------------------
# DATA DICTIONARY / COLUMN GLOSSARY
# ---------------------------
with st.expander("ℹ️ What do these columns mean? (Data Dictionary)"):
    st.markdown("""
| Column | Meaning |
|---|---|
| **CreditScore** | The customer's credit score (300–900). Higher generally means lower credit risk. |
| **Geography** | The country/region the customer banks in (e.g., France, Germany, Spain). |
| **Gender** | Customer's gender as recorded by the bank. |
| **Age** | Customer's age in years. |
| **Tenure** | Number of years the customer has been with the bank. |
| **Balance** | Current account balance held by the customer. |
| **NumOfProducts** | Number of bank products the customer holds (e.g., savings account, credit card, loan). |
| **HasCrCard** | Whether the customer holds a credit card with the bank (1 = Yes, 0 = No). |
| **IsActiveMember** | Whether the customer is actively engaging with the bank (1 = Active, 0 = Inactive). |
| **EstimatedSalary** | Customer's estimated annual salary. |
| **churn_probability** | The model's predicted likelihood (0–100%) that this customer will leave the bank. |
| **segment** | Risk/value grouping: *High Risk–High Value*, *High Risk–Low Value*, or *Low Risk*. |
| **primary_driver** | The single factor (via SHAP) contributing most to that customer's predicted churn risk. |
| **auto_offer** | The retention offer automatically recommended based on the primary driver. |
""")

# ---------------------------
# OFFER TEMPLATES (mapped to SHAP's top churn driver per customer)
# ---------------------------
OFFER_MAP = {
    "NumOfProducts": "Bundled product offer — discounted fee on a second product to deepen engagement",
    "IsActiveMember": "Relationship manager outreach + engagement incentive to reactivate the account",
    "Balance": "Invitation to a premium banking tier with preferential rates",
    "Age": "Loyalty / life-stage-appropriate banking package",
    "Tenure": "Loyalty bonus recognizing years with the bank",
    "CreditScore": "Fee waiver or financial wellness consultation",
    "Geography": "Localized service improvement / regional outreach",
    "EstimatedSalary": "Fee waiver or cashback offer tailored to income bracket",
    "HasCrCard": "Credit card upgrade or annual fee waiver offer",
    "Gender": "General retention outreach"
}

# ---------------------------
# LOAD & CACHE DATA + MODEL (unchanged demo data/model)
# ---------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("bank_churn.csv")

    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_col = "Exited"
    X = df.drop(columns=target_col)
    y = df[target_col]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    results_df = X_test.copy()
    results_df["churn_probability"] = probs
    results_df["actual_exited"] = y_test.values

    value_col = "Balance" if "Balance" in results_df.columns else "EstimatedSalary"

    def segment(row):
        high_risk = row["churn_probability"] >= 0.5
        high_value = row[value_col] >= results_df[value_col].median()
        if high_risk and high_value:
            return "High Risk - High Value"
        elif high_risk and not high_value:
            return "High Risk - Low Value"
        else:
            return "Low Risk"

    results_df["segment"] = results_df.apply(segment, axis=1)

    action_map = {
        "High Risk - High Value": "Priority retention call + personalized offer",
        "High Risk - Low Value": "Automated email/SMS retention campaign",
        "Low Risk": "No action needed, monitor periodically"
    }
    results_df["recommended_action"] = results_df["segment"].map(action_map)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return (results_df, auc, report, cm, feature_importance, value_col,
            model, scaler, label_encoders, feature_columns, action_map, df, X_test_scaled)

with st.spinner("Training model..."):
    (results_df, auc, report, cm, feature_importance, value_col,
     model, scaler, label_encoders, feature_columns, action_map, full_df, X_test_scaled) = load_and_train()

# ---------------------------
# SHAP EXPLAINER (cached) — powers automatic personalized offers
# ---------------------------
@st.cache_resource
def get_tree_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_resource
def compute_top_drivers(_model, X_scaled, _feature_columns):
    explainer = get_tree_explainer(_model)
    raw = explainer.shap_values(X_scaled, check_additivity=False)
    if isinstance(raw, list):
        vals = raw[1]  # class 1 = churn
    else:
        vals = raw
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    top_idx = np.argmax(vals, axis=1)
    return [_feature_columns[i] for i in top_idx]

with st.spinner("Computing personalized offer drivers (SHAP)..."):
    at_risk_mask = results_df["churn_probability"] >= 0.5
    results_df["primary_driver"] = "N/A"
    results_df["auto_offer"] = "No action needed, monitor periodically"

    if at_risk_mask.sum() > 0:
        X_at_risk_scaled = X_test_scaled[at_risk_mask.values]
        top_drivers = compute_top_drivers(model, X_at_risk_scaled, feature_columns)

        # FIX: build index-aligned Series instead of assigning a raw list
        # positionally into a boolean-masked .loc — this avoids the
        # ValueError from ambiguous length/dtype inference in pandas.
        at_risk_index = results_df.index[at_risk_mask]
        driver_series = pd.Series(top_drivers, index=at_risk_index, dtype="object")
        offer_series = driver_series.map(lambda d: OFFER_MAP.get(d, "General retention outreach"))

        results_df.loc[at_risk_index, "primary_driver"] = driver_series
        results_df.loc[at_risk_index, "auto_offer"] = offer_series

# ---------------------------
# TOP METRICS
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("ROC-AUC Score", f"{auc:.3f}")
col1.caption("(How well the model tells churners apart from non-churners — closer to 1 is better)")

col2.metric("Recall (Churners)", f"{report['1']['recall']:.2f}")
col2.caption("(% of actual churners the model successfully catches)")

col3.metric("Precision (Churners)", f"{report['1']['precision']:.2f}")
col3.caption("(Of everyone flagged as a churner, % that are correct)")

high_value_at_risk = (results_df["segment"] == "High Risk - High Value").sum()
avg_value = results_df[value_col].mean() * 0.05
revenue_protected_default = high_value_at_risk * avg_value * 0.30
col4.metric("Est. Revenue Protected", f"${revenue_protected_default:,.0f}")
col4.caption("(Estimated annual revenue saved by acting on high-value at-risk customers)")

st.divider()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Performance",
    "🚨 At-Risk Customers",
    "🔍 Feature Importance",
    "📤 Test Your Own Data",
    "🎯 What-If Simulator",
    "💰 ROI Calculator"
])

with tab1:
    st.subheader("Confusion Matrix")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            annot_kws={"size": 10}, cbar=False
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, width="content")

    st.subheader("Segment Distribution")
    seg_counts = results_df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig2 = px.bar(seg_counts, x="segment", y="count", color="segment",
                  title="Customers by Risk Segment")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, width="stretch")

with tab2:
    st.subheader("Customers Predicted to Churn")
    st.caption("`primary_driver` and `auto_offer` are generated automatically per customer using SHAP.")

    segment_filter = st.multiselect(
        "Filter by segment",
        options=results_df["segment"].unique(),
        default=["High Risk - High Value", "High Risk - Low Value"]
    )

    filtered = results_df[results_df["segment"].isin(segment_filter)].sort_values(
        "churn_probability", ascending=False
    )

    st.write(f"Showing {len(filtered)} customers")
    st.dataframe(
        filtered[["churn_probability", "segment", "primary_driver", "auto_offer", value_col, "Age", "Tenure"]]
        .style.format({"churn_probability": "{:.2%}"}),
        width="stretch"
    )

    csv = filtered.to_csv(index=False)
    st.download_button("Download filtered list as CSV", csv, "at_risk_customers.csv", "text/csv")

with tab3:
    st.subheader("What Drives Churn Predictions")
    fig3 = px.bar(feature_importance.head(10), x="importance", y="feature", orientation="h",
                  title="Top 10 Features Driving Churn Risk")
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    st.plotly_chart(fig3, width="stretch")

with tab4:
    st.subheader("Upload Your Own Bank Customer Data")
    st.write(
        "Upload a CSV or Excel file with customer data to get churn predictions from the "
        "same trained model. Your file should contain the same feature columns used in training:"
    )
    st.code(", ".join(feature_columns))

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)

            st.write("Preview of uploaded data:")
            st.dataframe(user_df.head(), width="stretch")

            for col in ["RowNumber", "CustomerId", "Surname", "Exited"]:
                if col in user_df.columns:
                    user_df = user_df.drop(columns=col)

            missing_cols = [c for c in feature_columns if c not in user_df.columns]

            if missing_cols:
                st.error(
                    f"Your file is missing required columns: {', '.join(missing_cols)}. "
                    f"Please include all of: {', '.join(feature_columns)}"
                )
            else:
                user_df = user_df[feature_columns].copy()

                encode_errors = []
                for col, le in label_encoders.items():
                    if col in user_df.columns:
                        unseen = set(user_df[col].astype(str)) - set(le.classes_)
                        if unseen:
                            encode_errors.append(
                                f"Column '{col}' has unrecognized values: {', '.join(unseen)}. "
                                f"Expected one of: {', '.join(le.classes_)}"
                            )
                        else:
                            user_df[col] = le.transform(user_df[col].astype(str))

                if encode_errors:
                    for err in encode_errors:
                        st.error(err)
                else:
                    user_scaled = scaler.transform(user_df)
                    user_probs = model.predict_proba(user_scaled)[:, 1]

                    output_df = user_df.copy()
                    output_df["churn_probability"] = user_probs

                    value_col_user = value_col if value_col in output_df.columns else output_df.columns[0]
                    median_val = output_df[value_col_user].median()

                    def segment_user(row):
                        high_risk = row["churn_probability"] >= 0.5
                        high_value = row[value_col_user] >= median_val
                        if high_risk and high_value:
                            return "High Risk - High Value"
                        elif high_risk and not high_value:
                            return "High Risk - Low Value"
                        else:
                            return "Low Risk"

                    output_df["segment"] = output_df.apply(segment_user, axis=1)
                    output_df["recommended_action"] = output_df["segment"].map(action_map)

                    output_df["primary_driver"] = "N/A"
                    output_df["auto_offer"] = "No action needed, monitor periodically"
                    output_at_risk_mask = output_df["churn_probability"] >= 0.5

                    if output_at_risk_mask.sum() > 0:
                        with st.spinner("Computing personalized offers (SHAP)..."):
                            explainer = get_tree_explainer(model)
                            at_risk_scaled = user_scaled[output_at_risk_mask.values]
                            raw = explainer.shap_values(at_risk_scaled, check_additivity=False)
                            if isinstance(raw, list):
                                vals = raw[1]
                            else:
                                vals = raw
                                if vals.ndim == 3:
                                    vals = vals[:, :, 1]
                            top_idx = np.argmax(vals, axis=1)
                            drivers = [feature_columns[i] for i in top_idx]

                            # FIX: same index-aligned assignment as above
                            at_risk_index = output_df.index[output_at_risk_mask]
                            driver_series = pd.Series(drivers, index=at_risk_index, dtype="object")
                            offer_series = driver_series.map(
                                lambda d: OFFER_MAP.get(d, "General retention outreach")
                            )

                            output_df.loc[at_risk_index, "primary_driver"] = driver_series
                            output_df.loc[at_risk_index, "auto_offer"] = offer_series

                    output_df = output_df.sort_values("churn_probability", ascending=False)

                    st.success(f"Predictions generated for {len(output_df)} customers.")

                    seg_counts_user = output_df["segment"].value_counts().reset_index()
                    seg_counts_user.columns = ["segment", "count"]
                    fig_user = px.bar(seg_counts_user, x="segment", y="count", color="segment",
                                       title="Uploaded Customers by Risk Segment")
                    st.plotly_chart(fig_user, width="stretch")

                    st.dataframe(
                        output_df.style.format({"churn_probability": "{:.2%}"}),
                        width="stretch"
                    )

                    result_csv = output_df.to_csv(index=False)
                    st.download_button(
                        "Download predictions as CSV",
                        result_csv,
                        "uploaded_data_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Could not process the file: {e}")

with tab5:
    st.subheader("What-If Simulator")
    st.write("Enter a single customer's details below to see their live predicted churn risk and auto-generated offer.")

    geo_options = list(label_encoders["Geography"].classes_) if "Geography" in label_encoders else ["France", "Germany", "Spain"]
    gender_options = list(label_encoders["Gender"].classes_) if "Gender" in label_encoders else ["Male", "Female"]

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", geo_options)
        gender = st.selectbox("Gender", gender_options)

    with sim_col2:
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        num_products = st.slider("Number of Products", 1, 4, 2)

    with sim_col3:
        balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    if st.button("Predict Churn Risk", type="primary"):
        sim_input = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": estimated_salary
        }

        sim_df = pd.DataFrame([sim_input])
        sim_df = sim_df[feature_columns]

        for col, le in label_encoders.items():
            if col in sim_df.columns:
                sim_df[col] = le.transform(sim_df[col].astype(str))

        sim_scaled = scaler.transform(sim_df)
        sim_prob = model.predict_proba(sim_scaled)[0][1]

        if sim_prob >= 0.5:
            if balance >= full_df["Balance"].median():
                seg = "High Risk - High Value"
            else:
                seg = "High Risk - Low Value"
        else:
            seg = "Low Risk"

        explainer = get_tree_explainer(model)
        raw = explainer.shap_values(sim_scaled, check_additivity=False)
        if isinstance(raw, list):
            vals = raw[1]
        else:
            vals = raw
            if vals.ndim == 3:
                vals = vals[:, :, 1]
        top_driver = feature_columns[int(np.argmax(vals[0]))]
        auto_offer = OFFER_MAP.get(top_driver, "General retention outreach")

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Churn Probability", f"{sim_prob:.1%}")
        res_col2.metric("Risk Segment", seg)

        st.progress(min(float(sim_prob), 1.0))

        st.markdown(f"**Primary churn driver (SHAP):** `{top_driver}`")
        st.markdown(f"**Auto-generated personalized offer:** {auto_offer}")

with tab6:
    st.subheader("ROI Calculator")
    st.write(
        "Adjust the assumptions below to see how the estimated revenue protected changes. "
        "This reflects the business trade-offs a retention team would actually weigh."
    )

    roi_col1, roi_col2, roi_col3 = st.columns(3)

    with roi_col1:
        revenue_yield_pct = st.slider(
            "Annual revenue yield per customer (% of balance)",
            min_value=1, max_value=15, value=5
        ) / 100

    with roi_col2:
        retention_success_rate = st.slider(
            "Expected retention campaign success rate (%)",
            min_value=5, max_value=80, value=30
        ) / 100

    with roi_col3:
        cost_per_offer = st.number_input(
            "Average cost per retention offer ($)",
            min_value=0, value=50, step=10
        )

    avg_value_roi = results_df[value_col].mean() * revenue_yield_pct
    gross_revenue_protected = high_value_at_risk * avg_value_roi * retention_success_rate
    total_campaign_cost = high_value_at_risk * cost_per_offer
    net_revenue_protected = gross_revenue_protected - total_campaign_cost

    st.divider()
    roi_res1, roi_res2, roi_res3 = st.columns(3)
    roi_res1.metric("High-Value At-Risk Customers", f"{high_value_at_risk}")
    roi_res2.metric("Gross Revenue Protected", f"${gross_revenue_protected:,.0f}")
    roi_res3.metric("Net Revenue Protected (after campaign cost)", f"${net_revenue_protected:,.0f}")

    st.caption(
        f"Calculation: {high_value_at_risk} high-value at-risk customers × "
        f"${avg_value_roi:,.0f} estimated annual value each × {retention_success_rate:.0%} retention success "
        f"− ${total_campaign_cost:,.0f} total campaign cost = ${net_revenue_protected:,.0f} net protected."
    )
