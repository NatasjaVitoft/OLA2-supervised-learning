import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def random_forest_regression(df):

    target_variable = "quality_score"

    n_estimators = st.slider("Select the number of trees", min_value=10, max_value=500, value=100, step=10)

    st.write("### Model 1: Using All Features")
    all_columns = df.columns.tolist()
    all_columns.remove(target_variable)  

    X_all = df[all_columns]
    y_all = df[target_variable]

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    model_all = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model_all.fit(X_train_all, y_train_all)

    y_pred_all = model_all.predict(X_test_all)
    mse_all = mean_squared_error(y_test_all, y_pred_all)
    r2_all = r2_score(y_test_all, y_pred_all)

    st.write(f"**Mean Squared Error**: {mse_all:.2f}")
    st.write(f"**R² Score**: {r2_all:.2f}")

    st.write("### Feature Importance (Model 1)")
    importances_all = model_all.feature_importances_
    feature_names_all = all_columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_all, importances_all)
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance (Model 1 - All Features, {n_estimators} Trees)")
    plt.tight_layout()

    st.pyplot(plt)

    st.write("### Model 2: Dropping 'Category Two Defects' and 'Moisture'")
    columns_to_drop = ["Category Two Defects", "Moisture"]
    reduced_columns = [col for col in all_columns if col not in columns_to_drop]

    X_reduced = df[reduced_columns]
    y_reduced = df[target_variable]

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.2, random_state=42)
    model_reduced = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model_reduced.fit(X_train_reduced, y_train_reduced)

    y_pred_reduced = model_reduced.predict(X_test_reduced)
    mse_reduced = mean_squared_error(y_test_reduced, y_pred_reduced)
    r2_reduced = r2_score(y_test_reduced, y_pred_reduced)

    st.write(f"**Mean Squared Error**: {mse_reduced:.2f}")
    st.write(f"**R² Score**: {r2_reduced:.2f}")

    st.write("### Feature Importance (Model 2 - Reduced Features)")
    importances_reduced = model_reduced.feature_importances_
    feature_names_reduced = reduced_columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_reduced, importances_reduced)
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance (Model 2 - Reduced Features, {n_estimators} Trees)")
    plt.tight_layout()

    st.pyplot(plt)

    ##################################################################

    residuals_all = y_test_all - y_pred_all
    residuals_reduced = y_test_reduced - y_pred_reduced

    st.write("### Residuals Plot (Model 1 - All Features)")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_all, residuals_all)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot (Model 1 - All Features, {n_estimators} Trees)')
    st.pyplot(plt)

    st.write("### Residuals Plot (Model 2 - Reduced Features)")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_reduced, residuals_reduced)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot (Model 2 - Reduced Features, {n_estimators} Trees)')
    st.pyplot(plt)

    #################################################################

    st.write("### Predicted vs Actual (Model 1 - All Features)")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_all, y_pred_all)
    plt.plot([y_test_all.min(), y_test_all.max()], [y_test_all.min(), y_test_all.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual (Model 1 - All Features, {n_estimators} Trees)')
    st.pyplot(plt)

    st.write("### Predicted vs Actual (Model 2 - Reduced Features)")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_reduced, y_pred_reduced)
    plt.plot([y_test_reduced.min(), y_test_reduced.max()], [y_test_reduced.min(), y_test_reduced.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual (Model 2 - Reduced Features, {n_estimators} Trees)')
    st.pyplot(plt)