import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import zscore
import scipy.cluster.hierarchy as ch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

parent_dir = os.path.dirname(os.path.abspath(__file__))  
data_path = os.path.join(parent_dir, '..', 'coffee_quality.csv')  
df = pd.read_csv(data_path)

target_variable = "quality_score"


page = st.sidebar.selectbox("Select a model", ("Random forest regression", "PCA and clustering"))

st.title("Coffee Quality Models")

if page == "Random forest regression":
    st.write("### Random Forest Models")

    st.write("### Model 1: Using All Features")
    all_columns = df.columns.tolist()
    all_columns.remove(target_variable)  

    X_all = df[all_columns]
    y_all = df[target_variable]

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    model_all = RandomForestRegressor(n_estimators=100, random_state=42)
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
    plt.title("Feature Importance (Model 1 - All Features)")
    plt.tight_layout()

    st.pyplot(plt)

    st.write("### Model 2: Dropping 'Category Two Defects' and 'Moisture'")
    columns_to_drop = ["Category Two Defects", "Moisture"]
    reduced_columns = [col for col in all_columns if col not in columns_to_drop]

    X_reduced = df[reduced_columns]
    y_reduced = df[target_variable]

    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.2, random_state=42)
    model_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
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
    plt.title("Feature Importance (Model 2 - Reduced Features)")
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
    plt.title('Residuals Plot (Model 1 - All Features)')
    st.pyplot(plt)

    st.write("### Residuals Plot (Model 2 - Reduced Features)")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_reduced, residuals_reduced)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot (Model 2 - Reduced Features)')
    st.pyplot(plt)


if page == "Clustering":
    
    st.header("Clustering")
    st.write("The k-means algorithm needs to know the number of clusters to find in the dataset." \
    "try and find the amount of cluster that makes the most sense!")

    df_pca = pd.read_pickle("../data/coffee_pca.pkl")
    n_clusters = st.number_input(min_value=1, max_value=10, label="Number clusters")

    if (st.button("Predict k-means") and n_clusters):
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10) 
        prediction = kmeans.fit_predict(df_pca)

        plt.scatter(df_pca['PC_1'], df_pca['PC_2'], c=prediction, s=50, cmap='viridis')
        plt.xlabel('PC_1')
        plt.ylabel('PC_2')
        plt.grid(True)
        st.pyplot(plt)
    
    # Dendogram for agglomerative clustering
    st.header("Agglomerative Clustering")
    st.write("For finding out the optimal number of cluster by euclidean distance of data points. One can use a dendogram to visualize it potential cluster amount.")
    dendo_png = mpimg.imread("../data/dendogram.png")
    st.image(dendo_png)

    n_clusters_agg = st.number_input(min_value=1, max_value=10, label="Number clusters", key="2")
    if (st.button("Predict Agglomerative clustering") and n_clusters_agg):
        model = AgglomerativeClustering(n_clusters_agg, linkage = 'ward')
        aggmodel_pred = model.fit_predict(df_pca)

        plt.scatter(df_pca['PC_1'], df_pca['PC_2'], c=aggmodel_pred, s=50, cmap='viridis')
        plt.xlabel('PC_1')
        plt.ylabel('PC_2')
        plt.grid(True)
        st.pyplot(plt)

    st.header("DBSCAN")
    st.write("Dbscan can automatically determine how many clusters are present using a different set of parameters. ")
    st.markdown("#### **eps**")
    st.write("""
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
            """)
    st.markdown("#### **min_samples**")
    st.write("""
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, 
            whereas if it is set to a lower value, the found clusters will be more sparse.
            """)

    eps = st.number_input(min_value=0.1, max_value=10.0, value=0.40, label="eps", key="3")
    min_s = st.number_input(min_value=1, max_value=10, value=6, label="Min samples", key="4")

    if (st.button("Predict DBSCAN") and eps and min_s):
        dbscan = DBSCAN(eps=eps, min_samples=min_s)
        dbscan_pred = dbscan.fit_predict(df_pca)

        plt.scatter(df_pca['PC_1'], df_pca['PC_2'], c=dbscan_pred, s=50, cmap='viridis')
        plt.xlabel('PC_1')
        plt.ylabel('PC_2')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)





