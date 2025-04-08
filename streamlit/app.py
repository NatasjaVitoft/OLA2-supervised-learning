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
from classification import classification_model_bins, classification_model_quantiles
from random_forest import random_forest_regression

parent_dir = os.path.dirname(os.path.abspath(__file__))  
data_path = os.path.join(parent_dir, '..', 'coffee_quality.csv')  
df = pd.read_csv(data_path)

target_variable = "quality_score"

page = st.sidebar.selectbox("Select a model", ("Random forest regression", "Random forest classification", "Clustering"))

st.title("Coffee Quality Models")

if page == "Random forest regression":

    st.header("Random forest regression")
    st.write(random_forest_regression(df))

if page == "Random forest classification":
    
    st.header("Random forest classification")
    st.write(classification_model_bins(df))
    st.write(classification_model_quantiles(df))

if page == "Clustering":
    
    st.header("Clustering")
    st.write("The k-means algorithm needs to know the number of clusters to find in the dataset." \
    "try and find the amount of cluster that makes the most sense!")

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'coffee_pca.pkl')
    df_pca = pd.read_pickle(data_path)
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
    data_path_dendo = os.path.join(os.path.dirname(__file__), 'data', 'dendogram.png')
    dendo_png = mpimg.imread(data_path_dendo)
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





