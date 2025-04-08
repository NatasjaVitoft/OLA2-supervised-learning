import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def classification_model_quantiles(df_class):

    n_bins = 3  
    labels = ["Low", "Medium", "High"]
    df_class["quality_category"] = pd.qcut(df_class["quality_score"], q=n_bins, labels=labels)

    st.write("First few rows of quality_score and quality_category:")
    st.write(df_class[["quality_score", "quality_category"]].head(10))

    category_counts = df_class["quality_category"].value_counts()
    st.write("### Category Distribution:")
    st.write(category_counts)

    X = df_class.drop(columns=["quality_score", "quality_category"])
    y = df_class["quality_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    clf_report = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"### Model Accuracy: {accuracy:.2f}")

    report_df = pd.DataFrame(clf_report).transpose()

    st.write("### Classification Report:")
    st.dataframe(report_df.style.background_gradient(cmap="coolwarm"))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)

    st.write("### Confusion Matrix:")
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)



def classification_model_bins(df_class):

    bins = [0, 75, 85, 100]
    labels = ["Low", "Medium", "High"]

    df_class["quality_category"] = pd.cut(df_class["quality_score"], bins=bins, labels=labels, right=False)

    st.write("First few rows of quality_score and quality_category:")
    st.write(df_class[["quality_score", "quality_category"]].head(10))

    category_counts = df_class["quality_category"].value_counts()
    st.write("### Category Distribution:")
    st.write(category_counts)

    X = df_class.drop(columns=["quality_score", "quality_category"])
    y = df_class["quality_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    clf_report = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"### Model Accuracy: {accuracy:.2f}")

    report_df = pd.DataFrame(clf_report).transpose()

    st.write("### Classification Report:")
    st.dataframe(report_df.style.background_gradient(cmap="coolwarm"))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)

    st.write("### Confusion Matrix:")
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

