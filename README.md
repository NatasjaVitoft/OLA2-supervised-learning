# OLA2-supervised-learning

By Natasja Vitoft and Lasse Baggesgård Hansen

A model training experiment of Coffee quality data, based on dataset:
https://www.kaggle.com/datasets/ohoodaljohani/coffeequalityarabicabeansdatacleaneddataset


```
-- root random_forest.ipynb, cluster_main.ipynb (Main notebook files)
     |
     -- pdf -- Distance_measurement.pdf (Distance measurement presentation)
     |
     -- streamlit -- app.py (Streamlit application)
```


## Streamlit application 

We made an interactive streamlit application with the purpose of making the model parameters interactive. It's fun to see how the parameters change the outcome in real-time!

Follow these steps to get it running locally:

---

**Step 1: Clone the repository**

git clone https://github.com/NatasjaVitoft/OLA2-supervised-learning.git

**Step 2: Create and activate enviroment**

python -m venv venv
.\venv\Scripts\activate

**Step 3: Install requiremnts**

pip install -r requirements.txt

**Step 4: Run streamlit app**

streamlit run streamlit/app.py

