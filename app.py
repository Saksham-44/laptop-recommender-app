import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

le_Company = pickle.load(open('le_Company.pkl', 'rb'))
le_TypeName = pickle.load(open('le_TypeName.pkl', 'rb'))
le_Gpu = pickle.load(open('le_Gpu.pkl', 'rb'))
le_cpu_cleaned = pickle.load(open('le_cpu_cleaned.pkl', 'rb'))
le_Storage_type = pickle.load(open('le_Storage_type.pkl', 'rb'))
le_OpSys = pickle.load(open('le_OpSys.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
df_model = pickle.load(open('df_model.pkl', 'rb'))
df_model_scaled = pickle.load(open('df_model_scaled.pkl', 'rb'))
df_original = pickle.load(open('df_original.pkl', 'rb'))

st.title("💻 Laptop Recommender")

budget = st.number_input("Enter your budget (₹)", min_value=10000, max_value=350000, step=5000)
purpose = st.selectbox("Purpose", le_TypeName.classes_)
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
storage_type = st.selectbox("Storage Type", le_Storage_type.classes_)

def recommend_laptop(budget, purpose, ram, storage_type):
    purpose_enc = le_TypeName.transform([purpose])
    storage_enc = le_Storage_type.transform([storage_type])
    user_df = pd.DataFrame([[df_model['Company'].mean(), purpose_enc[0], df_model['Inches'].mean(), ram, df_model['Gpu'].mean(), df_model['OpSys'].mean(), df_model['cpu_cleaned'].mean(), storage_enc[0], budget]], columns=df_model.columns)
    user_scaled = scaler.transform(user_df)
    budget_filter = df_original[df_original['Price'] <= budget].index
    df_model_filtered = df_model_scaled.iloc[budget_filter]
    scores = cosine_similarity(user_scaled, df_model_filtered)
    top3 = scores[0].argsort()[-3:][::-1]
    return df_original[['Company', 'TypeName', 'Cpu', 'Ram', 'Memory', 'Gpu', 'Price']].iloc[df_model_filtered.index[top3]]

if st.button("Recommend Laptops"):
    results = recommend_laptop(budget, purpose, ram, storage_type)
    st.write(results)