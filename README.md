# Laptop Recommender System

A machine learning project that recommends the top 3 laptops based on user preferences like budget, purpose, RAM, and storage type.

## Dataset
Laptop Specifications and Price Prediction Dataset from Kaggle containing 1,303 laptops with detailed hardware specifications.

## How It Works
1. User inputs budget, purpose (Gaming/Ultrabook etc.), RAM, and storage preference
2. Input is encoded and scaled to match the training data format
3. Cosine Similarity compares user requirements against all laptops in the dataset
4. Top 3 most similar laptops within budget are returned

## Data Cleaning & EDA
- Extracted CPU brand from full CPU string using apply() and lambda
- Created Storage_type feature (SSD/HDD/SSD+HDD) from messy Memory column
- Performed EDA on price distribution, GPU types, RAM, screen size, and OS

## Tech Stack
- Python, Pandas, Scikit-learn, Streamlit
- MinMaxScaler for feature normalization
- LabelEncoder for categorical encoding
- Cosine Similarity for recommendation logic

## How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas scikit-learn streamlit numpy`
3. Run: `streamlit run app.py`
