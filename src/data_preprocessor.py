import pandas as pd
import numpy as np
import os

def get_processed_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='inner', suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.contains('_drop')] 
    df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)
    
    # Cleaning & Imputation
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').replace(-1, np.nan).fillna(df['Rating'].median())
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    
    # HEURISTIC: Confidence Score
    m = df['Number of Reviews'].quantile(0.75)
    C = df['Rating'].mean()
    df['Confidence_Score'] = (df['Number of Reviews']/(df['Number of Reviews']+m) * df['Rating']) + (m/(df['Number of Reviews']+m) * C)
    
    df['Description'] = df['Description'].fillna("No description available")
    df['Ranks and Genre'] = df['Ranks and Genre'].fillna("General")
    df['tags'] = df['Book Name'] + " " + df['Author'] + " " + df['Description'] + " " + df['Ranks and Genre']
    
    return df