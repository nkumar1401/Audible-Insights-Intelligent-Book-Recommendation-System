import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np

def get_processed_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Merge logic
    df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='inner', suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.contains('_drop')] 
    df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)

    # --- THE FIX: Clean columns BEFORE joining them ---
    cols_to_join = ['Book Name', 'Author', 'Description', 'Ranks and Genre']
    
    for col in cols_to_join:
        # Fill NaN with empty string so the addition doesn't fail
        df[col] = df[col].fillna('').astype(str).str.strip()

    # Create metadata - If all columns are empty, we use a fallback word
    df['metadata'] = (
        df['Book Name'] + " " + 
        df['Author'] + " " + 
        df['Description'] + " " + 
        df['Ranks and Genre']
    ).str.lower()

    # Final Safety: If metadata is just spaces, put a generic tag
    df.loc[df['metadata'].str.strip() == "", 'metadata'] = "audible book discovery"

    # Numeric cleaning
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(df['Rating'].median() if not df['Rating'].empty else 0)
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    
    # Confidence Score
    m = df['Number of Reviews'].quantile(0.75) if len(df) > 0 else 0
    C = df['Rating'].mean() if len(df) > 0 else 0
    df['Confidence_Score'] = (df['Number of Reviews']/(df['Number of Reviews']+m+1e-6) * df['Rating']) + (m/(df['Number of Reviews']+m+1e-6) * C)

    df = df.reset_index(drop=True)
    return df