import requests
import pandas as pd
import os
import time

def update_diverse_books():
    # 1. Configuration & Path Setup
    CATEGORIES = ["technology", "spirituality", "fiction", "business", "history", "biography"]
    REQUIRED_COLUMNS = ['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price', 'Description', 'Ranks and Genre']
    FILE_PATH = "raw_data/Audible_Catlog_Advanced_Features.csv"
    
    # Create the raw_data directory if it doesn't exist
    if not os.path.exists("raw_data"):
        print("Creating 'raw_data' directory...")
        os.makedirs("raw_data")
        
    all_new_data = []

    # 2. Data Acquisition
    for category in CATEGORIES:
        print(f"Fetching newest books for: {category}...")
        # Google Books API call focusing on newest releases
        api_url = f"https://www.googleapis.com/books/v1/volumes?q=subject:{category}&orderBy=newest&maxResults=10"
        
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                items = response.json().get('items', [])
                for item in items:
                    volume = item.get('volumeInfo', {})
                    book_data = {
                        'Book Name': volume.get('title', 'Unknown Title'),
                        'Author': ", ".join(volume.get('authors', ['Unknown Author'])),
                        'Rating': volume.get('averageRating', 0),
                        'Number of Reviews': volume.get('ratingsCount', 0),
                        'Price': 0, # Placeholder for your pricing logic
                        'Description': volume.get('description', 'No description available'),
                        'Ranks and Genre': f"{category}, " + ", ".join(volume.get('categories', []))
                    }
                    all_new_data.append(book_data)
            
            # Professional practice: Avoid hitting rate limits
            time.sleep(1) 
        except Exception as e:
            print(f"Error fetching {category}: {e}")

    if not all_new_data:
        print("⚠️ No new data fetched from API. Exiting to protect existing files.")
        return

    # 3. Processing & Cleaning Gate
    new_df = pd.DataFrame(all_new_data)
    new_df = new_df[REQUIRED_COLUMNS] 

    # --- ELITE CLEANING LAYER ---
    # Automatically rejects rows containing web error messages found in previous scrapes
    error_keywords = ["robot", "browser is accepting cookies", "traffic is piling up", "Oops!"]
    clean_mask = ~new_df['Description'].str.contains('|'.join(error_keywords), case=False, na=False)
    new_df = new_df[clean_mask]
    
    # Remove duplicates within the new batch
    new_df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)

    # 4. Smart Merge Logic (Anti-Conflict)
    if os.path.exists(FILE_PATH):
        try:
            # Load existing data, skipping bad lines caused by Git merge markers
            existing_df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
            
            # Combine old and new datasets
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Deduplicate: Keep the 'last' entry (the newest one)
            combined_df.drop_duplicates(subset=['Book Name', 'Author'], keep='last', inplace=True)
            
            # Final Safety: Clean the whole combined dataset again for robot errors
            final_mask = ~combined_df['Description'].str.contains('|'.join(error_keywords), case=False, na=False)
            combined_df = combined_df[final_mask]

            # Save the clean, unified version
            combined_df.to_csv(FILE_PATH, index=False)
            print(f"✅ Smart Merge Complete. Total high-quality records: {len(combined_df)}")
            
        except Exception as e:
            print(f"Merge failed due to file corruption. Saving new data as fallback. Error: {e}")
            new_df.to_csv(FILE_PATH, index=False)
    else:
        new_df.to_csv(FILE_PATH, index=False)
        print(f"Initial file created with {len(new_df)} records.")

if __name__ == "__main__":
    update_diverse_books()