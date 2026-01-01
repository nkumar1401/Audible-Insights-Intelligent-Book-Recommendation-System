import requests
import pandas as pd
import os
import time

def update_diverse_books():
    # 1. Define your Global-Local categories
    CATEGORIES = ["technology", "spirituality", "fiction", "business", "history", "biography"]
    REQUIRED_COLUMNS = ['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price', 'Description', 'Ranks and Genre']
    FILE_PATH = "raw_data/Audible_Catlog_Advanced_Features.csv"
    
    # NEW: Create the raw_data directory if it doesn't exist
    if not os.path.exists("raw_data"):
        print("Creating 'raw_data' directory...")
        os.makedirs("raw_data")
    all_new_data = []

    for category in CATEGORIES:
        print(f"Fetching newest books for: {category}...")
        # API call focusing on newest releases in India for this category
        api_url = f"https://www.googleapis.com/books/v1/volumes?q=subject:{category}+country:IN&orderBy=newest&maxResults=10"
        
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
                        'Price': 0, # API placeholder
                        'Description': volume.get('description', 'No description available'),
                        'Ranks and Genre': f"{category}, " + ", ".join(volume.get('categories', []))
                    }
                    all_new_data.append(book_data)
            # Sleep to avoid hitting API rate limits (Professional practice)
            time.sleep(1) 
        except Exception as e:
            print(f"Error fetching {category}: {e}")

    # 2. Convert and Clean
    new_df = pd.DataFrame(all_new_data)
    new_df = new_df[REQUIRED_COLUMNS] # schema gatekeeping
    
    # Remove duplicates before appending
    new_df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)

    # 3. Save to CSV
    if os.path.exists(FILE_PATH):
        # We append without the header so your main CSV stays clean
        new_df.to_csv(FILE_PATH, mode='a', header=False, index=False)
    else:
        new_df.to_csv(FILE_PATH, index=False)
    
    print(f"Update Complete. Added {len(all_new_data)} diverse titles.")
    # Verification Logic
    final_df = pd.read_csv(FILE_PATH)
    print(f"Update Verified: Dataset now contains {len(final_df)} total records.")

if __name__ == "__main__":
    update_diverse_books()