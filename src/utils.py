import requests

def get_book_cover(book_name):
    """Fetches a real book cover from Google Books API."""
    try:
        query = book_name.replace(" ", "+")
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{query}&maxResults=1"
        response = requests.get(url, timeout=2)
        data = response.json()
        
        # Extract the thumbnail link
        if "items" in data:
            return data["items"][0]["volumeInfo"]["imageLinks"]["thumbnail"]
    except:
        pass
    # Professional fallback if internet fails or book not found
    return f"https://via.placeholder.com/180x260/232f3e/ffffff?text={book_name[:15]}"