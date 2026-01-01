import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_ai_reasoning(user_query, book_title, description):
    """
    Leverages Llama 3 to explain the synergy between the user's mood 
    and the recommended book.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Reasoning unavailable: API Key missing."
            
        client = Groq(api_key=api_key)
        
        # System prompt to keep the founder's tone professional and concise
        prompt = f"""
        User Interest: {user_query}
        Recommended Book: {book_title}
        Book Description: {description[:500]}
        
        In exactly 2 sentences, explain why this book is a must-read for the user.
        Focus on the unique value proposition and growth potential.
        """
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "This title was selected for its high semantic alignment with your intellectual trajectory."