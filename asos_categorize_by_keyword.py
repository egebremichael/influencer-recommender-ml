import json
import pandas as pd
import requests

url = "https://asos2.p.rapidapi.com/categories/list"

querystring = {"country":"US","lang":"en-US"}

headers = {
	"X-RapidAPI-Key": "f8f91464a8mshfabc82f63b3ad7dp1c2959jsn67f3f6088bcf",
	"X-RapidAPI-Host": "asos2.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)


def classify_title(title):
    """
    Classify the title based on keywords.
    """
    fitness_keywords = ["Activewear", "SALE Activewear", "Gym", "Sports", "Running", "Training", "Ski & Snowboard", "Leggings", "Outdoors", "Yoga & Studio", "Soccer"]
    beauty_keywords = [
        "Face + Body", "Makeup", "Skin care", "Body care", "Hair care", "Tools & Accessories",
        "Suncare & Tanning", "Wellness", "Skinvestments", "Caudalie", "Charlotte Tilbury",
        "COSRX", "e.l.f.", "Elemis", "ghd", "Kristin Ess Hair", "Lottie", "MAC", "Olaplex",
        "Revolution", "Sunday Riley", "The Ordinary", "SALE Face + Body"
    ]
    
    title_lower = title.lower()
    if any(keyword.lower() in title_lower for keyword in fitness_keywords):
        return "fitness"
    elif any(keyword.lower() in title_lower for keyword in beauty_keywords):
        return "beauty"
    else:
        return "fashion"

def extract_titles_and_classify(items, path=None):
    """
    Recursively extract titles from the JSON data and classify them.
    """
    titles = []

    for item in items:
        if 'content' in item and 'title' in item['content'] and item['content']['title']:
            title = item['content']['title']
            category = classify_title(title)
            
            titles.append({'title': title, 'category': category})

        if 'children' in item:
            titles.extend(extract_titles_and_classify(item['children'], path=item.get('alias', path)))

    return titles

data = response.json()['navigation']
titles_and_categories = extract_titles_and_classify(data)


df = pd.DataFrame(titles_and_categories)
df.to_csv('titles_with_categories.csv', index=False)  
df.head()
