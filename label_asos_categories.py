import requests
import csv

categories_url = "https://asos2.p.rapidapi.com/categories/list"
categories_params = {
  "country": "US"
}
headers = {
    'X-RapidAPI-Key': 'f8f91464a8mshfabc82f63b3ad7dp1c2959jsn67f3f6088bcf',
    'X-RapidAPI-Host': 'asos2.p.rapidapi.com'
}


categories_response = requests.get(categories_url, headers=headers, params=categories_params)
categories_data = categories_response.json()

def collect_category_titles(categories):
    titles = []
    
    for category in categories:
        title = category.get('content', {}).get('title')
        if title:
            titles.append(title)
        
        # Handle nested categories
        children = category.get('children', [])
        titles.extend(collect_category_titles(children))
    
    return titles


navigation_data = categories_data.get('navigation', [])
titles = collect_category_titles(navigation_data)

with open("asos_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    header = ["title"]
    writer.writerow(header)

    for title in titles:
        row = [title]
        writer.writerow(row)


print("CSV file created :)")