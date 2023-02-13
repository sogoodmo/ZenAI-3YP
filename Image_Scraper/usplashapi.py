import requests
import os
import random
import csv 

def scrape_images():
    url = "https://api.unsplash.com/search/photos"
    query = "person standing"
    headers = {
        "Authorization": "Client-ID -ikuRA7H_4F1zHu4htLfSZketAHG4ezrJEUEanOCaO0"
    }

    # Make the initial search request
    response = requests.get(url, headers=headers, params={"query": query, "page": 1, "per_page": 20})

    if response.status_code == 200:
        search_data = response.json()
        total_pages = search_data["total_pages"]
        results = search_data["results"]

        # Create a directory to save the images
        if not os.path.exists("images"):
            os.makedirs("images")

        # Download the first 20 images
        for i, photo in enumerate(results):
            photo_url = photo["urls"]["full"]
            response = requests.get(photo_url)
            with open(f"images/img_{i}.jpg", "wb") as f:
                f.write(response.content)

        input('Press to continue')

        # Download the remaining images in batches of 20
        for page in range(2, total_pages + 1):
            response = requests.get(url, headers=headers, params={"query": query, "page": page, "per_page": 20})
            if response.status_code == 200:
                search_data = response.json()
                results = search_data["results"]
                for i, photo in enumerate(results):
                    photo_url = photo["urls"]["full"]
                    response = requests.get(photo_url)
                    with open(f"images/img_{(page - 1) * 20 + i}.jpg", "wb") as f:
                        f.write(response.content)

    else:
        print("Search request failed with status code:", response.status_code)

def split_csv():
    with open('images/training.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)

    random.shuffle(data)
    split_index = int(0.8 * len(data))
    training_data = data[:split_index]
    testing_data = data[split_index:]

    with open('training_.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(training_data)

    with open('testing_.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(testing_data)

    split_csv()