import sys
import os
import json
import time
import hashlib
from datetime import datetime
from trendspy import Trends
from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1:27017')
db = client.google_trends
collection = db.data_by_filter

# save to mongodb
def insert_or_update_db(data):
    result = collection.find_one({"id": data.get("id")})
    if result:
        collection.update_one({"id": data.get("id")}, {"$set": data})
        print(f'Updated: {data.get("id")}')
    else:
        collection.insert_one(data)
        print(f'Inserted: {data.get("id")}')

# show all data in db
def show_db_collection_all():
    documents = collection.find()
    for doc in documents:
        print(doc)

def get_trends_data(input_json):
    keyword = input_json.get("keyword")
    geo = input_json.get("geo")
    timeframe = input_json.get("timeframe")
    cat = input_json.get("category")
    gprop = input_json.get("gprop")
    headers = {
        "referer": "https://www.google.com/"
    }
    
    if keyword == "":
        raise ValueError('keyword cannot be empty')
    if gprop not in ['', 'images', 'news', 'youtube', 'froogle']:
        raise ValueError('gprop must be empty (to indicate web), images, news, youtube, or froogle')    
    if timeframe not in ['now 1-H', 'now 4-H', 'now 1-d', 'now 7-d', 'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', 'all']:
        raise ValueError('timeframe must be now 1-H, now 4-H, now 1-d, now 7-d, today 1-m, today 3-m, today 12-m, today 5-y, or all')
    
    print(f"[+] Getting trends data for {keyword} in {geo} for {timeframe}")
    tr = Trends(request_delay=3, retries=3)
    
    print(f"[++] Getting interest_over_time for {keyword} in {geo} for {timeframe}")
    interest_over_time = tr.interest_over_time(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    print("Waiting for 3 seconds...")
    time.sleep(3)
    print(f"[++] Getting interest_by_region for {keyword} in {geo} for {timeframe}")
    interest_by_region = tr.interest_by_region(keyword, geo=geo, timeframe=timeframe, cat=cat, gprop=gprop)
    print("Waiting for 3 seconds...")
    time.sleep(3)
    print(f"[++] Getting related_topics for {keyword} in {geo} for {timeframe}")
    related_topics = tr.related_topics(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    for key in related_topics:
        related_topics[key] = related_topics[key].to_dict('records')
    print("Waiting for 3 seconds...")
    time.sleep(3)
    print(f"[++] Getting related_queries for {keyword} in {geo} for {timeframe}")
    related_queries = tr.related_queries(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    for key in related_queries:
        related_queries[key] = related_queries[key].to_dict('records')
    print("Waiting for 3 seconds...")
    time.sleep(3)
    
    trends_data = {
        "id": hashlib.sha1(f"{keyword}_{geo}_{timeframe}_{cat}_{gprop}".encode()).hexdigest(),
        "interest_over_time": interest_over_time.to_dict('records'),
        "interest_by_region": interest_by_region.to_dict('records'),
        "related_topics": related_topics,
        "related_queries": related_queries,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    insert_or_update_db(trends_data)
    print("Data saved to MongoDB")

def main():
    input_json = {
        "keyword": "python",
        "geo": "",
        "timeframe": "today 12-m",
        "category": 0,
        "gprop": ""
    }
    get_trends_data(input_json)

if __name__ == "__main__":
    main()