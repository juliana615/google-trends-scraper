import sys
import requests
import json
import time
import re
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient

client = MongoClient('mongodb://127.0.0.1:27017')
db = client.google_trends
collection = db.data_by_filter

today = datetime.now()
TIME_SELECTOR = {
    "past_hour": {
        "name": "Past hour",
        "time_range": "now 1-H",
        "time_from": (today - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "time_to": today.strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_from": (today - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_to": (today - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "resolution": "MINUTE",
    },
    "past_4_hours": {
        "name": "Past 4 hours",
        "time_range": "now 4-H",
        "time_from": (today - timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S'),
        "time_to": today.strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_from": (today - timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_to": (today - timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S'),
        "resolution": "MINUTE",
    },
    "past_day": {
        "name": "Past day",
        "time_range": "now 1-d",
        "time_from": (today - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "time_to": today.strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_from": (today - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_to": (today - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S'),
        "resolution": "EIGHT_MINUTE",
    },
    "past_7_days": {
        "name": "Past 7 days",
        "time_range": "now 7-d",
        "time_from": (today - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S'),
        "time_to": today.strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_from": (today - timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%S'),
        "compare_time_to": (today - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S'),
        "resolution": "HOUR",
    },
    "past_30_days": {
        "name": "Past 30 days",
        "time_range": "today 1-m",
        "time_from": (today - timedelta(days=30)).strftime('%Y-%m-%d'),
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": (today - timedelta(days=61)).strftime('%Y-%m-%d'),
        "compare_time_to": (today - timedelta(days=31)).strftime('%Y-%m-%d'),
        "resolution": "DAY",
    },
    "past_90_days": {
        "name": "Past 90 days",
        "time_range": "today 3-m",
        "time_from": (today - timedelta(days=90)).strftime('%Y-%m-%d'),
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": (today - timedelta(days=181)).strftime('%Y-%m-%d'),
        "compare_time_to": (today - timedelta(days=91)).strftime('%Y-%m-%d'),
        "resolution": "DAY",
    },
    "past_12_months": {
        "name": "Past 12 months",
        "time_range": "today 12-m",
        "time_from": (today - relativedelta(months=12)).strftime('%Y-%m-%d'),
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": (today - relativedelta(years=2)).strftime('%Y-%m-%d'),
        "compare_time_to": (today - relativedelta(months=12) - timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolution": "WEEK",
    },
    "past_5_years": {
        "name": "Past 5 years",
        "time_range": "today 5-y",
        "time_from": (today - relativedelta(years=5)).strftime('%Y-%m-%d'),
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": (today - relativedelta(years=10)).strftime('%Y-%m-%d'),
        "compare_time_to": (today - relativedelta(years=5) - timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolution": "WEEK",
    },
    "2004_present": {
        "name": "2004 - present",
        "time_range": "all",
        "time_from": '2004-01-01',
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": '2004-01-01',
        "compare_time_to": '2005-01-01',
        "resolution": "MONTH",
    },
    "custom_range": {
        "name": "Custom time range",
        "time_range": "",
        "time_from": 0,
        "time_to": today.strftime('%Y-%m-%d'),
        "compare_time_from": 0,
        "compare_time_to": 0,
    },
}
    
def insert_or_update_db(data):
    
    result = collection.find_one({"id": data.get("id")})
    if result:
        collection.update_one({"id": data.get("id")}, {"$set": data})
        print(f'Updated: {data.get("id")}')
    else:
        collection.insert_one(data)
        print(f'Inserted: {data.get("id")}')

def show_db_collection_all():
    documents = collection.find()
    for doc in documents:
        print(doc)
        
def get_token_and_header():
    with open('./session.txt', 'r', encoding='utf-8') as fh_session:
        session = fh_session.read()
    match_token = re.search(r'token=([A-Za-z0-9_-]+)"', session)
    if match_token:
        token = match_token.group(1)
    else:
        token = ''
        
    match_header = re.search(r'"headers":\s*({.*?})\s*,\s*"body"', session, re.DOTALL)
    if match_header:
        header_string = match_header.group(1)
        header = json.loads(header_string)
    else:
        header = {}
        
    return [token, header]

def get_response(url, header):
    # print(f'url: {url}')
    # print(f'header: {header}')
    response = requests.get(url, headers=header)
    if response.status_code == 200:
        print('Response is okay')
        # print(f'response: {response.text}')
        # print(f'response: {response.content}')
        result_text = response.text
        result_text = result_text.removeprefix(')]}\',\n')
        result = json.loads(result_text)
    else:
        print(f'Response error: {response.status_code}')
        result = {}
    
    print('Sleeping for 1 second...')
    time.sleep(1)
    return result

def get_parts(slug, country, time_from, time_to, compare_time_from, compare_time_to, time_range, keyword, token, header, keyword_type, resolution, backend, user_country_code, hl='en-US', tz='-480', language='en', user_type='USER_TYPE_LEGIT_USER'):
    if slug == 'multiline':
        req = f'%7B%22time%22:%22{time_from}+{time_to}%22,%22resolution%22:%22{resolution}%22,%22locale%22:%22{hl}%22,%22comparisonItem%22:%5B%7B%22geo%22:%7B%22country%22:%22{country}%22%7D,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22{keyword}%22%7D%5D%7D%7D%5D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22{backend}%22,%22category%22:0%7D,%22userConfig%22:%7B%22userType%22:%22{user_type}%22%7D%7D'
    elif slug == 'comparedgeo':
        location_resolution = 'REGION'
        req = f'%7B%22time%22:%22{time_from}+{time_to}%22,%22resolution%22:%22{location_resolution}%22,%22locale%22:%22{hl}%22,%22comparisonItem%22:%5B%7B%22geo%22:%7B%22country%22:%22{country}%22%7D,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22{keyword}%22%7D%5D%7D%7D%5D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22{backend}%22,%22category%22:0%7D,%22userConfig%22:%7B%22userType%22:%22{user_type}%22%7D%7D'
    elif slug == 'relatedsearches':
        if keyword_type == 'QUERY':
            req = f'%7B%22restriction%22:%7B%22geo%22:%7B%22country%22:%22{country}%22%7D,%22time%22:%22{time_from}+{time_to}%22,%22originalTimeRangeForExploreUrl%22:%22{time_range}%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22{keyword}%22%7D%5D%7D%7D,%22keywordType%22:%22{keyword_type}%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%22{compare_time_from}+{compare_time_to}%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D,%22language%22:%22{language}%22,%22userCountryCode%22:%22{user_country_code}%22,%22userConfig%22:%7B%22userType%22:%22{user_type}%22%7D%7D'
        elif keyword_type == 'ENTITY':
            req = f'%7B%22restriction%22:%7B%22geo%22:%7B%22country%22:%22{country}%22%7D,%22time%22:%22{time_from}+{time_to}%22,%22originalTimeRangeForExploreUrl%22:%22{time_range}%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22{keyword}%22%7D%5D%7D%7D,%22keywordType%22:%22{keyword_type}%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%22{compare_time_from}+{compare_time_to}%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22{backend}%22,%22category%22:0%7D,%22language%22:%22{language}%22,%22userCountryCode%22:%22{user_country_code}%22,%22userConfig%22:%7B%22userType%22:%22{user_type}%22%7D%7D'
        else:
            print(f'keyword_type mismatch: {keyword_type}')
            sys.exit(1)
    else:
        print(f'slug mismatch: {slug}')
        sys.exit(1)
        
    url = f'https://trends.google.com/trends/api/widgetdata/relatedsearches?hl={hl}&tz={tz}&req={req}&token={token}'
    print(url)
    return get_response(url, header)

def get_full_data(country, time_from, time_to, compare_time_from, compare_time_to, time_range, keyword, token, header, resolution, backend, user_country_code):
    print(f'[+] Retrieving full data of country: {country}, keyword: {keyword}, time_range: {time_range}')
    
    # get miltiline
    print(f'[++] Retrieving multiline ...')
    multi_line = get_parts('multiline', country, time_from, time_to, '', '', '', keyword, token, header, '', resolution, backend, user_country_code)
    
    # get compared geo
    print(f'[++] Retrieving comparedgeo ...')
    compared_geo = get_parts('comparedgeo', country, time_from, time_to, '', '', '', keyword, token, header, '', resolution, backend, user_country_code)
    
    # get related topics
    print(f'[++] Retrieving related topics ...')
    related_topics = get_parts('relatedsearches', country, time_from, time_to, compare_time_from, compare_time_to, time_range, keyword, token, header, 'ENTITY', '', backend, user_country_code)
    
    # get related queries
    print(f'[++] Retrieving related queries ...')
    related_queries = get_parts('relatedsearches', country, time_from, time_to, compare_time_from, compare_time_to, time_range, keyword, token, header, 'QUERY', '', backend, user_country_code)
    
    return {
        "id": hash(country + time_from + time_to + keyword),
        "multi_line": multi_line,
        "compared_geo": compared_geo,
        "related_topics": related_topics,
        "related_queries": related_queries,
    }
    
def main():
    country = 'US'
    # past_hour, past_4_hours, past_day, past_7_days, past_30_days, past_90_days, past_12_months, past_5_years, 2004_present
    time_selector = 'past_12_months'
    keyword = 'startup'
    backend = 'CM'
    # backend = 'IZG'
    user_country_code = 'US'
    [token, header] = get_token_and_header()
    time_from = TIME_SELECTOR.get(time_selector).get('time_from')
    time_to = TIME_SELECTOR.get(time_selector).get('time_to')
    compare_time_from = TIME_SELECTOR.get(time_selector).get('compare_time_from')
    compare_time_to = TIME_SELECTOR.get(time_selector).get('compare_time_to')
    time_range = TIME_SELECTOR.get(time_selector).get('time_range')
    resolution = TIME_SELECTOR.get(time_selector).get('resolution')
    
    result = get_full_data(country, time_from, time_to, compare_time_from, compare_time_to, time_range, keyword, token, header, resolution, backend, user_country_code)
    # insert_or_update_db(result)
    
if __name__ == '__main__':
    main()