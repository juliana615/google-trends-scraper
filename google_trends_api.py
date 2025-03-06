import sys
import os
import requests
import json
import time
import re
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from urllib.parse import urlencode, urljoin

client = MongoClient('mongodb://127.0.0.1:27017')
db = client.google_trends
collection = db.data_by_filter

HEADER_GET = os.path.join(os.getcwd(), 'session_get.txt')
HEADER_POST = os.path.join(os.getcwd(), 'session.txt')

BASE_TRENDS_URL = 'https://trends.google.com/trends'
EXPLORE_URL = f'{BASE_TRENDS_URL}/api/explore'
INTEREST_OVER_TIME_URL = f'{BASE_TRENDS_URL}/api/widgetdata/multiline'
INTEREST_BY_REGION_URL = f'{BASE_TRENDS_URL}/api/widgetdata/comparedgeo'
RELATED_QUERIES_URL = f'{BASE_TRENDS_URL}/api/widgetdata/relatedsearches'

today = datetime.now()
TIME_SELECTOR = {
    "past_hour": {
        "name": "Past hour",
        "timeframe": "now 1-H",
    },
    "past_4_hours": {
        "name": "Past 4 hours",
        "timeframe": "now 4-H",
    },
    "past_day": {
        "name": "Past day",
        "timeframe": "now 1-d",
    },
    "past_7_days": {
        "name": "Past 7 days",
        "timeframe": "now 7-d",
    },
    "past_30_days": {
        "name": "Past 30 days",
        "timeframe": "today 1-m",
    },
    "past_90_days": {
        "name": "Past 90 days",
        "timeframe": "today 3-m",
    },
    "past_12_months": {
        "name": "Past 12 months",
        "timeframe": "today 12-m",
    },
    "past_5_years": {
        "name": "Past 5 years",
        "timeframe": "today 5-y",
    },
    "2004_present": {
        "name": "2004 - present",
        "timeframe": "all",
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

def get_google_cookies():
    return dict(filter(lambda i: i[0] == 'NID', requests.get(
        f'{BASE_TRENDS_URL}/explore/?geo=US'
    ).cookies.items()))
    
def get_response(url, header, params, method='GET', trim_chars=0):
    # print(f'url: {url}')
    # print(f'header: {header}')
    print(f'{url}?{urlencode(params)}')
    if method == 'GET':
        # response = requests.get(url, headers=header, params=params)
        response = requests.get(f'{url}?{urlencode(params)}')
    else:
        # response = requests.post(url, headers=header, params=params)
        response = requests.post(f'{url}?{urlencode(params)}', headers=header)
    if response.status_code == 200:
        print('Response is okay')
        # print(f'response: {response.text}')
        # print(f'response: {response.content}')
        result_text = response.text
        result_text = result_text[trim_chars:]
        result = json.loads(result_text)
    else:
        print(f'Response error: {response.status_code}')
        print(f'error text: {response.text}')
        sys.exit(1)
        # result = {}
    
    print('Sleeping for 1 second...')
    time.sleep(1)
    return result

def get_headers(method):
    if method == 'POST':
        filepath = HEADER_POST
    else:
        filepath = HEADER_GET
        
    with open(filepath, 'r', encoding='utf-8') as fh_header:
        session = json.load(fh_header)
        headers = session.get('headers')
    return headers
        
def get_token(header, params):
    result = get_response(EXPLORE_URL, header, params, method='POST', trim_chars=4)
    print(f'get_token share text: {result.get("shareText")}')
    widgets = result.get('widgets')
    interest_over_time = {
        "id": widgets[0].get('id'),
        "title": widgets[0].get('title'),
        "token": widgets[0].get('token'),
        "req": widgets[0].get('request')
    }
    interest_by_region = {
        "id": widgets[1].get('id'),
        "title": widgets[1].get('title'),
        "token": widgets[1].get('token'),
        "req": widgets[1].get('request')
    }
    related_topics = {
        "id": widgets[2].get('id'),
        "title": widgets[2].get('title'),
        "token": widgets[2].get('token'),
        "req": widgets[2].get('request')
    }
    related_queries = {
        "id": widgets[3].get('id'),
        "title": widgets[3].get('title'),
        "token": widgets[3].get('token'),
        "req": widgets[3].get('request')
    }
    
    return [interest_over_time, interest_by_region, related_topics, related_queries]

def get_interest_over_time(header, params):
    return get_response(INTEREST_OVER_TIME_URL, header, params, 'GET', trim_chars=5)

def get_interest_by_region(header, params):
    return get_response(INTEREST_BY_REGION_URL, header, params, 'GET', trim_chars=5)

def get_related_queries(header, params):
    return get_response(RELATED_QUERIES_URL, header, params, 'GET', trim_chars=5)

def get_full_data(header, hl, tz, widgets):
    print(f'[+] Retrieving full data...')
    
    # get interest over time
    params_interest_over_time = {
        "hl": hl,
        "tz": tz,
        "req": widgets[0].get('req'),
        "token": widgets[0].get('token')
    }
    print(f'[++] Get interest over time...')
    interest_over_time = get_interest_over_time(header, params_interest_over_time)
    
    # get interest by region
    params_interest_by_region = {
        "hl": hl,
        "tz": tz,
        "req": widgets[1].get('req'),
        "token": widgets[1].get('token')
    }
    print(f'[++] Get interest by region...')
    interest_by_region = get_interest_by_region(header, params_interest_by_region)

    # get realted topics
    params_related_topics = {
        "hl": hl,
        "tz": tz,
        "req": widgets[2].get('req'),
        "token": widgets[2].get('token')
    }
    print(f'[++] Get realted topics...')
    related_topics = get_related_queries(header, params_related_topics)

    # get related queries
    params_related_queries = {
        "hl": hl,
        "tz": tz,
        "req": widgets[3].get('req'),
        "token": widgets[3].get('token')
    }
    print(f'[++] Get related queries...')
    related_queries = get_related_queries(header, params_related_queries)

    return {
        "id": hash(''.join(widget.get('token', '') for widget in widgets[:4])),
        "interest_over_time": interest_over_time,
        "interest_by_region": interest_by_region,
        "related_topics": related_topics,
        "related_queries": related_queries,
    }
    
def get_result_from_filter(keyword: str, country: str, timeframe: str='today 12-m', category: int=0, gprop: str=''):
    if gprop not in ['', 'images', 'news', 'youtube', 'froogle']:
        raise ValueError('gprop must be empty (to indicate web), images, news, youtube, or froogle')    
    if timeframe not in ['now 1-H', 'now 4-H', 'now 1-d', 'now 7-d', 'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', 'all']:
        raise ValueError('timeframe must be now 1-H, now 4-H, now 1-d, now 7-d, today 1-m, today 3-m, today 12-m, today 5-y, or all')    
    
    hl = 'en-US'
    tz = -480
    # header = {'accept-language': hl}
    
    token_params = {
        "hl": hl,
        "tz": tz,
        "req": {
            "comparisonItem": [
                {"keyword": keyword, "time": timeframe, "geo": country}
            ],
            "category": category,
            "property": gprop
        }
    }
    
    # headers_post = get_headers('POST')
    # widgets = get_token(headers_post, token_params)
    # print(widgets)
    
    headers_get = get_headers('GET')
    widgets = [
                {
                    "id": "TIMESERIES",
                    "title": "Interest over time",
                    "token": "APP6_UEAAAAAZ8qqUWfNWho19jUMQDvU5ohgLR03VWP-",
                    "req": {
                    "time": "2024-03-06 2025-03-06",
                    "resolution": "WEEK",
                    "locale": "en-US",
                    "comparisonItem": [
                        {
                        "geo": {},
                        "complexKeywordsRestriction": {
                            "keyword": [
                            {
                                "type": "BROAD",
                                "value": "toy"
                            }
                            ]
                        }
                        }
                    ],
                    "requestOptions": {
                        "property": "",
                        "backend": "IZG",
                        "category": 0
                    },
                    "userConfig": {
                        "userType": "USER_TYPE_LEGIT_USER"
                    }
                    },
                },
                {
                    "id": "GEO_MAP",
                    "title": "Interest by region",
                    "token": "APP6_UEAAAAAZ8qqUVksEOPqNckvWr6AI4XVaccuji9N",
                    "req": {
                    "geo": {},
                    "comparisonItem": [
                        {
                        "time": "2024-03-06 2025-03-06",
                        "complexKeywordsRestriction": {
                            "keyword": [
                            {
                                "type": "BROAD",
                                "value": "toy"
                            }
                            ]
                        }
                        }
                    ],
                    "resolution": "COUNTRY",
                    "locale": "en-US",
                    "requestOptions": {
                        "property": "",
                        "backend": "IZG",
                        "category": 0
                    },
                    "userConfig": {
                        "userType": "USER_TYPE_LEGIT_USER"
                    }
                    },
                },
                {
                    "id": "RELATED_TOPICS",
                    "title": "Related topics",
                    "token": "APP6_UEAAAAAZ8qqUUrTTCxnl5edWFYuPr45OFNf1dsp",
                    "req": {
                    "restriction": {
                        "geo": {},
                        "time": "2024-03-06 2025-03-06",
                        "originalTimeRangeForExploreUrl": "today 12-m",
                        "complexKeywordsRestriction": {
                        "keyword": [
                            {
                            "type": "BROAD",
                            "value": "toy"
                            }
                        ]
                        }
                    },
                    "keywordType": "ENTITY",
                    "metric": [
                        "TOP",
                        "RISING"
                    ],
                    "trendinessSettings": {
                        "compareTime": "2023-03-06 2024-03-05"
                    },
                    "requestOptions": {
                        "property": "",
                        "backend": "IZG",
                        "category": 0
                    },
                    "language": "en",
                    "userCountryCode": "US",
                    "userConfig": {
                        "userType": "USER_TYPE_LEGIT_USER"
                    }
                    },
                },
                {
                    "id": "RELATED_QUERIES",
                    "title": "Related queries",
                    "token": "APP6_UEAAAAAZ8qqUZKKw3SkHxLioRvoKlaL3Yw_ctLO",
                    "req": {
                    "restriction": {
                        "geo": {},
                        "time": "2024-03-06 2025-03-06",
                        "originalTimeRangeForExploreUrl": "today 12-m",
                        "complexKeywordsRestriction": {
                        "keyword": [
                            {
                            "type": "BROAD",
                            "value": "toy"
                            }
                        ]
                        }
                    },
                    "keywordType": "QUERY",
                    "metric": [
                        "TOP",
                        "RISING"
                    ],
                    "trendinessSettings": {
                        "compareTime": "2023-03-06 2024-03-05"
                    },
                    "requestOptions": {
                        "property": "",
                        "backend": "IZG",
                        "category": 0
                    },
                    "language": "en",
                    "userCountryCode": "US",
                    "userConfig": {
                        "userType": "USER_TYPE_LEGIT_USER"
                    }
                    },
                }
                ]
    result = get_full_data(headers_get, hl, tz, widgets)
    print(result)
    # insert_or_update_db(result)
        
def main():
    keyword = 'startup'
    country = 'US'
    
    # past_hour, past_4_hours, past_day, past_7_days, past_30_days, past_90_days, past_12_months, past_5_years, 2004_present!@#
    
    timeframe = 'today 12-m'
    
    # 0: All categories, 1: Art & entertainment, ...
    category = 0
    
    # '': Web Search, 'images': Image search, 'news': News Search 'youtube': YouTube Search, 'froogle': Google Shopping Search
    gprop = ''
    
    get_result_from_filter(keyword, country, timeframe, category, gprop)
    
if __name__ == '__main__':
    main()