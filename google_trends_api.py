import requests
import json
import time
import re
from datetime import datetime

'''
{"restriction":{"geo":{"country":"US"},"time":"2004-01-01 2025-03-02","originalTimeRangeForExploreUrl":"all","complexKeywordsRestriction":{"keyword":[{"type":"BROAD","value":"startup"}]}},"keywordType":"QUERY","metric":["TOP","RISING"],"trendinessSettings":{"compareTime":"2004-01-01 2005-01-01"},"requestOptions":{"property":"","backend":"IZG","category":0},"language":"en","userCountryCode":"US","userConfig":{"userType":"USER_TYPE_LEGIT_USER"}}
'''

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

def get_related_queries(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header):
    hl = 'en-US'
    tz = '-480'
    keyword_type = 'QUERY'
    language = 'en'
    user_country_code = 'US'
    user_type = 'USER_TYPE_LEGIT_USER'
    req = f'%7B%22restriction%22:%7B%22geo%22:%7B%22country%22:%22{country}%22%7D,%22time%22:%22{time_from}+{time_to}%22,%22originalTimeRangeForExploreUrl%22:%22all%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22{keyword}%22%7D%5D%7D%7D,%22keywordType%22:%22{keyword_type}%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%22{compare_time_from}+{compare_time_to}%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D,%22language%22:%22{language}%22,%22userCountryCode%22:%22{user_country_code}%22,%22userConfig%22:%7B%22userType%22:%22{user_type}%22%7D%7D'
    
    url = f'https://trends.google.com/trends/api/widgetdata/relatedsearches?hl={hl}&tz={tz}&req={req}&token={token}'
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
    
def get_related_topics(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header):
    pass

def get_compared_geo(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header):
    pass

def get_multi_line(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header):
    pass

def get_full_data(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header):
    # get miltiline
    multi_line = get_multi_line(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header)
    
    # get compared geo
    compared_geo = get_compared_geo(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header)
    
    # get related queries
    related_topics = get_related_topics(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header)
    
    # get related queries
    related_queries = get_related_queries(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header)

def main():
    today = datetime.today()
    country = 'US'
    time_from = '2004-01-01'
    time_to = today.strftime("%Y-%m-%d")
    print(f'time_to: {time_to}')
    compare_time_from = '2004-01-01'
    compare_time_to = '2005-01-01'
    keyword = 'startup'

    [token, header] = get_token_and_header()
    # print('token: ', token)
    # print('header: ', header)
    # print('cookie: ', header.get('sec-ch-ua-platform'))
    
    # related_queries = get_related_queries(country, time_from, time_to, compare_time_from, compare_time_to, keyword, token, header)
    # print(related_queries)

if __name__ == '__main__':
    main()