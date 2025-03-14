import requests
import json

headers = {
    "User-Agent": "PostmanRuntime/7.29.2",  # Mimic Postman
    "Accept": "*/*",
    "Connection": "keep-alive"
}

# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Brave/91.0.4472.124', 'Accept': 'application/json, text/plain, */*', 'Connection': 'keep-alive'}

def ensure_list(item):
	return list(item) if hasattr(item, '__iter__') and not isinstance(item, str) and not isinstance(item, dict) else [item]

def _encode_items(self, keywords, timeframe="today 12-m", geo=''):
    data = list(map(ensure_list, [keywords, timeframe, geo]))
    lengths = list(map(len, data))
    max_len = max(lengths)
    if not all(max_len % length == 0 for length in lengths):
        raise ValueError(f"Ambiguous input sizes: unable to determine how to combine inputs of lengths {lengths}")
    data = [item * (max_len // len(item)) for item in data]
    items = [dict(zip(['keyword', 'time', 'geo'], values)) for values in zip(*data)]
    return items

def _encode_request(params):
    if 'resolution' in params:
        print('res')
    if 'keyword' in params:
        print('asdf')
        keywords = ensure_list(params.pop('keyword'))
        if len(keywords) != 1:
            raise ValueError("This endpoint only supports a single keyword")
        params['keywords'] = keywords

    items = _encode_items(
        keywords  = params['keywords'],
        timeframe = params.get('timeframe', "today 12-m"),
        geo		  = params.get('geo', '')
    )
    
    req = {'req': json.dumps({
        'comparisonItem': items,
        'category': params.get('cat', 0),
        'property': params.get('gprop', '')
    })}

    req.update({'hl': 'en', 'tz': 360})
    return req

# params = {'req': '{"time": "2024-03-13 2025-03-13", "resolution": "WEEK", "locale": "en-US", "comparisonItem": [{"geo": {}, "complexKeywordsRestriction": {"keyword": [{"type": "BROAD", "value": "discord"}]}}], "requestOptions": {"property": "", "backend": "IZG", "category": 0}, "userConfig": {"userType": "USER_TYPE_EMBED_OVER_QUOTA"}}', 'token': 'APP6_UEAAAAAZ9OYk7Umzl_6jYCv3F7mcCXWHPfMhhZh', 'hl': 'en-US', 'tz': 360}

# params = {'req': '{"geo": {}, "comparisonItem": [{"time": "2024-03-13 2025-03-13", "complexKeywordsRestriction": {"keyword": [{"type": "BROAD", "value": "pizza"}]}}], "resolution": "COUNTRY", "locale": "en-US", "requestOptions": {"property": "", "backend": "IZG", "category": 0}, "userConfig": {"userType": "USER_TYPE_EMBED_OVER_QUOTA"}, "includeLowSearchVolumeGeos": false}', 'token': 'APP6_UEAAAAAZ9Ppx9QIj-oOESyGuDr-XNR_VYWFLXzR', 'hl': 'en-US', 'tz': 360}

params = {'req': '{"geo": {}, "comparisonItem": [{"time": "2024-03-13 2025-03-13", "complexKeywordsRestriction": {"keyword": [{"type": "BROAD", "value": "cake"}]}}], "resolution": "COUNTRY", "locale": "en-US", "requestOptions": {"property": "", "backend": "IZG", "category": 0}, "userConfig": {"userType": "USER_TYPE_EMBED_OVER_QUOTA"}, "includeLowSearchVolumeGeos": false}', 'token': 'APP6_UEAAAAAZ9P6L2BuKoohWajurqqpLkcs1FsDxKrO', 'hl': 'en-US', 'tz': 360}

# {"geo":{"country":"US"},"comparisonItem":[{"time":"2024-03-13 2025-03-13","complexKeywordsRestriction":{"keyword":[{"type":"BROAD","value":"coffee"}]}}],"resolution":"REGION","locale":"en-US","requestOptions":{"property":"","backend":"IZG","category":0},"userConfig":{"userType":"USER_TYPE_LEGIT_USER"}}

# params = _encode_request(params)

# response = requests.get("https://trends.google.com/trends/api/widgetdata/multiline", params=params, headers=headers)
response = requests.get("https://trends.google.com/trends/api/widgetdata/comparedgeo", params=params, headers=headers)

print(response.status_code, response.text)
