import sys
import os
import json
import hashlib
import requests
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from typing import Dict, List, Union, Optional, Any
from time import sleep,time
from enum import Enum
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, quote_plus
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient

BATCH_URL			    = f'https://trends.google.com/_/TrendsUi/data/batchexecute'
HOT_TRENDS_URL          = f'https://trends.google.com/trends/hottrends/visualize/internal/data'

# ----------- API LINKS -------------
API_URL  				= f'https://trends.google.com/trends/api'
API_EXPLORE_URL 		= f'{API_URL}/explore'
API_GEO_DATA_URL        = f'{API_URL}/explore/pickers/geo'
API_CATEGORY_URL        = f'{API_URL}/explore/pickers/category'
API_TOPCHARTS_URL       = f'{API_URL}/topcharts'
API_AUTOCOMPLETE        = f'{API_URL}/autocomplete/'
DAILY_SEARCHES_URL 		= f'{API_URL}/dailytrends'
REALTIME_SEARCHES_URL   = f'{API_URL}/realtimetrends'

API_TOKEN_URL 			= f'https://trends.google.com/trends/api/widgetdata'
API_TIMELINE_URL 		= f'{API_TOKEN_URL}/multiline'
API_MULTIRANGE_URL 		= f'{API_TOKEN_URL}/multirange'
API_GEO_URL 			= f'{API_TOKEN_URL}/comparedgeo'
API_RELATED_QUERIES_URL = f'{API_TOKEN_URL}/relatedsearches'

# ----------- EMBED LINKS -------------
EMBED_URL               = f'https://trends.google.com/trends/embed/explore'
EMBED_GEO_URL           = f'{EMBED_URL}/GEO_MAP'
EMBED_TOPICS_URL        = f'{EMBED_URL}/RELATED_TOPICS'
EMBED_QUERIES_URL       = f'{EMBED_URL}/RELATED_QUERIES'
EMBED_TIMESERIES_URL    = f'{EMBED_URL}/TIMESERIES'

# --------------- RSS ----------------- 
DAILY_RSS 				= f'https://trends.google.com/trends/trendingsearches/daily/rss'
REALTIME_RSS            = f'https://trends.google.com/trending/rss'

# --- TIME DELAY ----
TIME_DELAY = 6

# ------ TREND TOPICS ---------
TREND_TOPICS = {
    1: "Autos and Vehicles",
    2: "Beauty and Fashion",
    3: "Business and Finance",
    20: "Climate",
    4: "Entertainment",
    5: "Food and Drink",
    6: "Games",
    7: "Health",
    8: "Hobbies and Leisure",
    9: "Jobs and Education",
    10: "Law and Government",
    11: "Other",
    13: "Pets and Animals",
    14: "Politics",
    15: "Science",
    16: "Shopping",
    17: "Sports",
    18: "Technology",
    19: "Travel and Transportation"
}

# ------ CONVERTER CONSTANTS -----------------
_RELATED_QUERIES_DESIRED_COLUMNS  = ['query','topic','title','type','mid','value']

# ------ UTIL CONSTANTS -----------------
_HEX_TO_CHAR_DICT = {
	r'\x7b':'{',
	r'\x7d':'}',
	r'\x22':'"',
	r'\x5d':']',
	r'\x5b':'[',
		'\\\\':'\\'
}
_tag_pattern = re.compile(r'<([\w:]+)>(.*?)</\1>', re.DOTALL)

# ------- TIMEFRAME CONSTANTS -------------
__all__ = ['convert_timeframe', 'timeframe_to_timedelta', 'verify_consistent_timeframes']


# --- TIMEFRAME UTILS ---
# Regular expression pattern to validate date strings in the format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH'
VALID_DATE_PATTERN = r'^\d{4}-\d{2}-\d{2}(T\d{2})?$'

# Set of fixed timeframes supported by an external API
FIXED_TIMEFRAMES = {'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d', 'today 1-m', 'today 3-m', 'today 5-y', 'today 12-m', 'all'}

# Date format strings for standard and datetime with hour formats
DATE_FORMAT = "%Y-%m-%d"
DATE_T_FORMAT = "%Y-%m-%dT%H"

# Regular expression pattern to validate offset strings like '10-d', '5-H', etc.
OFFSET_PATTERN = r'\d+[-]?[Hdmy]$'

# Mapping of units (H, d, m, y) to relativedelta arguments
UNIT_MAP = {'H': 'hours', 'd': 'days', 'm': 'months', 'y': 'years'}


def _is_valid_date(date_str):
	# Checks if the given string matches the valid date pattern
	return bool(re.match(VALID_DATE_PATTERN, date_str))


def _is_valid_format(offset_str):
	# Checks if the given string matches the valid offset pattern
	return bool(re.match(OFFSET_PATTERN, offset_str))


def _extract_time_parts(offset_str):
	# Extracts numerical value and unit (H, d, m, y) from the offset string
	match = re.search(r'(\d+)[-]?([Hdmy]+)', offset_str)
	if match:
		return int(match.group(1)), match.group(2)
	return None


def _decode_trend_datetime(date_str):
	# Parses the date string into a datetime object based on whether it includes time ('T' character)
	return datetime.strptime(date_str, DATE_T_FORMAT) if 'T' in date_str else datetime.strptime(date_str, DATE_FORMAT)


def _process_two_dates(date_part_1, date_part_2):
	isT1 = 'T' in date_part_1
	isT2 = 'T' in date_part_2
	if (not isT1) and (not isT2):
		return f'{date_part_1} {date_part_2}'

	# Processes two date parts and returns the formatted result
	date_1 = _decode_trend_datetime(date_part_1)
	date_2 = _decode_trend_datetime(date_part_2)

	# Adjust date formatting if only one of the dates includes hour information
	if (isT1) and (not isT2):
		date_2 += timedelta(days=1)
		date_2 = date_2.replace(hour=0)
	elif (not isT1) and (isT2):
		date_1 = date_1.replace(hour=0)

	# Ensure the difference between dates does not exceed 7 days when time information is included
	if ('T' in date_part_1 or 'T' in date_part_2) and abs((date_1 - date_2).days) > 7:
		raise ValueError(f'Date difference cannot exceed 7 days for format with hours: {date_part_1} {date_part_2}')

	# Return the formatted result with both dates including hours
	return f'{date_1.strftime(DATE_T_FORMAT)} {date_2.strftime(DATE_T_FORMAT)}'


def _process_date_with_offset(date_part_1, offset_part):
	# Processes a date part with an offset to calculate the resulting timeframe
	date_1 = _decode_trend_datetime(date_part_1)
	count, unit = _extract_time_parts(offset_part)

	# Calculate the offset using relativedelta
	raw_diff = relativedelta(**{UNIT_MAP[unit]: count})
	if unit in {'m', 'y'}:
		# Special handling for months and years: adjust based on the current UTC date
		now = datetime.now(timezone.utc)
		end_date = now - raw_diff
		raw_diff = now - end_date

	# Raise an error if the offset exceeds 7 days for formats that include time
	if 'T' in date_part_1 and ((unit == 'd' and count > 7) or (unit == 'H' and count > 7 * 24)):
		raise ValueError(f'Offset cannot exceed 7 days for format with time: {date_part_1} {offset_part}. Use YYYY-MM-DD format or "today".')

	# Determine the appropriate date format based on the unit (hours/days or months/years)
	date_format = DATE_T_FORMAT if 'T' in date_part_1 else DATE_FORMAT
	return f'{(date_1 - raw_diff).strftime(date_format)} {date_1.strftime(date_format)}'


def convert_timeframe(timeframe, convert_fixed_timeframes_to_dates=False):
	"""
	Converts timeframe strings to Google Trends format.

	Supports multiple formats:
	1. Fixed timeframes ('now 1-H', 'today 12-m', etc.)
	2. Date ranges ('2024-01-01 2024-12-31')
	3. Date with offset ('2024-03-25 5-m')
	4. Hour-specific ranges ('2024-03-25T12 2024-03-25T15')

	Parameters:
		timeframe (str): Input timeframe string
		convert_fixed_timeframes_to_dates (bool): Convert fixed timeframes to dates

	Returns:
		str: Converted timeframe string in Google Trends format

	Raises:
		ValueError: If timeframe format is invalid
	"""
	# If the timeframe is in the fixed set and conversion is not requested, return as is
	if (timeframe in FIXED_TIMEFRAMES) and (not convert_fixed_timeframes_to_dates):
		return timeframe
	
	# Replace 'now' and 'today' with the current datetime in the appropriate format
	utc_now = datetime.now(timezone.utc)
	if convert_fixed_timeframes_to_dates and timeframe=='all':
		return '2024-01-01 {}'.format(utc_now.strftime(DATE_FORMAT))

	timeframe = timeframe.replace('now', utc_now.strftime(DATE_T_FORMAT)).replace('today', utc_now.strftime(DATE_FORMAT))

	# Split the timeframe into two parts
	parts = timeframe.split()
	if len(parts) != 2:
		raise ValueError(f"Invalid timeframe format: {timeframe}. Expected format: '<date> <offset>' or '<date> <date>'.")

	date_part_1, date_part_2 = parts

	# Process the timeframe based on its parts
	if _is_valid_date(date_part_1):
		if _is_valid_date(date_part_2):
			# Process if both parts are valid dates
			return _process_two_dates(date_part_1, date_part_2)
		elif _is_valid_format(date_part_2):
			# Process if the second part is a valid offset
			return _process_date_with_offset(date_part_1, date_part_2)

	raise ValueError(f'Could not process timeframe: {timeframe}')

def timeframe_to_timedelta(timeframe):
	result = convert_timeframe(timeframe, convert_fixed_timeframes_to_dates=True)
	date_1, date_2 = result.split()
	datetime_1 = _decode_trend_datetime(date_1)
	datetime_2 = _decode_trend_datetime(date_2)
	return (datetime_2 - datetime_1)

def verify_consistent_timeframes(timeframes):
	"""
	Verifies that all timeframes have consistent resolution.

	Google Trends requires all timeframes in a request to have the same
	data resolution (e.g., hourly, daily, weekly).

	Parameters:
		timeframes (list): List of timeframe strings

	Returns:
		bool: True if timeframes are consistent

	Raises:
		ValueError: If timeframes have different resolutions
	"""
	if isinstance(timeframes, str):
		return True

	timedeltas = list(map(timeframe_to_timedelta, timeframes))
	if all(td == timedeltas[0] for td in timedeltas):
		return True
	else:
		raise ValueError(f"Inconsistent timeframes detected: {[str(td) for td in timedeltas]}")

# Define the mapping between time range, resolution, and its range
def get_resolution_and_range(timeframe):
	delta = timeframe_to_timedelta(timeframe)
	if delta < timedelta(hours=5):
		return "1 minute", "delta < 5 hours"
	elif delta < timedelta(hours=36):
		return "8 minutes", "5 hours <= delta < 36 hours"
	elif delta < timedelta(hours=72):
		return "16 minutes", "36 hours <= delta < 72 hours"
	elif delta < timedelta(days=8):
		return "1 hour", "72 hours <= delta < 8 days"
	elif delta < timedelta(days=270):
		return "1 day", "8 days <= delta < 270 days"
	elif delta < timedelta(days=1900):
		return "1 week", "270 days <= delta < 1900 days"
	else:
		return "1 month", "delta >= 1900 days"

# Function to check if all timeframes have the same resolution
def check_timeframe_resolution(timeframes):
	timeframes = ensure_list(timeframes)
	resolutions = list(map(get_resolution_and_range, timeframes))

	# Extract only resolutions (without ranges) to check if they are the same
	resolution_values = [r[0] for r in resolutions]

	# Check if all resolutions are the same
	deltas = [timeframe_to_timedelta(timeframe) for timeframe in timeframes]
	if len(set(resolution_values)) > 1:
		# If there are differences, output an error message with details
		error_message = "Error: Different resolutions detected for the timeframes:\n"
		for timeframe, delta, (resolution, time_range) in zip(timeframes, deltas, resolutions):
			error_message += (
				f"Timeframe: {timeframe}, Delta: {delta}, "
				f"Resolution: {resolution} (based on range: {time_range})\n"
			)
		raise ValueError(error_message)
	
	min_delta, min_timeframe = min(zip(deltas, timeframes))
	max_delta, max_timeframe = max(zip(deltas, timeframes))
	
	if max_delta >= min_delta * 2:
		raise ValueError(
			f"Error: The maximum delta {max_delta} (from timeframe {max_timeframe}) "
			f"should be less than twice the minimum delta {min_delta} (from timeframe {min_timeframe})."
		)
# --- END TIMEFRAME UTILS ---


# --- TREND KEYWORD ---
class TrendKeyword:
    """
    Represents a trending search term with associated metadata.

    This class encapsulates information about a trending keyword, including
    its search volume, related news, geographic information, and timing data.

    Attributes:
        keyword (str): The trending search term
        news (list): Related news articles
        geo (str): Geographic location code
        started_timestamp (tuple): When the trend started
        ended_timestamp (tuple): When the trend ended (if finished)
        volume (int): Search volume
        volume_growth_pct (float): Percentage growth in search volume
        trend_keywords (list): Related keywords
        topics (list): Related topics
        news_tokens (list): Associated news tokens
        normalized_keyword (str): Normalized form of the keyword
    """
    def __init__(self, item: list):
        (
            self.keyword,
            self.news, # news!
            self.geo,
            self.started_timestamp,
            self.ended_timestamp,
            self._unk2,
            self.volume,
            self._unk3,
            self.volume_growth_pct,
            self.trend_keywords,
            self.topics,
            self.news_tokens,
            self.normalized_keyword
        ) = item
        if self.news:
            self.news = list(map(NewsArticle.from_api, self.news))

    @property
    def topic_names(self):
        """Returns a list of topic names for the trend's topic IDs."""
        return [TREND_TOPICS.get(topic_id, f"Unknown Topic ({topic_id})") for topic_id in self.topics]

    def _convert_to_datetime(self, raw_time):
        """Converts time in seconds to a datetime object with UTC timezone, if it exists."""
        return datetime.fromtimestamp(raw_time, tz=timezone.utc) if raw_time else None

    @property
    def is_trend_finished(self) -> bool:
        """Checks if the trend is finished."""
        return self.ended_timestamp is not None

    def hours_since_started(self) -> float:
        """Returns the number of hours elapsed since the trend started."""
        if not self.started_timestamp:
            return 0
        delta = datetime.now(tz=timezone.utc) - datetime.fromtimestamp(self.started_timestamp[0], tz=timezone.utc)
        return delta.total_seconds() / 3600

    def __repr__(self):
        """Returns a complete string representation for object reconstruction."""
        # Convert NewsArticle objects back to their original form
        news_data = self.news
        if self.news:
            news_data = [
                {
                    'title': article.title,
                    'url': article.url,
                    'source': article.source,
                    'time': article.time,
                    'picture': article.picture,
                    'snippet': article.snippet
                } for article in self.news
            ]
        
        # Create list of all components in initialization order
        components = [
            self.keyword,
            news_data,
            self.geo,
            self.started_timestamp,
            self.ended_timestamp,
            self._unk2,
            self.volume,
            self._unk3,
            self.volume_growth_pct,
            self.trend_keywords,
            self.topics,
            self.news_tokens,
            self.normalized_keyword
        ]
        
        return f"{self.__class__.__name__}({components!r})"

    def __str__(self):
        """Returns a human-readable string representation."""
        timeframe = datetime.fromtimestamp(self.started_timestamp[0]).strftime('%Y-%m-%d %H:%M:%S')
        if self.is_trend_finished:
            timeframe += ' - ' + datetime.fromtimestamp(self.ended_timestamp[0]).strftime('%Y-%m-%d %H:%M:%S')
        else:
            timeframe += ' - now'
            
        s =    f'Keyword        : {self.keyword}'
        s += f'\nGeo            : {self.geo}'
        s += f'\nVolume         : {self.volume} ({self.volume_growth_pct}%)'
        s += f'\nTimeframe      : {timeframe}'
        s += f'\nTrend keywords : {len(self.trend_keywords)} keywords ({truncate_string(",".join(self.trend_keywords), 50)})'
        s += f'\nNews tokens    : {len(self.news_tokens)} tokens'
        return s

    def brief_summary(self):
        """Returns an informative summary of the trend."""
        parts = [f"[{self.geo}] {self.keyword}: {self.volume:,} searches"]
        
        if self.trend_keywords:
            parts.append(f"{len(self.trend_keywords)} related keywords")
        if self.topics:
            topic_list = ", ".join(self.topic_names)
            parts.append(f"topics: {topic_list}")
        if self.news:
            parts.append(f"{len(self.news)} news articles")
        
        return ", ".join(parts)

    def _repr_pretty_(self, p, cycle):
        """Integration with IPython's pretty printer."""
        if cycle:
            p.text("[...]")
        else:
            p.text(self.brief_summary())
            
    def __format__(self, format_spec):
        """Implements formatting for f-strings and format() method."""
        return self.brief_summary()
        
    def __str__(self):
        return self.brief_summary()

class TrendKeywordLite:
    """
    A lightweight version of TrendKeyword for simple trend representation.

    This class provides a simplified view of trending keywords, primarily used
    for RSS feeds and basic trending data.

    Attributes:
        keyword (str): The trending search term
        volume (str): Approximate search volume
        trend_keywords (list): Related keywords
        link (str): URL to more information
        started (int): Unix timestamp when the trend started
        picture (str): URL to related image
        picture_source (str): Source of the picture
        news (list): Related news articles
    """
    def __init__(self, keyword, volume, trend_keywords, link, started, picture, picture_source, news):
        self.keyword = keyword
        self.volume = volume
        self.trend_keywords = trend_keywords
        self.link = link
        self.started = None
        self.picture = picture
        self.picture_source = picture_source
        self.news = news
        if started:
            self.started = self._parse_pub_date(started)
        elif news:
            self.started = min([item.time for item in news])

    @staticmethod
    def _parse_pub_date(pub_date):
        return int(datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z').timestamp())

    @classmethod
    def from_api(cls, data):
        title = data.get('title')
        if isinstance(title, dict):
            title = title.get('query')
        volume          = data.get('formattedTraffic') or data.get('approx_traffic')
        trend_keywords  = ([item.get('query') for item in data.get('relatedQueries', [])])
        trend_keywords  = trend_keywords or (data.get('description', '').split(', ') if 'description' in data else None)
        trend_keywords  = trend_keywords or list(set([word for item in data.get('idsForDedup', '') for word in item.split(' ')]))
        link            = data.get('shareUrl') or data.get('link')
        started         = data.get('pubDate')
        picture         = data.get('picture') or data.get('image', {}).get('imageUrl')
        picture_source  = data.get('picture_source') or data.get('image', {}).get('source')
        articles        = data.get('articles') or data.get('news_item') or []

        return cls(
            keyword			= title,
            volume			= volume,
            trend_keywords 	= trend_keywords,
            link			= link,
            started         = started,
            picture         = picture,
            picture_source  = picture_source,
            news            = [NewsArticle.from_api(item) for item in ensure_list(articles)]
        )

    def __repr__(self):
        return f"TrendKeywordLite(title={self.keyword}, traffic={self.volume}, started={self.started})"

    def __str__(self):
        s  =   'Keyword        : {}'.format(self.keyword)
        s += '\nVolume         : {}'.format(self.volume) if self.volume else ''
        s += '\nStarted        : {}'.format(datetime.fromtimestamp(self.started).strftime('%Y-%m-%d %H:%M:%S')) if self.started else ''
        s += '\nTrend keywords : {} keywords ({})'.format(len(self.trend_keywords), truncate_string(','.join(self.trend_keywords), 50)) if self.trend_keywords else ''
        s += '\nNews           : {} news'.format(len(self.news)) if self.news else ''
        return s

# --- END TREND KEYWORD ---


# --- TREND LIST ---
class TrendList(list):
    """
    A list-like container for trending topics with additional filtering capabilities.
    Inherits from list to maintain all standard list functionality.
    """
    
    def __init__(self, trends: List[TrendKeyword]):
        super().__init__(trends)
    
    def filter_by_topic(self, topic: Union[int, str, List[Union[int, str]]]) -> 'TrendList':
        """
        Filter trends by topic ID or name.
        
        Args:
            topic: Topic identifier. Can be:
                - int: Topic ID (e.g., 18 for Technology)
                - str: Topic name (e.g., 'Technology')
                - list of int/str: Multiple topics (matches any)
        
        Returns:
            TrendList: New TrendList containing only trends matching the specified topic(s)
        """
        topics = [topic] if not isinstance(topic, list) else topic
        
        name_to_id = {name.lower(): id_ for id_, name in TREND_TOPICS.items()}
        
        topic_ids = set()
        for t in topics:
            if isinstance(t, int):
                topic_ids.add(t)
            elif isinstance(t, str):
                topic_id = name_to_id.get(t.lower())
                if topic_id:
                    topic_ids.add(topic_id)
                    
        filtered = [
            trend for trend in self 
            if any(topic_id in trend.topics for topic_id in topic_ids)
        ]
        
        return TrendList(filtered)
    
    def get_topics_summary(self) -> dict:
        """
        Get a summary of topics present in the trends.
        
        Returns:
            dict: Mapping of topic names to count of trends
        """
        topic_counts = {}
        for trend in self:
            for topic_id in trend.topics:
                topic_name = TREND_TOPICS.get(topic_id, f"Unknown ({topic_id})")
                topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
        return dict(sorted(topic_counts.items(), key=lambda x: (-x[1], x[0])))
    
    def __str__(self) -> str:
        """Return string representation of the trends."""
        if not self:
            return "[]"
        return "[\n " + ",\n ".join(trend.brief_summary() for trend in self) + "\n]"

# --- END TREND LIST ---


# --- UTILS ---
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class LRUCache(OrderedDict):
    def __init__(self, maxsize=128):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

def ensure_list(item):
	return list(item) if hasattr(item, '__iter__') and not isinstance(item, str) and not isinstance(item, dict) else [item]

def extract_column(data, column, default: Any = None, f=None):
	if f is None:
		return [item.get(column, default) for item in data]
	return [f(item.get(column, default)) for item in data]

def flatten_data(data, columns):
    return [{**{kk: vv for k in columns if k in d for kk, vv in d[k].items()},
             **{k: v for k, v in d.items() if k not in columns}} 
            for d in data]

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def filter_data(data, desired_columns):
	desired_columns = set(desired_columns)
	return [{k: v for k, v in item.items() if k in desired_columns} for item in data]

def decode_escape_text(text):
	for k,v in _HEX_TO_CHAR_DICT.items():
		text = text.replace(k, v)
		
	if r'\x' in text:
		text = re.sub(r'\\x[0-9a-fA-F]{2}', lambda match:chr(int(match.group(0)[2:], 16)), text)
	return text

def parse_xml_to_dict(text, prefix=''):
	item_dict = {}
	for tag, content in _tag_pattern.findall(text):
		content = parse_xml_to_dict(content.strip(), tag+'_')
		tag = tag.replace(prefix, '')
		if tag in item_dict:
			if not isinstance(item_dict[tag], list):
				item_dict[tag] = [item_dict[tag]]
			item_dict[tag].append(content)
			continue
		item_dict[tag] = content
	if not item_dict:
		return text
	return item_dict

def get_utc_offset_minutes():
    """
    Returns the local time offset from UTC in minutes.
    Positive values for time zones ahead of UTC (eastward),
    negative values for time zones behind UTC (westward).
    """
    # Get current local time
    now = datetime.now()
    
    # Get offset in seconds
    utc_offset = -time.timezone
    
    # Account for daylight saving time if active
    if time.localtime().tm_isdst:
        utc_offset += 3600  # Add one hour in seconds
    
    # Convert seconds to minutes
    return utc_offset // 60

def parse_time_ago(time_ago):
	if not time_ago:
		return None
	
	match = re.match(r'(\d+)\s*(\w+)', time_ago)
	if not match:
		return None
	
	value, unit = match.groups()
	value = int(value)

	if 'h' in unit:
		delta = timedelta(hours=value)
	elif 'd' in unit:
		delta = timedelta(days=value)
	elif 'm' in unit:
		delta = timedelta(minutes=value)
	else:
		delta = timedelta(0)

	now = datetime.now(timezone.utc)
	timestamp = int((now - delta).replace(microsecond=0).timestamp())
	return timestamp

def truncate_string(s, max_length):
    if len(s) > max_length:
        return s[:max_length - 3] + '...'
    return s

# --- END UTILS ---


# --- NEWS ARTICLE ---
class NewsArticle:
    """
    Represents a news article related to a trending topic.

    This class handles both dictionary and list-based article data from
    various Google Trends API endpoints.

    Parameters:
        title (str): Article title
        url (str): Article URL
        source (str): News source name
        picture (str): URL to article image
        time (str or int): Publication time or timestamp
        snippet (str): Article preview text

    Note:
        If time is provided as a string with 'ago' format (e.g., '2 hours ago'),
        it will be automatically converted to a timestamp.
    """
    def __init__(self, title=None, url=None, source=None, picture=None, time=None, snippet=None, article_ids=None):
        self.title = title
        self.url = url
        self.source = source
        self.picture = picture
        self.time = time
        if isinstance(self.time, str) and ('ago' in self.time):
            self.time = parse_time_ago(self.time)
        self.snippet = snippet

    @classmethod
    def from_api(cls, data):
        if isinstance(data, dict):
            return cls(
                title=data.get('title') or data.get('articleTitle'),
                url=data.get('url'),
                source=data.get('source'),
                picture=data.get('picture') or data.get('image', {}).get('imageUrl'),
                time=data.get('time') or data.get('timeAgo'),
                snippet=data.get('snippet')
            )
        elif isinstance(data, list):
            return cls(
                title=data[0],
                url=data[1],
                source=data[2],
                time=data[3][0] if data[3] else None,
                picture=data[4] if len(data) > 4 else None
            )
        else:
            raise ValueError("Unsupported data format: must be dict or list")

    def __repr__(self):
        return f"NewsArticle(title={self.title!r}, url={self.url!r}, source={self.source!r}, " \
               f"picture={self.picture!r}, time={self.time!r}, snippet={self.snippet!r})"

    def __str__(self):
        s =    'Title   : {}'.format(self.title)
        s += '\nURL     : {}'.format(self.url) if self.url else ''
        s += '\nSource  : {}'.format(self.source) if self.source else ''
        s += '\nPicture : {}'.format(self.picture) if self.picture else ''
        s += '\nTime    : {}'.format(datetime.fromtimestamp(self.time).strftime('%Y-%m-%d %H:%M:%S')) if self.time else ''
        s += '\nSnippet : {}'.format(self.snippet) if self.snippet else ''
        return s
    
# --- END NEWS ARTICLE ---


# --- HIERARCHICAL SEARCH ---

def flatten_tree(node, parent_id='', result=None, join_ids=True):
    """
    Recursively transforms a tree structure into a flat list.
    
    Args:
        node (dict): Tree node with 'name', 'id' and optional 'children' keys
        parent_id (str): Parent node ID
        result (list): Accumulated result
        join_ids (bool): Whether to join IDs with parent (True for geo, False for categories)
        
    Returns:
        list: List of dictionaries with name and id
    """
    if result is None:
        result = []
    
    current_id = node['id']
    # Join IDs only for geographical data
    if join_ids and parent_id:
        full_id = f"{parent_id}-{current_id}"
    else:
        full_id = current_id
    
    result.append({
        'name': node['name'],
        'id': full_id
    })
    
    if 'children' in node:
        for child in node['children']:
            flatten_tree(child, full_id if join_ids else '', result, join_ids)
    
    return result

class HierarchicalIndex:
    """
    An index for efficient searches in hierarchical Google Trends data structures.

    This class provides fast lookups for hierarchical data like locations and categories,
    supporting both exact and partial matching of names.

    Examples:
        - Geographical hierarchies (Country -> Region -> City)
        - Category hierarchies (Main category -> Subcategory)

    Methods:
        add_item(item): Add an item to the index
        exact_search(name): Find exact match for name
        partial_search(query): Find items containing the query
        id_search(id_query): Find by ID (supports both exact and partial matching)
    """
    
    def __init__(self, items: List[dict], partial_id_search: bool = True):
        """
        Initialize the search index.
        
        Args:
            items (List[dict]): List of dictionaries with 'name' and 'id'
            partial_id_search (bool): Whether to allow partial ID matches 
                (True for geo locations, False for categories)
        """
        # Main storage: dict with lowercase name as key
        self.name_to_item: Dict[str, dict] = {}
        
        # Inverted index for partial matching
        self.word_index: Dict[str, List[str]] = {}
        
        # Store search mode
        self.partial_id_search = partial_id_search
        
        # Build indexes
        for item in items:
            self.add_item(item)
    
    def add_item(self, item: dict) -> None:
        """
        Add a single item to the index.
        
        Args:
            item (dict): Dictionary with 'name' and 'id'
        """
        name = item['name'].lower()
        
        # Add to main storage
        self.name_to_item[name] = item
        
        # Split name into words and add to inverted index
        words = set(re.split(r'\W+', name))
        for word in words:
            if word:
                if word not in self.word_index:
                    self.word_index[word] = []
                self.word_index[word].append(name)
    
    def exact_search(self, name: str) -> Optional[dict]:
        """
        Perform exact name search (case-insensitive).
        
        Args:
            name (str): Name to search for
            
        Returns:
            Optional[dict]: Item dictionary if found, None otherwise
        """
        return self.name_to_item.get(name.lower())
    
    def partial_search(self, query: str) -> List[dict]:
        """
        Perform partial name search (case-insensitive).
        
        Args:
            query (str): Search query string
            
        Returns:
            List[dict]: List of matching item dictionaries
        """
        query = query.lower()
        results = set()
        
        # Search for partial matches in word index
        for word, items in self.word_index.items():
            if query in word:
                results.update(items)
        
        # Also check if query matches any part of full names
        for name in self.name_to_item:
            if query in name:
                results.add(name)
        
        # Return found items
        return [self.name_to_item[name] for name in results]
    
    def id_search(self, id_query: str) -> List[dict]:
        """
        Search by ID.
        
        Args:
            id_query (str): ID or partial ID to search for
            
        Returns:
            List[dict]: List of matching item dictionaries
        """
        if self.partial_id_search:
            # For geo data - allow partial matches
            return [item for item in self.name_to_item.values() 
                   if id_query in item['id']]
        else:
            # For categories - only exact matches
            return [item for item in self.name_to_item.values() 
                   if item['id'] == id_query]

def create_hierarchical_index(tree_data: dict, join_ids: bool = True) -> HierarchicalIndex:
    """
    Create a complete search system from a hierarchical tree structure.
    
    Args:
        tree_data (dict): Original tree structure
        join_ids (bool): Whether to join IDs with parent 
            (True for geo locations, False for categories)
        
    Returns:
        HierarchicalIndex: Initialized search system
    """
    # First flatten the tree
    flat_items = flatten_tree(tree_data, join_ids=join_ids)
    # Then create and return the search index
    return HierarchicalIndex(flat_items, partial_id_search=join_ids)

# --- END HIERARCHICAL SEARCH ---


# --- CONVERTER ---
class TrendsDataConverter:
	"""
	Converts raw Google Trends API responses to pandas DataFrames.

	This class provides static methods for converting various types of
	Google Trends data into more usable formats.

	Methods:
		interest_over_time: Converts timeline data
		related_queries: Converts related queries data
		geo_data: Converts geographic data
		suggestions: Converts search suggestions
		rss_items: Parses RSS feed items
	"""
	@staticmethod
	def token_to_bullets(token_data):
		items = token_data.get('request', {}).get('comparisonItem', [])
		bullets = [item.get('complexKeywordsRestriction', {}).get('keyword', [''])[0].get('value','') for item in items]
		metadata = [next(iter(item.get('geo', {'':'unk'}).values()), 'unk') for item in items]
		if len(set(metadata))>1:
			bullets = [b+' | '+m for b,m in zip(bullets, metadata)]
		metadata = [item.get('time', '').replace('\\', '') for item in items]
		if len(set(metadata))>1:
			bullets = [b+' | '+m for b,m in zip(bullets, metadata)]

		return bullets

	@staticmethod
	def interest_over_time(widget_data, keywords, time_as_index=True):
		"""
		Converts interest over time data to a pandas DataFrame.

		Parameters:
			widget_data (dict): Raw API response data
			keywords (list): List of keywords for column names
			time_as_index (bool): Use time as DataFrame index

		Returns:
			pandas.DataFrame: Processed interest over time data
		"""
		timeline_data = widget_data
		timeline_data = timeline_data.get('default', timeline_data)
		timeline_data = timeline_data.get('timelineData', timeline_data)
		if not timeline_data:
			return pd.DataFrame(columns=keywords)


		df_data = np.array(extract_column(timeline_data, 'value')).reshape(len(timeline_data), -1)
		df_data = dict(zip(keywords, df_data.T))
		if ('isPartial' in timeline_data[-1]) or any('isPartial' in row for row in timeline_data):
			df_data['isPartial'] = extract_column(timeline_data, 'isPartial', False)


		timestamps = extract_column(timeline_data, 'time', f=lambda x:int(x) if x else None)
		timestamps = np.array(timestamps, dtype='datetime64[s]').astype('datetime64[ns]')
		# timestamps += np.timedelta64(get_utc_offset_minutes(), 'm')
		if time_as_index:
			return pd.DataFrame(df_data, index=pd.DatetimeIndex(timestamps, name='time [UTC]'))
		return pd.DataFrame({'time':timestamps, **df_data})

	@staticmethod
	def multirange_interest_over_time(data, bullets=None):
		data = data.get('default', {}).get('timelineData', [{}])
		if not 'columnData' in data[0]:
			return pd.DataFrame()

		num_parts = len(data[0]['columnData'])
		if bullets is None:
			bullets = ['keyword_'+str(i) for i in range(num_parts)]

		df_data = {}
		for i in range(num_parts):
			timeline_data = [item['columnData'][i] for item in data]
			df_data[bullets[i]] = extract_column(timeline_data, 'value', f=lambda x:x if x!=-1 else None)

			if ('isPartial' in timeline_data[-1]) or any('isPartial' in row for row in timeline_data):
				df_data['isPartial_'+str(i)] = extract_column(timeline_data, 'isPartial', False)

			timestamps = extract_column(timeline_data, 'time', f=lambda ts:int(ts) if ts else None)
			timestamps = np.array(timestamps, dtype='datetime64[s]').astype('datetime64[ns]')
			df_data['index_'+str(i)] = timestamps
		return pd.DataFrame(df_data)

	@staticmethod
	def related_queries(widget_data):
		ranked_data 	 = widget_data.get('default',{}).get('rankedList')
		if not ranked_data:
			return {'top':pd.DataFrame(), 'rising':pd.DataFrame()}	
		
		result           = {}
		result['top']    = pd.DataFrame(flatten_data(filter_data(ranked_data[0]['rankedKeyword'], _RELATED_QUERIES_DESIRED_COLUMNS), ['topic']))
		result['rising'] = pd.DataFrame(flatten_data(filter_data(ranked_data[1]['rankedKeyword'], _RELATED_QUERIES_DESIRED_COLUMNS), ['topic']))
		return result

	@staticmethod
	def geo_data(widget_data, bullets=None):
		data = widget_data.get('default', {}).get('geoMapData', [])
		filtered_data = list(filter(lambda item:item['hasData'][0], data))
		if not filtered_data:
			return pd.DataFrame()
		
		num_keywords = len(filtered_data[0]['value'])
		if not bullets:
			bullets = ['keyword_'+str(i) for i in range(num_keywords)]

		found_cols = set(filtered_data[0].keys()) & {'coordinates', 'geoCode', 'geoName', 'value'}
		df_data = {}
		df_data['geoName'] = extract_column(filtered_data, 'geoName')
		if 'geoCode' in found_cols:
			df_data['geoCode'] = extract_column(filtered_data, 'geoCode')
		if 'coordinates' in found_cols:
			df_data['lat'] = extract_column(filtered_data, 'coordinates', f=lambda x:x['lat'])
			df_data['lng'] = extract_column(filtered_data, 'coordinates', f=lambda x:x['lng'])

		values = np.array(extract_column(filtered_data, 'value')).reshape(len(filtered_data), -1)
		for keyword,values_row in zip(bullets, values.T):
			df_data[keyword] = values_row
		return pd.DataFrame(df_data)
	
	@staticmethod
	def suggestions(data):
		return pd.DataFrame(data['default']['topics'])
	
	@staticmethod
	def rss_items(data):
		item_pattern = re.compile(r'<item>(.*?)</item>', re.DOTALL)
		items = list(map(lambda item:parse_xml_to_dict(item, 'ht:'), item_pattern.findall(data)))
		return items
	
	@staticmethod
	def trending_now_showcase_timeline(data, request_timestamp=None):
		lens = [len(item[1]) for item in data]
		min_len, max_len = min(lens), max(lens)
		if min_len in {30,90,180,42}:
			max_len = min_len + 1

		time_offset = 480 if max_len < 32 else 14400 if max_len < 45 else 960

		timestamp = int(request_timestamp or datetime.now(timezone.utc).timestamp())
		timestamps = [timestamp // time_offset * time_offset - time_offset * i for i in range(max_len+2)][::-1]
		timestamps = np.array(timestamps, dtype='datetime64[s]').astype('datetime64[ns]')
		if (timestamp%time_offset) <= 60: # Time delay determined empirically
			df_data = {item[0]:item[1][-min_len:] for item in data}
			df = pd.DataFrame(df_data, index=timestamps[:-1][-min_len:])
			return df
		
		res = {}
		for item in data:
			res[item[0]] = np.pad(item[1], (0, max_len - len(item[1])), mode='constant', constant_values=0)
		df = pd.DataFrame(res, index=timestamps[-max_len:])
		return df

# --- END CONVERTER ---

# --- CLIENT ---
class TrendsQuotaExceededError(Exception):
    """Raised when the Google Trends API quota is exceeded for related queries/topics."""
    def __init__(self):
        super().__init__(
            "API quota exceeded for related queries/topics. "
            "To resolve this, you can try:\n"
            "1. Use a different referer in request headers:\n"
            "   tr.related_queries(keyword, headers={'referer': 'https://www.google.com/'})\n"
            "2. Use a different IP address by configuring a proxy:\n"
            "   tr.set_proxy('http://proxy:port')\n"
            "   # or\n"
            "   tr = Trends(proxy={'http': 'http://proxy:port', 'https': 'https://proxy:port'})\n"
            "3. Wait before making additional requests"
        )

class BatchPeriod(Enum): # update every 2 min
	'''
	Time periods for batch operations.
	'''
	Past4H  = 2 #31 points (new points every 8 min)
	Past24H = 3 #91 points (every 16 min)
	Past48H = 5 #181 points (every 16 min)
	Past7D  = 4 #43 points (every 4 hours) 
	
class Trends:
	"""
	A client for accessing Google Trends data.

	This class provides methods to analyze search trends, get real-time trending topics,
	and track interest over time and regions.

	Parameters:
		hl (str): Language and country code (e.g., 'en-US'). Defaults to 'en-US'.
		tzs (int): Timezone offset in minutes. Defaults to current system timezone.
		use_entity_names (bool): Whether to use entity names instead of keywords. 
			Defaults to False.
		proxy (str or dict): Proxy configuration. Can be a string URL or a dictionary
			with protocol-specific proxies. Examples:
			- "http://user:pass@10.10.1.10:3128"
			- {"http": "http://10.10.1.10:3128", "https": "http://10.10.1.10:1080"}
	"""
		
	def __init__(self, language='en', tzs=360, request_delay=1., max_retries=3, use_enitity_names = False, proxy=None, **kwargs):
		"""
		Initialize the Trends client.
		
		Args:
			language (str): Language code (e.g., 'en', 'es', 'fr').
			tzs (int): Timezone offset in minutes. Defaults to 360.
			request_delay (float): Minimum time interval between requests in seconds. Helps avoid hitting rate limits and behaving like a bot. Set to 0 to disable.
			max_retries (int): Maximum number of retry attempts for failed requests. Each retry includes exponential backoff delay of 2^(max_retries-retries) seconds for rate limit errors (429, 302).
			use_enitity_names (bool): Whether to use entity names instead of keywords.
			proxy (str or dict): Proxy configuration.
			**kwargs: Additional arguments for backwards compatibility.
				- hl (str, deprecated): Old-style language code (e.g., 'en' or 'en-US').
				If provided, will be used as fallback when language is invalid.
		"""
		if isinstance(language, str) and len(language) >= 2:
			self.language = language[:2].lower()
		elif 'hl' in kwargs and isinstance(kwargs['hl'], str) and len(kwargs['hl']) >= 2:
			self.language = kwargs['hl'][:2].lower()
		else:
			self.language = 'en'
	
		# self.hl = hl
		self.tzs = tzs or -int(datetime.now().astimezone().utcoffset().total_seconds()/60)
		self._default_params = {'hl': self.language, 'tz': tzs}
		self.use_enitity_names = use_enitity_names
		self.session = requests.session()
		self._headers = {'accept-language': self.language}
		self._geo_cache = {}
		self._category_cache = {}  # Add category cache
		self.request_delay = request_delay
		self.max_retires = max_retries
		self.last_request_times = {0,1}
		# Initialize proxy configuration
		self.set_proxy(proxy)
	
	def set_proxy(self, proxy=None):
		"""
		Set or update proxy configuration for the session.

		Args:
			proxy (str or dict, optional): Proxy configuration. Can be:
				- None: Remove proxy configuration
				- str: URL for all protocols (e.g., "http://10.10.1.10:3128")
				- dict: Protocol-specific proxies (e.g., {"http": "...", "https": "..."})
		"""
		if isinstance(proxy, str):
			# Convert string URL to dictionary format
			proxy = {
				'http': proxy,
				'https': proxy
			}
		
		# Update session's proxy configuration
		self.session.proxies.clear()
		if proxy:
			self.session.proxies.update(proxy)

	def _extract_keywords_from_token(self, token):
		if self.use_enitity_names:
			return [item['text'] for item in token['bullets']]
		else :
			return [item['complexKeywordsRestriction']['keyword'][0]['value'] for item in token['request']['comparisonItem']]

	@staticmethod
	def _parse_protected_json(response: requests.models.Response):
		"""
		Parses JSON data from a protected API response.

		Args:
			response (requests.models.Response): Response object from requests

		Returns:
			dict: Parsed JSON data

		Raises:
			ValueError: If response status is not 200, content type is invalid,
					or JSON parsing fails
		"""
		valid_content_types = {'application/json', 'application/javascript', 'text/javascript'}
		content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
		
		if (response.status_code != 200) or (content_type not in valid_content_types):
			raise ValueError(f"Invalid response: status {response.status_code}, content type '{content_type}'")

		try:
			json_data = response.text.split('\n')[-1]
			return json.loads(json_data)
		except json.JSONDecodeError:
			raise ValueError("Failed to parse JSON data")

	def _encode_items(self, keywords, timeframe="today 12-m", geo=''):
		data = list(map(ensure_list, [keywords, timeframe, geo]))
		lengths = list(map(len, data))
		max_len = max(lengths)
		if not all(max_len % length == 0 for length in lengths):
			raise ValueError(f"Ambiguous input sizes: unable to determine how to combine inputs of lengths {lengths}")
		data = [item * (max_len // len(item)) for item in data]
		items = [dict(zip(['keyword', 'time', 'geo'], values)) for values in zip(*data)]
		return items

	def _encode_request(self, params):
		if 'keyword' in params:
			keywords = ensure_list(params.pop('keyword'))
			if len(keywords) != 1:
				raise ValueError("This endpoint only supports a single keyword")
			params['keywords'] = keywords

		items = self._encode_items(
			keywords  = params['keywords'],
			timeframe = params.get('timeframe', "today 12-m"),
			geo		  = params.get('geo', '')
		)
		
		req = {'req': json.dumps({
			'comparisonItem': items,
			'category': params.get('cat', 0),
			'property': params.get('gprop', '')
		})}

		req.update(self._default_params)
		return req

	def _get(self, url, params=None, headers=None):
		"""
		Make HTTP GET request with retry logic and proxy support.
		
		Args:
			url (str): URL to request
			params (dict, optional): Query parameters
			
		Returns:
			requests.Response: Response object
			
		Raises:
			ValueError: If response status code is not 200
			requests.exceptions.RequestException: For network-related errors
		"""
		retries = self.max_retires
		response_code = 429
		response_codes = []
		last_response = None
		req = None
		while (retries > 0):
			try:

				if self.request_delay:
					min_time = min(self.last_request_times)
					sleep_time = max(0, self.request_delay - (time() - min_time))
					sleep(sleep_time)
					self.last_request_times = (self.last_request_times - {min_time,}) | {time(),}

				req = self.session.get(url, params=params, headers=headers)
				last_response = req
				response_code = req.status_code
				response_codes.append(response_code)

				if response_code == 200:
					return req
				else:
					if response_code in {429,302}:
						sleep(2**(self.max_retires-retries))
					retries -= 1
				
			except Exception as e:
				if retries == 0:
					raise
				retries -= 1

		if response_codes.count(429) > len(response_codes) / 2:
			current_delay = self.request_delay or 1
			print(f"\nWarning: Too many rate limit errors (429). Consider increasing request_delay "
				f"to Trends(request_delay={current_delay*2}) before Google implements a long-term "
				f"rate limit!")
		last_response.raise_for_status()

	@classmethod
	def _extract_embedded_data(cls, text):
		pattern = re.compile(r"JSON\.parse\('([^']+)'\)")
		matches = pattern.findall(text)
		# If matches found, decode and return result
		if matches:
			return json.loads(decode_escape_text(matches[0]))  # Take first match
		print("Failed to extract JSON data")

	def _token_to_data(self, token):
		URL = {
			'fe_line_chart': 		API_TIMELINE_URL,
			'fe_multi_range_chart':	API_MULTIRANGE_URL,
			'fe_multi_heat_map':    API_GEO_URL,
			'fe_geo_chart_explore': API_GEO_URL,
			'fe_related_searches':	API_RELATED_QUERIES_URL
		}[token['type']]

		params = {'req': json.dumps(token['request']), 'token': token['token']}
		params.update(self._default_params)
		# req    = self.session.get(URL, params=params)
		req    = self._get(URL, params=params)
		data   = Trends._parse_protected_json(req)
		return data

	def _get_token_data(self, url, params=None, request_fix=None, headers=None, raise_quota_error=False):
		"""
		Internal method to get token data from Google Trends API.
		
		Handles both 'keyword' and 'keywords' parameters for backward compatibility
		and convenience.
		"""

		params 	= self._encode_request(params)
		req 	= self._get(url, params=params, headers=headers)
		token 	= self._extract_embedded_data(req.text)

		if request_fix is not None:
			token = {**token, 'request':{**token['request'], **request_fix}}

		if raise_quota_error:
			user_type = token.get('request', {}).get('userConfig', {}).get('userType', '')
			if user_type == "USER_TYPE_EMBED_OVER_QUOTA":
				raise TrendsQuotaExceededError()

		data 	= self._token_to_data(token)
		return token, data

	def _get_batch(self, req_id, data):
		req_data = json.dumps([[[req_id,f"{json.dumps(data)}", None,"generic"]]])
		post_data  = f'f.req={req_data}'
		headers = {
			"content-type": "application/x-www-form-urlencoded;charset=UTF-8"
		}
		req = self.session.post(BATCH_URL, post_data, headers=headers)
		return req

	def interest_over_time(self, keywords, timeframe="today 12-m", geo='', cat=0, gprop='', return_raw = False, headers=None):
		"""
		Retrieves interest over time data for specified keywords.
		
		Parameters:
			keywords (str or list): Keywords to analyze.
			timeframe : str or list
				Defines the time range for querying interest over time. It can be specified as a single string or a list. 
				Supported formats include:

				- 'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d'
				- 'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y'
				- 'all' for all available data
				- 'YYYY-MM-DD YYYY-MM-DD' for specific date ranges
				- 'YYYY-MM-DDTHH YYYY-MM-DDTHH' for hourly data (if less than 8 days)

				Additional flexible formats:
				
				1. **'now {offset}'**: Timeframes less than 8 days (e.g., 'now 72-H' for the last 72 hours).
				2. **'today {offset}'**: Larger periods starting from today (e.g., 'today 5-m' for the last 5 months).
				3. **'date {offset}'**: Specific date with offset (e.g., '2024-03-25 5-m' for 5 months back from March 25, 2024).

				**Note:** Offsets always go backward in time.

				Resolutions based on timeframe length:
				
				- `< 5 hours`: 1 minute
				- `5 hours <= delta < 36 hours`: 8 minutes
				- `36 hours <= delta < 72 hours`: 16 minutes
				- `72 hours <= delta < 8 days`: 1 hour
				- `8 days <= delta < 270 days`: 1 day
				- `270 days <= delta < 1900 days`: 1 week
				- `>= 1900 days`: 1 month

				Restrictions:
				- **Same resolution**: All timeframes must have the same resolution.
				- **Timeframe length**: Maximum timeframe cannot be more than twice the length of the minimum timeframe.
			geo (str): Geographic location code (e.g., 'US' for United States).
			cat (int): Category ID. Defaults to 0 (all categories).
			gprop (str): Google property filter.
			return_raw (bool): If True, returns raw API response.

		Returns:
			pandas.DataFrame or raw API response
			Processed trending keywords data or raw API data if `return_raw=True`
		"""
		check_timeframe_resolution(timeframe)
		timeframe = list(map(convert_timeframe, ensure_list(timeframe)))

		token, data = self._get_token_data(EMBED_TIMESERIES_URL, locals(), headers=headers)
		if return_raw:
			return token, data

		if token['type']=='fe_line_chart':
			keywords = self._extract_keywords_from_token(token)
			return TrendsDataConverter.interest_over_time(data, keywords=keywords)
		if token['type']=='fe_multi_range_chart':
			bullets = TrendsDataConverter.token_to_bullets(token)
			return TrendsDataConverter.multirange_interest_over_time(data, bullets=bullets)
		return data
	
	def related_queries(self, keyword, timeframe="today 12-m", geo='', cat=0, gprop='', return_raw = False, headers=None):
		"""
        Retrieves related queries for a single search term.
        
        Args:
            keyword (str): A single keyword to analyze
            timeframe (str): Time range for analysis
            geo (str): Geographic location code
            cat (int): Category ID
            gprop (str): Google property filter
            return_raw (bool): If True, returns raw API response
            headers (dict, optional): Custom request headers. Can be used to set different referer
                                    to help bypass quota limits
        
        Raises:
            TrendsQuotaExceededError: When API quota is exceeded
			
		Parameters:
			dict: Two DataFrames containing 'top' and 'rising' related queries
			
		Example:
			>>> tr = Trends()
			>>> related = tr.related_queries('python')
			>>> print("Top queries:")
			>>> print(related['top'])
			>>> print("\nRising queries:")
			>>> print(related['rising'])
		"""
		headers = headers or {"referer": "https://trends.google.com/trends/explore"}
		token, data = self._get_token_data(EMBED_QUERIES_URL, locals(), headers=headers, raise_quota_error=True)
		if return_raw:
			return token, data
		return TrendsDataConverter.related_queries(data)
	
	def related_topics(self, keyword, timeframe="today 12-m", geo='', cat=0, gprop='', return_raw = False, headers=None):
		"""
		Retrieves related topics for a single search term.
		
		Parameters:
            keyword (str): A single keyword to analyze
            timeframe (str): Time range for analysis
            geo (str): Geographic location code
            cat (int): Category ID
            gprop (str): Google property filter
            return_raw (bool): If True, returns raw API response
            headers (dict, optional): Custom request headers. Can be used to set different referer
                                    to help bypass quota limits
        
        Raises:
            TrendsQuotaExceededError: When API quota is exceeded
			
		Example:
			>>> tr = Trends()
			>>> related = tr.related_topics('python')
			>>> print("Top topics:")
			>>> print(related['top'])
			>>> print("\nRising topics:")
			>>> print(related['rising'])
		"""
		headers = headers or {"referer": "https://trends.google.com/trends/explore"}
		token, data = self._get_token_data(EMBED_TOPICS_URL, locals(), headers=headers, raise_quota_error=True)
		if return_raw:
			return token, data
		return TrendsDataConverter.related_queries(data)


	def interest_by_region(self, keywords, timeframe="today 12-m", geo='', cat=0, gprop='', resolution=None, inc_low_vol=False, return_raw=False):
		"""
		Retrieves geographical interest data based on keywords and other parameters.

		Parameters:
			keywords (str or list): Search keywords to analyze.
			timeframe (str): Time range for analysis (e.g., "today 12-m", "2022-01-01 2022-12-31")
			geo (str): Geographic region code (e.g., "US" for United States)
			cat (int): Category ID (default: 0 for all categories)
			gprop (str): Google property filter
			resolution (str): Geographic resolution level:
				- 'COUNTRY' (default when geo is empty)
				- 'REGION' (states/provinces)
				- 'CITY' (cities)
				- 'DMA' (Designated Market Areas)
			inc_low_vol (bool): Include regions with low search volume
			return_raw (bool): Return unprocessed API response data

		Returns:
			pandas.DataFrame or dict: Processed geographic interest data, or raw API response if return_raw=True
		"""
		if (not resolution):
			resolution = 'COUNTRY' if ((geo=='') or (not geo)) else 'REGION'

		data_injection = {'resolution': resolution, 'includeLowSearchVolumeGeos': inc_low_vol}
		token, data = self._get_token_data(EMBED_GEO_URL, locals(), request_fix=data_injection)
		if return_raw:
			return token, data
		
		bullets = TrendsDataConverter.token_to_bullets(token)
		return TrendsDataConverter.geo_data(data, bullets)
	
	def suggestions(self, keyword, language=None, return_raw=False):
		params = {'hz':language, 'tz':self.tzs} if language else self._default_params
		encoded_keyword = keyword.replace("'", "")
		encoded_keyword = quote(encoded_keyword, safe='-')
		req  = self._get(API_AUTOCOMPLETE+encoded_keyword, params)
		data = self._parse_protected_json(req)
		if return_raw:
			return data
		return TrendsDataConverter.suggestions(data)

	def hot_trends(self):
		req = self.session.get(HOT_TRENDS_URL)
		return json.loads(req.text)

	def top_year_charts(self, year='2023', geo='GLOBAL'):
		"""
		https://trends.google.com/trends/yis/2023/GLOBAL/
		"""
		params = {'date':year, 'geo':geo, 'isMobile':False}
		params.update(self._default_params)
		req = self._get(API_TOPCHARTS_URL, params)
		data = self._parse_protected_json(req)
		return data

	def trending_stories(self, geo='US', category='all', max_stories=200, return_raw=False):
		'''
		Old API
		category: all: "all",  business: "b",  entertainment: "e",  health: "m",  sicTech: "t",  sports: "s",  top: "h"
		'''
		forms = {'ns': 15, 'geo': geo, 'tz': self.tzs, 'hl': 'en', 'cat': category, 'fi' : '0', 'fs' : '0', 'ri' : max_stories, 'rs' : max_stories, 'sort' : 0}
		url = 'https://trends.google.com/trends/api/realtimetrends'
		req = self._get(url, forms)
		data = self._parse_protected_json(req)
		if return_raw:
			return data
		
		data = data.get('storySummaries', {}).get('trendingStories', [])
		data = [TrendKeywordLite.from_api(item) for item in data]
		return data

	def daily_trends_deprecated(self, geo='US', return_raw=False):
		params = {'ns': 15, 'geo': geo, 'hl':'en'}
		params.update(self._default_params)
		req = self._get(DAILY_SEARCHES_URL, params)
		data = self._parse_protected_json(req)
		if return_raw:
			return data
		data = data.get('default', {}).get('trendingSearchesDays', [])
		data = [TrendKeywordLite.from_api(item) for day in data for item in day['trendingSearches']]
		return data

	def daily_trends_deprecated_by_rss(self, geo='US', safe=True, return_raw=False):
		'''
		Only last 20 daily news
		'''

		params = {'geo':geo, 'safe':safe}
		req  = self._get(DAILY_RSS, params)
		if return_raw:
			return req.text
		data = TrendsDataConverter.rss_items(req.text)
		data = list(map(TrendKeywordLite.from_api, data))
		return data
	
	def trending_now(self, geo='US', language='en', hours=24, num_news=0, return_raw=False):
		"""
		Retrieves trending keywords that have seen significant growth in popularity within the last specified number of hours.

		Parameters:
		-----------
		geo : str, optional
			The geographical region for the trends, default is 'US' (United States).
		language : str, optional
			The language of the trends, default is 'en' (English).
		hours : int, optional
			The time window (in hours) for detecting trending keywords. Minimum value is 1, and the maximum is 191. Default is 24.
		num_news : int, optional
			NOT RECOMMENDED to use as it significantly slows down the function. The feature for fetching news associated with the trends is rarely used on the platform. 
			If you want trending keywords with news, consider using `trending_now_by_rss` instead. Default is 0.
		return_raw : bool, optional
			If set to True, the function returns the raw data directly from the API. Default is False, meaning processed data will be returned.

		Returns:
		--------
		dict or raw API response
			Processed trending keywords data or raw API data if `return_raw=True`.
		"""
		req_data = [None, None, geo, num_news, language, hours, 1]
		req = self._get_batch('i0OFE', req_data)
		data = self._parse_protected_json(req)
		if return_raw:
			return data

		data = json.loads(data[0][2])
		data = TrendList(map(TrendKeyword, data[1]))
		return data

	def trending_now_by_rss(self, geo='US', return_raw=False):
		"""
		Retrieves trending keywords from the RSS feed for a specified geographical region.

		Parameters:
		-----------
		geo : str, optional
			The geographical region for the trends, default is 'US' (United States).
		return_raw : bool, optional
			If set to True, the function returns the raw data directly from the API. Default is False, meaning processed data will be returned.

		Returns:
		--------
		Union[dict, List[TrendKeywordLite]]
			A dictionary with raw RSS feed data if `return_raw=True`, or a list of `TrendKeyword` objects otherwise.
		"""
		params = {'geo':geo}
		req  = self._get(REALTIME_RSS, params)
		if return_raw:
			return req.text
		data = TrendsDataConverter.rss_items(req.text)
		data = list(map(TrendKeywordLite.from_api, data))
		return data
	
	def trending_now_news_by_ids(self, news_ids, max_news=3, return_raw=False):
		req = self._get_batch('w4opAf', [news_ids, max_news])
		data = self._parse_protected_json(req)
		if return_raw:
			return data

		data = json.loads(data[0][2])
		data = list(map(NewsArticle.from_api, data[0]))
		return data
	
	def trending_now_showcase_timeline(self, keywords, geo='US', timeframe=BatchPeriod.Past24H, return_raw=False):
		req_data = [None,None,[[geo, keyword, timeframe.value, 0, 3] for keyword in keywords]]
		request_timestamp = int(datetime.now(timezone.utc).timestamp())
		req  = self._get_batch('jpdkv', req_data)
		data = self._parse_protected_json(req)
		if return_raw:
			return data
		
		data = json.loads(data[0][2])[0]
		data = TrendsDataConverter.trending_now_showcase_timeline(data, request_timestamp)
		return data
	
	def categories(self, find: str = None, language: str = None) -> List[dict]:
		"""
		Search for categories in Google Trends data.
		
		This function retrieves and caches category data from Google Trends API, then performs
		a partial search on the categories. The results are cached by language to minimize API calls.
		
		Args:
			find (str, optional): Search query for categories. If None or empty string,
				returns all available categories. Defaults to None.
			language (str, optional): Language code for the response (e.g., 'en', 'es').
				If None, uses the instance's default language. Defaults to None.
		
		Returns:
			List[dict]: List of matching categories. Each category is a dictionary containing:
				- name (str): Category name in the specified language
				- id (str): Category identifier
		
		Examples:
			>>> trends = Trends()
			>>> # Find all categories containing "computer"
			>>> computer_cats = trends.categories(find="computer")
			>>> # Find all categories in Spanish
			>>> spanish_cats = trends.categories(language="es")
			>>> # Find specific category in German
			>>> tech_cats = trends.categories(find="Technologie", language="de")
		"""
		cur_language = language or self.language
		
		if cur_language not in self._category_cache:
			req = self._get(API_CATEGORY_URL, {'hl': cur_language, 'tz': self.tzs})
			data = self._parse_protected_json(req)
			self._category_cache[cur_language] = create_hierarchical_index(data, join_ids=False)
		
		if not find:
			return list(self._category_cache[cur_language].name_to_item.values())
			
		return self._category_cache[cur_language].partial_search(find)

	def geo(self, find: str = None, language: str = None) -> List[dict]:
		"""
		Search for geographical locations in Google Trends data.
		
		This function retrieves and caches geographical data from Google Trends API, then performs
		a partial search on the locations. The results are cached by language to minimize API calls.
		
		Args:
			find (str, optional): Search query for locations. If None or empty string,
				returns all available locations. Defaults to None.
			language (str, optional): Language code for the response (e.g., 'en', 'es').
				If None, uses the instance's default language. Defaults to None.
		
		Returns:
			List[dict]: List of matching locations. Each location is a dictionary containing:
				- name (str): Location name in the specified language
				- id (str): Location identifier (e.g., 'US-NY' for New York, United States)
		
		Examples:
			>>> trends = GoogleTrends()
			>>> # Find all locations containing "York"
			>>> locations = trends.geo(find="York")
			>>> # Find all locations in Spanish
			>>> spanish_locations = trends.geo(language="es")
			>>> # Find specific location in German
			>>> berlin = trends.geo(find="Berlin", language="de")
		
		Note:
			- Results are cached by language to improve performance
			- API response is parsed and structured for efficient searching
			- Case-insensitive partial matching is used for searches
		"""
		# Use provided language or fall back to instance default
		cur_language = language or self.language
		
		# Check if we need to fetch and cache data for this language
		if cur_language not in self._geo_cache:
			# Fetch geographical data from Google Trends API
			data = self._get(API_GEO_DATA_URL,
							{'hl': cur_language, 'tz': self.tzs})
			data = self._parse_protected_json(data)
			# Create and cache search system for this language
			self._geo_cache[cur_language] = create_hierarchical_index(data)
		
		# Perform partial search (empty string returns all locations)
		if not find:
			return list(self._geo_cache[cur_language].name_to_location.values())
			
		return self._geo_cache[cur_language].partial_search(find)

# --- END CLIENT ---

# --------- MongoDB Initialization ---------------
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
    tr = Trends(request_delay=TIME_DELAY, retries=3)
    
    print(f"[++] Getting interest_over_time for {keyword} in {geo} for {timeframe}")
    interest_over_time = tr.interest_over_time(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    print(f"Waiting for {TIME_DELAY} seconds...")
    sleep(TIME_DELAY)
    print(f"[++] Getting interest_by_region for {keyword} in {geo} for {timeframe}")
    interest_by_region = tr.interest_by_region(keyword, geo=geo, timeframe=timeframe, cat=cat, gprop=gprop)
    print(f"Waiting for {TIME_DELAY} seconds...")
    sleep(TIME_DELAY)
    print(f"[++] Getting related_topics for {keyword} in {geo} for {timeframe}")
    related_topics = tr.related_topics(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    for key in related_topics:
        related_topics[key] = related_topics[key].to_dict('records')
    print(f"Waiting for {TIME_DELAY} seconds...")
    sleep(TIME_DELAY)
    print(f"[++] Getting related_queries for {keyword} in {geo} for {timeframe}")
    related_queries = tr.related_queries(keyword, geo=geo, headers=headers, timeframe=timeframe, cat=cat, gprop=gprop)
    for key in related_queries:
        related_queries[key] = related_queries[key].to_dict('records')
    print(f"Waiting for {TIME_DELAY} seconds...")
    sleep(TIME_DELAY)
    
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