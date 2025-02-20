# Contains shared Polygon.io API utilities
import os
import requests
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from typing import Dict, List, Optional, Tuple
import logging
import time

load_dotenv()
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
BASE_URL = "https://api.polygon.io"

    # Existing fetch_paginated_data implementation

def fetch_paginated_data(url: str, params: Dict = None) -> List[Dict]:
    """Handle Polygon pagination with loop prevention"""
    results = []
    next_url = url
    retries = 0
    max_retries = 3
    timeout = 10
    max_pages = 20  # Safety net to prevent infinite loops
    pages_fetched = 0
    seen_cursors = set()
    original_params = params.copy() if params else {}

    logging.info(f"Starting pagination for {next_url}")
    
    while next_url and pages_fetched < max_pages:
        try:
            parsed = urlparse(next_url)
            query = parse_qs(parsed.query)

            # Remove potential conflict parameters
            for key in ["timestamp.gte", "timestamp.lte", "sort", "order"]:
                if key in query:
                    del query[key]
            
            # Merge with original params
            for key, value in original_params.items():
                if isinstance(value, list):
                    query[key] = value
                else:
                    query[key] = [str(value)]
            
            # Extract cursor for loop detection
            cursor = query.get('cursor', [None])[0]
            if cursor in seen_cursors:
                logging.warning(f"Detected duplicate cursor {cursor}, stopping pagination")
                break
            seen_cursors.add(cursor)
            
            # Clean and fetch
            query['apiKey'] = POLYGON_API_KEY
            next_url = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
            
            logging.info(f"Fetching page {pages_fetched + 1}")
            response = requests.get(next_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check for empty results
            records = data.get('results', data.get('ticks', data.get('tickers', [])))
            if not records:
                logging.info("No more records found")
                break
                
            results.extend(records)
            pages_fetched += 1
            logging.info(f"Received {len(records)} records (total: {len(results)})")
            
            # Update next URL
            next_url = data.get('next_url')
            time.sleep(0.3)

        except Exception as e:
            if retries >= max_retries:
                logging.error(f"Aborting after {max_retries} retries: {str(e)}")
                break
            retries += 1
            sleep_time = min(2 ** retries, 10)
            logging.warning(f"Retry {retries}/{max_retries} in {sleep_time}s: {str(e)}")
            time.sleep(sleep_time)
    
    logging.info(f"Completed pagination after {pages_fetched} pages")
    return results

