import argparse
from datetime import datetime
from src.py.util.corporate_actions import corporate_actions_manager
# from src.py.util.api_client import PolygonAPIClient
import logging

def main():
    parser = argparse.ArgumentParser(description='Backfill Corporate Actions to S3')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--bucket', default='quant-trader-data-gintoki', help='S3 bucket name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('CA Backfill')

    # Initialize API Client
    # api = PolygonAPIClient()

    try:
        logger.info(f"Backfilling corporate actions from {args.start} to {args.end}")
        
        # Fetch and upload splits/dividends
        corporate_actions_manager.fetch_market_actions(args.start, args.end)
        success = corporate_actions_manager.upload_market_actions(args.bucket, args.start, args.end)
        
        if success:
            logger.info("Corporate actions backfill completed successfully")
        else:
            logger.error("Corporate actions backfill failed")

    except Exception as e:
        logger.error(f"Corporate action backfill error: {str(e)}")

if __name__ == '__main__':
    main()
