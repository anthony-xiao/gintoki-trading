  #!/bin/bash
# macOS-compatible date calculations
START_DATE="2024-01-01"
END_DATE=$(date -v-2d +%Y-%m-%d)
# END_DATE="2020-12-31"
TICKERS="TSLA NVDA AMD GME MRNA AMAT FTNT INTC PLTR SNPS LMT DHR HCA UPS CPB"
# TICKERS="TSLA"

docker run --rm \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e POLYGON_API_KEY=$POLYGON_API_KEY \
  historical-fetcher:latest \
  python -m src.py.data_ingestion.historical_data_fetcher \
  $TICKERS \
  --start $START_DATE \
  --end $END_DATE \
  --threads 6