import { HistoricalFetcher } from '../../src/py/data_ingestion/historical_data_fetcher.py';
import { S3Client } from '@aws-sdk/client-s3';
import { python } from 'pythonia';

const { HistoricalFetcher } = await python('data_ingestion.historical_data_fetcher')

const TEST_RANGE = { 
  start: '2025-02-20', 
  end: '2025-02-21' 
};

export async function testHistoricalIngest() {
  const s3 = new S3Client({ region: 'us-west-2' });
  const fetcher = new HistoricalFetcher();
  
  await fetcher.fetch_and_store('AAPL', TEST_RANGE.start, TEST_RANGE.end);
  
  // Verify Parquet files
  const paths = [
    `historical/AAPL/aggregates/minute/${TEST_RANGE.start}_to_${TEST_RANGE.end}.parquet`,
    `historical/AAPL/trades/${TEST_RANGE.start}.parquet`
  ];
  
  for (const path of paths) {
    const exists = await s3.headObject({
      Bucket: process.env.AWS_S3_BUCKET,
      Key: path
    }).then(() => true).catch(() => false);
    
    assert.ok(exists, `Missing S3 file: ${path}`);
  }
}
