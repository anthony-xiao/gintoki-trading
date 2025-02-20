import { PolygonRealtime } from '../../src/js/core/services/data/polygon-realtime.mjs';
import { strict as assert } from 'node:assert';
import { S3Client } from '@aws-sdk/client-s3';

const TEST_SYMBOL = 'AAPL';
const TEST_DURATION = 5000; // 5 seconds

export async function testDataPipeline() {
  const s3 = new S3Client({ region: 'us-west-2' });
  const ingestor = new PolygonRealtime();
  
  try {
    // Start ingestion
    ingestor.subscribe([TEST_SYMBOL]);
    
    // Wait for data flow
    await new Promise(r => setTimeout(r, TEST_DURATION));
    
    // Verify S3 writes
    const objects = await s3.listObjectsV2({
      Bucket: process.env.AWS_S3_BUCKET,
      Prefix: `processed/ticks/${TEST_SYMBOL}`
    });
    
    assert.ok(objects.Contents.length > 0, 
      'No data written to S3');
    console.log('✅ Data pipeline test passed');

  } finally {
    ingestor.ws.close();
  }
}

// Run test if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  await testDataPipeline();
}
