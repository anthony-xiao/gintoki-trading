import { websocketClient } from '@polygon.io/client-js';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { TickProcessor } from './tick-processor.mjs';
import dotenv from 'dotenv';

dotenv.config();

const s3 = new S3Client({ region: 'us-west-2' });
const BATCH_SIZE = 20;
let tickBuffer = [];

export class PolygonRealtime {
  constructor() {
    this.symbolCache = new Map();
    this.ws = websocketClient(process.env.POLYGON_API_KEY).stocks();
    this.tickProcessor = new TickProcessor();

    this.ws.onopen = () => {
      console.log('✅ Connected to Polygon');
      this.authenticate();
      setTimeout(() => this.subscribe(['AAPL', 'MSFT', 'TSLA']), 500);
    };

  this.ws.onmessage = async (msg) => {
    try {
      console.log(JSON.parse(msg.data).flatMap(m => m.ev))
      const rawData = JSON.parse(msg.data).filter(m => m.ev === 'A'); // Only process aggregates
    
      const processor = new TickProcessor();
      const processedTicks = await processor.process(rawData);
      processedTicks.forEach(tick => {
        console.log('tick', tick)
        tickBuffer.push(tick);
        console.log(tickBuffer.length)
        if(tickBuffer.length >= BATCH_SIZE) {
          this.flushToS3();
        }
    });

  } catch (error) {
    console.error('Error processing message:', error);
  }
  this.startHealthMonitor();
}};

  authenticate() {
    this.ws.send(JSON.stringify({
      action: 'auth',
      params: process.env.POLYGON_API_KEY
    }));
  }

  subscribe(symbols) {
    const normalizedSymbols = symbols.map(s => s.toUpperCase());
    const subscriptions = normalizedSymbols.flatMap(sym => 
      ['T', 'Q', 'A'].map(type => `${type}.${sym}`)
    );
    console.log(`📡 Subscribing to: ${subscriptions.join(', ')}`);
    this.ws.send(JSON.stringify({
      action: 'subscribe',
      params: subscriptions.join(',')
    }));
  }

  async flushToS3() {
    const batch = tickBuffer.splice(0, BATCH_SIZE);
    const date = new Date().toISOString().split('T')[0];
    
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET,
      Key: `realtime/${symbol}/${Date.now()}.json`,
      Body: JSON.stringify(batch)
    }));
    
    console.log(`📦 Saved ${batch.length} ticks to S3`);
  }

  startHealthMonitor() {
    setInterval(() => {
      this.symbolCache.forEach((cache, symbol) => {
        console.log(`📊 ${symbol} buffer: ${cache.priceWindow.length}/200`);
      });
    }, 5000);
  }
}

// Start service
new PolygonRealtime();
