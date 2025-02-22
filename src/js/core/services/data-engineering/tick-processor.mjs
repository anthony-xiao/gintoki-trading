// src/js/core/services/data/tick-processor.mjs
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { TechnicalIndicators } from './technical-indicators.mjs';
import { python } from 'pythonia';
import dotenv from 'dotenv';

// const py = python({ pythonPath: process.env.PYTHONPATH });
const pd = await python('pandas');
const ca = await python('util.corporate_actions');

dotenv.config();

const s3 = new S3Client({ region: 'us-west-2' });

export class TickProcessor {
  constructor() {
    this.symbolCache = new Map();
    this.batchSize = 100;
    this.currentBatch = [];
    setInterval(() => {
      this.symbolCache.forEach((v,k) => {
        console.log(`🔍 ${k} buffer: ${v.priceWindow.length} items`);
        console.log(`🔍 ${k} buffer volume: ${v.volumeWindow.length} items`);

      });
    }, 5000);
      setInterval(() => this.logCacheState(), 10000);
    if (!TickProcessor.instance) {
      this.symbolCache = new Map();
      TickProcessor.instance = this;
    }
    return TickProcessor.instance;
  }

  async process(rawMessages) {
    try {
      const processed = [];
      
      for (const msg of rawMessages) {
        if (!this.validateMessage(msg)) continue;
        
        const normalized = this.normalizeMessage(msg);
        if (!this.applyQualityChecks(normalized)) continue;
        
        const withFeatures = this.enrichWithFeatures(normalized);
        const adjusted = await this.applyCorporateActions(withFeatures);
  
        // Get symbol from normalized message
        const symbol = normalized.symbol;
        console.log(adjusted)
        if (this.isFeatureReady(symbol)) {
          processed.push(adjusted);
          await this.handleBatch(adjusted);
        } else {
          const cache = this.symbolCache.get(symbol) || { priceWindow: [] };
          console.log(`Buffering data for ${symbol} (${cache.priceWindow.length}/15)`);
        }
      }
      
      return processed;
    } catch (error) {
      console.error('Tick processing error:', error);
      return [];
    }
  }
  

  validateMessage(msg) {
    const type = msg.ev;
    console.log(`📩 Received ${type} message for ${msg.sym}`);
  
    const requiredFields = {
      'A': ['sym', 'o', 'h', 'l', 'c', 'v', 's'],
      'Q': ['sym', 'bp', 'ap', 'bs', 'as', 't'],
      'T': ['sym', 'p', 's', 't', 'c']
    };
    const isValid = requiredFields[type]?.every(field => {
      const valid = msg[field] !== undefined;
      if (!valid) console.warn(`❌ Missing ${field} in ${type} message`);
      return valid;
    });
    
    return isValid;

    // return requiredFields[msg.ev]?.every(field => msg[field] !== undefined);
  }

  normalizeMessage(msg) {
    const base = {
      symbol: msg.sym.toUpperCase(), // Force uppercase
      timestamp: new Date(msg.t || msg.s).getTime(),
      type: msg.ev,
      exchange: msg.x || 'DEFAULT'
    };

    switch(msg.ev) {
      case 'A':
        return {
          ...base,
          open: parseFloat(msg.o),
          high: parseFloat(msg.h),
          low: parseFloat(msg.l),
          close: parseFloat(msg.c),
          volume: parseInt(msg.v),
          vwap: parseFloat(msg.vw || 0),
          startTime: msg.s,
          endTime: msg.e
        };

      case 'Q':
        return {
          ...base,
          bidPrice: parseFloat(msg.bp),
          askPrice: parseFloat(msg.ap),
          bidSize: parseInt(msg.bs),
          askSize: parseInt(msg.as)
        };

      case 'T':
        return {
          ...base,
          price: parseFloat(msg.p),
          size: parseInt(msg.s),
          conditions: msg.c || []
        };

      default:
        return null;
    }
  }

  applyQualityChecks(msg) {
    let checks = [];
    
    switch(msg.type) {
      case 'A':
        checks = [
          msg.high >= msg.low,
          msg.volume > 0,
          msg.close >= 0.01
        ];
        break;
      case 'Q':
        checks = [
          msg.bidPrice <= msg.askPrice,
          msg.bidSize > 0,
          msg.askSize > 0
        ];
        break;
      case 'T':
        checks = [
          msg.price > 0,
          msg.size > 0
        ];
        break;
    }
  
    const passed = checks.every(check => check);
    if (!passed) {
      console.warn(`⚠️ Failed quality checks for ${msg.symbol} ${msg.type}:`, checks);
    }
    
    return passed;
  }

  isFeatureReady(symbol) {
    const cache = this.symbolCache.get(symbol);
    return cache?.priceWindow?.length >= 15;
  }


  enrichWithFeatures(msg) {
     // Only calculate features for aggregates (A)
    if (msg.type !== 'A') {
      return {
        ...msg,
        features: null // No features for non-aggregates
      };
    }
    const symbol = msg.symbol.toUpperCase(); // Normalize symbol casing

    if (!this.symbolCache.has(symbol)) {
      console.log(`📦 Initializing NEW cache for ${symbol}`);
      this.symbolCache.set(symbol, {
        priceWindow: [],
        volumeWindow: [],
        initializedAt: Date.now() // Track cache age
      });
    }
  
    const cache = this.symbolCache.get(symbol);

    if (!cache.priceWindow) {
      cache.priceWindow = [];
    }  
    
    if (msg.type === 'A') {
      console.log(`➕ Adding ${symbol} close=${msg.close} volume=${msg.volume}`);
      cache.priceWindow.push(msg.close);
      cache.volumeWindow.push(msg.volume);
      // cache.priceWindow = [...cache.priceWindow, msg.close];
      // cache.volumeWindow = [...cache.volumeWindow, msg.volume];
  
      // Maintain window size
      if (cache.priceWindow.length > 200) {
        cache.priceWindow = cache.priceWindow.slice(-200);
        cache.volumeWindow = cache.volumeWindow.slice(-200);
      }
      console.log(`➕ Added ${symbol} close=${msg.close} (window: ${cache.priceWindow.length}/200)`);
    }
  
    const canCalculate = cache.priceWindow.length >= 15;
  
    return {
      ...msg,
      features: {
        rsi: canCalculate ? 
          TechnicalIndicators.calculateRSI(cache.priceWindow) : null,
        obv: canCalculate ? 
          TechnicalIndicators.calculateOBV(cache.priceWindow, cache.volumeWindow) : null,
        vwap: msg.vwap,
        buffer_ready: canCalculate // New feature
      }
    };
  }
  

  async applyCorporateActions(msg) {
    // Only process aggregates (A) for corporate actions
    if (msg.type !== 'A') return msg; 
    try {
      const df = await pd.DataFrame([msg]);
      const adjusted = await ca.corporate_actions_manager.apply_adjustments(df, msg.symbol);
      return {
        ...msg,
        ...await adjusted.to_dict('records')[0],
        _adjusted: true
      };
    } catch (error) {
      console.error('Corporate action adjustment failed:', error);
      return msg;
    }
  }

  async handleBatch(msg) {
    this.currentBatch.push(msg);
    if (this.currentBatch.length >= this.batchSize) {
      await this.flushToS3();
      this.currentBatch = [];
    }
  }

  async flushToS3() {
    const date = new Date().toISOString().split('T')[0];
    const timestamp = Date.now();
    
    try {
      // Add filtering here
      const filteredBatch = this.currentBatch.filter(
        t => t.features?.buffer_ready
      );
      
      // Replace currentBatch with filteredBatch
      await s3.send(new PutObjectCommand({
        Bucket: process.env.AWS_S3_BUCKET,
        Key: `processed/ticks/${date}/${timestamp}.parquet`,
        Body: JSON.stringify(filteredBatch), // Use filtered data
        ContentType: 'application/json'
      }));
      
      console.log(`📦 Saved ${filteredBatch.length} valid ticks (from ${this.currentBatch.length} raw)`);
    } catch (error) {
      console.error('S3 upload failed:', error);
      throw error;
    }
  }
  

  logCacheState() {
    // console.log('🔍 Current Cache State:');
    this.symbolCache.forEach((value, key) => {
      console.log(`- ${key}: ${value.priceWindow.length} price points`);
    });
  }
  
}
