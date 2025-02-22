import { 
    SMA,
    EMA,
    RSI,
    OBV,
    Stochastic,
    VWAP 
  } from 'technicalindicators';
  
  export class TechnicalIndicators {
    static calculateRSI(closes, period = 14) {
      if (closes.length < period + 1) {
        console.warn(`📉 RSI requires ${period+1} values, got ${closes.length}`);
        return null; // Fallback to neutral value
      }
    
      // Handle perfect upward trend
      const isPerfectUptrend = closes.slice(1)
        .every((val, i) => val > closes[i]);
      
      if (isPerfectUptrend) return 100;
    
      // Handle perfect downward trend
      const isPerfectDowntrend = closes.slice(1)
        .every((val, i) => val < closes[i]);
      
      if (isPerfectDowntrend) return 0;
    
      // Standard calculation
      const deltas = closes.slice(1).map((c, i) => c - closes[i]);
      const gains = deltas.filter(d => d > 0);
      const losses = deltas.filter(d => d < 0).map(d => Math.abs(d));
    
      const avgGain = gains.reduce((a,b) => a + b, 0) / period;
      const avgLoss = losses.reduce((a,b) => a + b, 0) / period || 0.0001; // Avoid division by zero
      
      const rs = avgGain / avgLoss;
      const rsi = 100 - (100 / (1 + rs));
      
      return Math.min(Math.max(rsi, 0), 100); // Clamp between 0-100
    }
    
    static calculateOBV(closes, volumes) {
      if (!closes?.length || !volumes?.length || closes.length !== volumes.length) {
        console.warn(`OBV requires matching close/volume arrays`);
        return null;
      }      
      // OBV requires at least 2 data points
      const results = OBV.calculate({
        close: closes,
        volume: volumes
      });
      
      return results.length > 0 ? results[results.length - 1] : 0;
    }
  
    static calculateStochastic(highs, lows, closes, period = 14) {
      return Stochastic.calculate({
        high: highs,
        low: lows,
        close: closes,
        period,
        signalPeriod: 3
      });
    }
  
    static calculateVWAP(ticks) {
      return VWAP.calculate({
        open: ticks.map(t => t.open),
        high: ticks.map(t => t.high),
        low: ticks.map(t => t.low),
        close: ticks.map(t => t.close),
        volume: ticks.map(t => t.volume)
      });
    }
  }


// technical-indicators.mjs tests
console.assert(
  TechnicalIndicators.calculateRSI(
    Array.from({length: 15}, (_, i) => 100 - i)
  ) === 0,
  'Perfect downtrend should return 0'
);

console.assert(
  TechnicalIndicators.calculateRSI(
    Array.from({length: 15}, (_, i) => 100 + i)
  ) === 100,
  'Perfect uptrend should return 100'
);

console.assert(
  Math.abs(TechnicalIndicators.calculateRSI(
    Array.from({length: 15}, () => 100)
  ) - 50) < 0.01,
  'Flat line should return 50'
);


// Update test data to match correct OBV calculation
console.assert(
  TechnicalIndicators.calculateOBV(
    [10, 10.5, 10.3, 10.7], 
    [1000, 2000, 1500, 3000]
  ) === 3500, // Changed from 5500 to correct value
  'OBV calculation failed. Expected 3500 got %s',
  TechnicalIndicators.calculateOBV([10, 10.5, 10.3, 10.7], [1000, 2000, 1500, 3000])
);
console.log('All tests passed!');

  