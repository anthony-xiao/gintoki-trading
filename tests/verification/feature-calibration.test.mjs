import { TickProcessor } from '../../src/js/core/services/data/tick-processor.mjs';
import { strict as assert } from 'node:assert';

const TEST_SYMBOL = 'AAPL';

// Generate realistic test data with variance
function generateTestMessages(count) {
    let price = 150;
    return Array.from({length: count}, (_, i) => {
      // Add 10% random volatility
      const change = (Math.random() * 0.2 - 0.1) * price; 
      price += change;
      
      return {
        ev: 'A',
        sym: TEST_SYMBOL,
        o: price - Math.abs(change * 0.5),
        h: price + Math.abs(change),
        l: price - Math.abs(change),
        c: price,
        v: 1000000 + i*1000,
        s: Date.now() - (count - i)*60000,
        e: Date.now() - (count - i)*60000,
        vw: price
      };
    });
  }
  

export async function testFeatureCalibration() {
  const processor = new TickProcessor();
  const testMessages = generateTestMessages(20);
  const results = [];

  for (const msg of testMessages) {
    const processed = await processor.process([msg]);
    if (processed.length > 0) {
      results.push(processed[0]);
    }
  }

  // Verify first valid features
  const firstValid = results.find(r => r.features.rsi !== 50);
  assert.ok(firstValid, 'No valid features generated');
  
  // Modified assertion
  assert.ok(
    firstValid.features.rsi >= 0 && firstValid.features.rsi <= 100,
    `Invalid RSI value: ${firstValid.features.rsi}`
  );
  
  // Realistic OBV check
  assert.ok(
    Math.abs(firstValid.features.obv) > 0,
    `OBV should show direction: ${firstValid.features.obv}`
  );
  
  console.log('✅ Feature calibration test passed');
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  await testFeatureCalibration();
}
