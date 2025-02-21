import { strict as assert } from 'node:assert';
import { TickProcessor } from '../../src/js/core/services/data/tick-processor.mjs';

const TEST_TICKS = [
  // Valid aggregate
  {
    ev: 'A',
    sym: 'AAPL',
    o: 150.0, h: 155.0, l: 149.5, c: 153.2,
    v: 1000000, s: Date.now() - 10000, e: Date.now()
  },
  // Invalid trade
  { ev: 'T', sym: 'AAPL', p: 'invalid', s: 100, t: Date.now() }
];

export function testTickProcessing() {
  const processor = new TickProcessor();
  
  // Test validation
  const validated = processor.process(TEST_TICKS);
  assert.equal(validated.length, 1, 'Should filter invalid ticks');
  
  // Test normalization
  const normalized = validated[0];
  assert.ok(normalized.symbol === 'AAPL', 'Symbol normalization failed');
  assert.ok(typeof normalized.close === 'number', 'Number conversion failed');
  
  // Test feature enrichment
  assert.ok('features' in normalized, 'Missing features');
  assert.ok('rsi' in normalized.features, 'RSI calculation missing');
}
