// Quick test script (test-tick-processor.mjs)
import { TickProcessor } from './tick-processor.mjs';

const testMessages = [
  { ev: 'A', sym: 'AAPL', o: 150, h: 155, l: 149, c: 153, v: 1000000, s: 1627660800000 }, 
  { ev: 'Q', sym: 'MSFT', bp: 300, ap: 301, bs: 500, as: 400, t: 1627660801000 },
  { ev: 'T', sym: 'GOOG', p: 2700, s: 100, t: 1627660802000, c: ['@'] }
];

const processor = new TickProcessor();
processor.process(testMessages)
  .then(ticks => console.log('Processed Ticks:', ticks))
  .catch(console.error);
