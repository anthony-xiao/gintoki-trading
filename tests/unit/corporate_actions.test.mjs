import { strict as assert } from 'node:assert';
import { python } from 'pythonia';

const ca = await python('util.corporate_actions')

export async function testCorporateActions() {
  // Test split adjustment
  const test_data = pd.DataFrame({
    timestamp: ['2023-01-01', '2023-01-02'],
    close: [100, 100]
  });
  
  ca.corporate_actions_manager.split_map['AAPL'] = {
    '2023-01-02': 4 // 4:1 split
  };

  const adjusted = ca.corporate_actions_manager.apply_adjustments(test_data, 'AAPL');
  assert.equal(adjusted.iloc[1].close, 25, 'Split adjustment failed');
}
