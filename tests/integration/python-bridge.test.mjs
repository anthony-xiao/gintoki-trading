import { test } from 'node:test'
import assert from 'node:assert'
import pythonBridge from 'jspybridge'

test('Python bridge initialization', async () => {
  const py = pythonBridge({
    python: 'python3',
    env: { PYTHONPATH: process.env.PYTHONPATH }
  })
  
  await py.ex`import sys`
  const path = await py`sys.path`
  assert.ok(path.includes('/app/src/py'), 'Python path not configured correctly')
  
  await py.end()
})
