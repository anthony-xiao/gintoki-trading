import { $ } from 'zx'
import dotenv from 'dotenv'

dotenv.config()

await $`node tests/integration/data-pipeline.test.mjs`
