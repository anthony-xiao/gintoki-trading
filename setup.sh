#!/bin/bash

# Create root directory structure
mkdir -p {.github/workflows,config,src/{js/core/{services/{alpaca,data},utils,bridges},py/ml_core/{training,models,utils},shared/protocols},data/{raw,processed,features},infrastructure/{terraform,cloud-init},monitoring/grafana/dashboards,tests/{js/unit,py/unit},scripts}

# Create core files
touch .env.example Dockerfile docker-compose.{prod,dev}.yml Makefile package.json requirements.txt README.md ARCHITECTURE.md

# Create config files
cat > config/alpaca.json <<EOF
{
  "paper": true,
  "max_position_size": 0.05,
  "max_daily_loss": 0.15
}
EOF

cat > config/risk_params.yaml <<EOF
risk_model:
  stop_loss: 0.02
  trailing_stop: 0.025
  max_leverage: 4
  volatility_window: 21
EOF

# Create JS core files
cat > src/js/core/services/data/polyfeed.mjs <<'EOF'
import { WebSocketClient } from 'polygon.io';
import { S3Client } from '@aws-sdk/client-s3';

const polyWS = new WebSocketClient(process.env.POLYGON_KEY);
const s3 = new S3Client({ region: 'us-west-2' });

polyWS.on('message', async (msg) => {
  // Data processing logic
});
EOF

# Create Python training files
cat > src/py/ml_core/training/train.py <<'EOF'
import tensorflow as tf
import shap

class AdaptiveTrader(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(128)
    
    def call(self, inputs):
        return self.lstm(inputs)
EOF

# Create CI/CD pipeline
cat > .github/workflows/ci.yml <<EOF
name: CI Pipeline
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm install && npm test
EOF

# Create basic .gitignore
cat > .gitignore <<EOF
.env
node_modules/
__pycache__/
*.pyc
.DS_Store
data/
models/
EOF

# Create Dockerfile
cat > Dockerfile <<EOF
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["node", "src/js/core/services/data/polyfeed.mjs"]
EOF

echo "Project structure created successfully!"
echo "Next steps:"
echo "1. chmod +x setup.sh && ./setup.sh"
echo "2. Edit .env.example and rename to .env"
echo "3. Run 'npm install && pip install -r requirements.txt'"
