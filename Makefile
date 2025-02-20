.PHONY: install train start monitor

install:
    npm install
    pip install -r requirements.txt

train:
    python -m src.py.ml_core.training.train --production

start:
    docker-compose -f docker-compose.prod.yml up --build

monitor:
    open http://localhost:3000/dashboard
