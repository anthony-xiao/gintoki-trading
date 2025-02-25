import os
import boto3
from datetime import datetime, timedelta

def lambda_handler(event, context):
    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=1)
    
    # Get configured tickers
    ssm = boto3.client('ssm')
    tickers = ssm.get_parameter(Name='/trading/tickers')['Parameter']['Value'].split(',')
    
    # Trigger ECS task
    ecs = boto3.client('ecs')
    response = ecs.run_task(
        cluster='historical-fetcher',
        taskDefinition='historical-fetcher-daily',
        count=1,
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-123456'],
                'securityGroups': ['sg-123456'],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [{
                'name': 'fetcher',
                'command': [
                    "python", "-m", "src.py.data_ingestion.historical_data_fetcher",
                    "--start", start_date.strftime('%Y-%m-%d'),
                    "--end", end_date.strftime('%Y-%m-%d'),
                    "--tickers", *tickers
                ]
            }]
        }
    )
    return response
