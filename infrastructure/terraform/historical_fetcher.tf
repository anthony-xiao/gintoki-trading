resource "aws_cloudwatch_event_rule" "daily_fetch" {
  name        = "daily-historical-fetch"
  description = "Triggers daily historical data collection"
  schedule_expression = "cron(0 23 * * ? *)" # 11PM UTC
}

resource "aws_lambda_function" "daily_fetcher" {
  filename      = "${path.module}/../../src/py/infrastructure/daily-fetcher-lambda.zip"
  function_name = "historical-fetcher-daily"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "daily-fetcher-lambda.lambda_handler"
  runtime       = "python3.11"
  timeout       = 900
}

output "corporate_actions_paths" {
  description = "S3 paths for corporate actions storage"
  value = {
    splits    = "s3://quant-trader-data-gintoki/historical/{ticker}/corporate_actions/splits/"
    dividends = "s3://quant-trader-data-gintoki/historical/{ticker}/corporate_actions/dividends/"
  }
}
