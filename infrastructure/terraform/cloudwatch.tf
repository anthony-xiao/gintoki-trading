resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "TradingDataPipeline"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 24
        height = 6
        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", "historical-fetcher-daily"]
          ]
          period = 300
          stat   = "Sum"
          region = "ap-southeast-2"
          title  = "Daily Fetch Executions"
        }
      }
    ]
  })
}

output "dashboard_url" {
  value = "https://console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}
