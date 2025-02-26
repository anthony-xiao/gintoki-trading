#!/bin/bash
# Launch p4d.24xlarge spot instance
aws ec2 run-instances \
  --image-id ami-0c20b8c3851e4abec \
  --instance-type p4d.24xlarge \
  --key-name trading-key \
  --security-group-ids sg-123456 \
  --subnet-id subnet-123456 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-training}]' \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=15.0}'
