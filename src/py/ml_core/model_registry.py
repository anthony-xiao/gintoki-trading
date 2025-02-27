import boto3
import hashlib
import os
from datetime import datetime

class EnhancedModelRegistry:
    def __init__(self, bucket: str = 'quant-trader-data-gintoki'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.version_prefix = "enhanced_v"
        
    def save_enhanced_model(self, local_path: str, model_type: str):
        """Save enhanced model with versioned naming"""
        version = f"{self.version_prefix}{datetime.now().strftime('%Y%m%d%H%M')}"
        s3_key = f"models/{version}_{model_type}_{os.path.basename(local_path)}"
        
        # Generate checksum
        with open(local_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
            
        # Upload model and metadata
        self.s3.upload_file(local_path, self.bucket, s3_key)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{s3_key}.metadata",
            Body=f'{{"checksum":"{checksum}","created":"{datetime.utcnow().isoformat()}"}}'
        )
        return s3_key

    def list_enhanced_models(self):
        """List all enhanced model versions"""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"models/{self.version_prefix}"
        )
        return [obj['Key'] for obj in response.get('Contents', [])]
