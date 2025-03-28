import boto3
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_latest_model_version(bucket: str, prefix: str, extension: str = '.h5') -> Optional[str]:
    """
    Find the latest model version from S3 based on version number and timestamp.
    Version format: V{number}_{timestamp}{extension}
    """
    try:
        s3_client = boto3.client('s3')
        
        # List all objects in the prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No models found in s3://{bucket}/{prefix}")
            return None
            
        # Get all model files with the correct extension
        model_files = [obj['Key'] for obj in response['Contents'] 
                      if obj['Key'].endswith(extension)]
        
        if not model_files:
            logger.warning(f"No {extension} files found in s3://{bucket}/{prefix}")
            return None
            
        # Extract version numbers and timestamps
        version_info = []
        for file in model_files:
            # Extract filename from path
            filename = file.split('/')[-1]
            
            # Parse version number and timestamp
            match = re.match(r'V(\d+)_(\d+)', filename)
            if match:
                version_num = int(match.group(1))
                timestamp = int(match.group(2))
                version_info.append((version_num, timestamp, file))
        
        if not version_info:
            logger.warning(f"No valid versioned files found in s3://{bucket}/{prefix}")
            return None
            
        # Sort by version number (descending) and then timestamp (descending)
        version_info.sort(key=lambda x: (-x[0], -x[1]))
        
        # Get the latest version
        latest_version = version_info[0][2]
        logger.info(f"Found latest model version: {latest_version}")
        
        return latest_version
        
    except Exception as e:
        logger.error(f"Error finding latest model version: {str(e)}")
        raise 