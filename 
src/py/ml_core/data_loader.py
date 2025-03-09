    def _load_s3_data(self, prefix: str) -> pd.DataFrame:
        """Load and concatenate parquet files from S3"""
        prefix = prefix.lower().strip('/')  # Normalize path case
        s3_uri = f"s3://{self.bucket}/{prefix}"
        logger = logging.getLogger("training")
        logger.debug(f"📁 Investigating: {s3_uri}")
        logger.debug(f"Normalized S3 search path: {prefix}")
                # Fix 2: Process even if 'Contents' key exists but list is empty
                for obj in objects:
                    if any(obj['Key'].lower().endswith(ext) for ext in ('.parquet', '.parq', '.snappy.parquet')):
                        logger.debug(f"📦 Found compatible Parquet file: {obj['Key']}")
                        logger.debug(f"📂 Full S3 path: s3://{self.bucket}/{obj['Key']}")
                        logger.debug(f"📏 File size: {obj['Size']} bytes")
                        response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                        dfs.append(pd.read_parquet(BytesIO(response['Body'].read())))
    def _process_quotes(self, ticker: str) -> pd.DataFrame:
        """Process raw quotes into spread features"""
        ticker = ticker.strip().lower()
        quotes_path = f'historical/{ticker}/quotes/'.lower()
        logger = logging.getLogger("training")
        logger.debug("🏃 Starting quote processing for %s at %s", ticker, quotes_path)
        try:
            quotes = self._load_s3_data(quotes_path)
