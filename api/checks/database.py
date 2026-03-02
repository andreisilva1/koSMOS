import asyncio
import os

from dotenv import load_dotenv
from fastapi import HTTPException
from botocore.exceptions import ClientError

load_dotenv()
ALLOW_LOCAL_FALLBACK = (
    os.getenv("ALLOW_LOCAL_FALLBACK")
    if os.getenv("ALLOW_LOCAL_FALLBACK") is not None
    else 1
)


def check_aws_connection(
    client,
    bucket_name: str,
    AWS_ACCESS_KEY_ID: str,
    AWS_DEFAULT_REGION: str,
    AWS_SECRET_ACCESS_KEY: str,
):
    if AWS_ACCESS_KEY_ID and AWS_DEFAULT_REGION and AWS_SECRET_ACCESS_KEY:
        try:
            client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            raise HTTPException(status_code=500, detail="Connection error.")
    if not bool(ALLOW_LOCAL_FALLBACK):
        raise HTTPException(status_code=500, detail="Connection error.")
    return False
