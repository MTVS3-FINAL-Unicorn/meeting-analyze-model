import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 AWS 자격 증명 및 설정 불러오기
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

client = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION
                      )

async def upload_file_to_s3(file_path, key):
    bucket = 'jurassic-park'
    
    try:
        # 파일 업로드
        client.upload_file(file_path, bucket, key)

    except FileNotFoundError:
        print("The file was not found")
        
    except NoCredentialsError:
        print("Credentials not available")


if __name__ == '__main__':
    file_path = 'ComfyUI/output/AnimateDiff_00009.mp4'
    bucket = 'jurassic-park'
    key = 'comfy_result.wav' 
