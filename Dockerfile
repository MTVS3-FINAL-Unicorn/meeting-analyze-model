FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime


# app 기본 파일 생성
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 포트 설정
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host=0.0.0.0",  "--port=8000"]

