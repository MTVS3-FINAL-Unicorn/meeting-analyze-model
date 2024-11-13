import base64
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
from utils.upload_s3 import post_wordcloud
from utils.handle_server_data import aggregate_meeting_tokens, aggregate_question_tokens
from AnalyzeMeeting.embedding_vector_model import EmbeddingVectorModel
from AnalyzeMeeting.stt import STTWhisper
from AnalyzeMeeting.text_organize import remove_stopwords, tokenize_text
from AnalyzeMeeting.topic_model import TopicModel
from AnalyzeMeeting.sentiment_model import SentimentAnalyzer
from typing import List
from contextlib import asynccontextmanager
from uuid import uuid1
from typing import Dict, List, Tuple, Union

from AnalyzeMeeting.gen_wordcloud import make_wordcloud

# 환경변수 로드
load_dotenv()

logging.basicConfig(
    filename='app.log',        # 로그 파일 이름
    filemode='a',               # 'a'는 기존 파일에 추가, 'w'는 파일을 덮어쓰기
    level=logging.INFO,         # 로그 레벨 설정
    encoding='utf-8-sig',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_vector_model, sentiment_analyzer, tokens, stt_whisper
    embedding_vector_model = EmbeddingVectorModel()
    sentiment_analyzer = SentimentAnalyzer()
    tokens = {}
    stt_whisper = STTWhisper()    
    yield
    
    
# FastAPI와 템플릿 설정
app = FastAPI(
    title="좌담회 분석 API 문서",
    description="좌담회 데이터 관리, 분석 기능",
    version="1.0.0",
    lifespan=lifespan
    )

# 텍스트 답변 처리 엔드포인트
class SubmitTextIn(BaseModel):
    surveyQuestion: str
    textResponse: str
    userId: int
    meetingId: int
    corpId: int
    questionId: int
    
class SubmitTextOut(BaseModel):
    result: str
    
@app.post("/submit-text", response_model=SubmitTextOut, tags=['Submit meeting data'])
async def submit_text_response(response: SubmitTextIn):
    print(f'endpoint: /submit-text, response: {response}')
    logging.info(f"/submit-text : {response}")
    corp_id, meeting_id, question_id, user_id, text_response = response.corpId, response.meetingId, response.questionId, response.userId, response.textResponse
    if corp_id not in tokens:
            tokens[corp_id] = {meeting_id: {question_id: {user_id: {'tokens':[],'answers':[]}}}}
    if meeting_id not in tokens[corp_id]:
        tokens[corp_id][meeting_id] = {}
    if question_id not in tokens[corp_id][meeting_id]:
        tokens[corp_id][meeting_id][question_id] = {}
    if user_id not in tokens[corp_id][meeting_id][question_id]:
        tokens[corp_id][meeting_id][question_id][user_id] = {'answers':[], 'tokens':[]}
    tokens[corp_id][meeting_id][question_id][user_id]['tokens'].extend(remove_stopwords(tokenize_text(text_response)))
    tokens[corp_id][meeting_id][question_id][user_id]['answers'].extend(text_response)
    
    return {"result": "텍스트 응답이 성공적으로 처리되었습니다."}

# 음성 답변 처리 엔드포인트 (음성 파일을 STT로 변환)
class SubmitVoiceIn(BaseModel):
    surveyQuestion: str 
    voiceResponse: str
    userId: int 
    meetingId: int 
    corpId: int
    questionId: int

class SubmitVoiceOut(BaseModel):
    result: str
    text_data: str
    
@app.post("/submit-voice", response_model=SubmitVoiceOut, tags=['Submit meeting data'])
async def submit_voice_response(response: SubmitVoiceIn):
    try:
        logging.info(f"/submit-voice: {response}")
        print(f'endpoint: /submit-voice, response: {response}')
        
        corp_id, meeting_id, question_id, user_id, voice_response = response.corpId, response.meetingId, response.questionId, response.userId, response.voiceResponse
        voice_data = base64.b64decode(voice_response)
        text_response = stt_whisper.transcribe(voice_data)
        if corp_id not in tokens:
            tokens[corp_id] = {meeting_id: {question_id: {user_id: {'tokens':[],'answers':[]}}}}
        if meeting_id not in tokens[corp_id]:
            tokens[corp_id][meeting_id] = {}
        if question_id not in tokens[corp_id][meeting_id]:
            tokens[corp_id][meeting_id][question_id] = {}
        if user_id not in tokens[corp_id][meeting_id][question_id]:
            tokens[corp_id][meeting_id][question_id][user_id] = {'answers':[], 'tokens':[]}
        tokens[corp_id][meeting_id][question_id][user_id]['tokens'].extend(remove_stopwords(tokenize_text(text_response)))
        tokens[corp_id][meeting_id][question_id][user_id]['answers'].extend(text_response)
        
        return {
            "result": "음성 응답이 성공적으로 처리되었습니다.",
            "text_data": text_response
                }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error: {e}')


'''최종 분석 처리 엔드포인트'''
class AnalyzeAllIn(BaseModel):
    corpId: int
    meetingId: int

class AnalyzeAllOut(BaseModel):
    result: str
    topic_result: Dict
    wordcloud_filename: str
    sentiment_result: Dict[str, Dict]
    
@app.post("/analyze-all", response_model=AnalyzeAllOut, tags=['Analyze all questions'])
async def analyze_all(response: AnalyzeAllIn):
    try:
        print(f'endpoint: /analyze-all, response: {response}')
        logging.info(f"endpoint: /analyze-all: {response}")
        corp_id, meeting_id = response.corpId, response.meetingId
        all_tokens = aggregate_meeting_tokens(corp_id, meeting_id, tokens)
        topic_model = TopicModel(all_tokens)
        json_result = topic_model.make_lda_json()
        
        embedding_vector_model.make_token_embeddings(all_tokens)
        port = embedding_vector_model.find_available_port()
        embedding_vector_model.run_tensorboard('127.0.0.1', port)
        
        file_name = f"wordcloud_{str(uuid1())}.png"
        output_image_path = f"./data/{file_name}"
        make_wordcloud(tokens=all_tokens, mask_image_path=None, width=800, height=400, output_image_name=file_name)
        
        await post_wordcloud(output_image_path, file_name, meeting_id)
        
        sentiment_result = sentiment_analyzer.analyze_token_sentiment(all_tokens)
        
        return {"result":"모든 분석이 성공적으로 완료되었습니다.", "topic_result":json_result, "wordcloud_filename":file_name, "sentiment_result":sentiment_result}
    
    except KeyError:
        if corp_id not in tokens:
            raise HTTPException(status_code=404, detail="corpId에 해당하는 데이터가 존재하지 않습니다.")
        elif meeting_id not in tokens[corp_id]:
            raise HTTPException(status_code=404, detail='meetingId에 해당하는 데이터가 존재하지 않습니다.')
    except Exception as e:
        if str(e) =='empty vocabulary; perhaps the documents only contain stop words':
            raise HTTPException(status_code=404, detail="분석할 데이터가 존재하지 않습니다.")
        raise HTTPException(status_code=400, detail=f"error: {e}")


'''질문 별 분석 엔드포인트''' 
# 토픽 분석 
class AnalyzeTopicIn(BaseModel):
    responses: List

class AnalyzeTopicOut(BaseModel):
    result: str
    topic_result: Dict
    
@app.post("/analyze-topic", response_model=AnalyzeTopicOut, tags=['Analyze each question'])
async def analyze_topic(response: AnalyzeTopicIn):
    logging.info(f"endpoint: /analyze-topic : {response}")
    responses = response.responses
    question_tokens = aggregate_question_tokens(responses)
    topic_model = TopicModel(question_tokens)
    json_result = topic_model.make_lda_json()
    
    return {"result":"토픽 분석이 성공적으로 완료되었습니다.", "topic_result":json_result}


# 임베딩 분석, 각 답변에 대해 거리 확인
class AnalyzeEmbeddingIn(BaseModel):
    responses: list
    
class AnalyzeEmbeddingOut(BaseModel):
    result: str
    tensorboard_url: str
    
@app.post("/analyze-embedding", response_model=AnalyzeEmbeddingOut, tags=['Analyze each question'])
async def analyze_embedding(response: AnalyzeEmbeddingIn):
    logging.info(f"endpoint: /analyze-embedding : {response}")
    responses = response.responses
    embedding_vector_model.make_sentence_embeddings(responses)
    port = embedding_vector_model.find_available_port()
    embedding_vector_model.run_tensorboard('127.0.0.1', port)
    
    return {"result":"임베딩 분석이 성공적으로 완료되었습니다.", "tensorboard_url":f"http://127.0.0.1:{port}/#projector"}

# 워드 클라우드
class GenerateWordcloudIn(BaseModel):
    responses: list

class GenerateWordcloudOut(BaseModel):
    result: str
    wordcloud_filename: str

@app.post("/generate-wordcloud", response_model=GenerateWordcloudOut, tags=['Analyze each question'])
async def generate_wordcloud(response: GenerateWordcloudIn):
    logging.info(f"endpoint: /generate-wordcloud : {response}")
    responses = response.responses
    meeting_id = responses[0]['meetingId']
    print(f'endpoint: /generate-wordcloud, response: {response}')
    tokens = remove_stopwords(tokenize_text(aggregate_question_tokens(responses)))
    file_name = f"wordcloud_{str(uuid1())}.png"
    output_image_path = f"./data/{file_name}"
    make_wordcloud(tokens=tokens, mask_image_path=None, width=800, height=400, output_image_name=file_name)
    await post_wordcloud(output_image_path, file_name, meeting_id)
    
    return {"result":"워드 클라우드가 성공적으로 생성되었습니다.", "wordcloud_filename":file_name}

# 감정분석
class AnalyzeSentimentIn(BaseModel):
    responses: list
    mostCommonK: int
    
class AnalyzeSentimentOut(BaseModel):
    result: str
    sentiment_result: Dict[str, Union[Dict, List]]
    token_count: Dict[str, int]
    most_common_token: List[Tuple[str, int]]
    
@app.post("/analyze-sentiment", response_model=AnalyzeSentimentOut, tags=['Analyze each question'])
async def analyze_sentiment(response: AnalyzeSentimentIn):
    logging.info(f"endpoint: /analyze-sentiment : {response}")
    sent_result, token_count, most_common_token = sentiment_analyzer.analyze_sentence_sentiment(response.responses, most_k=response.mostCommonK)
    return {"result":"감정 분석이 성공적으로 완료되었습니다.", "sentiment_result":sent_result, "token_count":token_count, "most_common_token":most_common_token}

# #시연용 분셕 사이트
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse
# templates = Jinja2Templates(directory="./templates")
# @app.get('/index.html')
# async def serve_html():
#     try:
#         with open("./index.html", "r") as f:
#             html_content = f.read()
#         return HTMLResponse(content=html_content, status_code=200)
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="HTML file not found.")