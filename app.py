import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Union
from uuid import uuid1

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel

from AnalyzeMeeting.embedding_vector_model import EmbeddingVectorAnalyzer
from AnalyzeMeeting.gen_wordcloud import make_wordcloud
from AnalyzeMeeting.make_script import MeetingScript
from AnalyzeMeeting.make_summary import summary_model
from AnalyzeMeeting.sentiment_model import SentimentAnalyzer
from AnalyzeMeeting.stt import STTWhisper
from AnalyzeMeeting.text_organize import remove_stopwords, tokenize_text
from AnalyzeMeeting.topic_model import TopicModel
from utils.handle_server_data import aggregate_meeting_tokens, aggregate_question_tokens
from utils.upload_s3 import post_wordcloud

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    filename='app.log',        # 로그 파일 이름
    filemode='a',               # 'a'는 기존 파일에 추가, 'w'는 파일을 덮어쓰기
    level=logging.INFO,         # 로그 레벨 설정
    encoding='utf-8-sig',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_analyzer, meetings, stt_whisper, embedding_model
    sentiment_analyzer = SentimentAnalyzer()
    meetings = {}
    stt_whisper = STTWhisper()
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
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
    
    corp_id, meeting_id, question_id, user_id, text_response, survey_question = response.corpId, response.meetingId, response.questionId, response.userId, response.textResponse, response.surveyQuestion
    
    if corp_id not in meetings:
        meetings[corp_id] = {meeting_id:MeetingScript(corp_id, meeting_id)}
    if meeting_id not in meetings[corp_id]:
        meetings[corp_id][meeting_id] = MeetingScript(corp_id, meeting_id)
        
    meeting_script = meetings[corp_id][meeting_id]
    meeting_script.add_question(question_id, survey_question)
    meeting_script.add_answer(question_id, text_response, user_id)
    
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
        
        corp_id, meeting_id, question_id, user_id, voice_response, survey_question = response.corpId, response.meetingId, response.questionId, response.userId, response.voiceResponse, response.surveyQuestion
        voice_data = base64.b64decode(voice_response)
        text_response = stt_whisper.transcribe(voice_data)
        if corp_id not in meetings:
            meetings[corp_id] = {meeting_id: MeetingScript(corp_id, meeting_id)}
        if meeting_id not in meetings[corp_id]:
            meetings[corp_id][meeting_id] = MeetingScript(corp_id, meeting_id)

        meeting_script = meetings[corp_id][meeting_id]
        meeting_script.add_question(question_id, survey_question)
        meeting_script.add_answer(question_id, text_response, user_id)
        
        logging.info(f"/submit-voice result: {text_response}")
        return {
            "result": "음성 응답이 성공적으로 처리되었습니다.",
            "text_data": text_response
                }
        
    except Exception as e:
        logging.error("Error: %s", e)
        raise HTTPException(status_code=400, detail=f'Error: {e}')


'''최종 분석 처리 엔드포인트'''
class MeetingScriptIn(BaseModel):
    corpId: int
    meetingId: int
class MeetingScriptOut(BaseModel):
    result: str
    script: List[Dict[str, List[str]]]

@app.post("/meeting-script", tags=['Analyze all questions'])
async def meeting_script(response: MeetingScriptIn):
    corp_id, meeting_id = response.corpId, response.meetingId
    meeting_script = meetings[corp_id][meeting_id] 
    script = meeting_script.to_script_format()
    return {"result": "스크립트가 성공적으로 생성되었습니다.", "script": script}

# TODO: 요약 llm 생성 및 요약 기능
class MeetingSummaryIn(BaseModel):
    corpId: int
    meetingId: int
class MeetingSummaryOut(BaseModel):
    result: str
    summary: str
    
@app.post("/meeting-summary", tags=['Analyze all questions'])
async def meeting_summary(response: MeetingSummaryIn):
    corp_id, meeting_id = response.corpId, response.meetingId
    meeting_script = meetings[corp_id][meeting_id]
    script = meeting_script.to_script_format()
    summary = summary_model.exec(script)
    print(summary)
    return {"result":"요약이 성공적으로 완료되었습니다.", "summary":summary}
        
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
        # logging
        print(f'endpoint: /analyze-all, response: {response}')
        logging.info(f"endpoint: /analyze-all: {response}")
        
        corp_id, meeting_id = response.corpId, response.meetingId
        all_tokens = aggregate_meeting_tokens(corp_id, meeting_id, tokens)
        topic_model = TopicModel(all_tokens)
        json_result = topic_model.make_lda_json()
        embedding_vector_analyzer = EmbeddingVectorAnalyzer(corp_id=corp_id, meeting_id=meeting_id, embedding_model=embedding_model)
        embedding_vector_analyzer.make_token_embeddings(all_tokens)
        port = embedding_vector_analyzer.find_available_port()
        embedding_vector_analyzer.run_tensorboard('127.0.0.1', port)
        
        file_name = f"wordcloud_{str(uuid1())}.png"
        output_image_path = f"./data/{file_name}"
        make_wordcloud(tokens=all_tokens, mask_image_path=None, width=800, height=400, output_image_name=file_name)
        
        await post_wordcloud(output_image_path, file_name, meeting_id)
        
        sentiment_result = sentiment_analyzer.analyze_token_sentiment(all_tokens)
        
        return {"result":"모든 분석이 성공적으로 완료되었습니다.", "topic_result":json_result, "wordcloud_filename":file_name, "sentiment_result":sentiment_result, "tensorboard_url":f"http://127.0.0.1:{port}/#projector"}
    
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
    corpId: int
    meetingId: int
    questionId: int
    
class AnalyzeEmbeddingOut(BaseModel):
    result: str
    tensorboard_url: str
    
@app.post("/analyze-embedding", response_model=AnalyzeEmbeddingOut, tags=['Analyze each question'])
async def analyze_embedding(response: AnalyzeEmbeddingIn):
    logging.info(f"endpoint: /analyze-embedding : {response}")
    responses, corp_id, meeting_id, question_id = response.responses, response.corpId, response.meetingId, response.questionId
    embedding_vector_analyzer = EmbeddingVectorAnalyzer(corp_id=corp_id, meeting_id=meeting_id, question_id=question_id, embedding_model=embedding_model)
    embedding_vector_analyzer.make_sentence_embeddings(responses)
    port = embedding_vector_analyzer.find_available_port()
    embedding_vector_analyzer.run_tensorboard('127.0.0.1', port)
    
    return {"result":"임베딩 분석이 성공적으로 완료되었습니다.", "tensorboard_url":f"http://127.0.0.1:{port}/#projector"}

# 워드 클라우드
class GenerateWordcloudIn(BaseModel):
    responses: list

class GenerateWordcloudOut(BaseModel):
    result: str
    wordcloud_filename: str

@app.post("/generate-wordcloud", response_model=GenerateWordcloudOut, tags=['Analyze each question'])
async def generate_wordcloud(response: GenerateWordcloudIn):
    # logging
    logging.info(f"endpoint: /generate-wordcloud : {response}")
    print(f'endpoint: /generate-wordcloud, response: {response}')
    
    responses = response.responses
    meeting_id = responses[0]['meetingId']
    
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