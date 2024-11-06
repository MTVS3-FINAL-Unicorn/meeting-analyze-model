import base64

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.upload_s3 import upload_file_to_s3
from utils.handle_server_data import aggregate_meeting_tokens, aggregate_question_tokens
from AnalyzeMeeting.embedding_vector_model import EmbeddingVectorModel
from AnalyzeMeeting.stt import stt_whisper
from AnalyzeMeeting.text_organize import remove_stopwords, tokenize_text
from AnalyzeMeeting.topic_model import TopicModel
import os

from AnalyzeMeeting.gen_wordcloud import make_wordcloud

# 환경변수 로드
load_dotenv()

# FastAPI와 템플릿 설정
app = FastAPI(
    title="좌담회 분석 API 문서",
    description="좌담회 데이터 관리, 분석 기능",
    version="1.0.0"
)
templates = Jinja2Templates(directory="./templates")

# embedding_model 생성
embedding_vector_model = EmbeddingVectorModel()

# 토큰 저장소
tokens = {}

# 텍스트 답변 처리 엔드포인트
class TextResponse(BaseModel):
    surveyQuestion: str
    textResponse: str
    userId: int
    meetingId: int
    corpId: int
    questionId: int

@app.post("/submit-text-response")
async def submit_text_response(response: TextResponse):
    corpId, meetingId, questionId, userId = response.corpId, response.meetingId, response.questionId, response.userId
    if corpId not in tokens:
            tokens[corpId] = {meetingId: []}
    if meetingId not in tokens[corpId]:
        tokens[corpId][meetingId] = {}
    if questionId not in tokens[corpId][meetingId]:
        tokens[corpId][meetingId][questionId] = {}
    if userId not in tokens[corpId][meetingId][questionId]:
        tokens[corpId][meetingId][questionId][userId] = []
    tokens[corpId][meetingId][questionId][userId].extend(remove_stopwords(tokenize_text(response.textResponse)))
    
    return {"result": "텍스트 응답이 성공적으로 처리되었습니다."}

# 음성 답변 처리 엔드포인트 (음성 파일을 STT로 변환)
class VoiceResponse(BaseModel):
    surveyQuestion: str 
    voiceResponse: str
    userId: int 
    meetingId: int 
    corpId: int
    questionId: int
@app.post("/submit-voice-response")
async def submit_voice_response(response: VoiceResponse):
    try:
        corpId, meetingId, questionId, userId, voiceResponse = response.corpId, response.meetingId, response.questionId, response.userId, response.voiceResponse
        voice_data = base64.b64decode(voiceResponse)
        text_response = stt_whisper(voice_data)
        if corpId not in tokens:
            tokens[corpId] = {meetingId: []}
        if meetingId not in tokens[corpId]:
            tokens[corpId][meetingId] = {}
        if questionId not in tokens[corpId][meetingId]:
            tokens[corpId][meetingId][questionId] = {}
        if userId not in tokens[corpId][meetingId][questionId]:
            tokens[corpId][meetingId][questionId][userId] = []
        tokens[corpId][meetingId][questionId][userId].extend(remove_stopwords(tokenize_text(text_response)))
        
        return {
            "result": "음성 응답이 성공적으로 처리되었습니다.",
            "text_data": text_response
                }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error: {e}')

# # 설문 선택 답변 처리 엔드포인트
# class ChoiceResponse(BaseModel):
#     surveyQuestion: str
#     choiceResponse: str
#     userId: int
#     meetingId: int
#     corpId: int
#     questionId: int
    
# @app.post("/submit-choice-response")
# async def submit_choice_response(response: ChoiceResponse):
#     # 선택형 설문 데이터를 저장
#     return {"result": "설문 응답이 성공적으로 처리되었습니다."}

# 최종 분석 처리 엔드포인트
class AnalyzeOverallResponse(BaseModel):
    corpId: int
    meetingId: int
    
@app.post("/analyze-overall-responses")
async def analyze_overall_responses(response: AnalyzeOverallResponse):
    try:
        overall_tokens = aggregate_meeting_tokens(response.corpId, response.meetingId, tokens)
        topic_model = TopicModel(overall_tokens, response.corpId, response.meetingId)
        json_result = topic_model.make_lda_json()
        
        embedding_vector_model.make_embeddings(overall_tokens, response.corpId, response.meetingId)
        embedding_vector_model.run_tensorboard('0.0.0.0', '7779')
        
        output_image_path = f"./data/wordcloud_{response.corpId}_{response.meetingId}.png"
        wordcloud_result = make_wordcloud(tokens=overall_tokens, mask_image_path=None, width=800, height=400, output_image_path=output_image_path)
        key = f"wordcloud_{response.corpId}_{response.meetingId}.png"
        await upload_file_to_s3(wordcloud_result, key)
        AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
        file_url = f"https://jurassic-park.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{key}"
        return {"topic_result":json_result, "wordcloud_result":file_url}
    
    except KeyError:
        if response.corpId not in tokens:
            raise HTTPException(status_code=404, detail="corpId에 해당하는 데이터가 존재하지 않습니다.")
        elif response.meetingId not in tokens[response.corpId]:
            raise HTTPException(status_code=404, detail='meetingId에 해당하는 데이터가 존재하지 않습니다.')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"error: {e}")

class AnalyzeTopic(BaseModel):
    corpId: int
    meetingId: int
    topicId: int
    questionId: int
    
@app.post("analyze-topic")
async def analyze_topic(response: AnalyzeTopic):
    topic_model = TopicModel(tokens[response.corpId][response.meetingId], response.corpId, response.meetingId)
    json_result = topic_model.make_lda_json()
    
    return {"topic_result":json_result}


# 임베딩 분석 엔드포인트 , 각 답변에 대해 거리
class AnalyzeEmbedding(BaseModel):
    corpId: int
    meetingId: int
    questionId: int
    
@app.post("analyze-embedding")
async def analyze_embedding(response: AnalyzeEmbedding):
    pass

class GenerateWordcloud(BaseModel):
    corpId: int
    meetingId: int
    questionId: int
    
@app.post("generate-wordcloud")
async def generate_wordcloud(response: GenerateWordcloud):
    pass

class AnalyzeSentiment(BaseModel):
    corpId: int
    meetingId: int
    questionId: int
@app.post("analyze-sentiment")
async def analyze_sentiment(response: AnalyzeSentiment):
    pass