import json
import aiofiles
import os
import glob

def aggregate_meeting_tokens(corpId, meetingId, tokens):
    """특정 corpId와 meetingId의 모든 토큰을 합친 리스트를 반환합니다."""
    if corpId not in tokens or meetingId not in tokens[corpId]:
        return []

    aggregated_tokens = []

    for _, users in tokens[corpId][meetingId].items():
        for _, user_tokens in users.items():
            aggregated_tokens.extend(user_tokens)

    return aggregated_tokens

def aggregate_question_tokens(corpId, meetingId, questionId, tokens):
    """특정 corpId, meetingId, questionId의 모든 토큰을 합친 리스트를 반환합니다."""
    # corpId와 meetingId가 존재하는지 확인
    if corpId not in tokens or meetingId not in tokens[corpId] or questionId not in tokens[corpId][meetingId]:
        return []

    # 모든 토큰을 저장할 리스트 초기화
    aggregated_tokens = []

    # meetingId 내부의 모든 questionId와 userId의 토큰을 합침
    for _, user_token in tokens[corpId][meetingId][questionId].items():
        aggregated_tokens.extend(user_token)

    return aggregated_tokens

async def preload_all_tokens():
    """서버 시작 시 모든 JSON 파일을 메모리에 로드."""
    cache = {}
    
    files = glob.glob('./data/tokens_*.json')
    for file_path in files:
        corpId, meetingId = map(int, file_path.split('_')[1:3])
        async with aiofiles.open(file_path, 'r') as f:
            data = json.loads(await f.read())
            if corpId not in cache:
                cache[corpId] = {}
            cache[corpId][meetingId] = data
    
    return cache

async def save_tokens_to_json(corpId, meetingId, questionId, tokens):
    """corpId와 meetingId별로 분석된 토큰을 JSON 파일로 저장"""
    file_path = f"./data/tokens_{corpId}_{meetingId}.json"
    # 기존 파일이 있으면 읽어서 병합
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, 'r') as f:
            existing_data = json.loads(await f.read())
            # 기존 데이터와 현재 tokens 데이터를 병합
            for questionId, users in tokens[corpId][meetingId].items():
                if questionId not in existing_data:
                    existing_data[questionId] = {}
                for userId, user_tokens in users.items():
                    if userId in existing_data[questionId]:
                        existing_data[questionId][userId].extend(user_tokens)
                    else:
                        existing_data[questionId][userId] = user_tokens
            data_to_save = existing_data
    else:
        data_to_save = tokens[corpId][meetingId]

    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(data_to_save, ensure_ascii=False))