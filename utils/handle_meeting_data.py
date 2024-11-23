def aggregate_meeting_tokens(corp_id, meeting_id, tokens):
    """특정 corp_id와 meeting_id의 모든 토큰을 합친 리스트를 반환합니다."""
    if corp_id not in tokens or meeting_id not in tokens[corp_id]:
        return []

    aggregated_tokens = []

    for _, users in tokens[corp_id][meeting_id].items():
        for _, user_response in users.items():
            aggregated_tokens.extend(user_response['tokens'])

    return aggregated_tokens

def aggregate_question_tokens(responses):
    """특정 corp_id, meeting_id, question_id의 모든 토큰을 합친 리스트를 반환합니다."""
    # 모든 토큰을 저장할 리스트 초기화
    aggregated_tokens = []
    for response in responses:
        aggregated_tokens.append(response['answer'])

    return aggregated_tokens

def aggregate_question_sentences(corp_id, meeting_id, question_id, tokens):
    """특정 corp_id, meeting_id, question_id의 모든 토큰을 합친 리스트를 반환합니다."""
    # corp_id와 meeting_id가 존재하는지 확인
    if corp_id not in tokens or meeting_id not in tokens[corp_id] or question_id not in tokens[corp_id][meeting_id]:
        return []

    # 모든 토큰을 저장할 리스트 초기화
    aggregated_sentences = []

    # meeting_id 내부의 모든 question_id와 user_id의 토큰을 합침
    for _, user_response in tokens[corp_id][meeting_id][question_id].items():
        aggregated_sentences.append(user_response['answer'])

    return aggregated_sentences
