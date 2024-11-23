from typing import Dict, List

import pandas as pd

from AnalyzeMeeting.text_organize import remove_stopwords, tokenize_text


class MeetingScript():
    def __init__(self, corp_id: int, meeting_id: int):
        self.corp_id = corp_id
        self.meeting_id = meeting_id
        self.questions = pd.DataFrame(columns=['question_id', 'question_text'])
        self.data = pd.DataFrame(columns=['question_id', 'user_id', 'answer', 'tokens'])

    def add_question(self, question_id: str, question_text: str):
        # 질문 중복 확인 후 추가
        if not self.questions[self.questions['question_id'] == question_id].empty:
            print(f"Question '{question_id}' already exists.")
        else:
            new_question = pd.DataFrame([{
                'question_id': question_id,
                'question_text': question_text,
            }])
            self.questions = pd.concat([self.questions, new_question], ignore_index=True)

    def add_answer(self, question_id: str, answer: str, user_id: int):
        tokenized_answer = remove_stopwords(tokenize_text(answer))
        
        new_answer = pd.DataFrame([{
            'question_id': question_id,
            'user_id': user_id,
            'answer': answer,
            'tokens': tokenized_answer,
        }])
        self.data = pd.concat([self.data, new_answer], ignore_index=True)

    def get_question_text(self, question_id: str) -> str:
        # 특정 question_id에 대한 question_text 반환
        question_row = self.questions[self.questions['question_id'] == question_id]['question_text']
        return question_row.iloc[0] if not question_row.empty else ""

    def get_answers(self, question_id: str) -> pd.DataFrame:
        # 특정 question_id에 대한 모든 답변을 포함하는 DataFrame 반환
        return self.data[(self.data['question_id'] == question_id) & self.data['answer'].notna()][['user_id', 'answer']]

    def get_tokens(self, question_id: str) -> pd.DataFrame:
        # 특정 question_id에 대한 모든 토큰화된 답변 반환
        return self.data[(self.data['question_id'] == question_id) & self.data['tokens'].notna()][['user_id', 'tokens']]

    def get_all_data(self) -> pd.DataFrame:
        # 질문과 답변을 병합하여 반환 시 corp_id와 meeting_id 추가
        all_data = pd.merge(self.data, self.questions, on="question_id", how="left")
        all_data['corp_id'] = self.corp_id
        all_data['meeting_id'] = self.meeting_id
        return all_data
    
    def to_script_format(self) -> List[Dict[str, List[str]]]:
        # 전체 데이터를 질문과 답변 구조로 변환
        script = []
        questions = self.questions['question_id'].unique()
        
        for question_id in questions:
            question_text = self.get_question_text(question_id)
            answers = self.get_answers(question_id)['answer'].tolist()
            script.append({
                "question": question_text,
                "answer": answers
            })
        
        return script