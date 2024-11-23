from typing import Dict, List

import pandas as pd

from utils.handle_text import remove_stopwords, tokenize_text


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

    def get_all_tokens(self) -> List[str]:
        """
            모든 질문의 답변에서 토큰을 추출하여 하나의 리스트로 반환합니다.
        """
        all_tokens = self.data['tokens'].dropna().explode().tolist()
        return all_tokens
    
    def get_all_data(self) -> pd.DataFrame:
        # 질문과 답변을 병합하여 반환 시 corp_id와 meeting_id 추가
        all_data = pd.merge(self.data, self.questions, on="question_id", how="left")
        all_data['corp_id'] = self.corp_id
        all_data['meeting_id'] = self.meeting_id
        return all_data
    
    def to_script_format(self) -> List[Dict[str, List[Dict[str, str]]]]:
        # 전체 데이터를 질문과 답변 구조로 변환
        script = []
        questions = self.questions['question_id'].unique()
        
        for question_id in questions:
            question_text = self.get_question_text(question_id)
            answers_df = self.get_answers(question_id)
            
            # 각 답변에 대해 user_id와 answer를 포함하는 리스트 생성
            answers = answers_df.to_dict(orient="records")
            
            script.append({
                "question": question_text,
                "answers": answers,
            })
        
        return script

    
if __name__ == "__main__":
    meeting_script = MeetingScript(corp_id=1, meeting_id=1)
    meeting_script.add_question(question_id=1, question_text="가장 좋아하는 음식은?")
    meeting_script.add_answer(question_id=1, answer="저는 매운 음식을 정말 좋아해요! 저는 특히 떡볶이를 자주 먹습니다. 매콤달콤한 양념이 입안에서 맴돌고, 쫄깃한 떡과 어우러져 매일 먹어도 질리지 않는 음식 중 하나입니다. 때로는 라면사리를 넣어 함께 먹기도 하고, 어묵과 계란을 추가하면 더욱 풍성한 맛을 느낄 수 있어요.", user_id=1)
    # print(meeting_script.to_script_format())
    print(meeting_script.get_all_tokens())