import re
from collections import Counter, defaultdict

from utils.handle_text import remove_stopwords, tokenize_text


class SentimentAnalyzer:
    def __init__(self, sentence_model, token_model):
        # 모델 초기화
        self.sentence_model = sentence_model
        self.token_model = token_model

    def analyze_sentence_sentiment(self, responses, most_k=5):
        result = defaultdict(dict)
        all_tokens = []
        for response in responses:
            answer = response['answer']
            score = self.extract_float_from_response(self.sentence_model.exec(answer)) 
            result[answer]['sentiment_score'] = score
            if 'tokens' not in result[answer]:
                result[answer]['tokens'] = []
            tokens = remove_stopwords(tokenize_text(answer))
            result[answer]['tokens'].extend(tokens)
            all_tokens.extend(tokens)
        
        token_counter = Counter(all_tokens)
        most_common_token = token_counter.most_common(most_k)
        return result, token_counter, most_common_token

    def analyze_token_sentiment(self, tokens):
        token_counter = Counter(tokens)
        unique_tokens = list(token_counter.keys())

        token_scores = self.token_model(unique_tokens)

        result = {
            token: {
                'freq': token_counter[token],
                'sentiment_score': score['score']
            }
            for token, score in zip(unique_tokens, token_scores)
        }

        return result

    def extract_float_from_response(self, response):
        # 정규식을 이용해 숫자 (소수점 포함) 추출
        match = re.search(r'\d+\.\d+', response)
        if match:
            return float(match.group())  # 매칭된 숫자를 float으로 변환
        else:
            raise ValueError("숫자를 찾을 수 없습니다.")  # 숫자가 없을 경우 에러 처리

if __name__ == "__main__":
    # review_list = ['분홍색', '테마', '시즌', '정말', '부드러운', '이미지', '같아요', '고객', '좋아할', '같네요', '개인', '초록색', '브랜드', '정체', '있다고', '생각', '분홍색', '흔한', '느낌', '같아요', '차라리', '이번', '초록색', '어떨까', '초록색', '자연', '이미지', '강해서', '요즘', '트렌드', '같아요', '분홍색', '상큼', '여성', '이미지', '젊은', '인기', '많을', '초록색', '테마', '시원하고', '깨끗한', '이미지', '있을', '같아요', '분홍색', '정말', '제품', '꽃잎', '성분', '생각', '브랜드', '이미지', '관성', '유지', '초록색', '적합할', '같습니다', '분홍색', '화장품', '패키지', '같아요', '초록색', '환경', '메시지', '강화할', '있어서', '좋을', '같아요', '분홍색', '화사한', '느낌', '부담', '사용', '있을', '같아요', '초록색', '브랜드', '생각', '자연', '관련', '브랜드', '라면', '분홍색', '따뜻하고', '부드러운', '이미지', '주기', '때문', '소비자', '긍정', '반응', '있을', '같아요', '초록색', '자연스러운', '이미지', '강화하는', '좋을', '같아요', '친환경', '느낌', '강해요', '분홍색', '꽃잎', '영감', '만큼', '제품', '컬러', '생각', '초록색', '시각', '강렬한', '인상', '있어서', '좋을', '같아요', '분홍색', '고급스러운', '느낌', '있다고', '생각', '초록색', '브랜드', '자연', '이미지', '더욱', '있을', '같습니다', '분홍색', '소비자', '편안한', '느낌', '있어서', '좋다고', '생각', '초록색', '세련된', '이미지', '있을', '같아요']
    # print(analyze_token_sentiment(review_list))
    
    # sentence
    responses = [
        {
            "indivId": 1,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "저는 분홍색 패키지가 더 예쁜 것 같아요. 화사한 느낌이 좋아요.",
            "sentiment_score": 0.85
        },
        {
            "indivId": 2,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "초록색 패키지가 더 자연스럽고 신선한 느낌이 들어요. 친환경적인 이미지를 줘서 좋아요.",
            "sentiment_score": 0.90
        },
        {
            "indivId": 3,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "분홍색 패키지는 너무 흔한 느낌이 들어서 별로예요. 차라리 초록색이 나은 것 같아요.",
            "sentiment_score": 0.25
        },
        {
            "indivId": 4,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "초록색 패키지가 더 고급스러운 느낌이에요. 요즘 트렌드와도 잘 맞는 것 같아요.",
            "sentiment_score": 0.80
        },
        {
            "indivId": 5,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "저는 분홍색 패키지가 더 여성스럽고 화사해서 끌려요. 밝은 느낌이 좋습니다.",
            "sentiment_score": 0.75
        },
        {
            "indivId": 6,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "초록색이 자연스럽고 깔끔해서 더 좋은 것 같아요. 환경을 생각하는 브랜드 같아서 인상적입니다.",
            "sentiment_score": 0.88
        },
        {
            "indivId": 7,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "분홍색 패키지는 너무 부담스러워요. 너무 밝아서 제 취향이 아니에요.",
            "sentiment_score": 0.30
        },
        {
            "indivId": 8,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "초록색이 더 편안한 느낌이 들어서 좋아요. 요즘 자연스러운 색상이 유행인 것 같아요.",
            "sentiment_score": 0.83
        },
        {
            "indivId": 9,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "분홍색이 좀 더 독특한 느낌이라서 신선하게 다가옵니다. 초록색은 너무 흔해요.",
            "sentiment_score": 0.65
        },
        {
            "indivId": 10,
            "meetingId": 3,
            "questionType": "서술형",
            "questionId": 1,
            "answer": "초록색 패키지가 더 안정감을 주는 느낌이에요. 지속 가능성을 생각하는 느낌이 들어서 더 좋아요.",
            "sentiment_score": 0.87
        }
    ]
    sent_model = SentimentAnalyzer()
    print(sent_model.analyze_sentence_sentiment(responses))