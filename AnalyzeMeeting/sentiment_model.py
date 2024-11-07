from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# load model
tokenizer = AutoTokenizer.from_pretrained("jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("jaehyeong/koelectra-base-v3-generalized-sentiment-analysis")
sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    # target reviews
    review_list = ['분홍색', '테마', '시즌', '정말', '부드러운', '이미지', '같아요', '고객', '좋아할', '같네요', '개인', '초록색', '브랜드', '정체', '있다고', '생각', '분홍색', '흔한', '느낌', '같아요', '차라리', '이번', '초록색', '어떨까', '초록색', '자연', '이미지', '강해서', '요즘', '트렌드', '같아요', '분홍색', '상큼', '여성', '이미지', '젊은', '인기', '많을', '초록색', '테마', '시원하고', '깨끗한', '이미지', '있을', '같아요', '분홍색', '정말', '제품', '꽃잎', '성분', '생각', '브랜드', '이미지', '관성', '유지', '초록색', '적합할', '같습니다', '분홍색', '화장품', '패키지', '같아요', '초록색', '환경', '메시지', '강화할', '있어서', '좋을', '같아요', '분홍색', '화사한', '느낌', '부담', '사용', '있을', '같아요', '초록색', '브랜드', '생각', '자연', '관련', '브랜드', '라면', '분홍색', '따뜻하고', '부드러운', '이미지', '주기', '때문', '소비자', '긍정', '반응', '있을', '같아요', '초록색', '자연스러운', '이미지', '강화하는', '좋을', '같아요', '친환경', '느낌', '강해요', '분홍색', '꽃잎', '영감', '만큼', '제품', '컬러', '생각', '초록색', '시각', '강렬한', '인상', '있어서', '좋을', '같아요', '분홍색', '고급스러운', '느낌', '있다고', '생각', '초록색', '브랜드', '자연', '이미지', '더욱', '있을', '같습니다', '분홍색', '소비자', '편안한', '느낌', '있어서', '좋다고', '생각', '초록색', '세련된', '이미지', '있을', '같아요']
    # predict
    for idx, review in enumerate(review_list):
        pred = sentiment_classifier(review)
        print(f'{review}\n>> {pred[0]}')