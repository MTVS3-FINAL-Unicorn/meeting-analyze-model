import re

from konlpy.tag import Okt


def tokenize_text(text, token_len=2):
    text = re.sub(r'[^ㄱ-ㅣ가-힣\s]', '', text)
    okt = Okt()
    okt_data = okt.pos(text)
    word_list = [word for word, pos in okt_data if (pos in ['Noun', 'Adjective']) and (len(word) >= token_len) ]
    sentence = ' '.join(word_list)
    return sentence

def remove_stopwords(text):
    tokens = text.split(' ')
    stopwords = [
    "이", "그", "저", "을", "를", "은", "는", "이다", "있다", "없다", "에", "에서", 
    "와", "과", "로", "으로", "의", "도", "에게", "한테", "그리고", "그러나", 
    "그래서", "하지만", "또한", "즉", "나", "너", "우리", "당신", "저희", "그", 
    "그녀", "이것", "그것", "저것", "어떤", "이러한", "그런", "아주", "매우", 
    "보다", "보다도", "그냥", "무조건", "하지만", "그러나"
    ]
    meaningful_words = [w for w in tokens if not w in stopwords]
    return meaningful_words