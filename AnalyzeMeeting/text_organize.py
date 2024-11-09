import re
from typing import Union, List
from konlpy.tag import Okt


def tokenize_text(text: Union[str, List[str]], token_len: int = 2) -> list:
    """ 한국어 문장을 konlpy okt로 토큰화하여 공백으로 합쳐진 문자열을 반환합니다.

    Args:
        text (_type_):  str | list
        token_len (int, optional):  int, Defaults to 2.

    Returns:
        _type_: str
    """
    okt = Okt()

    # 입력이 리스트일 경우
    if isinstance(text, list):
        tokens = [token for sentence in text for token in tokenize_text(sentence, token_len=token_len)]
        return tokens

    # 입력이 문자열일 경우
    text = re.sub(r'[^ㄱ-ㅣ가-힣\s]', '', text)
    okt_data = okt.pos(text)
    word_list = [word for word, pos in okt_data if (pos in ['Noun', 'Adjective']) and (len(word) >= token_len)]
    return word_list

def remove_stopwords(tokens:List[str]) -> list:
    stopwords = [
    "이", "그", "저", "을", "를", "은", "는", "이다", "있다", "없다", "에", "에서", 
    "와", "과", "로", "으로", "의", "도", "에게", "한테", "그리고", "그러나", 
    "그래서", "하지만", "또한", "즉", "나", "너", "우리", "당신", "저희", "그", 
    "그녀", "이것", "그것", "저것", "어떤", "이러한", "그런", "아주", "매우", 
    "보다", "보다도", "그냥", "무조건", "하지만", "그러나"
    ]
    meaningful_words = [w for w in tokens if not w in stopwords]
    return meaningful_words