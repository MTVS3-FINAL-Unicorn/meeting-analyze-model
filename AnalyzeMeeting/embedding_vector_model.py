import os
import random
import socket
import subprocess

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from tensorboard.plugins import projector

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TODO: 하나의 서버에서 여러 개의 tensorboard 열 수 있게 하기
    # 1. log 폴더 설정
    # 2. port 설정
    
class EmbeddingVectorAnalyzer():
    def __init__(self, corp_id, meeting_id, embedding_model, question_id=None):
        self.token_list = None
        self.corp_id = corp_id
        self.meeting_id = meeting_id
        self.question_id = question_id
        self.embedding_model = embedding_model
        self.embedding_vectors = None
        self.embedding_vectors_array = None

        if self.question_id:
            self.LOG_DIR = f"logs/{self.corp_id}/{self.meeting_id}/{self.question_id}"
        else:
            self.LOG_DIR = f"logs/{self.corp_id}/{self.meeting_id}/all"
    
    def make_token_embeddings(self, token_list):
        self.token_list = token_list
        self.embedding_vectors = self.embedding_model.embed_documents(token_list)
        self.embedding_vectors_array = np.array(self.embedding_vectors)
    
    def make_sentence_embeddings(self, responses):
        self.token_list = []
        for response in responses:
            self.token_list.append(response['answer'])
        self.embedding_vectors = self.embedding_model.embed_documents(self.token_list)
        self.embedding_vectors_array = np.array(self.embedding_vectors)
    
    def make_checkpoint(self, log_dir=None):
        # 로그 저장 경로 설정
        if log_dir:
            self.LOG_DIR = log_dir
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # 메타데이터 파일 저장 (각 벡터에 해당하는 문장을 포함)
        metadata_path = os.path.join(self.LOG_DIR, 'metadata.tsv')
        with open(metadata_path, 'w', encoding='utf-8-sig') as f:
            for token in self.token_list:
                f.write(f"{token}\n")

        # TensorFlow 변수로 임베딩 데이터를 저장
        embedding_var = tf.Variable(self.embedding_vectors, name='token_embeddings')

        # 설정 파일 생성
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'

        # 설정 저장
        projector.visualize_embeddings(self.LOG_DIR, config)

        # 임베딩 체크포인트 저장
        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint.save(os.path.join(self.LOG_DIR, "embedding.ckpt"))
        
        print(f"TensorBoard 데이터와 로그는 {self.LOG_DIR} 디렉토리에 저장되었습니다.")
        
    def run_tensorboard(self, host, port):
        try:
            self.make_checkpoint()
            # subprocess를 통해 tensorboard 실행
            subprocess.Popen(['tensorboard', '--logdir', self.LOG_DIR, '--host', host, '--port', port])
            print(f"TensorBoard가 {self.LOG_DIR}에서 실행 중입니다. http://{host}:{port} 에서 확인하세요.")
        except Exception as e:
            print(f"TensorBoard 실행 중 오류가 발생했습니다: {e}")
            
    def find_available_port(self):
        while True:
            # 1024 ~ 65535 사이의 포트 번호 생성
            port = random.randint(1024, 65535)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # 포트가 사용 중인지 확인
                result = sock.connect_ex(('localhost', port))
                if result != 0:
                    # 사용 가능한 포트일 경우 반환
                    return str(port)
                
    def exce(self, responses, host, port):
        self.make_sentence_embeddings(responses)
        self.run_tensorboard(host, port)
        

if __name__ == '__main__':
    tokens = ['분홍색', '테마', '시즌', '정말', '부드러운', '이미지', '같아요', '고객', '좋아할', '같네요', '개인', '초록색', '브랜드', '정체', '있다고', '생각', '분홍색', '흔한', '느낌', '같아요', '차라리', '이번', '초록색', '어떨까', '초록색', '자연', '이미지', '강해서', '요즘', '트렌드', '같아요', '분홍색', '상큼', '여성', '이미지', '젊은', '인기', '많을', '초록색', '테마', '시원하고', '깨끗한', '이미지', '있을', '같아요', '분홍색', '정말', '제품', '꽃잎', '성분', '생각', '브랜드', '이미지', '관성', '유지', '초록색', '적합할', '같습니다', '분홍색', '화장품', '패키지', '같아요', '초록색', '환경', '메시지', '강화할', '있어서', '좋을', '같아요', '분홍색', '화사한', '느낌', '부담', '사용', '있을', '같아요', '초록색', '브랜드', '생각', '자연', '관련', '브랜드', '라면', '분홍색', '따뜻하고', '부드러운', '이미지', '주기', '때문', '소비자', '긍정', '반응', '있을', '같아요', '초록색', '자연스러운', '이미지', '강화하는', '좋을', '같아요', '친환경', '느낌', '강해요', '분홍색', '꽃잎', '영감', '만큼', '제품', '컬러', '생각', '초록색', '시각', '강렬한', '인상', '있어서', '좋을', '같아요', '분홍색', '고급스러운', '느낌', '있다고', '생각', '초록색', '브랜드', '자연', '이미지', '더욱', '있을', '같습니다', '분홍색', '소비자', '편안한', '느낌', '있어서', '좋다고', '생각', '초록색', '세련된', '이미지', '있을', '같아요']
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    embedding_vector_analyzer = EmbeddingVectorAnlalyzer(1, 1, embedding_model)
    embedding_vector_analyzer.make_token_embeddings(tokens)
    port = embedding_vector_analyzer.find_available_port()
    embedding_vector_analyzer.run_tensorboard('127.0.0.1', port)
