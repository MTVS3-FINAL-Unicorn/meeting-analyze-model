import os

import torch

# from dotenv import load_dotenv
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

from AnalyzeMeeting.llm_model import LLMModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
llama_embedding_model = OllamaEmbeddings(model='llama3.2')
openai_embedding_model = OpenAIEmbeddings()

summary_prompt = '''
                        당신은 도움이 되는 텍스트분석전문가입니다.
                        답변에는 요약한 내용만을 제시합니다.
                    '''
summary_user_prompt_template = PromptTemplate.from_template('''
                                                                다음 텍스트를 요약하되, 주요 포인트에 중점을 두고 아래와 같은 형식으로 요약하세요:
                                                                요약은 간단하게 나타내고 주요 포인트에 집중해주세요
                                                                용량 과 단위도 나타내세요

                                                                1. 요약:
                                                                2. 주요 포인트:
                                                                    - 포인트 1
                                                                    - 포인트 2
                                                                    - 포인트 3

                                                                요약할 내용이 없다면, 요약을 생략해도 됩니다.
                                                                다음은 요약할 텍스트입니다:
                                                                {text}                                         
                                                                ''')

summary_model = LLMModel(ChatOpenAI(model='gpt-4o-mini'), None, device, summary_prompt, summary_user_prompt_template)