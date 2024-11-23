from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

class LLMModel():
    def __init__(self, model, tokenizer, device, prompt, user_prompt_template):
        self.prompt = prompt
        self.user_prompt_template = user_prompt_template
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def strip_noise_from_text(self, text):
        return self.model.invoke(text + ' 이 내용에서, 의미가 없는 문자열을 제거한 뒤, 온전히 그 내용만 돌려줘').content
    
    
    def exec(self, text):
        llm_prompt_result = self.user_prompt_template.format(text=text)
        messages = [SystemMessage(content=self.prompt) ,
                    HumanMessage(content=llm_prompt_result)]

        return self.model(messages).content
