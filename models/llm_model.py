import torch
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

class LLMModel():
    def __init__(self, model, prompt, user_prompt_template):
        self.prompt = prompt
        self.user_prompt_template = user_prompt_template
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def exec(self, text):
        llm_prompt_result = self.user_prompt_template.format(text=text)
        messages = [SystemMessage(content=self.prompt) ,
                    HumanMessage(content=llm_prompt_result)]

        return self.model(messages).content
