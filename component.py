from long_memory.component import WeaviateLongMemory
from short_memory.component import WeaviateShortMemory
from prompt import compress_conversation

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import json
import os
import re

load_dotenv()

class MemLLM():
    def __init__(self, short_memory:WeaviateShortMemory=None, long_memory:WeaviateLongMemory=None, user:str="deafult", context_limit:int=1000):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.context_limit = context_limit
        if not short_memory:
            self.short_memory = WeaviateShortMemory(user=user)
        else:
            self.short_memory = short_memory
        if not long_memory:
            self.long_memory = WeaviateLongMemory(user=user)
        else:
            self.long_memory = long_memory
        self.msg_queue = []
        self.pre_msg_queue = []
        self.prompt = ""
        
    # 運行
    def run(slef):
        # 運行
        # 輸入
        # 檢查輸入大小
        # 進行搜尋
        # 組裝 prompt
        # 回答
        # 檢查整體 prompt 大小
        # 重複
        # session end
        # 清空 pre msg, msg queue
        # 將 short memory 紀錄導入 long memory
        return
    
    def _llm_create(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return completion.choices[0].message.content
    # 壓縮到 short memory
    def _extract_to_short_memory(self):
        print("Extracting chat log to short memory...")
        # 導入到 short memory
        self.short_memory.add_chatlogs(self.msg_queue)
        # 將對話紀錄壓縮成一句話
        json_res = self._llm_create(compress_conversation.format(chat_logs=self.msg_queue))
        res = json.loads(re.search(r"```json(.*?)```", json_res, re.DOTALL).group(1).strip())
        # 清空對話紀錄
        self.msg_queue.clear()
        # 將壓縮句子加入
        self.pre_msg_queue.append(res['text'])
        print("Done.")
    # 將 short memory 導入至 long memory
    def _save_to_long_memory(self):
        data = self.short_memory.dump_memory(True)
        self.long_memory.add_chat_logs(data)
        print('Short memory dump to Long memory.')
    # 檢查 input limit
    def _check_input_limit(self, input:str):
        token_number = self._count_token(input)
        # 目前限制在不超過一半的 context limit 大小
        if token_number/2 > self.context_limit:
            print(f"Input too large, input token:{token_number}")
            return True
        else:
            return False
    # 處理對話紀錄的 context limit 大小
    def _handle_context_limit(self):
        token_number = self._count_token(str(self.msg_queue))
        if token_number >= self.context_limit:
            self._extract_to_short_memory()
            return
        else:
            return
    def _count_token(self, text:str, model='gpt-4o-mini'):
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    # 從 memory 中找尋記憶
    def _get_memory(self, query:str):
        long_mem_result = self.long_memory.get_memory(query)
        short_mem_result = self.short_memory.get_memory(query)
        pass