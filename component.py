from long_memory.component import WeaviateLongMemory
from short_memory.component import WeaviateShortMemory
from prompt import compress_conversation_prompt, generate_answer_prompt

from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
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
        self.pre_msg = ""
        self.current_prompt = ""
        
    # 運行
    def run(self, end_clear=False):
        # 運行
        print("\033[33mStarting chat, type 'exit' to exit\033[0m")
        while True:
            # 輸入
            user_query = input(":")
            if user_query=="exit":
                break
            # 檢查輸入大小
            if self._check_input_limit(user_query):
                print("\033[33mPlease try again.\033[0m")
            else:
                # 進行搜尋
                long_mem_result, short_mem_result = self.get_memory(user_query)
                # 組裝 prompt
                generate_prompt = generate_answer_prompt.format(
                    chat_history=f"{self.pre_msg}, {self.msg_queue}",
                    short_relevant_memory=str(short_mem_result),
                    long_relevant_memory=str(long_mem_result),
                    user_message=user_query)
                self.current_prompt = generate_prompt
                # 回答
                res = self._llm_create(generate_prompt)
                print(f"user:{user_query}")
                print(f"assistant:{res}")
                # 儲存對話
                msg = {
                    "user":user_query,
                    "assistant":res,
                    "time":datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                self.msg_queue.append(msg)
                # 處理 prompt 大小
                self._handle_context_limit()
                # 重複
        # session end
        if end_clear:
            # 清空 msg queue
            self.msg_queue.clear()
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
        print("\033[34mExtracting chat log to short memory...\033[0m")
        # 導入到 short memory
        self.short_memory.add_chatlogs(self.msg_queue)
        # 將對話紀錄壓縮成一句話
        res = self._llm_create(compress_conversation_prompt.format(chat_logs=self.msg_queue))
        # 清空對話紀錄
        self.msg_queue.clear()
        # 將壓縮句子加入
        self.pre_msg = res
        print("\033[34mDone.\033[0m")
    # 將 short memory 導入至 long memory
    def save_to_long_memory(self, clear_short_memory=True):
        data = self.short_memory.dump_memory(clear_short_memory)
        self.long_memory.add_chat_logs(data)
        print('\033[34mShort memory dump to Long memory done.\033[0m')
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
    def get_memory(self, query:str):
        """retrieve relevant memory from long memory & short memory

        Args:
            query (str): user query

        Returns:
            long_mem_result: the result find in long memory
            short_mem_result: the result find in short memory
        """
        print('\033[31mSearch long memory, short memory\033[0m')
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.long_memory.get_memory, query)
            future2 = executor.submit(self.short_memory.get_memory, query)
        long_mem_result = future1.result()
        short_mem_result = future2.result()
        return long_mem_result, short_mem_result