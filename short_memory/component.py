from short_memory.schema import LOG_SCHEMA
from short_memory.prompt import generate_keyword, hyde_generated

from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
import weaviate

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI

import requests
import copy
import os

load_dotenv()

class Base(ABC):
    # 添加
    @abstractmethod
    def add_chatlogs():
        pass
    # 取得
    @abstractmethod
    def get_memory():
        pass
    
class WeaviateShortMemory(Base):
    def __init__(self, weaviate_url="127.0.0.1", port=8080, user="deafult", model="gpt-4o-mini", ollama_url="http://localhost:11434/api/generate"):
        self.client = weaviate.connect_to_local(weaviate_url, port)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.ollama_url = ollama_url
        self.user = user
        self.chatlog_class_name = f"{user[0].upper()+user[1:]}_short_memory_chatlog"
        self._memory_exists()
        self.chatlog_class = self.client.collections.get(self.chatlog_class_name)
    
    def _memory_exists(self):
        if self._class_exists(self.chatlog_class_name):
            print(f"Detect existed {self.user} user short memory space, loading...")
        else:
            print("Detect empty short memory, create memory space...")
            chatlog_schema = copy.deepcopy(LOG_SCHEMA)
            chatlog_schema["class"] = self.chatlog_class_name
            self._create_class(chatlog_schema)
    
    # 檢查這個 class 存不存在    
    def _class_exists(self, class_name):
        return self.client.collections.exists(class_name)
    
    # 創建 class schema
    def _create_class(self, class_schema):
        self.client.collections.create_from_dict(class_schema)
        
    def _llm_create(self, prompt):
        gpt_family = ["gpt-4o-mini", "gpt-4o"]
        ollama_family = ["llama3.3", "llama3.1", "llama3.1:405b", "gemma2:27b", "qwen2.5:32b"]
        
        if self.model in gpt_family:
            messages = [{"role": "user", "content": prompt}]
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return completion.choices[0].message.content
        elif self.model in ollama_family:
            url = self.ollama_url
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream":False
            }

            response = requests.post(url, json=data)
            return response.json()['response']
    
    def add_chatlogs(self, chatlogs:list):
        """add chatlogs to short memory

        Args:
            chatlogs (list): a list of chatlog, chatlog needs be following format
                ```
                {
                    "user":"Hi, how are you today?",
                    "assistant":"fine, how about you?",
                    "time": "2024-10-31T04:18:00Z" (Optional)
                }
                ```
        """
        for chatlog in chatlogs:
            self.add_chatlog(chatlog)
    
    def add_chatlog(self, chatlog:dict):
        """add a chatlog to short memory

        Args:
            chatlog (dict): needs be following format
                ```
                {
                    "user":"Hi, how are you today?",
                    "assistant":"fine, how about you?",
                    "time": "2024-10-31T04:18:00Z" (Optional)
                }
                ```
        """
        if "assistant" and "user" not in chatlog:
            print(f"chat log format incorrect, should contain 'assistant' and 'user'\nChat log:{chatlog}")
        else:
            # save to weaviate
            data = {
                "text":f"user:{chatlog['user']}, assistant:{chatlog['assistant']}",
                "time":chatlog.get('time')
            }
            self._insert_weaviate(data)
    
    def get_memory(self, query:str, method="similarity", k=5, sort=True):
        """search relvant memory

        Args:
            query (str): the query you want to search.
            method (str): [similarity/keyword/BM25/HyDE] Search method, 'similarity' for vector-based similarity search,
                          'keyword' for substring matching, 'BM25' use bm25 algorithm, 'HyDE' use HyDE method then similarity search. Defaults to "similarity".
            k (int): number of relevant memory.
            sort (bool): If true, the retrieve chat log will sort with ascending, Defaults to True.
        """
        if method=="similarity":
            response = self.chatlog_class.query.near_text(
                query=query,
                limit=k,
                return_properties=["text", "time"],
                return_metadata=MetadataQuery(distance=True)
            )
        elif method=="keyword":
            # generate keywords by llm
            keyword_prompt = generate_keyword.format(query=query)
            keyword_list = eval(self._llm_create(keyword_prompt))
            print(f'Search by keywords:{keyword_list}')
            response = self.chatlog_class.query.fetch_objects(
                filters=Filter.by_property("text").contains_any(keyword_list),
                limit=k
            )
        elif method=="BM25":
            response = self.chatlog_class.query.bm25(
                query=query,
                return_metadata=MetadataQuery(score=True),
                limit=k
            )
        elif method=="HyDE":
            hyde_prompt = hyde_generated.format(query=query)
            hyde_query = self._llm_create(hyde_prompt)
            response = self.chatlog_class.query.near_text(
                query=hyde_query,
                limit=k,
                return_properties=["text", "time"],
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            print(f"Unknow method, only can be [similarity/keyword/BM25/HyDE]. Your method:{method}")
            return
        retrieve_memory = []
        for data in response.objects:
            retrieve_memory.append({"text":data.properties['text'], "time":data.properties['time']})
        if sort:
            retrieve_memory = sorted(retrieve_memory, key=lambda x: x["time"])
        retrieve_memory = [{"text": item["text"], "time": item["time"].strftime("%m/%d %H:%M")} for item in retrieve_memory]
        res = {
            "retrieve_memory":retrieve_memory
        }
        return res
    
    def _insert_weaviate(self, data, vector=None):
        data_id = self.chatlog_class.data.insert(
            properties=data,
            vector=vector
        )
        return data_id
    
    def show_memory(self):
        data = []
        for item in self.chatlog_class.iterator():
            data.append({
                "id":str(item.uuid),
                "text":item.properties['text'],
                "time":item.properties['time']
            })
        data = sorted(data, key=lambda x: x["time"])
        data = [{"text": item["text"], "time": item["time"].strftime("%Y-%m-%dT %H:%M")} for item in data]
        for log in data:
            print(log)
            
    def dump_memory(self, clear=True):
        data = []
        for item in self.chatlog_class.iterator():
            data.append({
                "text":item.properties['text'],
                "time":item.properties['time']
            })
        data = sorted(data, key=lambda x: x["time"])
        print("Dump short memory success.")
        if clear:
            print("clean the short memory...")
            self.client.collections.delete(self.chatlog_class_name)
            self._memory_exists()
        return data
    
    def del_memory(self):
        self.client.collections.delete(self.chatlog_class_name)
        self._memory_exists()