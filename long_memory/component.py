from abc import ABC, abstractmethod
from prompt import rewrite_prompt
from schema import CLASS_SCHEMA
from dotenv import load_dotenv
from numpy.linalg import norm
from datetime import datetime
from openai import OpenAI
from tools import tools
import numpy as np
import weaviate
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Base(ABC):
    # 儲存
    @abstractmethod
    def add_group_memory():
        pass
    # 取得
    @abstractmethod
    def get_relavant_memory():
        pass
    # 更新
    @abstractmethod
    def update_group_memory():
        pass
    @abstractmethod
    def update_single_memory():
        pass
    # 刪除
    @abstractmethod
    def del_group_memory():
        pass
    @abstractmethod
    def del_single_memory():
        pass
    @abstractmethod
    def del_all_memory():
        pass

class LongMemory(Base):
    def __init__(self, embedding_model='openai'):
        self.children = []
        if embedding_model=='openai':
            self.embedding_model = client.embeddings
        else:
            print('Not support other embedding model now.')
    class Node:
        def __init__(self, text, time=None, vector=None, origin_text=None, embedding_model='openai'):
            self.text = text
            self.origin_text = origin_text
            self.children = []
            if time:
                self.time = time
            else:
                self.time = datetime.now().strftime('%Y-%m-%d')
            if vector:
                self.vector = vector
            else:
                if embedding_model=='openai':
                    self.embedding_model = client.embeddings
                else:
                    print('Not support other embedding model now.')
                self.vector = self._create_embedding_vector()
        def _create_embedding_vector(self):
            response = self.embedding_model.create(
                input=self.text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        def add_child(self, child):
            self.children.append(child)
            
    def add_group_memory(self, group:dict):
        # 將 group 轉成 node
        description = group["description"]
        group_node = self.Node(text=description)
        for child in group["child"]:
            child_text = child["text"]
            child_time = child.get("time")
            child_vector = child.get("vector")
            child_origin_text = child.get("origin_text")
            child_node = self.Node(text=child_text, time=child_time, vector=child_vector, origin_text=child_origin_text)
            group_node.add_child(child_node)
        # 找出向量相似度最高的 group，如果超過 threshold 就合併 group 並且對 group description 進行更新
        threshold = 0.9
        max_score = -1
        most_similar_child = None
        for child in self.children:
            score = np.dot(group_node.vector, child.vector) / (norm(group_node.vector) * norm(child.vector))
            if score > threshold:
                if score > max_score:
                    max_score = score
                    most_similar_child = child
        if most_similar_child:
            rewrite, rewrite_des = self.rewrite_merge_group(most_similar_child.text, group_node.text)
            if rewrite=="True":
                most_similar_child.text = rewrite_des
            most_similar_child.children.extend(group_node.children)
        else:
            self.children.append(group_node)
        return
    def rewrite_merge_group(description_1, description_2):
        messages = [{"role": "user", "content": rewrite_prompt.format(description_1=description_1, description_2=description_2)}]
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice={
                "type":"function",
                "function":{"name":"rewrite_description"},
            }
        )
        res = eval(completion.choices[0].message.tool_calls[0].function.arguments)
        return res["rewrite"], res.get("description")
    def get_relavant_memory(self, query:str, vector=None, k=5):
        if vector:
            print("Searching with provided vector")
        else:
            vector = self._create_embedding(query)
        # 計算最相近 vector 的 group
        threshold = 0.25
        max_score = -1
        most_similar_group = None
        for child in self.children:
            score = np.dot(vector, child.vector) / (norm(vector) * norm(child.vector))
            if score > threshold:
                if score > max_score:
                    max_score = score
                    most_similar_group = child
        # 如果找不到相關記憶 (分數都不夠) 返回空
        if not most_similar_group:
            return
        # 在 group 內計算相近 vector 的記憶並且排序
        memory_list = []
        for child in most_similar_group.children:
            score = np.dot(vector, child.vector) / (norm(vector) * norm(child.vector))
            memory_info = {
                "score":score,
                "text":child.text,
                "time":child.time,
                "origin_text":child.origin_text,
            }
            memory_list.append(memory_info)
        memory_list.sort(key=lambda x: x['score'], reverse=True)
        result = {
            "group_description":most_similar_group.text,
            "memory":memory_list[:k],
        }
        return result
    def update_group_memory():
        pass
    def update_single_memory():
        pass
    def del_group_memory():
        pass
    def del_single_memory():
        pass
    def del_all_memory():
        pass
    def export_memory(self):
        group_list = []
        for group in self.children:
            group_text = group.text
            child_list = []
            for child in group.children:
                child_dict = {
                    "text":child.text,
                    "time":child.time,
                    "vector":child.vector, 
                    "origin_text":child.origin_text,
                }
                child_list.append(child_dict)
            group_dict = {
                "description":group_text,
                "child":child_list
            }
            group_list.append(group_dict)
            
        with open('long_memory.json', 'w') as f:
            json.dump(group_list, f)
        print('Memory export as long_memory.json')
        return
    def import_memory(self, memory_path:str):
        with open(memory_path, 'r') as f:
            memory = json.load(f)
        print(f'Loading memory with {memory_path}')
        for group in memory:
            self.add_group_memory(group)
        return
    def _create_embedding(self, text):
        response = self.embedding_model.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

class WeaviateLongMemory(Base):
    def __init__(self, weaviate_url, class_name = "base"):
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name
        
        if self._class_exists():
            print("Detect existed memory space, loading...")
        else:
            print("Detect empty memory, create memory space...")
            self._create_class(CLASS_SCHEMA)
    # 檢查這個 class 存不存在    
    def _class_exists(self):
        schema = self.client.schema.get()
        classes = [cls['class'] for cls in schema['classes']]
        return self.class_name in classes
    # 創建 class schema
    def _create_class(self, class_schema):
        self.client.schema.create_class(class_schema)
    # 取得本身 shema
    def get_schema(self):
        return self.client.schema.get()
    
    def add_group_memory():
        pass
    def get_relavant_memory():
        pass
    def update_group_memory():
        pass
    def update_single_memory():
        pass
    def del_group_memory():
        pass
    def del_single_memory():
        pass
    def del_all_memory():
        pass