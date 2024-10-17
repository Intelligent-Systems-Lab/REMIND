from weaviate.classes.query import MetadataQuery
from schema import GROUP_SCHEMA, CHILD_SCHEMA
from weaviate.classes.query import Filter
from abc import ABC, abstractmethod
from prompt import rewrite_prompt
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
    def get_relevant_memory():
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
            rewrite, rewrite_des = self._merge_similar_group(most_similar_child.text, group_node.text)
            if rewrite=="True":
                most_similar_child.text = rewrite_des
            most_similar_child.children.extend(group_node.children)
        else:
            self.children.append(group_node)
        return
    def _merge_similar_group(self, description_1, description_2):
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
        if res.get("description"):
            print(f'-----rewrite two group-----')
            print(f'First group description: {description_1}')
            print(f'Second group description: {description_2}')
            print(f'Merge description: {res.get("description")}')
            print(f'---------------------------')
        return res["rewrite"], res.get("description")
    def get_relevant_memory(self, query:str, vector=None, k=5):
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
    def __init__(self, weaviate_url="127.0.0.1", port=8080, user="deafult"):
        self.client = weaviate.connect_to_local(weaviate_url, port)
        self.group_class_name = f"{user[0].upper()+user[1:]}_long_memory_group"
        self.child_class_name = f"{user[0].upper()+user[1:]}_long_memory_child"
        
        if self._class_exists(self.group_class_name):
            print(f"Detect existed {user} user group memory space, loading...")
        else:
            print("Detect empty group memory, create memory space...")
            GROUP_SCHEMA["class"] = self.group_class_name
            self._create_class(GROUP_SCHEMA)
        if self._class_exists(self.child_class_name):
            print(f"Detect existed {user} user child memory space, loading...")
        else:
            print("Detect empty child memory, create memory space...")
            CHILD_SCHEMA["class"] = self.child_class_name
            CHILD_SCHEMA["properties"] = CHILD_SCHEMA["properties"].append({"name":"parent", "dataType": [f"{self.group_class_name}"]})
            self._create_class(CHILD_SCHEMA)
        self.group_class = self.client.collections.get(self.group_class_name)
        self.child_class = self.client.collections.get(self.child_class_name)
            
    # 檢查這個 class 存不存在    
    def _class_exists(self, class_name):
        return self.client.collections.exists(class_name)
    
    # 創建 class schema
    def _create_class(self, class_schema):
        self.client.collections.create_from_dict(class_schema)
        
    def get_schema(self):
        return self.client.collections.list_all()
    
    def add_group_memory(self, group:dict):
        
        group_description = group["description"]
        
        response = self.group_class.query.near_text(
            query=group_description,
            limit=1,
            return_properties=["text"],
            return_metadata=MetadataQuery(distance=True)
        )
        # 找出向量相似度最高的 group，如果小於 distance 就合併 group 並且對 group description 進行更新
        distance = 0.2
        if response.objects and response.objects[0].metadata.distance < distance:
            similar_group = response.objects[0]
            similar_group_des = similar_group.properties["text"]
            group_id = similar_group.uuid
            rewrite, rewrite_des = self._rewrite_merge_des(group_description, similar_group_des)
            if eval(rewrite):
                self.group_class.data.update(
                    uuid=group_id,
                    properties={
                        "text":rewrite_des
                    }
                )
            # 需要優化
            else:
                group_data = {
                    "time": datetime.now().isoformat(timespec='seconds') + 'Z',
                    "text": group_description
                }
                group_id = self._insert_weaviate(self.group_class, group_data)
        else:
            group_data = {
                "time": datetime.now().isoformat(timespec='seconds') + 'Z',
                "text": group_description
            }
            group_id = self._insert_weaviate(self.group_class, group_data)
        
        # 插入 child
        for child in group["child"]:
            child_data = {
                "time": child.get("time"),
                "text": child["text"],
                "origin_text": child.get("origin_text")
            }
            reference = {
                "parent":group_id
            }
            vector = child.get("vector")
            self._insert_weaviate(self.child_class, child_data, vector, reference)
        return
    def _insert_weaviate(self, class_collection, data, vector=None, references=None):
        data_id = class_collection.data.insert(
            properties=data,
            references=references,
            vector=vector
        )
        return data_id
    def _rewrite_merge_des(self, description_1, description_2):
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
        if res.get("description"):
            print(f'-----rewrite two group-----')
            print(f'First group description: {description_1}')
            print(f'Second group description: {description_2}')
            print(f'Merge description: {res.get("description")}')
            print(f'---------------------------')
        return res["rewrite"], res.get("description")
    def get_relevant_memory(self, query:str, k=5):
        response = self.group_class.query.near_text(
            query=query,
            limit=k,
            return_properties=["text"],
            return_metadata=MetadataQuery(distance=True)
        )
        similar_groups = response.objects
        
        if similar_groups:
            similar_group = similar_groups[0]
            other_groups = similar_groups[1:]
            group_id = similar_group.uuid
            group_description = similar_group.properties["text"]
            relative_memory = self.get_relavant_child(query, group_id)
            return group_description, relative_memory, other_groups
        else:
            return {"system": "Don't find relevant memory"}
    def get_relavant_child(self, query:str, group_id=None, k=5):
        if group_id:
            response = self.child_class.query.near_text(
                query=query,
                filters=Filter.by_ref("parent").by_id().equal(group_id),
                limit=k,
                return_properties=["time", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            response = self.child_class.query.near_text(
                query=query,
                limit=k,
                return_properties=["time", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        return response.objects
    def update_group_memory():
        pass
    def update_single_memory():
        pass
    def del_group_memory():
        pass
    def del_single_memory():
        pass
    def del_all_memory(self):
        self.client.collections.delete(self.child_class_name)
        self.client.collections.delete(self.group_class_name)