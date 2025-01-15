from long_memory.hotpot_prompt import summary_prompt, recall_search
from long_memory.schema import GROUP_SCHEMA, CHILD_SCHEMA, SEARCH_HISTORY

from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
import weaviate

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

import requests
import copy
import json
import os
import re

load_dotenv()

class Base(ABC):
    # 取得
    @abstractmethod
    def get_relevant_memory():
        pass
    @abstractmethod
    def del_memory():
        pass

class HotPotWeaviateLongMemory(Base):
    def __init__(self, weaviate_url="127.0.0.1", port=8080, user="deafult", model="gpt-4o-mini", ollama_url="http://localhost:11434/api/generate"):
        self.client = weaviate.connect_to_local(weaviate_url, port)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.ollama_url = ollama_url
        self.user = user
        self.group_class_name = f"{user[0].upper()+user[1:]}_hotpot_long_memory_group"
        self.child_class_name = f"{user[0].upper()+user[1:]}_hotpot_long_memory_child"
        self._memory_exists()
        self.group_class = self.client.collections.get(self.group_class_name)
        self.child_class = self.client.collections.get(self.child_class_name)
        self.recall_search_records = []
        self.sp_fact = []
    
    def _memory_exists(self):
        if self._class_exists(self.group_class_name):
            print(f"Detect existed {self.user} user group memory space, loading...")
        else:
            print("Detect empty group memory, create memory space...")
            group_schema = copy.deepcopy(GROUP_SCHEMA)
            group_schema["class"] = self.group_class_name
            properties = group_schema["properties"]
            properties.append({"name":"doc_name","dataType": ["text"],"moduleConfig": {"text2vec-openai": {"skip": "true"}}})
            group_schema["properties"] = properties
            group_schema["properties"].pop(0) # del time
            self._create_class(group_schema)
        if self._class_exists(self.child_class_name):
            print(f"Detect existed {self.user} user child memory space, loading...")
        else:
            print("Detect empty child memory, create memory space...")
            child_schema = copy.deepcopy(CHILD_SCHEMA)
            child_schema["class"] = self.child_class_name
            properties = child_schema["properties"]
            properties.append({"name":"parent", "dataType": [f"{self.group_class_name}"]})
            properties.append({"name":"doc_number","dataType": ["text"],"moduleConfig": {"text2vec-openai": {"skip": "true"}}})
            properties.pop(2)
            properties.pop(0) # del time
            child_schema["properties"] = properties
            self._create_class(child_schema)
    
    # 檢查這個 class 存不存在    
    def _class_exists(self, class_name):
        return self.client.collections.exists(class_name)
    
    # 創建 class schema
    def _create_class(self, class_schema):
        self.client.collections.create_from_dict(class_schema)
    
    def show_groups(self, limit=5):
        res = self.group_class.query.fetch_objects(
            limit=limit
        )
        for item in res.objects:
            print({
                "id":str(item.uuid),
                "doc_name":item.properties['doc_name'],
                "text":item.properties['text']
            })
        return
    
    def show_all_groups(self):
        for item in self.group_class.iterator():
            print({
                "id":str(item.uuid),
                "doc_name":item.properties['doc_name'],
                "text":item.properties['text']
            })
        return
    
    def show_all_children(self):
        for item in self.child_class.iterator():
            print({
                "id":str(item.uuid),
                "doc_number":item.properties['doc_number'],
                "text":item.properties['text']
            })
        return
    
    def group_count(self):
        count = 0
        for item in self.group_class.iterator():
            count+=1
        return count
    
    def child_count(self):
        count = 0
        for item in self.child_class.iterator():
            count+=1
        return count
    
    def dump_recall_records(self):
        for record in self.recall_search_records:
            print(json.dumps(record, indent=4))
        return self.recall_search_records
    
    def get_schema(self):
        return self.client.collections.list_all()
    
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
    
    def _llm_response_handler(self, response:str):
        """handle llm response format, especially for llama family"""
        try:
            return json.loads(response)
        except:
            response = response.strip()
            try:
                return json.loads(re.search(r"```json(.*?)```", response, re.DOTALL).group(1).strip())
            except:
                try:
                    return json.loads(re.search(r"```(.*?)```", response, re.DOTALL).group(1).strip())
                except:
                    return response
                
    def add_hotpot_doc(self, doc:list, summary_limit=50):
        doc_name = doc[0]
        doc_content = doc[1]
        
        prompt = summary_prompt.format(article=doc_content, summary_limit=summary_limit)
        res = self._llm_create(prompt)
        
        group_data = {
            "text": res,
            "doc_name": doc_name
        }
        group_id = self._insert_weaviate(self.group_class, group_data)
        
        for number, doc_paragraph in enumerate(doc_content):
            child_data = {
                "text": doc_paragraph.replace('"', ""), # handle json format error
                "doc_number": str(number)
            }
            reference = {
                "parent":group_id
            }
            self._insert_weaviate(self.child_class, data=child_data, references=reference)
        return

    def _insert_weaviate(self, class_collection, data, vector=None, references=None):
        data_id = class_collection.data.insert(
            properties=data,
            references=references,
            vector=vector
        )
        return data_id
    
    def get_relevant_memory(self, query:str, object_id=None, k=5):
        if object_id:
            response = self.group_class.query.fetch_objects(
                filters=Filter.by_id().equal(object_id)
            )
        else:
            response = self.group_class.query.near_text(
                query=query,
                limit=k,
                return_properties=["doc_name", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        similar_groups = response.objects
        
        if similar_groups:
            similar_group = similar_groups[0]
            other_groups = similar_groups[1:]
            group_id = similar_group.uuid
            group_description = {
                "doc_name":similar_group.properties['doc_name'],
                "text":similar_group.properties["text"]
            }
            relative_memory = self.get_relavant_child(query, group_id, k)
            retrieve_result = self._summary_retrieve_page(group_description, relative_memory, other_groups) 
            return retrieve_result
        else:
            return {"system": "Don't find relevant memory"}
        
    def get_memory(self, query:str, retrieve_number=5, recall=False, turn=4):
        self.recall_search_records.clear()
        self.sp_fact.clear()
        
        question = query
        if not recall:
            return self.get_relevant_memory(query=query, k=retrieve_number)
        else:
            search_records = []
            object_id = ""
            history = copy.deepcopy(SEARCH_HISTORY)
            for _ in range(turn):
                retrieve_result = self.get_relevant_memory(query=query, object_id=object_id, k=retrieve_number)
                p = recall_search.format(query=query, search_info=retrieve_result, search_history=history)
                llm_res = self._llm_create(p)
                res_dict = self._llm_response_handler(llm_res)
                # update search_history
                history['search times']+=1
                if query not in history['used queries']:
                    history['used queries'].append(query)
                if retrieve_result["closest_summary"] not in history['searched memory']:
                    history['searched memory'].append(retrieve_result["closest_summary"])
                try:
                    if res_dict.get('evidence'):
                        if type(res_dict.get('evidence'))==str:
                            history['evidence'].append(res_dict.get('evidence'))
                        else:
                            for e in res_dict.get('evidence'):
                                e['doc_name'] = retrieve_result["closest_summary"]['doc_name']
                                if e not in history['evidence']:
                                    history['evidence'].append(e)
                except:
                    print(res_dict)
                history['thought'] = res_dict['think']
                record = {
                    'search times':history['search times'],
                    'used query':query,
                    'searched memory':retrieve_result["closest_summary"],
                    'thought':res_dict['think'],
                    'evdience':res_dict.get('evidence')
                }
                search_records.append(record)
                
                action = res_dict['action']
                if action=='end':
                    search_records.append({
                        "end":res_dict['reason']
                    })
                    break
                elif action=='jump':
                    object_id = res_dict['id']
                elif action=='retry':
                    query=res_dict.get('query')
                    object_id = ""
                else:
                    print(f'Action unknow:{action}')
                    print(json.dumps(res_dict, indent=4))
                    break
            recall_search_record = {
                "query":question,
                "records":search_records
            }
            self.recall_search_records.append(recall_search_record)
            return history
    
    def _summary_retrieve_page(self, group_description:dict, relative_memory:list, other_groups:list):
        similar_snippets = []
        for m in relative_memory:
            similar_snippets.append({
                'text':m.properties['text'],
                'doc_number':m.properties['doc_number']
            })
        related_summaries = []
        for m in other_groups:
            related_summaries.append({
                'id':str(m.uuid),
                'doc_name':m.properties['doc_name'],
                'text':m.properties['text']
            })
        result = {
            "closest_summary":group_description,
            "similar_snippets":similar_snippets,
            "related_summaries":related_summaries
        }
        return result
    
    def get_relavant_child(self, query:str|list, group_id=None, k=5):
        if group_id:
            response = self.child_class.query.near_text(
                query=query,
                filters=Filter.by_ref("parent").by_id().equal(group_id),
                limit=k,
                return_properties=["doc_number", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            response = self.child_class.query.near_text(
                query=query,
                limit=k,
                return_properties=["doc_number", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        return response.objects
    def del_memory(self):
        self.client.collections.delete(self.child_class_name)
        self.client.collections.delete(self.group_class_name)
        self._memory_exists()