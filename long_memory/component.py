from long_memory.prompt import chatlog_classify_prompt, document_classify_prompt, recall_search, generate_keyword, hyde_generated
# from long_memory.prompt import rewrite_prompt
from long_memory.schema import GROUP_SCHEMA, CHILD_SCHEMA, SEARCH_HISTORY
# from long_memory.tools import tools

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
    def del_memory():
        pass

class WeaviateLongMemory(Base):
    def __init__(self, weaviate_url="127.0.0.1", port=8080, user="deafult", model="gpt-4o-mini", ollama_url="http://localhost:11434/api/generate", time_sort=True):
        self.client = weaviate.connect_to_local(weaviate_url, port)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.ollama_url = ollama_url
        self.user = user
        self.group_class_name = f"{user[0].upper()+user[1:]}_long_memory_group"
        self.child_class_name = f"{user[0].upper()+user[1:]}_long_memory_child"
        self._memory_exists()
        self.group_class = self.client.collections.get(self.group_class_name)
        self.child_class = self.client.collections.get(self.child_class_name)
        self.recall_search_records = []
        self.time_sort = time_sort
        
        self.origin_chat_logs=""
        self.classify_chat_logs=""
    
    def _memory_exists(self):
        if self._class_exists(self.group_class_name):
            print(f"Detect existed {self.user} user group memory space, loading...")
        else:
            print("Detect empty group memory, create memory space...")
            group_schema = copy.deepcopy(GROUP_SCHEMA)
            group_schema["class"] = self.group_class_name
            self._create_class(group_schema)
        if self._class_exists(self.child_class_name):
            print(f"Detect existed {self.user} user child memory space, loading...")
        else:
            print("Detect empty child memory, create memory space...")
            child_schema = copy.deepcopy(CHILD_SCHEMA)
            child_schema["class"] = self.child_class_name
            child_schema["properties"] = CHILD_SCHEMA["properties"].append({"name":"parent", "dataType": [f"{self.group_class_name}"]})
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
                "text":item.properties['text']
            })
        return
    
    def show_all_groups(self):
        for item in self.group_class.iterator():
            print({
                "id":str(item.uuid),
                "text":item.properties['text']
            })
        return
    
    def show_all_children(self):
        for item in self.child_class.iterator():
            print({
                "id":str(item.uuid),
                "text":item.properties['text']
            })
        return
    
    def group_count(self):
        count = 0
        for item in self.child_class.iterator():
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
    
    def add_article(self, article:list, summary_limit=100):
        article_list = [article[i:i + 15] for i in range(0, len(article), 15)]
        for count, article in enumerate(article_list):
            for i, log in enumerate(article):
                log['id'] = i
            log_set = set(range(0, len(article)))
            
            while 1:
                json_res = self._llm_create(document_classify_prompt.format(summary_limit=summary_limit,article=article))
                groups = self._llm_response_handler(json_res)
                classify_set = set()
                for group in groups['groups']:
                    classify_set.update(group.get('article_id'))
                if log_set != classify_set:
                    print(f"Articles not correct, missing id:{log_set.difference(classify_set)}, unknown id:{classify_set.difference(log_set)}, retry..")
                else:
                    break
                
            for group in groups['groups']:
                children = []
                for article_id in group["article_id"]:
                    log = [item for item in article if item.get('id') == article_id][0]
                    child = {
                        "text":log.get('text'),
                        "time":log.get('time')
                    }
                    children.append(child)
                group = {
                    "description":group["summary"],
                    "child":children
                }
                self.add_group_memory(group)
        print("\033[34mSave article to long memory done.\033[0m")
            
    def add_chat_logs(self, chat_logs:list, summary_limit=50):
        """add chat logs to long memory

        Args:
            chat_logs (list): list of chat log, chat log needs be following format
                ```
                {
                    "text":"user:Hi, how are you today?, assistant:fine, how about you?",
                    "time": "2024-10-31T04:18:00Z" (Optional)
                }
                ```
            summary_limit (int, optional): the limit number of the summary group. Defaults to 50.
        """
        # TODO: need a more clever way to classify chat_logs, if the origin text too large.
        chat_logs_list = [chat_logs[i:i + 15] for i in range(0, len(chat_logs), 15)]
        for count, chat_logs in enumerate(chat_logs_list):
            # print(f"\033[34m---Chat logs batch:{count+1}---\033[0m")
            for i, log in enumerate(chat_logs):
                log['id'] = i
            log_set = set(range(0, len(chat_logs)))
            
            while 1:
                # TODO: del time to reduce prompt use
                json_res = self._llm_create(chatlog_classify_prompt.format(summary_limit=summary_limit,chat_logs=chat_logs))
                groups = self._llm_response_handler(json_res)
                # 檢查有沒有資訊遺失
                classify_set = set()
                for group in groups['groups']:
                    classify_set.update(group.get('chat_logs'))
                # print(f"Saving record to self.origin_chat_logs, self.classify_chat_logs.")
                self.origin_chat_logs = chat_logs
                self.classify_chat_logs = groups
                if log_set != classify_set:
                    print(f"Chat logs not correct, missing id:{log_set.difference(classify_set)}, unknown id:{classify_set.difference(log_set)}, retry..")
                else:
                    # print(f"Fully classify")
                    break
                
            # print("\033[34mAdding to long memory...\033[0m")
            for group in groups['groups']:
                children = []
                for chat_log_id in group["chat_logs"]:
                    log = [item for item in chat_logs if item.get('id') == chat_log_id][0]
                    child = {
                        "text":log.get('text'),
                        "time":log.get('time')
                    }
                    children.append(child)
                group = {
                    "description":group["summary"],
                    "child":children
                }
                self.add_group_memory(group)
        print("\033[34mSave chat logs to long memory done.\033[0m")
            
    def add_group_memory(self, group:dict):
        
        group_description = group["description"]
        # 設置組別時間
        if group["child"][0].get("time"):
            group_time = group["child"][0].get("time")
        else:
            group_time = datetime.now().isoformat(timespec='seconds') + 'Z'
        
        """# 檢查是否有相似的 group
        response = self.group_class.query.near_text(
            query=group_description,
            limit=1,
            return_properties=["time", "text"],
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
                    "time": group_time,
                    "text": group_description
                }
                group_id = self._insert_weaviate(self.group_class, group_data)
        else:
            group_data = {
                "time": group_time,
                "text": group_description
            }
            group_id = self._insert_weaviate(self.group_class, group_data)"""
        group_data = {
            "time": group_time,
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
    
    """def _rewrite_merge_des(self, description_1, description_2):
        messages = [{"role": "user", "content": rewrite_prompt.format(description_1=description_1, description_2=description_2)}]
        completion = self.openai_client.chat.completions.create(
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
        return res["rewrite"], res.get("description")"""
    
    def get_relevant_memory(self, query:str, object_id=None, k=5, method="similarity"):
        if object_id:
            response = self.group_class.query.fetch_objects(
                filters=Filter.by_id().equal(object_id)
            )
        else:
            if method=="similarity":
                response = self.group_class.query.near_text(
                    query=query,
                    limit=k,
                    return_properties=["time", "text"],
                    return_metadata=MetadataQuery(distance=True)
                )
            elif method=="keyword":
                # generate keywords by llm
                keyword_prompt = generate_keyword.format(query=query)
                query = eval(self._llm_create(keyword_prompt))
                print(f'Search by keywords:{query}')
                response = self.group_class.query.fetch_objects(
                    filters=Filter.by_property("text").contains_any(query),
                    limit=k
                )
            elif method=="BM25":
                response = self.group_class.query.bm25(
                    query=query,
                    return_metadata=MetadataQuery(score=True),
                    limit=k
                )
            elif method=="HyDE":
                hyde_prompt = hyde_generated.format(query=query)
                query = self._llm_create(hyde_prompt)
                response = self.group_class.query.near_text(
                    query=query,
                    limit=k,
                    return_properties=["text", "time"],
                    return_metadata=MetadataQuery(distance=True)
                )
            else:
                print(f"Unknow method, only can be [similarity/keyword/BM25/HyDE]. Your method:{method}")
                return
        similar_groups = response.objects
        
        if similar_groups:
            similar_group = similar_groups[0]
            other_groups = similar_groups[1:]
            group_id = similar_group.uuid
            group_description = {
                "text":similar_group.properties["text"],
                "time":similar_group.properties['time'].strftime("%Y/%m/%d %H:%M")
            }
            relative_memory = self.get_relavant_child(query, group_id, k, method)
            retrieve_result = self._summary_retrieve_page(group_description, relative_memory, other_groups) 
            return retrieve_result
        else:
            return {"system": "Don't find relevant memory"}
        
    def get_memory(self, query:str, retrieve_number=5, recall=False, turn=4):
        question = query
        if not recall:
            return self.get_relevant_memory(query=query, k=retrieve_number)
        else:
            search_records = []
            object_id = ""
            history = copy.deepcopy(SEARCH_HISTORY)
            for _ in range(turn):
                retrieve_result = self.get_relevant_memory(query=query, object_id=object_id, k=retrieve_number)
                p = recall_search.format(current_time=datetime.now().strftime("%Y/%m/%d %H:%M"), query=query, search_info=retrieve_result, search_history=history)
                llm_res = self._llm_create(p)
                res_dict = self._llm_response_handler(llm_res)
                
                # update search_history
                history['search times']+=1
                if query not in history['used queries']:
                    history['used queries'].append(query)
                if retrieve_result["closest_summary"] not in history['searched memory']:
                    history['searched memory'].append(retrieve_result["closest_summary"])
                if res_dict.get('evidence'):
                    if type(res_dict.get('evidence'))==str:
                        history['evidence'].append(res_dict.get('evidence'))
                    else:
                        history['evidence'].extend(res_dict.get('evidence'))
                history['thought'] = res_dict['think']
                record = {
                    'search times':history['search times'],
                    'used query':query,
                    'searched memory':retrieve_result["closest_summary"],
                    'thought':res_dict['think'],
                    'evdience':history.get('evidence')
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
                'time':m.properties['time']
            })
        if self.time_sort:
            similar_snippets = sorted(similar_snippets, key=lambda x: x["time"])
            similar_snippets = [{"text": item["text"], "time": item["time"].strftime("%Y/%m/%d %H:%M")} for item in similar_snippets]
        related_summaries = []
        for m in other_groups:
            related_summaries.append({
                'id':str(m.uuid),
                'text':m.properties['text'],
                'time':m.properties['time'].strftime("%Y/%m/%d %H:%M")
            })
        result = {
            "closest_summary":group_description,
            "similar_snippets":similar_snippets,
            "related_summaries":related_summaries
        }
        return result
    
    def get_relavant_child(self, query:str|list, group_id=None, k=5, method="similarity"):
        if group_id:
            response = self.child_class.query.near_text(
                query=query,
                filters=Filter.by_ref("parent").by_id().equal(group_id),
                limit=k,
                return_properties=["time", "text"],
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            if method=="similarity" or method=="HyDE":
                response = self.child_class.query.near_text(
                    query=query,
                    limit=k,
                    return_properties=["time", "text"],
                    return_metadata=MetadataQuery(distance=True)
                )
            elif method=="keyword":
                response = self.child_class.query.fetch_objects(
                    filters=Filter.by_property("text").contains_any(query),
                    limit=k
                )
            elif method=="BM25":
                response = self.child_class.query.bm25(
                    query=query,
                    return_metadata=MetadataQuery(score=True),
                    limit=k
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
    def del_memory(self):
        self.client.collections.delete(self.child_class_name)
        self.client.collections.delete(self.group_class_name)
        self._memory_exists()