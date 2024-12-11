from long_memory.prompt import recall_search
from long_memory.schema import SEARCH_HISTORY

from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
import weaviate

from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

import copy
import json
import os
import re

load_dotenv()

class WeaviateLongMemoryNoTime():
    def __init__(self, weaviate_url="127.0.0.1", port=8080, user="deafult"):
        self.client = weaviate.connect_to_local(weaviate_url, port)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user = user
        self.group_class_name = f"{user[0].upper()+user[1:]}_long_memory_group"
        self.child_class_name = f"{user[0].upper()+user[1:]}_long_memory_child"
        self.group_class = self.client.collections.get(self.group_class_name)
        self.child_class = self.client.collections.get(self.child_class_name)
        self.recall_search_records = []
        
        self.origin_chat_logs=""
        self.classify_chat_logs=""
    
    def _llm_create(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return completion.choices[0].message.content
    
    def get_relevant_memory(self, query:str, object_id=None, k=5, method="similarity"):
        if object_id:
            response = self.group_class.query.fetch_objects(
                filters=Filter.by_id().equal(object_id)
            )
        else:
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
            group_description = {
                "text":similar_group.properties["text"]
            }
            relative_memory = self.get_relavant_child(query, group_id, k)
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
                res_dict = json.loads(re.search(r"```json(.*?)```", llm_res, re.DOTALL).group(1).strip())
                
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
            })
        related_summaries = []
        for m in other_groups:
            related_summaries.append({
                'id':str(m.uuid),
                'text':m.properties['text'],
            })
        result = {
            "closest_summary":group_description,
            "similar_snippets":similar_snippets,
            "related_summaries":related_summaries
        }
        return result
    
    def get_relavant_child(self, query:str, group_id=None, k=5):
        if group_id:
            response = self.child_class.query.near_text(
                query=query,
                filters=Filter.by_ref("parent").by_id().equal(group_id),
                limit=k,
                return_properties=["text"],
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            response = self.child_class.query.near_text(
                query=query,
                limit=k,
                return_properties=["text"],
                return_metadata=MetadataQuery(distance=True)
            )
        return response.objects