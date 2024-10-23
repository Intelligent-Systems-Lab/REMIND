rewrite_prompt = """You are a rewriter, you have two descriptions, they need to be merged to one description
first is {description_1}, second is {description_2}, if first has enough information that don't have to rewrite, 
if not, rewrite a new description"""

classify_prompt = """Watch the following chat logs, group chat records according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail as you can.
The chat logs need contain and need change name to assistant and user.
Don’t miss the conversation record.
```json
{{
    'groups':[
        {{
            'summary':'user walked in the park today and saw some dogs and a parrot that can speak chinese',
            'chat_logs':[
                {{
                    'assistant':'Hi, how are you today?',
                    'user':'Good. I walked in the park today'
                }},
                {{
                    'assistant':'That's really gread! do you see any pets?',
                    'user':'I saw dogs and a parrot, it can speak chinese!'
                }}
            ]
        }}
    ]
}}
```
Chat logs:{chat_logs}
"""

sublevel_prompt = """Watch the following article, group the article according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail as you can.
Don’t miss the origin article.
```json
{{
    'groups':[
        {{
            'summary':'abstract of AI future and challenge, and why it's important',
            'paragraph':[
                {{
                    'text':'AI has been invested in many fields and plays an important role...'
                }},
                {{
                    'text':'Most AI now is mainly driven by language models...'
                }}
            ]
        }}
    ]
}}
```
Article:{article}
"""
# (don't repeat previous evidence)
recursive_search = """You are a search expert and you are searching for memories related to query from the memory bank.
If you try searching several times, it is possible that you do not have this knowledge in your memory.
The following will display your current search information and search records.

Query:{query}

Information found: {search_info}

Search history, you will see during the entire search process: {search_history}

You have three actions and the output is in json format, you can write your thought into think field, 
relevant snippets paragraph into evidence field, 
They will add to the search_history
1.end: End the search when the information is sufficient
```json
{{
    "action":"end",
    "reason":"sufficient", # or insufficient
    "think":"",
    "evidence":"", 
}}
```
2.jump: jump to related_summaries to search
```json
{{
    "action":"jump",
    "id":"", # id of related_summaries
    "think":"",
    "evidence":"",
}}
```
3.retry: Search again using new keywords
```json
{{
    "action":"retry",
    "query':"", # search keywords, don’t be too similar to the previous keywords
    "think':"",
    "evidence":"",
}}
```
"""

search_history = {
    "search times":0,
    "used queries":[],
    "searched memory":[],
    "thought":"",
    "evidence":[],
}