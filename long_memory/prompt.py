rewrite_prompt = """You are a rewriter, you have two descriptions, they need to be merged to one description
first is {description_1}, second is {description_2}, if first has enough information that don't have to rewrite, 
if not, rewrite a new description"""

chatlog_classify_prompt = """Watch the following chat logs, you need to write the memory for youself,
Group chat records according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail like date, where or do what as you can.
Example:
Chat logs:[
    {{'id':1, 'text':'assistant:Hi, how are you today?, user:Good. I walked in the park today'}},
    {{'id':2, 'text':'user:I saw dogs and a parrot, it can speak chinese!, assistant:That's really gread!'}}
]
```json
{{
    "groups": [
        {{
            "summary": "user walked in the park today and saw some dogs and a parrot that can speak chinese",
            "chat_logs": [1, 2]
        }}
    ]
}}
```
Chat logs:{chat_logs}
"""

document_classify_prompt = """Watch the following article, group the article according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail as you can.
Don’t miss the origin article.
```json
{{
    "groups": [
        {{
            "summary": "abstract of AI future and challenge, and why it's important",
            "paragraph": [
                {{
                    "text": "AI has been invested in many fields and plays an important role..."
                }},
                {{
                    "text": "Most AI now is mainly driven by language models..."
                }}
            ]
        }}
    ]
}}
```
Article:{article}
"""
# (don't repeat previous evidence)
recall_search = """Your character is assistant and you are searching your memories related to query from the memory bank.
If you try searching several times, it is possible that you do not have this knowledge in your memory.
The following will display your current search information and search records.

Query:{query}

Information found: {search_info}

Search history, you will see during the entire search process: {search_history}

You have three actions and the output is in json format, you can write your thought into think field, 
the evidence field preferably original memory from similar_snippets, 
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
    "query':"", # search keywords, don't be too similar to the previous keywords
    "think':"",
    "evidence":"",
}}
```
"""