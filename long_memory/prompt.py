rewrite_prompt = """You are a rewriter, you have two descriptions, they need to be merged to one description
first is {description_1}, second is {description_2}, if first has enough information that don't have to rewrite, 
if not, rewrite a new description"""

chatlog_classify_prompt = """Watch the following chat logs, you need to write the memory for youself,
Group chat records according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail like date, where or specific things as you can.
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

hyde_generated="""You are a helpful assistant. Write a hypothetical passage that directly addresses the following query:
Query: {query}
The passage should provide a detailed and coherent response, even if the content is hypothetical. Ensure it is relevant to the query."""

generate_keyword="""Based on the query below, generate a list of key concepts or keywords that best capture its essential meaning 
for a memory retrieval system.
Respnse with this format, the keywords should less than 4
Example:["AI", "machine learning"]
Query: {query}"""

document_classify_prompt = """Watch the following article, group the article according to topics and summarize each group with json format.
Each summary can't over {summary_limit} and need as detail as you can.
Don’t miss the origin article.
Example:
Article:[
    {{'id':1, 'text':'Volcanism of the Mount Edziza volcanic complex in British Columbia, Canada, spans more than 7 million years.'}},
    {{'id':2, 'text':'The first magmatic cycle took place between 7.5 and 6 million years ago and is represented by the Raspberry, Little Iskut and Armadillo geological formations.'}}
]
```json
{{
    "groups": [
        {{
            "summary": "The volcanism of the Mount Edziza volcanic complex in British Columbia, Canada, spans over 7 million years. The initial magmatic cycle occurred between 7.5 and 6 million years ago, represented by the Raspberry, Little Iskut, and Armadillo geological formations.",
            "article_id": [1, 2]
        }}
    ]
}}
```
Article:{article}
"""
# (don't repeat previous evidence)
recall_search = """Your character is assistant and you are searching your memories related to query from the memory bank.
If you try searching several times, it is possible that you do not have this knowledge in your memory.
The searched memory is marked with time, so it can be used to make simple judgments.
{other_instruct}
The following will display your current search information and search records.
Time now:{current_time}

Query:{query}

Information found: {search_info}

Search history, you will see during the entire search process: {search_history}

You have three actions and the output is in json format, you can write your thought into think field, 
put key memory or what happen to evidence field as detail as possible.
These will add to the search_history

1.end: End the search when the information is sufficient, don't say it's insufficient without finding it
```json
{{
    "action":"end",
    "reason":"sufficient", # or insufficient
    "think":"",
    "evidence":[],
}}
```
2.jump: if you see something may help in the related_summaries, use jump to see it with more detail information
```json
{{
    "action":"jump",
    "id":"", # id of related_summaries
    "think":"",
    "evidence":[{{"text":"user:I like to go to park.."}}, {{"text":"user:I also like walking in.."}}],
}}
```
3.retry: Search again using new keywords, the keyword should be very different with previous keyword to get better search
```json
{{
    "action":"retry",
    "query":"", # search keywords, don't be too similar to the previous keywords
    "think':"",
    "evidence":[],
}}
```
"""