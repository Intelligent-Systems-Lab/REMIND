summary_prompt = """Watch the following article and summarize.
Summary can't over {summary_limit} words. The summary used to introduce the content of this article.
Article:{article}
"""

recall_search = """Your character is assistant and you are searching your memories related to query from the memory bank.
You must find enough information to answer the query.
The following will display your current search information and search records.

Query:{query}

Information found: {search_info}

Search history, you will see during the entire search process: {search_history}

You have three actions and the output is in json format, you can write your thought into think field, 
Put useful info that may be related to query into the evidence field.
These process will add to the search_history, so you don't need to repeat the evidence in the search history.
The search_history will provide to me in the end to answer the question.

example:
1.end: End the search when the information is sufficient
```json
{{
    "action":"end",
    "reason":"sufficient",
    "think":"",
    "evidence":[]
}}
```
2.jump: jump to related_summaries to search origin content
```json
{{
    "action":"jump",
    "id":"", # id of related_summaries
    "think":"",
    "evidence":[{{
        "text": ""
    }}]
}}
```
3.retry: Search again using new keywords
```json
{{
    "action":"retry",
    "query':"", # search keywords, don't be too similar to the previous keywords
    "think':"",
    "evidence":[]
}}
```
"""