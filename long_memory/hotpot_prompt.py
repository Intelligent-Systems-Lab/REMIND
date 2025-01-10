summary_prompt = """Watch the following article and summarize.
Summary can't over {summary_limit} words.
Article:{article}
"""
# (don't repeat previous evidence)
recall_search = """Your character is assistant and you are searching your memories related to query from the memory bank.
You need to find as sufficient as possible of information.
The following will display your current search information and search records.

Query:{query}

Information found: {search_info}

Search history, you will see during the entire search process: {search_history}

You have three actions and the output is in json format, you can write your thought into think field, 
Put similar_snippets that may be related to query into the evidence field.
These will add to the search_history

example:
1.end: End the search when the information is sufficient
```json
{{
    "action":"end",
    "reason":"sufficient", # or insufficient
    "think":"",
    "evidence":[]
}}
```
2.jump: jump to related_summaries to search
```json
{{
    "action":"jump",
    "id":"", # id of related_summaries
    "think":"",
    "evidence":[{{
        "text": " She was the mother of Şehzade Mehmed Seyfeddin and Esma Sultan of the Ottoman Empire.",
        "doc_id": "596095c7-03e8-44b6-87da-cda445de6045"
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