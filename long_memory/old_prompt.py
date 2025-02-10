old_chatlog_classify_prompt = """Please analyze the following chat logs. Your task is to classify the conversations into groups based on similar topics or themes and summarize each group. 
Follow these requirements:
1. Group chat records by topics or themes. Each group should include related conversations.
2. Summarize each group in JSON format, ensuring the summary covers all key points within the group.
3. Each summary must not exceed {summary_limit} words, make the summary concise but comprehensive.
4. Ensure every chat log is included in at least one group.
5. Use the following JSON format for the output.
{other_instruct}

Example:
Chat logs: [
    {{"id": 1, "text": "assistant: Hi, how are you today? user: Good. I walked in the park today."}},
    {{"id": 2, "text": "user: I saw dogs and a parrot, it can speak Chinese! assistant: That's really great!"}}
]
Output:
```json
{{
    "groups": [
        {{
            "summary": "User walked in the park, seeing dogs and a parrot that can speak Chinese. Assistant expresses enthusiasm.",
            "chat_logs": [1, 2]
        }}
    ]
}}
Chat logs:{chat_logs}
"""

old_recall_search = """Your role is assistant, and your task is to search your memory bank for information related to the provided query.
Try using different method to get the information.
{other_instruct}

### Current State
- **Time Now:**{current_time}
- **Question:**{question}

Information found: {search_info}
### Found Information
The following contains the search results:
1. **Closest Summary:** The most relevant memory group to the query.
2. **Similar Snippets:** Original content from the main memory group.
3. **Related Summaries:** Other potentially relevant memory groups (you can jump to retrieve their details).

Search history: {search_history}
### Search History
Includes the keywords used, and the information retrieved in previous turns.

### Actions
Respond in JSON format. Choose one of these actions and follow the specified format:
1.**End the search:** Use this when sufficient information has been found.
```json
{{
    "action":"end",
    "reason":"sufficient", # or insufficient
    "think":"",
    "evidence":["Similar Snippets supporting the answer, {{"text":"original text or dialog"}}"],
}}
```
2.**Jump to related summaries:** Sometimes information is hidden in other groups, even if it is not visible from the summary.
```json
{{
    "action":"jump",
    "id":"related_summary_id",
    "think":"",
    "evidence":[],
}}
```
3.**Retry with new keywords:** Use to re-search when no relevant information is found in the current search.
```json
{{
    "action":"retry",
    "keywords":"New search keywords, the keyword should be very different with previous keyword to get better search",
    "think":"",
    "evidence":[],
}}
```
Use the JSON and double quote format for the output:
"""