rewrite_prompt = """You are a rewriter, you have two descriptions, they need to be merged to one description
first is {description_1}, second is {description_2}, if first has enough information that don't have to rewrite, 
if not, rewrite a new description"""

chatlog_classify_prompt = """You are an advanced AI assistant tasked with classifying a large dataset of conversations into thematic groups. The dataset consists of thousands of conversations between two parties, covering various topics such as greetings, viewpoints, agreements, disagreements, facts, news, activities, memory, dreams, and changes of mind or viewpoints over time, etc. Your goal is to analyze the conversations, group them by theme, and ensure consistency in the grouping criteria.

### Instructions:

1. **Grouping by Theme**:
   - Analyze each conversation to identify its primary theme or topic.
   - Create new groups based on the identified themes and ensure that the grouping criteria remain consistent across all conversations.
   - If a conversation could fit into multiple themes, assign it to the most dominant or relevant theme based on its content and context.

3. **Output Format**:
   - Provide the results strictly in the JSON format outlined below.
   - Each group should include a concise yet comprehensive summary of its theme.
   - Each summary must not exceed {summary_limit} words.

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
{other_instruct}
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

recall_search = """You are an advanced AI assistant tasked with retrieving relevant information from a memory bank to answer a user's query. The memory bank contains grouped conversations and summaries. You will employ a strategy balancing **exploitation** (ending the search when sufficient information is found) and **exploration** (continuing to search for potentially better or complementary information) to deliver the most comprehensive and accurate response.
{other_instruct}

### Current State
- **Time Now:**{current_time}
- **Question:**{question}

### **Instructions**:

1. **Input Information**:
   - You are provided with:
     - The closest group's summary in closest_summary field (from the top-k group summaries).
     - The top-k conversations within that group in similar_snippets field(ranked by cosine similarity using embeddings).
     - Summaries of the remaining top-k groups in related_summaries field.
     - Search history, including:
       - Previously used queries (including rewritten ones).
       - Conversations or group summaries selected in each round (if any).

2. **Actions**:
   In each search round, you must decide on one of the following actions:
   - **End the search**: Choose this if the information already retrieved is sufficient to answer the query comprehensively.
   - **Jump to another group**: Select a different group if its summary suggests it may contain more relevant information.
   - **Retry with a rewritten query**: If the retrieved information is insufficient, rewrite the user query to trigger a fresh retrieval of top-k groups and their top-k conversations.

3. **Balancing Exploitation and Exploration**:
   - Aim to end the search promptly when the retrieved information seems sufficient, but occasionally prioritize exploration to uncover potentially overlooked or complementary details.
   - Exploration involves taking actions such as jump or retry even when the current information might seem sufficient. F
   - For instance, if a recent conversation mentions, "I like sushi," but an older one states, "I have no particular preference for Japanese or Chinese food," consider exploring further to ensure a comprehensive answer to a query like, "Sushi or Chinese for lunch?"
   - Keep exploration cautious and deliberate, as it may increase token usage and time. Always weigh the potential value of additional information against its cost.
   - If different groups contain similar information, jump to the group to see its original content and put it in the evidence, directly use groups summary may missing details.
4. **Search History Utilization**:
   - Record each round of the search process in the search history:
      . Include the selected group summaries and conversations, along with the action taken in that round (e.g., jump, retry, or end).
      . For exploration actions (jump or retry), clearly document the reasoning and ensure the rewritten query avoids similarity to previous unsuccessful attempts.
   - Use the search history to avoid repeating previously attempted queries or revisiting the same group unnecessarily.
   - When rewriting a query in a retry, ensure it is substantively different from previous queries while maintaining alignment with the user's intent.

5. **Output Format**:
   - Provide your decision in the following JSON format:
**End the search**
```json
{{
    "action":"end",
    "reason":"sufficient", # or insufficient
    "think":"",
    "evidence":[],
}}
```
**Jump to another group**
```json
{{
    "action":"jump",
    "id":"related_summary_id",
    "think":"",
    "evidence":[],
}}
```
**Retry with a rewritten query**
```json
{{
    "action":"retry",
    "keywords":"",
    "think":"",
    "evidence":[],
}}
```
6. **Limits**:
   - The maximum number of search rounds is capped at `4`. Use this limit effectively to balance thoroughness and efficiency.

7. **Evaluation Criteria**:
   - Your goal is to maximize the relevance and completeness of retrieved information.
   - Avoid premature termination of the search unless the evidence is unquestionably sufficient.
   - Ensure that rewritten queries aim to enhance precision or breadth without replicating past failures.

---

### **Example Output**:
```json
{{
    "action": "end",
    "reason": "The closest group’s summary and top conversations provide sufficient evidence to answer the query.",
    "think": "The retrieved conversations directly address the user’s query, and additional exploration is unlikely to yield significant new insights.",
    "evidence": [
        {{
            "text": "user:I like sushi",
            'time': '2024/11/01 12:04'
        }},
        {{
            "text": "user:I have no particular preference for Japanese or Chinese food.",
            'time': '2024/9/01 12:05'
        }}
    ]
}}
```

Input Information: {search_info}

Search history: {search_history}"""