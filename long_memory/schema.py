from datetime import datetime

# A chlid (memory) in the group
CHILD = {
    "text":str, # the content of the memory
    "time":datetime, # (optional)
    "vector":list, # the embedding vector of the memory, if no vector provided, using "text-embedding-3-small" automtically (optional)
    "origin_text":str, # original text of the memory (optional)
}

# A group content
GROUP = {
    "description":str, # the introduce of this memory group
    "child":list(CHILD) # the memories of the group
}

# Result of search in the weaviate
RELAVANT_MEMORY = {
    "group_description":str, # the introduce of this memory group
    "memory":list(CHILD) # relavant k memory
}

# Organize the RELAVANT_MEMORY and used in the prompt of recall_search
SEARCH_INFO = {
    "closest_summary":str,
    "similar_snippets":list,
    "related_summaries":list
}

# Search history used in the prompt of recall_search and final return in the recall stage
SEARCH_HISTORY = {
    "search_times":0,
    "used_keywords":[],
    "searched_memory":[],
    "thought":"",
    "evidence":[],
}

# Memory schema in the Weaviate
# https://weaviate.io/developers/weaviate/config-refs/schema
GROUP_SCHEMA = {
    "class": "long_memory_group",
    "description": "abstract of msg group",
    "vectorIndexType": "hnsw",
    "properties": [
        {
            "name": "time",
            "dataType": ["date"]
        },
        {
            "name": "text",
            "dataType": ["text"],
        },
    ],
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small"
        }
    }
}

CHILD_SCHEMA = {
    "class": "long_memory_child",
    "description": "chat history",
    "vectorIndexType": "hnsw",
    "properties": [
        {
            "name": "time",
            "dataType": ["date"]
        },
        {
            "name": "text",
            "dataType": ["text"]
        },
        {
            "name": "origin_text",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": "true"}
            }
        },
    ],
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small"
        }
    }
}