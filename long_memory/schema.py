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

# Result of get relavant memory
RELAVANT_MEMORY = {
    "group_description":str, # the introduce of this memory group
    "memory":list(CHILD) # relavant k memory
}

# Memory schema in the Weaviate
CLASS_SCHEMA = {
    "class": "base",
    "properties": [
        {
            "name": "time",
            "dataType": ["date"]
        },
        {
            "name": "text",
            "dataType": ["text"]
        }
    ]
}