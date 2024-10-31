LOG_SCHEMA = {
    "class": "short_memory_chatlog",
    "description": "chat log",
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