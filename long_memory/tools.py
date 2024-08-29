tools = [
  {
    "type": "function",
    "function": {
      "name": "rewrite_description",
      "parameters": {
        "type": "object",
        "properties": {
          "rewrite": {"type": "string",
                      "description":"Decide if need to rewrite",
                      "enum": ["True", "False"]},
          "description": {
            "type": "string",
            "description": "A proper new description, empty if no need rewrite",
          },
        },
        "required": ["rewrite"],
      },
    }
  }
]