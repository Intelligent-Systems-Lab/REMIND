rewrite_prompt = """You are a rewriter, you have two descriptions, they need to be merged to one description, first is {description_1}, second is {description_2}, if first has enough information that don't have to rewrite, if not, rewrite a new description"""

summary_prompt = "Watch the chat below. Summarize this chat record\n{chat_log}"