compress_conversation_prompt="""Please condense the following conversation into a single sentence of no more than 50 words,
preserving all key details and essential information.
Make the summary as clear and comprehensive as possible, capturing the core points of the dialogue.
Chat logs:{chat_logs}
"""

generate_answer_prompt="""You are an intelligent chatbot agent responsible for responding to user questions or engaging in conversation.
You will receive the following four inputs:
1.Chat History:The record of the current conversation.
2.Short-term Relevant Memory:Recent information from the latest interactions.
3.Long-term Relevant Memory:Older infomation from past interactions.
4.User message:The user message.
Based on these inputs, respond to the user, make the conversation feel coherent, natural, and personalized.

Chat History:{chat_history}
Short-term Relevant Memory:{short_relevant_memory}
Long-term Relevant Memory:{long_relevant_memory}
User message:{user_message}
"""