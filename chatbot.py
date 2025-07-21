from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os


load_dotenv()

# ✅ Step 1: Create HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Chat-compatible model
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=512,
    timeout=60
)

# ✅ Step 2: Pass it as `llm` into ChatHuggingFace
model = ChatHuggingFace(llm=llm)


chat_history = [
    SystemMessage(content='you are a helpful assistant')
]

while True:
    user_input = input ('you: ')
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:',result.content)

print(chat_history)

