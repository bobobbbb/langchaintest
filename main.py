import os

from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek-chat", model_provider="deepseek")

from langchain_core.messages import HumanMessage

# print(model.invoke([HumanMessage(content="Hi! I'm Max",name="Max")]))

# 这是一个内存检查点保存器，用于在对话过程中保存状态
# 类似于 agent 中的记忆模块，让 agent 能记住之前的对话内容
# 数据存储在内存中，程序重启后会丢失
from langgraph.checkpoint.memory import MemorySaver
# START：图的起始节点标识符
# MessagesState：预定义的状态模式，专门处理消息列表
# StateGraph：状态图类，用于构建工作流
from langgraph.graph import START, MessagesState, StateGraph

# # Define a new graph
# workflow = StateGraph(state_schema=MessagesState)


# # Define the function that calls the model
# def call_model(state: MessagesState):
#     response = model.invoke(state["messages"])
#     return {"messages": response}


# # Define the (single) node in the graph
# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# # Add memory
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}

# query = "Hi! I'm Bob."

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()  # output contains all messages in state

# query = "What's my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# config = {"configurable": {"thread_id": "abc234"}}

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# config = {"configurable": {"thread_id": "abc123"}}

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}

query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()