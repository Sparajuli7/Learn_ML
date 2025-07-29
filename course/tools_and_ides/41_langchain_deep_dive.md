# LangChain Deep Dive

## Overview
LangChain is a framework for developing applications powered by language models. This guide covers chains, agents, memory systems, and production deployments for 2025.

## Table of Contents
1. [LangChain Fundamentals](#langchain-fundamentals)
2. [Chains and Prompts](#chains-and-prompts)
3. [Agents and Tools](#agents-and-tools)
4. [Memory Systems](#memory-systems)
5. [Vector Stores](#vector-stores)
6. [Production Deployments](#production-deployments)

## LangChain Fundamentals

### Basic Setup
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any
import os

# Initialize language model
llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(temperature=0.7)

# Basic usage
response = llm("What is the capital of France?")
print(response)

# Chat model usage
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
response = chat_model(messages)
print(response.content)
```

### Prompt Templates
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Simple prompt template
template = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

prompt = template.format(country="France")
response = llm(prompt)

# Chat prompt template
system_template = "You are a helpful assistant that provides information about countries."
human_template = "What is the capital of {country}?"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

messages = chat_prompt.format_messages(country="France")
response = chat_model(messages)
```

## Chains and Prompts

### LLM Chain
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create LLM chain
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short summary about {topic}."
)

chain = LLMChain(llm=llm, prompt=template)

# Run chain
response = chain.run("artificial intelligence")
print(response)

# Run with multiple inputs
response = chain.run({"topic": "machine learning"})
print(response)
```

### Sequential Chains
```python
from langchain.chains import SimpleSequentialChain, SequentialChain

# First chain: Generate topic
topic_template = PromptTemplate(
    input_variables=["subject"],
    template="Generate a topic related to {subject}."
)
topic_chain = LLMChain(llm=llm, prompt=topic_template)

# Second chain: Write about topic
writing_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a paragraph about {topic}."
)
writing_chain = LLMChain(llm=llm, prompt=writing_template)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, writing_chain],
    verbose=True
)

response = overall_chain.run("technology")
print(response)

# Sequential chain with multiple inputs/outputs
topic_template = PromptTemplate(
    input_variables=["subject"],
    template="Generate a topic related to {subject}."
)
topic_chain = LLMChain(llm=llm, prompt=topic_template, output_key="topic")

writing_template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a {style} paragraph about {topic}."
)
writing_chain = LLMChain(llm=llm, prompt=writing_template, output_key="content")

overall_chain = SequentialChain(
    chains=[topic_chain, writing_chain],
    input_variables=["subject", "style"],
    output_variables=["topic", "content"],
    verbose=True
)

response = overall_chain({"subject": "science", "style": "informative"})
print(response)
```

### Router Chains
```python
from langchain.chains.router import MultiPromptRouter
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# Define different prompts for different topics
physics_template = PromptTemplate(
    input_variables=["question"],
    template="You are a physics expert. Answer this question: {question}"
)

math_template = PromptTemplate(
    input_variables=["question"],
    template="You are a mathematics expert. Answer this question: {question}"
)

# Create router
destinations = [
    "physics: For questions about physics",
    "math: For questions about mathematics"
]

default_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering questions about mathematics",
        "prompt_template": math_template
    }
]

router_chain = MultiPromptRouter.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    default_chain=LLMChain(llm=llm, prompt=default_prompt)
)

# Use router
response = router_chain.run("What is the speed of light?")
print(response)
```

## Agents and Tools

### Basic Agent
```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType

# Define tools
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet for current information"
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
response = agent.run("What is the latest news about AI?")
print(response)
```

### Custom Tools
```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for performing mathematical calculations"
    args_schema = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Evaluate mathematical expression"""
        try:
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error evaluating {expression}: {str(e)}"
    
    def _arun(self, expression: str) -> str:
        """Async version of _run"""
        return self._run(expression)

# Custom tool for weather
class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a city"
    args_schema = WeatherInput
    
    def _run(self, city: str) -> str:
        """Get weather for city (simplified)"""
        # In practice, you would call a weather API
        return f"Weather for {city}: Sunny, 25°C"
    
    def _arun(self, city: str) -> str:
        return self._run(city)

# Use custom tools
tools = [
    CalculatorTool(),
    WeatherTool()
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("What is 15 * 23 and what's the weather in Paris?")
print(response)
```

### Agent with Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create agent with memory
prompt = PromptTemplate(
    input_variables=["input", "chat_history", "agent_scratchpad"],
    template="You are a helpful assistant. Use the following tools to answer questions:\n\n{tools}\n\nUse this format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nQuestion: {input}\n{agent_scratchpad}"
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Run agent with memory
response1 = agent_executor.run("What is 2 + 2?")
response2 = agent_executor.run("What was the previous calculation?")
print(f"First response: {response1}")
print(f"Second response: {response2}")
```

## Memory Systems

### Conversation Buffer Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation
response1 = conversation.predict(input="Hi, my name is Alice.")
response2 = conversation.predict(input="What's my name?")
print(f"Response 1: {response1}")
print(f"Response 2: {response2}")
```

### Conversation Summary Memory
```python
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

# Create summary memory
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Long conversation
responses = []
responses.append(conversation.predict(input="Hi, I'm Alice."))
responses.append(conversation.predict(input="I work as a software engineer."))
responses.append(conversation.predict(input="I love programming in Python."))
responses.append(conversation.predict(input="What do you know about me?"))

for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

### Vector Store Memory
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Create memory
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs=dict(k=5))
)

# Add information to memory
memory.save_context(
    {"input": "Alice is a software engineer"},
    {"output": "I understand that Alice is a software engineer."}
)

memory.save_context(
    {"input": "Alice loves Python programming"},
    {"output": "I know that Alice loves Python programming."}
)

# Retrieve from memory
memory.load_memory_variables({"prompt": "Tell me about Alice"})
```

## Vector Stores

### Chroma Vector Store
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load documents
loader = TextLoader("data.txt")
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

# Search
query = "What is machine learning?"
docs = vectorstore.similarity_search(query, k=3)
for doc in docs:
    print(doc.page_content)
```

### Pinecone Vector Store
```python
from langchain.vectorstores import Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")

# Create vector store
index_name = "langchain-demo"
vectorstore = Pinecone.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=index_name
)

# Search
query = "What is deep learning?"
docs = vectorstore.similarity_search(query, k=3)
for doc in docs:
    print(doc.page_content)
```

### Retrieval QA Chain
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Ask questions
question = "What is the difference between supervised and unsupervised learning?"
answer = qa_chain.run(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Production Deployments

### FastAPI Integration
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import uvicorn

app = FastAPI()

# Define request/response models
class QueryRequest(BaseModel):
    question: str
    context: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

# Create chain
template = PromptTemplate(
    input_variables=["question", "context"],
    template="Context: {context}\nQuestion: {question}\nAnswer:"
)

chain = LLMChain(llm=llm, prompt=template)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Run chain
        response = chain.run({
            "question": request.question,
            "context": request.context
        })
        
        return QueryResponse(
            answer=response,
            sources=[]  # Add sources if using retrieval
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### LangChain with Streaming
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List

class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        print("Starting LLM...")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token, end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        print("\nLLM finished.")

# Use streaming
streaming_llm = OpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[CustomStreamingHandler()]
)

response = streaming_llm("Write a short story about a robot.")
```

### LangChain with Caching
```python
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
import langchain

# Enable caching
langchain.cache = InMemoryCache()

# Or use SQLite cache
langchain.cache = SQLiteCache(database_path=".langchain.db")

# Now all LLM calls will be cached
response1 = llm("What is the capital of France?")
response2 = llm("What is the capital of France?")  # This will use cache
```

## Conclusion

LangChain provides a comprehensive framework for building LLM applications. Key areas include:

1. **Chains**: Sequential and router chains for complex workflows
2. **Agents**: Tool-using agents with memory and reasoning
3. **Memory**: Various memory systems for conversation context
4. **Vector Stores**: Document storage and retrieval
5. **Production**: FastAPI integration, streaming, and caching

The framework continues to evolve with new features for more sophisticated LLM applications.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Examples](https://github.com/langchain-ai/langchain/tree/master/examples) 