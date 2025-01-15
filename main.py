import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
import ast
import re

load_dotenv()

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("Please set the 'OPENAI_API_KEY' environment variable")
    
    db = SQLDatabase.from_uri("sqlite:///chinook.sqlite")
    print(db.dialect)
    print(db.get_usable_table_names())
    db.run("SELECT * FROM Artist LIMIT 10;")

    llm = ChatOpenAI(model="gpt-4o-mini")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

    assert len(prompt_template.messages) == 1
    prompt_template.messages[0].pretty_print()
    system_message = prompt_template.format(dialect="SQLite", top_k=5)

    retriever_tool = get_retriever_tool(db)
    tools.append(retriever_tool)

    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )

    agent_prompt = f"{system_message}\n\n{suffix}"

    agent_executor = create_react_agent(llm, tools, state_modifier=agent_prompt)

    question = "How many albums does alis in chain have?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

def get_vector_store(db):
    cur_dir = os.path.dirname(__file__) 
    file_path = os.path.join(cur_dir, ".vectors")
    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = None

    if os.path.isfile(file_path):
        # The vectors file exists. Load the Vector store from file
        print(f"Initializing vector store from file :-) {file_path}")
        vector_store = InMemoryVectorStore.load(file_path, embeddings)
    else:
        print(f"Vectors file ({file_path}) not found. Creating it...")
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_texts(artists + albums)
        vector_store.dump(file_path)
        print("Done :-)")
    
    return vector_store

def get_retriever_tool(db):
    vector_store = get_vector_store(db)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    description = (
        "Use to look up values to filter on. Input is an approximate spelling "
        "of the proper noun, output is valid proper nouns. Use the noun most "
        "similar to the search."
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )

    return retriever_tool
    

if __name__ == "__main__":
    main()

