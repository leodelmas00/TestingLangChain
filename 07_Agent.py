import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_exa import ExaSearchRetriever

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

exa = ExaSearchRetriever(num_results=3)

user_query = "Is Akira Toriyama alive?"

print(f"Question: {user_query}")
print("-" * 50)

#Search with Exa
print("Searching with Exa...")
search_results = exa.invoke(user_query)
print(f"Found {len(search_results)} results")

#Limit result size to avoid TPM issues
results_text = str(search_results)[:1500]

#Ask LLM with truncated search results
prompt = f"""Based on these search results, answer concisely:

Search results:
{results_text}

Question: {user_query}

Answer:"""

response = llm.invoke(prompt)

print("-" * 50)
print(f"Agent response: {response.content}")