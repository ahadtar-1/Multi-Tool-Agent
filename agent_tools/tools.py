"""
The module comprises of the respective tools for the Multi-tool Agent. 
"""

import os
import re
import ast
import operator
import langchain
from langchain.tools import tool, ToolException
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PineconeException
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
index_name = "faqsampleindexjuly2026"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

math_operators = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod
}

calculator_schema = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"}
    },
    "required": ["expression"]
}

retriever_schema = {
    "type": "object",
    "properties": {
        "query": {"type": "string"}
    },
    "required": ["query"]
}

web_search_schema = {
    "type": "object",
    "properties": {
        "query": {"type": "string"}
    },
    "required": ["query"]
}


def safe_eval(tree_node):
    if isinstance(tree_node, ast.Constant) and isinstance(tree_node.value, (int, float)):
        return tree_node.value
    elif isinstance(tree_node, ast.BinOp):
        left_node = safe_eval(tree_node.left)
        right_node = safe_eval(tree_node.right)
        return math_operators[type(tree_node.op)](left_node, right_node)
    elif isinstance(tree_node, ast.UnaryOp):
        operand = safe_eval(tree_node.operand)
        return math_operators[type(tree_node.op)](operand)
    raise ValueError("Unsupported mathematical expression")


@tool(args_schema=calculator_schema)
def calculator(expression: str)-> str:
    """
    Evaluates all numeric and mathematical operations.

    Parameters
    ----------
    expression: str
        A string containing a valid mathematical expression.

    Returns
    --------
    str
        The result of the mathematical expression.
    
    """
    
    
    try:
        cleaned_expression = expression.replace('^', '**')
        cleaned_expression = re.sub(r'\bsqrt\s*\(([^)]+)\)',r'(\1)**0.5', cleaned_expression, flags=re.IGNORECASE)
        parsed_expression = ast.parse(cleaned_expression, mode='eval')
        result = safe_eval(parsed_expression.body)
        return str(result)
    except Exception as e:
        raise RuntimeError(f"Invalid mathematical expression '{expression}': {str(e)}")


@tool(args_schema=retriever_schema)
def knowledge_retriever(query: str)-> str:
    """
    Caters to only the queries relevant to Windows Server Licensing and Pricing FAQ, Windows Server 2012, and Windows Server 2012 Licensing and Pricing FAQ and not any other query or question which is not relevant to them. Retrieves text passages relevant to the specific query from a pinecone vector store.

    Parameters
    ----------
    query: str
        A natural-language string representing the user query relevant to Windows Server Licensing and Pricing FAQ, Windows Server 2012, and Windows Server 2012 Licensing and Pricing FAQ.

    Returns
    -------
    str
        The similar docs with respect to the query.
    
    """
    
    
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        similar_docs = vectorstore.similarity_search_with_score(query, k=1)

        if similar_docs == None:
            return "No similar docs found."
    
        for doc, score in similar_docs:
            similar_doc = doc.page_content
     
        if isinstance(similar_doc, str):
            splitted_text = similar_doc.split('\nAnswer: ') 
            answer = splitted_text[1]
            return answer  
    except Exception as e:
        raise RuntimeError("Please check your Pinecone API key and try again.")


@tool(args_schema=web_search_schema)
def web_search(query: str) -> str:
    """
    Caters to only the queries for any present day information or factual updates that occurred after January 1, 2025. Performs a real-time web search using Tavily. If the web search fails, do NOT fall back to internal knowledge, instead return the error message provided by the tool.

    Parameters
    ----------
    query: str
        A natural-language string representing the search query.

    Returns
    -------
    dict
        A dictionary of the top results, or an explicit error message.
    
    """
       
    
    try:
        tavily_search_func = TavilySearchAPIWrapper()
        results = tavily_search_func.results(query=query, max_results=2)
        if not results:
            return "WebSearchTool returned no relevant results."
        return str(results)
    except Exception as e:
        raise RuntimeError("Please check your Tavily API key and try again.")
