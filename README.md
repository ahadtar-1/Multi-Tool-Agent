# Multi-Tool-Agent

This project implements a Multi-Tool Agent for the purpose of answering queries related to calculations, private documents, and present-day information.

## Run the application locally

### Set up and Installation

This project can be run in a development enviroment which facilitates Python. For that purpose a conda environment should be created (**python 3.13**) to preserve the packages and dependencies. The requirements file should be executed after the conda environment is created to import the specific dependencies needed to run the project. Once the dependencies are imported then a .env file should be created and a Google AI API Key, Open AI API Key, Pinecone API Key, Tavily API Key, Langsmith API Key, Langsmith Project Name, and Langsmith Tracing Message must be inserted. 

```bash
conda create -n multitoolagent python=3.13

conda activate multitoolagent

pip install -r requirements.txt
```

### Case 1 - Run the application in a development environment

```bash
python app.py
```

### Case 2 - Run the application through external client

```bash
python app.py
```

#### APIs

#### Query Agent

A post request would be sent to the FastAPI application. It would comprise of a user query which would be sent to the Agent. The response sent back from the FastAPI application would be the response to the user's query.

#### API Endpoint

```
0.0.0.0:8080/agentchat/
```

#### Payload
```
{
    "query" : What is 1541 + 200?

    key must be the string "query"
    value must be the query 
}
```

### Tools and Technologies

* Python
* Langchain
* Gradio (Front-end)
* Pinecone (Private docs retrieval)
* Tavily (Web search)
* Langsmith (Monitoring and Tracing)
* Google Gemini 3.5 Flash-Lite (LLM for the Agent) 
* OpenAI text-embedding-3 large
* FastAPI (API Endpoints, Input Validation Guardrails)
