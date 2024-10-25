import os
from pathlib import Path
from dotenv import load_dotenv
import glob
from typing import List, Dict
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
import logging
from swarm_models import OpenAIChat

# Set up workspace directory
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['WORKSPACE_DIR'] = WORKSPACE_DIR

from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.swarm_router import SwarmRouter
from swarms_memory import ChromaDB
import subprocess

# Add to existing imports
from concurrent_workflow_async import AsyncConcurrentWorkflow

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

def load_text_documents(docs_folder: str) -> List[Dict[str, str]]:
    """
    Load all text documents from the specified folder.
    Returns a list of dictionaries containing document content and metadata.
    """
    documents = []
    docs_path = Path(docs_folder)
    
    # Create docs directory if it doesn't exist
    docs_path.mkdir(exist_ok=True)
    
    # Read all text files in the docs folder (both .txt and other text files)
    for text_file in docs_path.glob("*.*"):
        if text_file.suffix.lower() in ['.txt', '.md', '.text']:
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "id": str(text_file.stem),
                        "content": content,
                        "metadata": {
                            "source": str(text_file),
                            "type": "text",
                            "filename": text_file.name
                        }
                    })
            except Exception as e:
                print(f"Error reading file {text_file}: {str(e)}")
    
    return documents

# Add LightRAG configuration
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        "Meta-Llama-3.1-70B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1",
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="nomic-embed-text:latest",
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1",
    )

async def initialize_lightrag():
    try:
        workspace_dir = os.path.join(WORKSPACE_DIR, "lightrag_workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        print(f"Initializing LightRAG with workspace: {workspace_dir}")
        
        # Test embedding to get dimension
        test_text = ["This is a test sentence."]
        embedding = await embedding_func(test_text)
        embedding_dimension = embedding.shape[1]
        print(f"Embedding dimension: {embedding_dimension}")
        
        rag = LightRAG(
            working_dir=workspace_dir,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func
            ),
        )
        print("LightRAG initialized successfully")
        return rag
    except Exception as e:
        print(f"Error initializing LightRAG: {e}")
        return None

# Modify document loading to use both ChromaDB and LightRAG
async def load_and_process_documents(docs_folder: str):
    documents = load_text_documents(docs_folder)
    
    # Initialize LightRAG
    rag = await initialize_lightrag()
    
    if documents and rag:
        # Process documents sequentially to avoid event loop conflicts
        for doc in documents:
            try:
                # Add to ChromaDB for agent memory
                memory.add(document=doc["content"])
                
                # Add to LightRAG directly
                await rag.insert(doc["content"])
                print(f"Document {doc['id']} processed successfully with LightRAG")
            except Exception as e:
                print(f"Error processing document {doc['id']}: {str(e)}")
    
    return rag

# DOCUMENT_RETRIEVER_PROMPT
DOCUMENT_RETRIEVER_PROMPT = """You are a highly specialized document retrieval agent that uses both ChromaDB and LightRAG. Your tasks include:
1. Using LightRAG for advanced document retrieval with multiple search modes (naive, local, global, hybrid)
2. Falling back to ChromaDB when needed for historical context and agent memory
3. Combining results from both systems for comprehensive information retrieval
4. Implementing advanced RAG techniques for accurate retrieval
5. Providing specific quotes and references from the source documents
Provide a list of relevant documents along with their relevance scores and specific content matches."""

# Update document retriever agent to include LightRAG
class EnhancedDocumentRetrieverAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag = None
    
    async def set_rag(self, rag):
        self.rag = rag
        
    async def get_relevant_documents(self, query: str):
        results = []
        
        # Get results from LightRAG using different modes
        if self.rag:
            try:
                for mode in ["naive", "local", "global", "hybrid"]:
                    rag_result = await self.rag.query(
                        query, 
                        param=QueryParam(mode=mode)
                    )
                    results.append({"source": f"LightRAG ({mode})", "content": rag_result})
            except Exception as e:
                print(f"LightRAG query error: {e}")
        
        # Get results from ChromaDB - using query instead of search
        try:
            chroma_results = self.long_term_memory.query(query)
            # Combine characters into complete strings
            if isinstance(chroma_results, list):
                complete_text = ''.join(r if isinstance(r, str) else r.get('content', '') 
                                    for r in chroma_results)
                results.append({"source": "ChromaDB", "content": complete_text})
        except Exception as e:
            print(f"ChromaDB query error: {e}")
        
        return results

# Initialize ChromaDB with document loading
docs = load_text_documents("memory")
memory = ChromaDB(
    metric="cosine",
    n_results=5,
    output_dir="results",
    docs_folder="memory",
    verbose=True
)

# Add documents to ChromaDB one by one using the add method
if docs:
    for doc in docs:
        try:
            memory.add(
                document=doc["content"]  # Only pass the document content
            )
        except Exception as e:
            print(f"Error adding document {doc['id']}: {str(e)}")

# Initialize the language model
model = OpenAIChat(
    openai_api_base="https://api.sambanova.ai/v1",
    openai_api_key=api_key,
    model_name="Meta-Llama-3.1-70B-Instruct",
    temperature=0.1,
)

# Define specialized system prompts for each agent
SUMMARIZER_PROMPT = """You are an expert summarization agent. Your core competencies include:
1. Extracting key insights from complex documents
2. Summarizing information into concise, numbered bullet points
3. Ensuring each bullet point is no more than three sentences
4. Maintaining the original meaning and context of the information
Deliver clear, concise summaries that capture the essence of the documents while highlighting crucial information."""

CITATION_TRACER_PROMPT = """You are a specialized citation and source tracing agent. Your key responsibilities include:
1. Extracting citations from retrieved documents
2. Tracing citations back to their original ground truth documents
3. Handling various citation formats including APA, MLA, and Chicago
4. Ensuring the accuracy and completeness of citation information
Provide a list of citations along with their corresponding source documents."""

CONTEXTUAL_RELEVANCE_PROMPT = """You are a highly skilled contextual relevance agent. Your expertise covers:
1. Analyzing user context, including specific business functions or areas of interest
2. Tailoring responses based on the user's context
3. Ensuring the relevance of information to the user's specific needs
4. Adapting to various industry-specific and domain-specific contexts
Deliver context-aware insights that are directly relevant to the user's specific situation."""

OUTPUT_FORMATTER_PROMPT = """You are an expert output formatting agent. Your core competencies include:
1. Formatting responses in a concise, magazine-style format
2. Creating brief, numbered bullet points (no more than three sentences each)
3. Incorporating accurate citations linked to the original data source
4. Ensuring the output is clear, structured, and easy to read
Provide well-formatted responses that are both informative and visually appealing."""

QUALITY_CHECKER_PROMPT = """You are a specialized quality and accuracy checking agent. Your tasks include:
1. Evaluating the accuracy of summarized information against original documents
2. Assessing the relevance of the response to the user's query
3. Providing confidence scores for each piece of information
4. Identifying any potential inaccuracies or inconsistencies
Deliver a comprehensive quality report, including accuracy scores and any identified issues."""

# Initialize specialized agents
document_retriever_agent = EnhancedDocumentRetrieverAgent(
    agent_name="Document-Retriever",
    system_prompt=DOCUMENT_RETRIEVER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    long_term_memory=memory,
    dynamic_temperature_enabled=True,
    saved_state_path="document_retriever_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

summarizer_agent = Agent(
    agent_name="Summarizer",
    system_prompt=SUMMARIZER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="summarizer_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

citation_tracer_agent = Agent(
    agent_name="Citation-Tracer",
    system_prompt=CITATION_TRACER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="citation_tracer_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

contextual_relevance_agent = Agent(
    agent_name="Contextual-Relevance",
    system_prompt=CONTEXTUAL_RELEVANCE_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="contextual_relevance_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

output_formatter_agent = Agent(
    agent_name="Output-Formatter",
    system_prompt=OUTPUT_FORMATTER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="output_formatter_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

quality_checker_agent = Agent(
    agent_name="Quality-Checker",
    system_prompt=QUALITY_CHECKER_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="quality_checker_agent.json",
    user_name="summarize_system",
    retry_attempts=1,
    context_length=200000,
    output_type="string"
)

# Initialize the base agent list
print("Initializing agent list:")
agent_list = [
    contextual_relevance_agent,
    document_retriever_agent,
    summarizer_agent,
    citation_tracer_agent,
    quality_checker_agent,
    output_formatter_agent
]
print(f"Number of agents: {len(agent_list)}")
for agent in agent_list:
    print(f"- {agent.agent_name}")

# Modify the router run method to handle async operations
class AsyncSwarmRouter(SwarmRouter):
    def __init__(self, name, description, max_loops, agents, swarm_type):
        # Call parent class initialization first
        super().__init__(
            name=name,
            description=description,
            max_loops=max_loops,
            agents=agents,
            swarm_type=swarm_type
        )
        
        self.logs = []  # Add this line to store logs
        
        # Initialize base attributes
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents.copy()
        self.swarm_type = swarm_type
        
        # Setup logging with custom handler to capture logs
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        class LogCaptureHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
            
            def emit(self, record):
                log_entry = {
                    'timestamp': record.asctime if hasattr(record, 'asctime') else record.created,
                    'level': record.levelname,
                    'message': record.getMessage()
                }
                self.log_list.append(log_entry)
        
        # Add our custom handler
        if not self.logger.handlers:
            handler = LogCaptureHandler(self.logs)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Also add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if not self.agents or len(self.agents) == 0:
            raise ValueError("Agents list must not be empty")
            
        self.logger.info(f"Initializing AsyncSwarmRouter with {len(self.agents)} agents")
        if self.swarm_type == "ConcurrentWorkflow":
            # Pass a copy of the agents list to AsyncConcurrentWorkflow
            self.swarm = AsyncConcurrentWorkflow(agents=self.agents.copy())
            self.logger.info(f"Initialized AsyncConcurrentWorkflow with {len(self.agents)} agents")

    def get_logs(self):
        """Return the captured logs"""
        return self.logs

    async def async_run(self, task, *args, **kwargs):
        if not self.agents or len(self.agents) == 0:
            raise ValueError("No agents available for processing")
        try:
            self.logger.info(f"Running task with {len(self.agents)} agents")
            if self.swarm_type == "ConcurrentWorkflow":
                result = await self.swarm.async_process_agents(task, *args, **kwargs)
                return result
        except Exception as e:
            self.logger.error(f"Error occurred while running task: {str(e)}")
            raise e

# Move these functions outside the if __name__ == "__main__": block
async def initialize_system():
    # Initialize LightRAG and process documents
    rag = await load_and_process_documents("docs")
    
    # Set RAG for the existing document retriever agent
    await document_retriever_agent.set_rag(rag)
    
    # Verify agent list before router initialization
    if not agent_list or len(agent_list) == 0:
        raise ValueError("Agent list is empty before router initialization")
        
    # Initialize router with existing agent list
    router = AsyncSwarmRouter(
        name="summarize-ai-swarm",
        description="Analyze and summarize documents based on user queries",
        max_loops=1,
        agents=agent_list.copy(),  # Pass a copy of the agent list
        swarm_type="ConcurrentWorkflow"
    )
    
    return router

async def run_query(router, query):
    try:
        result = await router.async_run(query)
        print(result)
        
        for log in router.get_logs():
            print(f"{log.timestamp} - {log.level}: {log.message}")
            
        return result
    except Exception as e:
        print(f"Error running query: {e}")
        return None

# Keep the main block for direct script execution
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    async def main():
        try:
            router = await initialize_system()
            await run_query(
                router,
                "how could mycelium evolve the ability to harness the sun's power like a Dyson Sphere?"
            )
        except Exception as e:
            print(f"Error in main execution: {e}")
            raise e

    # Run with proper asyncio handling
    asyncio.run(main())
