import os
from pathlib import Path
from dotenv import load_dotenv
import glob
from typing import List, Dict

# Set up workspace directory
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['WORKSPACE_DIR'] = WORKSPACE_DIR

from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.swarm_router import SwarmRouter
from swarms_memory import ChromaDB
import subprocess

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

# Initialize ChromaDB with document loading
docs = load_text_documents("docs")
memory = ChromaDB(
    metric="cosine",
    n_results=5,
    output_dir="results",
    docs_folder="docs",
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
    openai_api_base="http://localhost:11434/v1",
    openai_api_key=api_key,
    model_name="hf.co/arcee-ai/SuperNova-Medius-GGUF:f16",
    temperature=0.1,
)

# Define specialized system prompts for each agent
DOCUMENT_RETRIEVER_PROMPT = """You are a highly specialized document retrieval agent. Your tasks include:
1. Analyzing user queries to understand the information needs
2. Retrieving relevant documents from the repository based on the query
3. Accessing and searching through the available text documents in the docs folder
4. Implementing advanced Retrieval-Augmented Generation (RAG) techniques for accurate retrieval
5. Providing specific quotes and references from the source documents
Provide a list of relevant documents along with their relevance scores and specific content matches."""

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
document_retriever_agent = Agent(
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

# Initialize the SwarmRouter
print("Initializing SwarmRouter with agents:")
agent_list = [
    document_retriever_agent,
    summarizer_agent,
    citation_tracer_agent,
    contextual_relevance_agent,
    output_formatter_agent,
    quality_checker_agent
]
print(f"Number of agents: {len(agent_list)}")
for agent in agent_list:
    print(f"- {agent.agent_name}")

router = SwarmRouter(
    name="summarize-ai-swarm",
    description="Analyze and summarize documents based on user queries",
    max_loops=1,
    agents=agent_list,
    swarm_type="ConcurrentWorkflow"
)

# Add debug output
print(f"Initializing SwarmRouter with {len(router.agents)} agents")

if __name__ == "__main__":
    # Run a comprehensive private equity document analysis task
    result = router.run(
        "how could mycelium evlove the ability to traverse the stars organically?"
    )
    print(result)

    # Retrieve and print logs
    for log in router.get_logs():
        print(f"{log.timestamp} - {log.level}: {log.message}")
