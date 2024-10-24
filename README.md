# Summarize App

The Summarize App is a sophisticated document analysis and summarization tool that leverages advanced AI models and frameworks to provide comprehensive insights from text documents. It utilizes Swarms, a multi-agent system, and LightRAG, a Retrieval-Augmented Generation (RAG) framework, to enhance document retrieval, summarization, and contextual analysis.

## Codebase Structure

- **`app.py`**: The main application file that initializes the system, loads documents, and sets up the agents and router.
- **`concurrent_workflow_async.py`**: Contains the `AsyncConcurrentWorkflow` class, which manages the concurrent execution of agents using asyncio.
- **`gradio_app.py`**: Provides a Gradio interface for interacting with the app, allowing users to input queries and view results.
- **`lightrag/`**: A directory containing the LightRAG framework, which includes modules for language model interaction, storage, and operations.
- **`.env` and `.env.example`**: Environment configuration files for setting API keys and other necessary configurations.

## Agent Workflow

The Summarize App uses a multi-agent system to perform various tasks related to document analysis and summarization. The agents are coordinated by a `SwarmRouter`, which manages their execution and communication. Here's how the workflow is structured:

1. **Initialization**: 
   - The system is initialized by loading text documents and setting up the LightRAG framework.
   - Agents are instantiated with specific roles and system prompts.

2. **Agent Execution**:
   - The `AsyncSwarmRouter` manages the execution of agents using the `AsyncConcurrentWorkflow` class.
   - Agents perform their tasks concurrently, with each agent focusing on a specific aspect of the document analysis process.

3. **Task Handling**:
   - Agents are designed to handle specific tasks such as document retrieval, summarization, citation tracing, contextual relevance analysis, output formatting, and quality checking.
   - Each agent uses a specialized system prompt to guide its operations and ensure task-specific outputs.

4. **Result Aggregation**:
   - The results from all agents are aggregated and processed to provide a comprehensive response to the user's query.
   - Logs are maintained for each agent's activity, allowing for transparency and debugging.

## Swarm Agents

The Swarm agents are specialized components that perform distinct tasks within the system. Each agent is initialized with a specific role and system prompt, guiding its operations. The key agents include:

- **Document Retriever Agent**: Utilizes LightRAG and ChromaDB for advanced document retrieval, combining results from both systems for comprehensive information retrieval.
- **Summarizer Agent**: Extracts key insights from documents and summarizes them into concise bullet points.
- **Citation Tracer Agent**: Extracts citations from retrieved documents and traces them back to their original sources.
- **Contextual Relevance Agent**: Analyzes user context to tailor responses based on specific business functions or areas of interest.
- **Output Formatter Agent**: Formats responses in a structured, magazine-style format, ensuring clarity and readability.
- **Quality Checker Agent**: Evaluates the accuracy and relevance of the information, providing confidence scores and identifying potential inaccuracies.

## LightRAG

LightRAG is a Retrieval-Augmented Generation framework that enhances document retrieval and analysis. It operates in multiple modes to provide accurate and context-aware responses:

- **Naive Mode**: Basic retrieval using vector embeddings.
- **Local Mode**: Focuses on local context and relationships within documents.
- **Global Mode**: Considers global context and relationships across documents.
- **Hybrid Mode**: Combines local and global contexts for comprehensive analysis.

LightRAG uses embeddings to represent text documents and queries, allowing for efficient and accurate retrieval. It also maintains a knowledge graph to store and manage relationships between entities and concepts.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/summarize-app.git
   cd summarize-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   - Copy `.env.example` to `.env` and fill in the necessary API keys and configurations.

## Usage

1. Start the Gradio app:
   ```bash
   python swarms-master/summarize/gradio_app.py
   ```

2. Access the app in your browser at `http://localhost:7860`.

3. Enter your query and let the app analyze and summarize the documents.
