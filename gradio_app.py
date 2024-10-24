import gradio as gr
import asyncio
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from app import initialize_system, run_query

async def query_function(user_query, progress=gr.Progress()):
    """Process the query and return structured results"""
    # Initialize the system only once when needed
    if not hasattr(query_function, 'router'):
        progress(0, desc="Initializing system...")
        query_function.router = await initialize_system()
    
    progress(0.2, desc="Starting query processing...")
    result = await run_query(query_function.router, user_query)
    
    progress(0.4, desc="Getting agent logs...")
    # Get logs from router to show agent outputs
    agent_logs = query_function.router.get_logs()
    
    # Format agent outputs into a readable string
    agent_outputs = "\n".join([
        f"[{log['timestamp']}] {log['level']}: {log['message']}"
        for log in agent_logs
    ])
    
    progress(0.6, desc="Processing results...")
    
    # Format the model output
    if isinstance(result, list):
        result = "\n\n".join(result)
    
    # Clean up the output
    model_output = str(result).strip()
    
    # Format the citations if they exist
    citations = "No citations available"
    if "Citations:" in model_output:
        parts = model_output.split("Citations:")
        model_output = parts[0].strip()
        citations = parts[1].strip()
    
    progress(1.0, desc="Complete!")
    
    # Return three components:
    # 1. The main model output
    # 2. Citations (if any)
    # 3. Agent processing logs
    return (
        f"### Model Output\n\n{model_output}",
        f"### Citations\n\n{citations}",
        f"### Processing Logs\n\n{agent_outputs}"
    )

def update_status():
    return "Processing complete"

def update_agent_activity():
    if hasattr(query_function, 'router'):
        logs = query_function.router.get_logs()
        return "\n".join([f"[{log['timestamp']}] {log['level']}: {log['message']}" 
                         for log in logs[-5:]])  # Show last 5 logs
    return "No activity yet"

# Create the demo using Blocks
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("""
        # Document Summarization and Analysis
        Enter a query to analyze and summarize documents using the AI system.
        """)
        
        # Input section
        with gr.Row():
            query_input = gr.Textbox(
                lines=2,
                placeholder="Enter your query here...",
                label="Query"
            )
        
        # Submit button
        submit_btn = gr.Button("Submit Query")
        
        # Results section
        with gr.Accordion("Results", open=True):
            model_output = gr.Markdown(label="Model Output")
            citations = gr.Markdown(label="Citations")
            processing_log = gr.Markdown(label="Processing Log")

        # Set up event handlers
        submit_btn.click(
            fn=query_function,
            inputs=query_input,
            outputs=[model_output, citations, processing_log]
        )

# Launch the demo in development mode
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
