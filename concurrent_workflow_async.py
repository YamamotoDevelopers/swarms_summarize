from swarms.structs.concurrent_workflow import ConcurrentWorkflow
import asyncio

class AsyncConcurrentWorkflow(ConcurrentWorkflow):
    async def async_process_agents(self, task, *args, **kwargs):
        """Process agents concurrently using asyncio"""
        async def process_agent(agent):
            try:
                if hasattr(agent, 'get_relevant_documents'):
                    return await agent.get_relevant_documents(task)
                else:
                    return agent.run(task)
            except Exception as e:
                print(f"Error processing agent {agent.agent_name}: {e}")
                return None

        # Create tasks for all agents
        tasks = [process_agent(agent) for agent in self.agents]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result, agent in zip(results, self.agents):
            if isinstance(result, Exception):
                print(f"Error from agent {agent.agent_name}: {result}")
            else:
                processed_results.append(result)
        
        return self.combine_results(processed_results)

    def combine_results(self, results):
        """Combine results from all agents"""
        combined = []
        for result in results:
            if isinstance(result, list):
                combined.extend(result)
            elif result is not None:
                combined.append(result)
        return combined
