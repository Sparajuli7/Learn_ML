# Agentic AI Advanced: Building Sophisticated Autonomous Agents

*"Advanced agentic AI systems combine multiple capabilities: reasoning, planning, tool use, and coordination to solve complex real-world problems."*

---

## 📚 Table of Contents

1. [Advanced Agent Architectures](#advanced-agent-architectures)
2. [Sophisticated Tool Integration](#sophisticated-tool-integration)
3. [Multi-Agent Coordination](#multi-agent-coordination)
4. [Advanced Planning and Reasoning](#advanced-planning-and-reasoning)
5. [Production Deployment](#production-deployment)
6. [Case Studies](#case-studies)
7. [Exercises and Projects](#exercises-and-projects)

---

## 🏗️ Advanced Agent Architectures

### Hierarchical Agent Systems

Building agents with multiple levels of abstraction and control.

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import asyncio

class AgentLayer(ABC):
    """Abstract base class for agent layers"""
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass
    
    @abstractmethod
    def can_handle(self, input_data: Any) -> bool:
        pass

class HighLevelPlanner(AgentLayer):
    """High-level strategic planning layer"""
    
    def can_handle(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data) > 100
    
    async def process(self, input_data: str) -> Dict[str, Any]:
        # Break down complex tasks into subtasks
        return {
            "type": "plan",
            "subtasks": [
                "research_phase",
                "analysis_phase", 
                "synthesis_phase",
                "delivery_phase"
            ],
            "dependencies": {
                "analysis_phase": ["research_phase"],
                "synthesis_phase": ["analysis_phase"],
                "delivery_phase": ["synthesis_phase"]
            }
        }

class TaskExecutor(AgentLayer):
    """Mid-level task execution layer"""
    
    def can_handle(self, input_data: Any) -> bool:
        return isinstance(input_data, dict) and "type" in input_data
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if input_data["type"] == "plan":
            return await self._execute_plan(input_data)
        return input_data
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for subtask in plan["subtasks"]:
            result = await self._execute_subtask(subtask)
            results[subtask] = result
        return {"type": "results", "data": results}
    
    async def _execute_subtask(self, subtask: str) -> Dict[str, Any]:
        # Simulate task execution
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "completed", "subtask": subtask}

class ToolUser(AgentLayer):
    """Low-level tool usage layer"""
    
    def __init__(self):
        self.tools = {
            "web_search": self._web_search,
            "calculator": self._calculator,
            "file_io": self._file_operations
        }
    
    def can_handle(self, input_data: Any) -> bool:
        return isinstance(input_data, dict) and "tool" in input_data
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = input_data["tool"]
        if tool_name in self.tools:
            return await self.tools[tool_name](input_data["params"])
        return {"error": f"Unknown tool: {tool_name}"}
    
    async def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        return {"result": f"Search results for: {query}", "source": "web"}
    
    async def _calculator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expression = params.get("expression", "")
        try:
            result = eval(expression)
            return {"result": result, "expression": expression}
        except:
            return {"error": "Invalid expression"}
    
    async def _file_operations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get("operation", "")
        filename = params.get("filename", "")
        return {"result": f"File {operation} on {filename}", "status": "success"}

class HierarchicalAgent:
    """Multi-layer agent system"""
    
    def __init__(self):
        self.layers = [
            HighLevelPlanner(),
            TaskExecutor(),
            ToolUser()
        ]
    
    async def process(self, input_data: Any) -> Any:
        """Process input through all applicable layers"""
        current_data = input_data
        
        for layer in self.layers:
            if layer.can_handle(current_data):
                current_data = await layer.process(current_data)
        
        return current_data

# Usage
async def main():
    agent = HierarchicalAgent()
    result = await agent.process("Research the latest developments in quantum computing and create a comprehensive report")
    print(result)

# asyncio.run(main())
```

### Memory-Augmented Agents

Agents with sophisticated memory systems for long-term learning.

```python
import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import hashlib

class MemorySystem:
    """Advanced memory system for agents"""
    
    def __init__(self, max_memory_size: int = 10000):
        self.max_memory_size = max_memory_size
        self.short_term_memory = []
        self.long_term_memory = {}
        self.episodic_memory = []
        self.semantic_memory = {}
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add new experience to memory"""
        # Add to short-term memory
        self.short_term_memory.append(experience)
        
        # Maintain memory size
        if len(self.short_term_memory) > self.max_memory_size:
            self._consolidate_memory()
    
    def _consolidate_memory(self):
        """Consolidate short-term memories into long-term memory"""
        for experience in self.short_term_memory:
            # Create memory key
            key = self._create_memory_key(experience)
            
            # Store in long-term memory
            if key not in self.long_term_memory:
                self.long_term_memory[key] = {
                    "count": 1,
                    "experiences": [experience],
                    "importance": self._calculate_importance(experience)
                }
            else:
                self.long_term_memory[key]["count"] += 1
                self.long_term_memory[key]["experiences"].append(experience)
        
        # Clear short-term memory
        self.short_term_memory = []
    
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to current query"""
        relevant_memories = []
        
        for key, memory in self.long_term_memory.items():
            relevance_score = self._calculate_relevance(query, memory)
            if relevance_score > 0.5:  # Threshold for relevance
                relevant_memories.append({
                    "memory": memory,
                    "relevance": relevance_score
                })
        
        # Sort by relevance and return top-k
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_memories[:top_k]
    
    def _create_memory_key(self, experience: Dict[str, Any]) -> str:
        """Create a unique key for memory storage"""
        content = str(experience.get("content", ""))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_importance(self, experience: Dict[str, Any]) -> float:
        """Calculate importance of an experience"""
        # Simple importance calculation
        factors = {
            "success": experience.get("success", False),
            "reward": experience.get("reward", 0),
            "complexity": len(str(experience.get("content", "")))
        }
        
        importance = 0.0
        if factors["success"]:
            importance += 0.3
        importance += min(factors["reward"] / 10.0, 0.4)
        importance += min(factors["complexity"] / 1000.0, 0.3)
        
        return importance
    
    def _calculate_relevance(self, query: str, memory: Dict[str, Any]) -> float:
        """Calculate relevance between query and memory"""
        # Simple relevance calculation using keyword matching
        query_words = set(query.lower().split())
        memory_content = str(memory.get("experiences", []))
        memory_words = set(memory_content.lower().split())
        
        intersection = query_words.intersection(memory_words)
        union = query_words.union(memory_words)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)

class MemoryAugmentedAgent:
    """Agent with sophisticated memory capabilities"""
    
    def __init__(self, memory_size: int = 10000):
        self.memory = MemorySystem(memory_size)
        self.current_context = {}
    
    def process_with_memory(self, input_data: str) -> Dict[str, Any]:
        """Process input using memory for context"""
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve_relevant_memories(input_data)
        
        # Update context with relevant memories
        self.current_context["relevant_memories"] = relevant_memories
        
        # Process input
        result = self._process_input(input_data)
        
        # Store experience in memory
        experience = {
            "input": input_data,
            "output": result,
            "context": self.current_context,
            "timestamp": "2025-01-01T12:00:00Z"
        }
        self.memory.add_experience(experience)
        
        return result
    
    def _process_input(self, input_data: str) -> Dict[str, Any]:
        """Process input using current context"""
        # Enhanced processing using memory context
        context_info = self.current_context.get("relevant_memories", [])
        
        return {
            "input": input_data,
            "output": f"Processed with {len(context_info)} relevant memories",
            "context_used": len(context_info),
            "enhanced_response": True
        }
```

---

## 🔧 Sophisticated Tool Integration

### Dynamic Tool Discovery and Loading

```python
import importlib
import inspect
from typing import Dict, Any, Callable
import requests

class ToolRegistry:
    """Registry for managing and discovering tools"""
    
    def __init__(self):
        self.tools = {}
        self.tool_descriptions = {}
        self.tool_metadata = {}
    
    def register_tool(self, name: str, func: Callable, description: str, metadata: Dict[str, Any] = None):
        """Register a new tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = description
        self.tool_metadata[name] = metadata or {}
    
    def discover_tools(self, query: str) -> List[str]:
        """Discover tools relevant to a query"""
        relevant_tools = []
        
        for name, description in self.tool_descriptions.items():
            if self._is_relevant(query, description):
                relevant_tools.append(name)
        
        return relevant_tools
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = self.tools[tool_name](**params)
            return {"result": result, "tool": tool_name, "status": "success"}
        except Exception as e:
            return {"error": str(e), "tool": tool_name, "status": "error"}
    
    def _is_relevant(self, query: str, description: str) -> bool:
        """Check if tool description is relevant to query"""
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())
        
        intersection = query_words.intersection(desc_words)
        return len(intersection) > 0

class AdvancedToolAgent:
    """Agent with sophisticated tool usage capabilities"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        self.tool_registry.register_tool(
            "web_search",
            self._web_search,
            "Search the web for current information",
            {"category": "information", "requires_internet": True}
        )
        
        self.tool_registry.register_tool(
            "data_analysis",
            self._data_analysis,
            "Analyze data and generate insights",
            {"category": "analysis", "requires_data": True}
        )
        
        self.tool_registry.register_tool(
            "file_operations",
            self._file_operations,
            "Read, write, and manipulate files",
            {"category": "storage", "requires_filesystem": True}
        )
    
    def process_with_tools(self, query: str) -> Dict[str, Any]:
        """Process query using appropriate tools"""
        # Discover relevant tools
        relevant_tools = self.tool_registry.discover_tools(query)
        
        # Execute tools and collect results
        results = {}
        for tool_name in relevant_tools:
            # Determine tool parameters from query
            params = self._extract_tool_params(query, tool_name)
            result = self.tool_registry.execute_tool(tool_name, params)
            results[tool_name] = result
        
        return {
            "query": query,
            "tools_used": relevant_tools,
            "results": results
        }
    
    def _extract_tool_params(self, query: str, tool_name: str) -> Dict[str, Any]:
        """Extract tool parameters from query"""
        # Simple parameter extraction
        if tool_name == "web_search":
            return {"query": query}
        elif tool_name == "data_analysis":
            return {"data": query}
        elif tool_name == "file_operations":
            return {"operation": "read", "filename": "data.txt"}
        return {}
    
    def _web_search(self, query: str) -> str:
        """Web search tool"""
        return f"Search results for: {query}"
    
    def _data_analysis(self, data: str) -> Dict[str, Any]:
        """Data analysis tool"""
        return {
            "insights": ["Trend 1", "Trend 2"],
            "statistics": {"count": 100, "mean": 50.5}
        }
    
    def _file_operations(self, operation: str, filename: str) -> str:
        """File operations tool"""
        return f"Performed {operation} on {filename}"
```

---

## 🤝 Multi-Agent Coordination

### Advanced Multi-Agent System

```python
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"

@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    message_type: str
    timestamp: str

class AdvancedMultiAgentSystem:
    """Sophisticated multi-agent coordination system"""
    
    def __init__(self, agents: List[Dict[str, Any]]):
        self.agents = {agent["name"]: agent for agent in agents}
        self.message_queue = asyncio.Queue()
        self.shared_memory = {}
        self.coordination_protocol = "consensus"
    
    async def coordinate_task(self, task: str) -> Dict[str, Any]:
        """Coordinate multiple agents to solve a complex task"""
        # Initialize task
        task_id = self._generate_task_id()
        self.shared_memory[task_id] = {
            "task": task,
            "status": "in_progress",
            "results": {},
            "dependencies": {}
        }
        
        # Create coordination plan
        plan = await self._create_coordination_plan(task)
        
        # Execute plan with agents
        results = await self._execute_coordinated_plan(plan, task_id)
        
        # Integrate results
        final_result = await self._integrate_results(results, task_id)
        
        return final_result
    
    async def _create_coordination_plan(self, task: str) -> Dict[str, Any]:
        """Create a coordination plan for the task"""
        # Analyze task requirements
        requirements = await self._analyze_task_requirements(task)
        
        # Assign roles to agents
        role_assignments = self._assign_roles(requirements)
        
        # Create workflow
        workflow = self._create_workflow(role_assignments)
        
        return {
            "requirements": requirements,
            "role_assignments": role_assignments,
            "workflow": workflow
        }
    
    async def _execute_coordinated_plan(self, plan: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute the coordinated plan"""
        results = {}
        workflow = plan["workflow"]
        
        for step in workflow:
            # Wait for dependencies
            await self._wait_for_dependencies(step, results)
            
            # Execute step
            step_result = await self._execute_step(step, task_id)
            results[step["name"]] = step_result
            
            # Update shared memory
            self.shared_memory[task_id]["results"][step["name"]] = step_result
        
        return results
    
    async def _execute_step(self, step: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute a single step in the workflow"""
        agent_name = step["agent"]
        agent = self.agents[agent_name]
        
        # Prepare input for agent
        input_data = self._prepare_agent_input(step, task_id)
        
        # Execute agent
        result = await agent["process"](input_data)
        
        # Send result to other agents if needed
        await self._broadcast_result(step, result, task_id)
        
        return result
    
    def _prepare_agent_input(self, step: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Prepare input data for an agent"""
        shared_data = self.shared_memory[task_id]
        
        return {
            "task": shared_data["task"],
            "step": step["name"],
            "context": shared_data["results"],
            "requirements": step.get("requirements", {})
        }
    
    async def _broadcast_result(self, step: Dict[str, Any], result: Dict[str, Any], task_id: str):
        """Broadcast result to other agents"""
        message = Message(
            sender=step["agent"],
            receiver="all",
            content=result,
            message_type="step_completion",
            timestamp="2025-01-01T12:00:00Z"
        )
        
        await self.message_queue.put(message)
    
    async def _integrate_results(self, results: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Integrate results from all agents"""
        # Use coordinator agent to integrate results
        coordinator = self.agents.get("coordinator")
        if coordinator:
            integration_input = {
                "task": self.shared_memory[task_id]["task"],
                "results": results,
                "context": self.shared_memory[task_id]
            }
            
            final_result = await coordinator["process"](integration_input)
            return final_result
        
        # Fallback integration
        return {
            "task_id": task_id,
            "status": "completed",
            "results": results,
            "summary": "Task completed by multi-agent system"
        }
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _assign_roles(self, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Assign roles to agents based on requirements"""
        role_assignments = {}
        
        for requirement in requirements.get("capabilities", []):
            # Find best agent for each requirement
            best_agent = self._find_best_agent(requirement)
            if best_agent:
                role_assignments[requirement] = best_agent
        
        return role_assignments
    
    def _find_best_agent(self, requirement: str) -> str:
        """Find the best agent for a given requirement"""
        # Simple agent selection based on capability matching
        for agent_name, agent in self.agents.items():
            capabilities = agent.get("capabilities", [])
            if requirement in capabilities:
                return agent_name
        
        return None
    
    def _create_workflow(self, role_assignments: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create workflow from role assignments"""
        workflow = []
        
        # Define workflow steps
        steps = [
            {"name": "research", "agent": "researcher", "dependencies": []},
            {"name": "analysis", "agent": "analyst", "dependencies": ["research"]},
            {"name": "writing", "agent": "writer", "dependencies": ["analysis"]},
            {"name": "review", "agent": "reviewer", "dependencies": ["writing"]}
        ]
        
        for step in steps:
            if step["name"] in role_assignments:
                step["agent"] = role_assignments[step["name"]]
                workflow.append(step)
        
        return workflow
    
    async def _wait_for_dependencies(self, step: Dict[str, Any], results: Dict[str, Any]):
        """Wait for step dependencies to complete"""
        dependencies = step.get("dependencies", [])
        
        for dep in dependencies:
            while dep not in results:
                await asyncio.sleep(0.1)  # Wait for dependency
    
    async def _analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """Analyze task to determine requirements"""
        # Simple requirement analysis
        requirements = {
            "capabilities": ["research", "analysis", "writing", "review"],
            "complexity": "high",
            "estimated_duration": "2 hours"
        }
        
        return requirements

# Usage example
async def main():
    agents = [
        {
            "name": "researcher",
            "capabilities": ["research"],
            "process": lambda x: {"type": "research", "result": "Research completed"}
        },
        {
            "name": "analyst", 
            "capabilities": ["analysis"],
            "process": lambda x: {"type": "analysis", "result": "Analysis completed"}
        },
        {
            "name": "writer",
            "capabilities": ["writing"],
            "process": lambda x: {"type": "writing", "result": "Writing completed"}
        },
        {
            "name": "coordinator",
            "capabilities": ["coordination"],
            "process": lambda x: {"type": "coordination", "result": "Integration completed"}
        }
    ]
    
    mas = AdvancedMultiAgentSystem(agents)
    result = await mas.coordinate_task("Create a comprehensive report on AI trends")
    print(result)

# asyncio.run(main())
```

---

## 🧠 Advanced Planning and Reasoning

### Hierarchical Task Networks

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    PRIMITIVE = "primitive"
    COMPOUND = "compound"
    ABSTRACT = "abstract"

@dataclass
class Task:
    name: str
    task_type: TaskType
    preconditions: List[str]
    effects: List[str]
    subtasks: Optional[List['Task']] = None
    method: Optional[str] = None

class HierarchicalTaskNetwork:
    """Advanced planning using hierarchical task networks"""
    
    def __init__(self):
        self.tasks = {}
        self.methods = {}
        self.current_state = set()
    
    def add_task(self, task: Task):
        """Add a task to the HTN"""
        self.tasks[task.name] = task
    
    def add_method(self, method_name: str, task_name: str, subtasks: List[str]):
        """Add a method for decomposing a task"""
        if task_name not in self.methods:
            self.methods[task_name] = []
        
        self.methods[task_name].append({
            "name": method_name,
            "subtasks": subtasks
        })
    
    def plan(self, goal_task: str, initial_state: set) -> List[str]:
        """Generate a plan using HTN planning"""
        self.current_state = initial_state.copy()
        
        # Start with the goal task
        plan = self._decompose_task(goal_task)
        
        return plan
    
    def _decompose_task(self, task_name: str) -> List[str]:
        """Recursively decompose a task"""
        if task_name not in self.tasks:
            return []
        
        task = self.tasks[task_name]
        
        if task.task_type == TaskType.PRIMITIVE:
            # Check if task can be executed
            if self._can_execute(task):
                self._execute_task(task)
                return [task.name]
            else:
                return []  # Task cannot be executed
        
        elif task.task_type == TaskType.COMPOUND:
            # Try to decompose using available methods
            if task_name in self.methods:
                for method in self.methods[task_name]:
                    plan = self._try_method(task_name, method)
                    if plan:
                        return plan
            
            return []
        
        return []
    
    def _try_method(self, task_name: str, method: Dict[str, Any]) -> List[str]:
        """Try to apply a method to decompose a task"""
        plan = []
        
        for subtask_name in method["subtasks"]:
            subtask_plan = self._decompose_task(subtask_name)
            if not subtask_plan:
                return []  # Method failed
            plan.extend(subtask_plan)
        
        return plan
    
    def _can_execute(self, task: Task) -> bool:
        """Check if a primitive task can be executed"""
        for precondition in task.preconditions:
            if precondition not in self.current_state:
                return False
        return True
    
    def _execute_task(self, task: Task):
        """Execute a primitive task"""
        # Add effects to current state
        for effect in task.effects:
            self.current_state.add(effect)

# Usage example
def create_htn_example():
    htn = HierarchicalTaskNetwork()
    
    # Add primitive tasks
    htn.add_task(Task("gather_data", TaskType.PRIMITIVE, [], ["data_available"]))
    htn.add_task(Task("analyze_data", TaskType.PRIMITIVE, ["data_available"], ["analysis_complete"]))
    htn.add_task(Task("write_report", TaskType.PRIMITIVE, ["analysis_complete"], ["report_complete"]))
    
    # Add compound tasks
    htn.add_task(Task("research_topic", TaskType.COMPOUND, [], ["research_complete"]))
    htn.add_task(Task("create_report", TaskType.COMPOUND, [], ["report_complete"]))
    
    # Add methods for decomposition
    htn.add_method("research_method", "research_topic", ["gather_data", "analyze_data"])
    htn.add_method("report_method", "create_report", ["research_topic", "write_report"])
    
    # Generate plan
    plan = htn.plan("create_report", set())
    return plan

# plan = create_htn_example()
# print(f"Generated plan: {plan}")
```

---

## 🚀 Production Deployment

### Scalable Agent Infrastructure

```python
import asyncio
import aiohttp
from typing import Dict, Any, List
import json
import logging

class AgentOrchestrator:
    """Orchestrator for managing multiple agents in production"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.monitoring = AgentMonitoring()
    
    async def start(self):
        """Start the agent orchestrator"""
        # Initialize agents
        await self._initialize_agents()
        
        # Start monitoring
        await self.monitoring.start()
        
        # Start task processing
        asyncio.create_task(self._process_tasks())
        
        logging.info("Agent orchestrator started")
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for processing"""
        task_id = self._generate_task_id()
        task["id"] = task_id
        task["status"] = "queued"
        
        await self.task_queue.put(task)
        return task_id
    
    async def get_result(self, task_id: str) -> Dict[str, Any]:
        """Get result for a task"""
        # Check if result is available
        result = await self.monitoring.get_task_result(task_id)
        return result
    
    async def _initialize_agents(self):
        """Initialize all agents"""
        for agent_config in self.config["agents"]:
            agent = await self._create_agent(agent_config)
            self.agents[agent_config["name"]] = agent
    
    async def _create_agent(self, config: Dict[str, Any]):
        """Create an agent instance"""
        agent_type = config["type"]
        
        if agent_type == "research":
            return ResearchAgent(config)
        elif agent_type == "analysis":
            return AnalysisAgent(config)
        elif agent_type == "writing":
            return WritingAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def _process_tasks(self):
        """Process tasks from the queue"""
        while True:
            task = await self.task_queue.get()
            
            try:
                # Update task status
                await self.monitoring.update_task_status(task["id"], "processing")
                
                # Process task
                result = await self._process_task(task)
                
                # Store result
                await self.monitoring.store_task_result(task["id"], result)
                
                # Update status
                await self.monitoring.update_task_status(task["id"], "completed")
                
            except Exception as e:
                logging.error(f"Error processing task {task['id']}: {e}")
                await self.monitoring.update_task_status(task["id"], "failed")
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task"""
        # Determine which agents to use
        required_agents = self._determine_required_agents(task)
        
        # Execute task with agents
        results = {}
        for agent_name in required_agents:
            agent = self.agents[agent_name]
            agent_result = await agent.process(task)
            results[agent_name] = agent_result
        
        # Integrate results
        final_result = await self._integrate_results(results, task)
        
        return final_result
    
    def _determine_required_agents(self, task: Dict[str, Any]) -> List[str]:
        """Determine which agents are required for a task"""
        required_agents = []
        
        if "research" in task.get("requirements", []):
            required_agents.append("research")
        
        if "analysis" in task.get("requirements", []):
            required_agents.append("analysis")
        
        if "writing" in task.get("requirements", []):
            required_agents.append("writing")
        
        return required_agents
    
    async def _integrate_results(self, results: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from multiple agents"""
        return {
            "task_id": task["id"],
            "results": results,
            "status": "completed",
            "timestamp": "2025-01-01T12:00:00Z"
        }
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())

class AgentMonitoring:
    """Monitoring system for agents"""
    
    def __init__(self):
        self.task_results = {}
        self.task_status = {}
        self.agent_metrics = {}
    
    async def start(self):
        """Start monitoring"""
        logging.info("Agent monitoring started")
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        self.task_status[task_id] = status
    
    async def store_task_result(self, task_id: str, result: Dict[str, Any]):
        """Store task result"""
        self.task_results[task_id] = result
    
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get task result"""
        return self.task_results.get(task_id, {"status": "not_found"})

# Example agent classes
class ResearchAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "research", "result": "Research completed"}

class AnalysisAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "analysis", "result": "Analysis completed"}

class WritingAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "writing", "result": "Writing completed"}
```

---

## 📊 Case Studies

### Case Study 1: Autonomous Research Assistant

A sophisticated agent system that can conduct comprehensive research autonomously.

**Key Features:**
- Multi-source data gathering
- Automated analysis and synthesis
- Report generation
- Continuous learning from feedback

**Implementation:**
```python
class AutonomousResearchAssistant:
    def __init__(self, api_key: str):
        self.agent_system = AdvancedMultiAgentSystem([
            {"name": "researcher", "capabilities": ["web_search", "academic_search"]},
            {"name": "analyst", "capabilities": ["data_analysis", "trend_analysis"]},
            {"name": "synthesizer", "capabilities": ["information_synthesis"]},
            {"name": "writer", "capabilities": ["report_writing"]}
        ])
    
    async def conduct_research(self, topic: str) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        task = {
            "type": "research",
            "topic": topic,
            "requirements": ["comprehensive_analysis", "trend_identification", "report_generation"]
        }
        
        result = await self.agent_system.coordinate_task(task)
        return result
```

### Case Study 2: Business Process Automation

An agent system that automates complex business workflows.

**Key Features:**
- Workflow orchestration
- Error handling and recovery
- Human-in-the-loop capabilities
- Audit trails and compliance

**Implementation:**
```python
class BusinessProcessAutomation:
    def __init__(self, workflow_config: Dict[str, Any]):
        self.orchestrator = AgentOrchestrator(workflow_config)
        self.workflow_engine = WorkflowEngine()
    
    async def automate_process(self, process_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Automate a business process"""
        workflow = self.workflow_engine.get_workflow(process_type)
        
        task = {
            "type": "business_process",
            "process": process_type,
            "data": data,
            "workflow": workflow
        }
        
        task_id = await self.orchestrator.submit_task(task)
        result = await self.orchestrator.get_result(task_id)
        
        return result
```

---

## 🧪 Exercises and Projects

### Exercise 1: Build a Hierarchical Agent

Create a three-layer agent system with planning, execution, and tool usage layers.

```python
# Your task: Implement a hierarchical agent system

class HierarchicalAgentSystem:
    def __init__(self):
        """
        Implement hierarchical agent system with three layers
        
        Requirements:
        1. High-level planning layer
        2. Mid-level execution layer  
        3. Low-level tool usage layer
        4. Communication between layers
        5. Error handling and recovery
        """
        # Initialize the three layers
        self.planning_layer = PlanningLayer()
        self.execution_layer = ExecutionLayer()
        self.tool_layer = ToolLayer()
        
        # Error handling and recovery
        self.error_log = []
    
    def process(self, task_description):
        """Process a high-level task through all layers"""
        try:
            # Step 1: High-level planning
            plan = self.planning_layer.create_plan(task_description)
            
            # Step 2: Mid-level execution
            execution_results = []
            for step in plan['steps']:
                result = self.execution_layer.execute_step(step)
                execution_results.append(result)
            
            # Step 3: Synthesize results
            final_result = self.planning_layer.synthesize_results(execution_results)
            
            return {
                'success': True,
                'plan': plan,
                'execution_results': execution_results,
                'final_result': final_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class PlanningLayer:
    def __init__(self):
        self.task_templates = {
            'research': ['gather_info', 'analyze_data', 'synthesize_findings'],
            'writing': ['outline', 'draft', 'revise', 'finalize'],
            'analysis': ['data_collection', 'processing', 'analysis', 'reporting']
        }
    
    def create_plan(self, task_description):
        """Create a high-level plan for the task"""
        task_type = self._classify_task(task_description)
        steps = self.task_templates.get(task_type, ['plan', 'execute', 'review'])
        
        plan = {
            'task_type': task_type,
            'description': task_description,
            'steps': [{'action': step, 'description': f"{step} for {task_description}"} for step in steps]
        }
        
        return plan
    
    def _classify_task(self, description):
        """Classify task type based on description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['research', 'study', 'investigate']):
            return 'research'
        elif any(word in description_lower for word in ['write', 'create', 'draft']):
            return 'writing'
        elif any(word in description_lower for word in ['analyze', 'examine', 'evaluate']):
            return 'analysis'
        else:
            return 'general'
    
    def synthesize_results(self, execution_results):
        """Synthesize results from execution layer"""
        if not execution_results:
            return "No results to synthesize"
        
        combined_result = "\n\n".join([
            f"Step {i+1}: {result.get('output', 'No output')}"
            for i, result in enumerate(execution_results)
        ])
        
        return f"Task completed successfully. Results:\n{combined_result}"


class ExecutionLayer:
    def __init__(self):
        self.execution_methods = {
            'gather_info': self._gather_information,
            'analyze_data': self._analyze_data,
            'synthesize_findings': self._synthesize_findings,
            'outline': self._create_outline,
            'draft': self._create_draft,
            'revise': self._revise_content,
            'finalize': self._finalize_content,
            'execute': self._execute_general
        }
    
    def execute_step(self, step):
        """Execute a single step"""
        action = step['action']
        description = step['description']
        
        if action in self.execution_methods:
            result = self.execution_methods[action](description)
            return {
                'step': action,
                'success': True,
                'output': result
            }
        else:
            return {
                'step': action,
                'success': False,
                'error': f"Unknown action: {action}"
            }
    
    def _gather_information(self, description):
        return f"Information gathered for: {description}"
    
    def _analyze_data(self, description):
        return f"Data analysis completed for: {description}"
    
    def _synthesize_findings(self, description):
        return f"Findings synthesized for: {description}"
    
    def _create_outline(self, description):
        return f"Outline created for: {description}"
    
    def _create_draft(self, description):
        return f"Draft created for: {description}"
    
    def _revise_content(self, description):
        return f"Content revised for: {description}"
    
    def _finalize_content(self, description):
        return f"Content finalized for: {description}"
    
    def _execute_general(self, description):
        return f"General execution completed for: {description}"


class ToolLayer:
    def __init__(self):
        self.available_tools = {
            'web_search': self._web_search,
            'database_query': self._database_query,
            'text_generation': self._text_generation,
            'general_tool': self._general_tool
        }
    
    def use_tool(self, tool_name, parameters):
        """Use a specific tool"""
        if tool_name in self.available_tools:
            result = self.available_tools[tool_name](parameters)
            return {'success': True, 'result': result}
        else:
            return {'success': False, 'error': f"Tool {tool_name} not available"}
    
    def _web_search(self, query):
        return f"Web search results for: {query}"
    
    def _database_query(self, query):
        return f"Database results for: {query}"
    
    def _text_generation(self, prompt):
        return f"Generated text for: {prompt}"
    
    def _general_tool(self, task):
        return f"General tool result for: {task}"
```

### Exercise 2: Multi-Agent Coordination

Build a system where agents can negotiate and coordinate to solve complex problems.

```python
# Your task: Implement a negotiation-based multi-agent system

class NegotiatingMultiAgentSystem:
    def __init__(self, agents: List[Dict[str, Any]]):
        """
        Implement negotiation-based multi-agent system
        
        Requirements:
        1. Agent negotiation protocols
        2. Conflict resolution mechanisms
        3. Resource allocation strategies
        4. Consensus building
        5. Dynamic task assignment
        """
        self.agents = {agent['name']: agent for agent in agents}
        self.negotiation_history = []
        self.consensus_threshold = 0.7
        
        # Initialize agent states
        for agent_name, agent in self.agents.items():
            agent['current_task'] = None
            agent['available_resources'] = agent['resources']
    
    def solve_task(self, task_description):
        """Solve a task using multi-agent negotiation"""
        try:
            # Step 1: Task decomposition
            subtasks = self._decompose_task(task_description)
            
            # Step 2: Task assignment
            assignment = self._assign_tasks(subtasks)
            
            # Step 3: Execute tasks
            execution_results = self._execute_tasks(assignment)
            
            # Step 4: Synthesize results
            final_result = self._synthesize_results(execution_results)
            
            return {
                'success': True,
                'task': task_description,
                'subtasks': subtasks,
                'execution_results': execution_results,
                'final_result': final_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'task': task_description
            }
    
    def _decompose_task(self, task_description):
        """Decompose task into subtasks"""
        task_lower = task_description.lower()
        
        subtasks = []
        
        if 'research' in task_lower or 'market' in task_lower:
            subtasks.append({
                'id': 'research',
                'description': 'Conduct market research and gather data',
                'required_capabilities': ['research'],
                'estimated_resources': 5
            })
        
        if 'analysis' in task_lower or 'analyze' in task_lower:
            subtasks.append({
                'id': 'analysis',
                'description': 'Analyze collected data and insights',
                'required_capabilities': ['analysis'],
                'estimated_resources': 4
            })
        
        if 'write' in task_lower or 'report' in task_lower:
            subtasks.append({
                'id': 'writing',
                'description': 'Write comprehensive report',
                'required_capabilities': ['writing'],
                'estimated_resources': 6
            })
        
        # Add general subtask if no specific ones found
        if not subtasks:
            subtasks.append({
                'id': 'general',
                'description': task_description,
                'required_capabilities': ['general'],
                'estimated_resources': 3
            })
        
        return subtasks
    
    def _assign_tasks(self, subtasks):
        """Assign tasks to agents based on capabilities"""
        assignment = {}
        
        for subtask in subtasks:
            best_agent = None
            best_score = 0
            
            for agent_name, agent in self.agents.items():
                # Check if agent has required capabilities
                if any(cap in agent['capabilities'] for cap in subtask['required_capabilities']):
                    # Calculate assignment score
                    score = self._calculate_assignment_score(agent, subtask)
                    
                    if score > best_score:
                        best_score = score
                        best_agent = agent_name
            
            assignment[subtask['id']] = {
                'agent': best_agent,
                'subtask': subtask,
                'status': 'assigned' if best_agent else 'unassigned'
            }
        
        return assignment
    
    def _calculate_assignment_score(self, agent, subtask):
        """Calculate how well an agent fits a subtask"""
        score = 0
        
        # Capability match
        capability_match = sum(1 for cap in subtask['required_capabilities'] 
                             if cap in agent['capabilities'])
        score += capability_match * 10
        
        # Resource availability
        if agent['available_resources'] >= subtask['estimated_resources']:
            score += 5
        
        # Current workload
        if agent['current_task'] is None:
            score += 3
        
        return score
    
    def _execute_tasks(self, assignment):
        """Execute assigned tasks"""
        execution_results = {}
        
        for subtask_id, assignment_info in assignment.items():
            if assignment_info['status'] == 'assigned':
                agent_name = assignment_info['agent']
                subtask = assignment_info['subtask']
                
                # Execute the task
                result = self._execute_subtask(agent_name, subtask)
                execution_results[subtask_id] = result
                
                # Update agent resources
                self.agents[agent_name]['available_resources'] -= subtask['estimated_resources']
                self.agents[agent_name]['current_task'] = subtask_id
            else:
                execution_results[subtask_id] = {
                    'success': False,
                    'error': 'No agent assigned',
                    'output': None
                }
        
        return execution_results
    
    def _execute_subtask(self, agent_name, subtask):
        """Execute a single subtask"""
        try:
            # Simulate task execution based on subtask type
            if subtask['id'] == 'research':
                output = f"Research completed by {agent_name}: Market data gathered and analyzed"
            elif subtask['id'] == 'analysis':
                output = f"Analysis completed by {agent_name}: Data insights and trends identified"
            elif subtask['id'] == 'writing':
                output = f"Writing completed by {agent_name}: Comprehensive report generated"
            else:
                output = f"General task completed by {agent_name}: {subtask['description']}"
            
            return {
                'success': True,
                'agent': agent_name,
                'subtask': subtask['id'],
                'output': output,
                'resources_used': subtask['estimated_resources']
            }
            
        except Exception as e:
            return {
                'success': False,
                'agent': agent_name,
                'subtask': subtask['id'],
                'error': str(e),
                'output': None
            }
    
    def _synthesize_results(self, execution_results):
        """Synthesize results from all agents"""
        successful_results = [
            result['output'] for result in execution_results.values()
            if result['success']
        ]
        
        if not successful_results:
            return "No successful task completions"
        
        # Combine all results
        combined_result = "\n\n".join(successful_results)
        
        return f"Multi-agent task completed successfully. Results:\n{combined_result}"

def test_negotiating_system():
    """Test the negotiating multi-agent system"""
    agents = [
        {"name": "agent1", "capabilities": ["research"], "resources": 10},
        {"name": "agent2", "capabilities": ["analysis"], "resources": 8},
        {"name": "agent3", "capabilities": ["writing"], "resources": 12}
    ]
    
    system = NegotiatingMultiAgentSystem(agents)
    result = system.solve_task("Create a comprehensive market analysis report")
    print(result)
```