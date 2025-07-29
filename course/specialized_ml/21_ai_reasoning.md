# AI Reasoning: Chain-of-Thought, Tree-of-Thought, and Logical Problem-Solving

*"The ability to reason step-by-step and explore multiple solution paths is what separates advanced AI from simple pattern matching."*

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
3. [Tree-of-Thought Reasoning](#tree-of-thought-reasoning)
4. [Advanced Reasoning Techniques](#advanced-reasoning-techniques)
5. [Implementation](#implementation)
6. [Applications](#applications)
7. [Exercises and Projects](#exercises-and-projects)

---

## 🎯 Introduction

AI reasoning represents the cutting edge of artificial intelligence, where models can think step-by-step, explore multiple solution paths, and arrive at logical conclusions. In 2025, reasoning capabilities have become essential for solving complex problems that require planning, analysis, and creative thinking.

### Key Reasoning Paradigms

1. **Chain-of-Thought (CoT)**: Step-by-step reasoning that mimics human problem-solving
2. **Tree-of-Thought (ToT)**: Exploring multiple reasoning paths simultaneously
3. **Logical Reasoning**: Formal logical inference and deduction
4. **Causal Reasoning**: Understanding cause-and-effect relationships
5. **Probabilistic Reasoning**: Dealing with uncertainty and probability

### 2025 Trends

- **Multi-step Reasoning**: Models that can break down complex problems into manageable steps
- **Self-Correction**: Models that can identify and fix their own reasoning errors
- **Multi-modal Reasoning**: Combining text, images, and other modalities in reasoning
- **Interactive Reasoning**: Models that can engage in back-and-forth reasoning with humans

---

## 🔗 Chain-of-Thought Reasoning

### Basic Chain-of-Thought Implementation

```python
import openai
from typing import List, Dict, Any

class ChainOfThoughtReasoner:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using chain-of-thought reasoning"""
        
        prompt = f"""
        Let's approach this problem step by step:
        
        Problem: {problem}
        
        Let me think through this step by step:
        1) First, I need to understand what's being asked
        2) Then, I'll identify the key components
        3) Next, I'll work through the solution step by step
        4) Finally, I'll verify my answer
        
        Let's start:
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        reasoning = response.choices[0].message.content
        
        # Extract the final answer
        answer = self._extract_answer(reasoning)
        
        return {
            "problem": problem,
            "reasoning": reasoning,
            "answer": answer,
            "method": "chain_of_thought"
        }
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract the final answer from reasoning text"""
        # Simple extraction - look for patterns like "Therefore" or "The answer is"
        lines = reasoning.split('\n')
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in ['therefore', 'answer is', 'conclusion']):
                return line.strip()
        return "Answer not clearly stated"

# Usage example
reasoner = ChainOfThoughtReasoner("your-api-key")
result = reasoner.solve_problem("If a train travels 120 km in 2 hours, what is its average speed?")
print(result["reasoning"])
```

### Advanced Chain-of-Thought with Self-Correction

```python
class SelfCorrectingCoTReasoner:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def solve_with_correction(self, problem: str) -> Dict[str, Any]:
        """Solve problem with self-correction capabilities"""
        
        # Initial reasoning
        initial_result = self._initial_reasoning(problem)
        
        # Self-evaluation
        evaluation = self._evaluate_reasoning(problem, initial_result["reasoning"])
        
        # Correction if needed
        if evaluation["needs_correction"]:
            corrected_result = self._correct_reasoning(problem, initial_result, evaluation)
            return corrected_result
        
        return initial_result
    
    def _initial_reasoning(self, problem: str) -> Dict[str, Any]:
        """Perform initial chain-of-thought reasoning"""
        prompt = f"""
        Solve this problem step by step:
        
        {problem}
        
        Think through each step carefully and show your work.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a careful problem solver."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return {
            "reasoning": response.choices[0].message.content,
            "step": "initial"
        }
    
    def _evaluate_reasoning(self, problem: str, reasoning: str) -> Dict[str, Any]:
        """Evaluate the quality of reasoning"""
        prompt = f"""
        Evaluate this reasoning for the following problem:
        
        Problem: {problem}
        
        Reasoning: {reasoning}
        
        Evaluate:
        1) Are all steps logical?
        2) Are there any mathematical errors?
        3) Is the final answer correct?
        4) What could be improved?
        
        Respond with a structured evaluation.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a critical evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Parse evaluation
        needs_correction = any(keyword in evaluation_text.lower() 
                             for keyword in ['error', 'incorrect', 'wrong', 'mistake'])
        
        return {
            "evaluation": evaluation_text,
            "needs_correction": needs_correction
        }
    
    def _correct_reasoning(self, problem: str, initial_result: Dict[str, Any], 
                          evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Correct reasoning based on evaluation"""
        prompt = f"""
        The previous reasoning had issues. Here's the evaluation:
        
        {evaluation['evaluation']}
        
        Please solve the problem again with corrections:
        
        {problem}
        
        Provide a corrected step-by-step solution.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a careful problem solver who learns from mistakes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return {
            "problem": problem,
            "original_reasoning": initial_result["reasoning"],
            "evaluation": evaluation["evaluation"],
            "corrected_reasoning": response.choices[0].message.content,
            "method": "self_correcting_chain_of_thought"
        }
```

---

## 🌳 Tree-of-Thought Reasoning

### Basic Tree-of-Thought Implementation

```python
from typing import List, Dict, Any, Optional
import heapq

class ThoughtNode:
    def __init__(self, thought: str, parent: Optional['ThoughtNode'] = None, 
                 value: float = 0.0, depth: int = 0):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.value = value
        self.depth = depth
    
    def add_child(self, child: 'ThoughtNode'):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1

class TreeOfThoughtReasoner:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_depth = 5
        self.max_breadth = 3
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve problem using tree-of-thought reasoning"""
        
        # Initialize root node
        root = ThoughtNode("Let's start solving this problem step by step.")
        
        # Build thought tree
        self._expand_tree(root, problem, 0)
        
        # Find best path
        best_path = self._find_best_path(root)
        
        return {
            "problem": problem,
            "best_path": best_path,
            "tree_structure": self._serialize_tree(root),
            "method": "tree_of_thought"
        }
    
    def _expand_tree(self, node: ThoughtNode, problem: str, depth: int):
        """Recursively expand the thought tree"""
        if depth >= self.max_depth:
            return
        
        # Generate next thoughts
        next_thoughts = self._generate_thoughts(node.thought, problem, depth)
        
        # Create child nodes
        for i, thought in enumerate(next_thoughts[:self.max_breadth]):
            child = ThoughtNode(thought, parent=node)
            child.value = self._evaluate_thought(thought, problem)
            node.add_child(child)
            
            # Recursively expand
            self._expand_tree(child, problem, depth + 1)
    
    def _generate_thoughts(self, current_thought: str, problem: str, depth: int) -> List[str]:
        """Generate next possible thoughts"""
        prompt = f"""
        Given this problem: {problem}
        
        Current reasoning: {current_thought}
        
        Generate 3-5 different next steps or thoughts to continue solving this problem.
        Each thought should be a logical next step in the reasoning process.
        
        Provide only the thoughts, one per line.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You generate logical next steps in problem-solving."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        thoughts = response.choices[0].message.content.strip().split('\n')
        return [thought.strip() for thought in thoughts if thought.strip()]
    
    def _evaluate_thought(self, thought: str, problem: str) -> float:
        """Evaluate the quality of a thought"""
        prompt = f"""
        Rate this reasoning step on a scale of 0-10:
        
        Problem: {problem}
        Reasoning step: {thought}
        
        Consider:
        - Is it logical?
        - Does it progress toward a solution?
        - Is it relevant to the problem?
        
        Respond with only a number between 0 and 10.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You evaluate reasoning quality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0  # Default value
    
    def _find_best_path(self, root: ThoughtNode) -> List[str]:
        """Find the best path through the thought tree"""
        best_path = []
        best_value = 0.0
        
        def dfs(node: ThoughtNode, path: List[str], total_value: float):
            nonlocal best_path, best_value
            
            current_path = path + [node.thought]
            current_value = total_value + node.value
            
            if not node.children:  # Leaf node
                if current_value > best_value:
                    best_value = current_value
                    best_path = current_path
            else:
                for child in node.children:
                    dfs(child, current_path, current_value)
        
        dfs(root, [], 0.0)
        return best_path
    
    def _serialize_tree(self, node: ThoughtNode) -> Dict[str, Any]:
        """Serialize tree structure for output"""
        return {
            "thought": node.thought,
            "value": node.value,
            "depth": node.depth,
            "children": [self._serialize_tree(child) for child in node.children]
        }

# Usage example
tot_reasoner = TreeOfThoughtReasoner("your-api-key")
result = tot_reasoner.solve_problem("How can we optimize a machine learning model for both accuracy and speed?")
print("Best reasoning path:")
for i, thought in enumerate(result["best_path"]):
    print(f"{i+1}. {thought}")
```

### Advanced Tree-of-Thought with Backtracking

```python
class AdvancedToTReasoner(TreeOfThoughtReasoner):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.backtrack_threshold = 3.0  # Minimum value to continue
    
    def solve_with_backtracking(self, problem: str) -> Dict[str, Any]:
        """Solve with backtracking when paths become unpromising"""
        
        root = ThoughtNode("Let's start solving this problem step by step.")
        
        # Build tree with backtracking
        self._expand_with_backtracking(root, problem, 0)
        
        # Find best path
        best_path = self._find_best_path(root)
        
        return {
            "problem": problem,
            "best_path": best_path,
            "tree_structure": self._serialize_tree(root),
            "method": "advanced_tree_of_thought_with_backtracking"
        }
    
    def _expand_with_backtracking(self, node: ThoughtNode, problem: str, depth: int):
        """Expand tree with backtracking for unpromising paths"""
        if depth >= self.max_depth:
            return
        
        # Generate thoughts
        next_thoughts = self._generate_thoughts(node.thought, problem, depth)
        
        # Evaluate and filter thoughts
        valid_thoughts = []
        for thought in next_thoughts[:self.max_breadth]:
            value = self._evaluate_thought(thought, problem)
            if value >= self.backtrack_threshold:
                valid_thoughts.append((thought, value))
        
        # Sort by value and take top thoughts
        valid_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        for thought, value in valid_thoughts:
            child = ThoughtNode(thought, parent=node)
            child.value = value
            node.add_child(child)
            
            # Continue expansion
            self._expand_with_backtracking(child, problem, depth + 1)
```

---

## 🧠 Advanced Reasoning Techniques

### Multi-Modal Reasoning

```python
class MultiModalReasoner:
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def reason_with_image(self, image_path: str, question: str) -> Dict[str, Any]:
        """Reason about an image and answer a question"""
        
        # Encode image
        import base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""
        Analyze this image and answer the question step by step:
        
        Question: {question}
        
        Think through this step by step:
        1) What do I see in the image?
        2) What are the key elements relevant to the question?
        3) How do these elements relate to the question?
        4) What is the logical conclusion?
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return {
            "question": question,
            "image": image_path,
            "reasoning": response.choices[0].message.content,
            "method": "multimodal_reasoning"
        }
```

### Logical Reasoning with Formal Logic

```python
class LogicalReasoner:
    def __init__(self):
        self.knowledge_base = []
        self.rules = []
    
    def add_fact(self, fact: str):
        """Add a logical fact to knowledge base"""
        self.knowledge_base.append(fact)
    
    def add_rule(self, premise: str, conclusion: str):
        """Add a logical rule"""
        self.rules.append((premise, conclusion))
    
    def reason(self, query: str) -> Dict[str, Any]:
        """Perform logical reasoning"""
        
        # Simple forward chaining
        derived_facts = set(self.knowledge_base)
        new_facts = True
        
        while new_facts:
            new_facts = False
            for premise, conclusion in self.rules:
                if self._evaluate_premise(premise, derived_facts):
                    if conclusion not in derived_facts:
                        derived_facts.add(conclusion)
                        new_facts = True
        
        # Check if query is entailed
        is_entailed = self._evaluate_query(query, derived_facts)
        
        return {
            "query": query,
            "knowledge_base": self.knowledge_base,
            "rules": self.rules,
            "derived_facts": list(derived_facts),
            "is_entailed": is_entailed,
            "method": "logical_reasoning"
        }
    
    def _evaluate_premise(self, premise: str, facts: set) -> bool:
        """Evaluate if a premise is satisfied by facts"""
        # Simple evaluation - in practice, use a proper logic parser
        return premise in facts
    
    def _evaluate_query(self, query: str, facts: set) -> bool:
        """Evaluate if query is entailed by facts"""
        return query in facts

# Usage example
reasoner = LogicalReasoner()
reasoner.add_fact("All birds can fly")
reasoner.add_fact("A penguin is a bird")
reasoner.add_rule("If X is a bird, then X can fly", "X can fly")

result = reasoner.reason("A penguin can fly")
print(f"Query entailed: {result['is_entailed']}")
```

---

## 💻 Implementation

### Building a Comprehensive Reasoning System

```python
class ComprehensiveReasoningSystem:
    def __init__(self, api_key: str):
        self.cot_reasoner = ChainOfThoughtReasoner(api_key)
        self.tot_reasoner = TreeOfThoughtReasoner(api_key)
        self.self_correcting_reasoner = SelfCorrectingCoTReasoner(api_key)
        self.logical_reasoner = LogicalReasoner()
    
    def solve_problem(self, problem: str, method: str = "auto") -> Dict[str, Any]:
        """Solve problem using specified reasoning method"""
        
        if method == "auto":
            # Choose method based on problem characteristics
            method = self._choose_method(problem)
        
        if method == "chain_of_thought":
            return self.cot_reasoner.solve_problem(problem)
        elif method == "tree_of_thought":
            return self.tot_reasoner.solve_problem(problem)
        elif method == "self_correcting":
            return self.self_correcting_reasoner.solve_with_correction(problem)
        elif method == "logical":
            return self.logical_reasoner.reason(problem)
        else:
            raise ValueError(f"Unknown reasoning method: {method}")
    
    def _choose_method(self, problem: str) -> str:
        """Choose the best reasoning method for a problem"""
        # Simple heuristic - in practice, use more sophisticated analysis
        
        if any(keyword in problem.lower() for keyword in ['logical', 'deduce', 'infer']):
            return "logical"
        elif any(keyword in problem.lower() for keyword in ['complex', 'multiple', 'options']):
            return "tree_of_thought"
        elif any(keyword in problem.lower() for keyword in ['calculate', 'solve', 'compute']):
            return "self_correcting"
        else:
            return "chain_of_thought"
    
    def compare_methods(self, problem: str) -> Dict[str, Any]:
        """Compare different reasoning methods on the same problem"""
        methods = ["chain_of_thought", "tree_of_thought", "self_correcting"]
        results = {}
        
        for method in methods:
            try:
                result = self.solve_problem(problem, method)
                results[method] = result
            except Exception as e:
                results[method] = {"error": str(e)}
        
        return {
            "problem": problem,
            "comparison": results
        }

# Usage example
reasoning_system = ComprehensiveReasoningSystem("your-api-key")

# Solve with automatic method selection
result = reasoning_system.solve_problem("What is the sum of the first 100 natural numbers?")

# Compare different methods
comparison = reasoning_system.compare_methods("How can we optimize a database query?")
```

---

## 🎯 Applications

### Mathematical Problem Solving

```python
class MathematicalReasoner:
    def __init__(self, api_key: str):
        self.reasoning_system = ComprehensiveReasoningSystem(api_key)
    
    def solve_math_problem(self, problem: str) -> Dict[str, Any]:
        """Solve mathematical problems with step-by-step reasoning"""
        
        # Enhance problem with mathematical context
        enhanced_problem = f"""
        Mathematical Problem: {problem}
        
        Please solve this step by step, showing all your work and reasoning.
        Make sure to:
        1) Identify the type of problem
        2) Write down the relevant formulas
        3) Show each calculation step
        4) Verify your answer
        """
        
        result = self.reasoning_system.solve_problem(enhanced_problem, "self_correcting")
        
        return {
            "original_problem": problem,
            "solution": result,
            "method": "mathematical_reasoning"
        }
```

### Code Debugging and Analysis

```python
class CodeReasoningSystem:
    def __init__(self, api_key: str):
        self.reasoning_system = ComprehensiveReasoningSystem(api_key)
    
    def debug_code(self, code: str, error_message: str = None) -> Dict[str, Any]:
        """Debug code using reasoning"""
        
        if error_message:
            problem = f"""
            Debug this code:
            
            Code:
            {code}
            
            Error:
            {error_message}
            
            Please analyze the code step by step to identify and fix the issue.
            """
        else:
            problem = f"""
            Analyze this code for potential issues:
            
            Code:
            {code}
            
            Please examine the code step by step for:
            1) Logic errors
            2) Performance issues
            3) Best practices violations
            4) Potential bugs
            """
        
        result = self.reasoning_system.solve_problem(problem, "tree_of_thought")
        
        return {
            "code": code,
            "error_message": error_message,
            "analysis": result,
            "method": "code_reasoning"
        }
```

---

## 🧪 Exercises and Projects

### Exercise 1: Implement Chain-of-Thought

```python
# Your task: Implement a basic chain-of-thought reasoner

class CustomChainOfThought:
    def __init__(self):
        """
        TODO: Implement this class
        
        Requirements:
        1. Break down problems into steps
        2. Generate intermediate reasoning
        3. Extract final answer
        4. Handle different problem types
        """
        pass

def test_custom_cot():
    """Test the custom chain-of-thought implementation"""
    reasoner = CustomChainOfThought()
    result = reasoner.solve("If 3x + 5 = 20, what is x?")
    print(result)
```

### Exercise 2: Build Tree-of-Thought Explorer

```python
# Your task: Implement a tree-of-thought explorer

class CustomTreeOfThought:
    def __init__(self, max_depth: int = 5, max_breadth: int = 3):
        """
        TODO: Implement this class
        
        Requirements:
        1. Generate multiple reasoning paths
        2. Evaluate path quality
        3. Find optimal solution path
        4. Handle backtracking
        """
        pass

def test_custom_tot():
    """Test the custom tree-of-thought implementation"""
    reasoner = CustomTreeOfThought()
    result = reasoner.solve("Design an algorithm to find the shortest path in a graph")
    print(result)
```

### Project: Intelligent Tutoring System

Build a reasoning-based tutoring system that can:

- Analyze student solutions step by step
- Identify reasoning errors
- Provide targeted feedback
- Generate similar practice problems

**Implementation Steps:**
1. Build problem parser and analyzer
2. Implement solution comparison logic
3. Create feedback generation system
4. Develop practice problem generator
5. Add adaptive difficulty adjustment

### Project: Code Review Assistant

Create an AI system that can review code using advanced reasoning:

- Analyze code logic and flow
- Identify potential bugs and issues
- Suggest improvements and optimizations
- Explain complex code sections

**Features:**
- Multi-language support
- Context-aware analysis
- Performance optimization suggestions
- Security vulnerability detection
- Best practices recommendations

---

## 📖 Further Reading

### Essential Papers

1. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022)
2. **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** (Yao et al., 2023)
3. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"** (Wang et al., 2022)
4. **"Large Language Models Are Human-Level Prompt Engineers"** (Zhou et al., 2022)

### Advanced Topics

1. **Neurosymbolic AI**: Combining neural networks with symbolic reasoning
2. **Causal Reasoning**: Understanding cause-and-effect relationships
3. **Probabilistic Programming**: Reasoning with uncertainty
4. **Formal Verification**: Proving correctness of AI reasoning
5. **Multi-Agent Reasoning**: Coordinated reasoning across multiple agents

### Tools and Frameworks

1. **LangChain**: Chain-of-thought and reasoning frameworks
2. **OpenAI Function Calling**: Tool use for reasoning
3. **PyTorch Geometric**: Graph-based reasoning
4. **SymPy**: Symbolic mathematics for reasoning
5. **Z3**: Theorem proving and logical reasoning

---

## 🎯 Key Takeaways

1. **Chain-of-Thought reasoning** enables step-by-step problem solving that mimics human thinking.

2. **Tree-of-Thought reasoning** explores multiple solution paths simultaneously for complex problems.

3. **Self-correction capabilities** allow AI systems to identify and fix their own reasoning errors.

4. **Multi-modal reasoning** combines different types of information (text, images, data) for comprehensive analysis.

5. **Formal logical reasoning** provides rigorous inference capabilities for structured problems.

6. **Advanced reasoning techniques** are essential for solving complex, real-world problems that require planning and analysis.

---

*"The future of AI lies not just in pattern recognition, but in the ability to reason, plan, and solve problems step by step."*

**Next: [Causal AI & Bayesian Methods](specialized_ml/19_causal_ai_bayesian.md) → Causal inference and probabilistic programming**