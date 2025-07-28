# AI Reasoning

## Overview
AI Reasoning encompasses the ability of artificial intelligence systems to draw logical conclusions, make inferences, and solve problems through various forms of reasoning. This includes logical reasoning, causal inference, probabilistic reasoning, and symbolic AI approaches.

## Logical Reasoning

### 1. Propositional Logic
```python
class PropositionalLogic:
    def __init__(self):
        self.symbols = set()
        self.knowledge_base = []
    
    def add_symbol(self, symbol):
        """Add a propositional symbol"""
        self.symbols.add(symbol)
    
    def add_knowledge(self, sentence):
        """Add a logical sentence to knowledge base"""
        self.knowledge_base.append(sentence)
    
    def evaluate(self, sentence, model):
        """Evaluate a sentence given a model (truth assignment)"""
        if isinstance(sentence, str):
            return model.get(sentence, False)
        elif sentence[0] == 'NOT':
            return not self.evaluate(sentence[1], model)
        elif sentence[0] == 'AND':
            return all(self.evaluate(arg, model) for arg in sentence[1:])
        elif sentence[0] == 'OR':
            return any(self.evaluate(arg, model) for arg in sentence[1:])
        elif sentence[0] == 'IMPLIES':
            return not self.evaluate(sentence[1], model) or self.evaluate(sentence[2], model)
        elif sentence[0] == 'IFF':
            return self.evaluate(sentence[1], model) == self.evaluate(sentence[2], model)
    
    def truth_table_enumeration(self, query):
        """Check if query is entailed by knowledge base using truth table"""
        symbols = list(self.symbols)
        
        for model in self.generate_models(symbols):
            if self.pl_true(model) and not self.evaluate(query, model):
                return False
        return True
    
    def generate_models(self, symbols):
        """Generate all possible truth assignments"""
        if not symbols:
            return [{}]
        
        rest = self.generate_models(symbols[1:])
        return [{symbols[0]: True} | model for model in rest] + \
               [{symbols[0]: False} | model for model in rest]
    
    def pl_true(self, model):
        """Check if knowledge base is true in model"""
        return all(self.evaluate(sentence, model) for sentence in self.knowledge_base)

# Example usage
kb = PropositionalLogic()
kb.add_symbol('P')
kb.add_symbol('Q')
kb.add_knowledge(['IMPLIES', 'P', 'Q'])
kb.add_knowledge('P')

# Check if Q is entailed
result = kb.truth_table_enumeration('Q')
print(f"KB entails Q: {result}")
```

### 2. First-Order Logic (FOL)
```python
class FirstOrderLogic:
    def __init__(self):
        self.constants = set()
        self.predicates = {}
        self.functions = {}
        self.knowledge_base = []
    
    def add_constant(self, constant):
        """Add a constant symbol"""
        self.constants.add(constant)
    
    def add_predicate(self, predicate, arity):
        """Add a predicate symbol with arity"""
        self.predicates[predicate] = arity
    
    def add_function(self, function, arity):
        """Add a function symbol with arity"""
        self.functions[function] = arity
    
    def parse_term(self, term):
        """Parse a first-order logic term"""
        if isinstance(term, str):
            if term in self.constants:
                return ('constant', term)
            else:
                return ('variable', term)
        else:
            function, args = term[0], term[1:]
            return ('function', function, [self.parse_term(arg) for arg in args])
    
    def parse_atom(self, atom):
        """Parse an atomic formula"""
        predicate, args = atom[0], atom[1:]
        return ('atom', predicate, [self.parse_term(arg) for arg in args])
    
    def substitute(self, expression, substitution):
        """Apply substitution to expression"""
        if isinstance(expression, str):
            return substitution.get(expression, expression)
        elif expression[0] == 'atom':
            predicate, args = expression[1], expression[2]
            new_args = [self.substitute(arg, substitution) for arg in args]
            return ('atom', predicate, new_args)
        elif expression[0] == 'NOT':
            return ('NOT', self.substitute(expression[1], substitution))
        elif expression[0] in ['AND', 'OR', 'IMPLIES', 'IFF']:
            return (expression[0], 
                   self.substitute(expression[1], substitution),
                   self.substitute(expression[2], substitution))
        elif expression[0] in ['FORALL', 'EXISTS']:
            var, body = expression[1], expression[2]
            return (expression[0], var, self.substitute(body, substitution))

# Example usage
fol = FirstOrderLogic()
fol.add_constant('john')
fol.add_constant('mary')
fol.add_predicate('loves', 2)

# Add knowledge: Everyone loves Mary
fol.knowledge_base.append(['FORALL', 'x', ['loves', 'x', 'mary']])

# Add knowledge: John loves Mary
fol.knowledge_base.append(['loves', 'john', 'mary'])
```

### 3. Resolution and Unification
```python
class Resolution:
    def __init__(self):
        self.clauses = []
    
    def add_clause(self, clause):
        """Add a clause to the knowledge base"""
        self.clauses.append(clause)
    
    def negate(self, literal):
        """Negate a literal"""
        if literal.startswith('NOT_'):
            return literal[4:]
        else:
            return f"NOT_{literal}"
    
    def resolve(self, clause1, clause2):
        """Resolve two clauses"""
        resolvents = []
        
        for literal1 in clause1:
            neg_literal1 = self.negate(literal1)
            if neg_literal1 in clause2:
                # Create resolvent by removing complementary literals
                new_clause = [l for l in clause1 if l != literal1] + \
                           [l for l in clause2 if l != neg_literal1]
                if new_clause not in resolvents:
                    resolvents.append(new_clause)
        
        return resolvents
    
    def resolution_procedure(self, query):
        """Resolution procedure to check entailment"""
        # Negate the query
        negated_query = [self.negate(literal) for literal in query]
        
        # Add negated query to clauses
        all_clauses = self.clauses + [negated_query]
        
        # Apply resolution until contradiction or no new clauses
        new_clauses = []
        while True:
            n = len(all_clauses)
            pairs = [(all_clauses[i], all_clauses[j]) 
                    for i in range(n) for j in range(i+1, n)]
            
            for ci, cj in pairs:
                resolvents = self.resolve(ci, cj)
                for resolvent in resolvents:
                    if not resolvent:  # Empty clause (contradiction)
                        return True
                    if resolvent not in all_clauses and resolvent not in new_clauses:
                        new_clauses.append(resolvent)
            
            if len(new_clauses) == 0:
                return False  # No contradiction found
            
            all_clauses.extend(new_clauses)
            new_clauses = []

# Example usage
resolver = Resolution()
resolver.add_clause(['P', 'Q'])  # P OR Q
resolver.add_clause(['NOT_P', 'R'])  # NOT P OR R
resolver.add_clause(['NOT_Q'])  # NOT Q

# Check if R is entailed
result = resolver.resolution_procedure(['R'])
print(f"KB entails R: {result}")
```

## Causal Reasoning

### 1. Causal Bayesian Networks
```python
import numpy as np
from scipy.stats import bernoulli

class CausalBayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpt = {}  # Conditional Probability Tables
    
    def add_node(self, node, parents=None):
        """Add a node to the network"""
        self.nodes[node] = parents or []
        self.cpt[node] = {}
    
    def add_edge(self, parent, child):
        """Add a directed edge from parent to child"""
        if child not in self.edges:
            self.edges[child] = []
        self.edges[child].append(parent)
        self.nodes[child] = self.edges[child]
    
    def set_cpt(self, node, parent_values, probability):
        """Set conditional probability"""
        key = (node, tuple(parent_values))
        self.cpt[key] = probability
    
    def get_probability(self, node, value, parent_values=None):
        """Get probability of node given parent values"""
        if parent_values is None:
            parent_values = []
        
        key = (node, tuple(parent_values))
        if key in self.cpt:
            return self.cpt[key] if value else 1 - self.cpt[key]
        else:
            return 0.5  # Default uniform probability
    
    def sample(self, evidence=None):
        """Sample from the network given evidence"""
        if evidence is None:
            evidence = {}
        
        # Topological sort
        visited = set()
        order = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.nodes.get(node, []):
                dfs(parent)
            order.append(node)
        
        for node in self.nodes:
            dfs(node)
        
        # Sample in topological order
        sample = evidence.copy()
        for node in order:
            if node not in sample:
                parent_values = [sample.get(parent, False) for parent in self.nodes.get(node, [])]
                prob = self.get_probability(node, True, parent_values)
                sample[node] = bernoulli.rvs(prob)
        
        return sample
    
    def do_intervention(self, node, value):
        """Perform do-intervention on a node"""
        # Create a new network with the intervention
        intervened_network = CausalBayesianNetwork()
        intervened_network.nodes = self.nodes.copy()
        intervened_network.edges = self.edges.copy()
        intervened_network.cpt = self.cpt.copy()
        
        # Remove incoming edges to the intervened node
        if node in intervened_network.edges:
            del intervened_network.edges[node]
        intervened_network.nodes[node] = []
        
        # Set the intervened node to the specified value
        intervened_network.cpt[(node, ())] = 1.0 if value else 0.0
        
        return intervened_network

# Example: Smoking -> Cancer -> Cough
cbn = CausalBayesianNetwork()
cbn.add_node('Smoking')
cbn.add_node('Cancer', ['Smoking'])
cbn.add_node('Cough', ['Cancer'])

# Set probabilities
cbn.set_cpt('Smoking', [], 0.3)  # P(Smoking) = 0.3
cbn.set_cpt('Cancer', [True], 0.8)  # P(Cancer|Smoking) = 0.8
cbn.set_cpt('Cancer', [False], 0.1)  # P(Cancer|not Smoking) = 0.1
cbn.set_cpt('Cough', [True], 0.9)  # P(Cough|Cancer) = 0.9
cbn.set_cpt('Cough', [False], 0.2)  # P(Cough|not Cancer) = 0.2

# Sample from the network
sample = cbn.sample()
print(f"Sample: {sample}")

# Perform intervention
intervened = cbn.do_intervention('Smoking', True)
intervened_sample = intervened.sample()
print(f"After do(Smoking=true): {intervened_sample}")
```

### 2. Counterfactual Reasoning
```python
class CounterfactualReasoner:
    def __init__(self, causal_network):
        self.network = causal_network
    
    def counterfactual_query(self, factual_evidence, intervention, query):
        """Answer counterfactual queries"""
        # Step 1: Abduction - infer unobserved variables
        factual_network = self.network
        abduction_result = self.abduce(factual_evidence)
        
        # Step 2: Action - perform intervention
        intervened_network = factual_network.do_intervention(
            intervention['node'], intervention['value']
        )
        
        # Step 3: Prediction - compute query in intervened world
        counterfactual_prob = self.predict(intervened_network, query, abduction_result)
        
        return counterfactual_prob
    
    def abduce(self, evidence):
        """Infer unobserved variables given evidence"""
        # Simple abduction using rejection sampling
        samples = []
        for _ in range(1000):
            sample = self.network.sample(evidence)
            samples.append(sample)
        
        return samples
    
    def predict(self, network, query, abduction_samples):
        """Predict query probability given samples"""
        count = 0
        for sample in abduction_samples:
            # Apply the same unobserved variables to intervened network
            intervened_sample = network.sample(sample)
            if self.evaluate_query(query, intervened_sample):
                count += 1
        
        return count / len(abduction_samples)
    
    def evaluate_query(self, query, sample):
        """Evaluate a query given a sample"""
        if isinstance(query, str):
            return sample.get(query, False)
        elif query[0] == 'AND':
            return all(self.evaluate_query(q, sample) for q in query[1:])
        elif query[0] == 'OR':
            return any(self.evaluate_query(q, sample) for q in query[1:])
        elif query[0] == 'NOT':
            return not self.evaluate_query(query[1], sample)

# Example counterfactual reasoning
reasoner = CounterfactualReasoner(cbn)

# Factual: Person smokes and has cancer
factual = {'Smoking': True, 'Cancer': True}

# Counterfactual: What if they hadn't smoked?
intervention = {'node': 'Smoking', 'value': False}
query = 'Cancer'

prob = reasoner.counterfactual_query(factual, intervention, query)
print(f"P(Cancer | do(Smoking=false), Smoking=true, Cancer=true) = {prob:.3f}")
```

## Probabilistic Reasoning

### 1. Bayesian Networks
```python
class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpt = {}
    
    def add_node(self, node, parents=None):
        """Add a node to the network"""
        self.nodes[node] = parents or []
        self.cpt[node] = {}
    
    def set_probability(self, node, parent_values, probability):
        """Set conditional probability"""
        key = (node, tuple(parent_values))
        self.cpt[key] = probability
    
    def get_probability(self, node, value, parent_values=None):
        """Get probability of node given parent values"""
        if parent_values is None:
            parent_values = []
        
        key = (node, tuple(parent_values))
        if key in self.cpt:
            return self.cpt[key] if value else 1 - self.cpt[key]
        else:
            return 0.5
    
    def variable_elimination(self, query_vars, evidence=None):
        """Variable elimination for exact inference"""
        if evidence is None:
            evidence = {}
        
        # Initialize factors
        factors = self.create_factors()
        
        # Eliminate variables not in query or evidence
        query_and_evidence = set(query_vars) | set(evidence.keys())
        all_vars = set(self.nodes.keys())
        elim_vars = all_vars - query_and_evidence
        
        for var in elim_vars:
            factors = self.eliminate_variable(factors, var)
        
        # Compute final probability
        result = self.compute_final_probability(factors, query_vars, evidence)
        return result
    
    def create_factors(self):
        """Create initial factors from CPTs"""
        factors = []
        for node in self.nodes:
            parents = self.nodes[node]
            factor = self.create_factor(node, parents)
            factors.append(factor)
        return factors
    
    def create_factor(self, node, parents):
        """Create a factor for a node"""
        factor = {}
        parent_values = self.generate_parent_combinations(parents)
        
        for pv in parent_values:
            for node_val in [True, False]:
                assignment = list(pv) + [node_val]
                prob = self.get_probability(node, node_val, list(pv))
                factor[tuple(assignment)] = prob
        
        return {'vars': parents + [node], 'table': factor}
    
    def generate_parent_combinations(self, parents):
        """Generate all combinations of parent values"""
        if not parents:
            return [()]
        
        rest = self.generate_parent_combinations(parents[1:])
        return [(True,) + r for r in rest] + [(False,) + r for r in rest]
    
    def eliminate_variable(self, factors, var):
        """Eliminate a variable from factors"""
        # Find factors containing the variable
        var_factors = [f for f in factors if var in f['vars']]
        other_factors = [f for f in factors if var not in f['vars']]
        
        if not var_factors:
            return factors
        
        # Multiply factors containing the variable
        product = self.multiply_factors(var_factors)
        
        # Sum out the variable
        marginalized = self.sum_out_variable(product, var)
        
        return other_factors + [marginalized]
    
    def multiply_factors(self, factors):
        """Multiply factors"""
        if len(factors) == 1:
            return factors[0]
        
        result = factors[0]
        for factor in factors[1:]:
            result = self.multiply_two_factors(result, factor)
        
        return result
    
    def multiply_two_factors(self, f1, f2):
        """Multiply two factors"""
        common_vars = set(f1['vars']) & set(f2['vars'])
        all_vars = list(set(f1['vars']) | set(f2['vars']))
        
        result_table = {}
        for assignment in self.generate_assignments(all_vars):
            val1 = self.get_factor_value(f1, assignment)
            val2 = self.get_factor_value(f2, assignment)
            result_table[assignment] = val1 * val2
        
        return {'vars': all_vars, 'table': result_table}
    
    def sum_out_variable(self, factor, var):
        """Sum out a variable from a factor"""
        new_vars = [v for v in factor['vars'] if v != var]
        result_table = {}
        
        for assignment in self.generate_assignments(new_vars):
            total = 0
            for var_val in [True, False]:
                full_assignment = list(assignment)
                var_idx = factor['vars'].index(var)
                full_assignment.insert(var_idx, var_val)
                total += factor['table'].get(tuple(full_assignment), 0)
            result_table[assignment] = total
        
        return {'vars': new_vars, 'table': result_table}
    
    def generate_assignments(self, vars):
        """Generate all assignments for variables"""
        if not vars:
            return [()]
        
        rest = self.generate_assignments(vars[1:])
        return [(True,) + r for r in rest] + [(False,) + r for r in rest]
    
    def get_factor_value(self, factor, assignment):
        """Get value from factor given assignment"""
        return factor['table'].get(assignment, 0)
    
    def compute_final_probability(self, factors, query_vars, evidence):
        """Compute final probability"""
        # Multiply remaining factors
        if factors:
            final_factor = self.multiply_factors(factors)
        else:
            return 1.0
        
        # Normalize
        total = 0
        for assignment in self.generate_assignments(final_factor['vars']):
            if self.consistent_with_evidence(assignment, final_factor['vars'], evidence):
                total += final_factor['table'].get(assignment, 0)
        
        if total == 0:
            return 0.0
        
        # Compute probability for query
        query_prob = 0
        for assignment in self.generate_assignments(final_factor['vars']):
            if self.consistent_with_evidence(assignment, final_factor['vars'], evidence):
                query_assignment = tuple(assignment[i] for i, var in enumerate(final_factor['vars']) 
                                      if var in query_vars)
                if all(query_assignment[i] for i, var in enumerate(query_vars)):
                    query_prob += final_factor['table'].get(assignment, 0)
        
        return query_prob / total
    
    def consistent_with_evidence(self, assignment, vars, evidence):
        """Check if assignment is consistent with evidence"""
        for var, value in evidence.items():
            if var in vars:
                var_idx = vars.index(var)
                if assignment[var_idx] != value:
                    return False
        return True
```

### 2. Markov Chain Monte Carlo (MCMC)
```python
import random

class MCMCSampler:
    def __init__(self, network):
        self.network = network
    
    def gibbs_sampling(self, query_vars, evidence, num_samples=10000):
        """Gibbs sampling for approximate inference"""
        # Initialize with random values
        current_state = self.initialize_state(evidence)
        samples = []
        
        for _ in range(num_samples):
            # Sample each variable in turn
            for var in self.network.nodes:
                if var not in evidence:
                    current_state[var] = self.sample_variable(var, current_state)
            
            # Record sample
            query_values = tuple(current_state[var] for var in query_vars)
            samples.append(query_values)
        
        # Compute probabilities
        return self.compute_probabilities(samples, query_vars)
    
    def initialize_state(self, evidence):
        """Initialize state with evidence and random values"""
        state = evidence.copy()
        for var in self.network.nodes:
            if var not in evidence:
                state[var] = random.choice([True, False])
        return state
    
    def sample_variable(self, var, state):
        """Sample a variable given current state"""
        parents = self.network.nodes[var]
        parent_values = [state[parent] for parent in parents]
        
        # Get conditional probability
        prob = self.network.get_probability(var, True, parent_values)
        
        # Sample based on probability
        return random.random() < prob
    
    def compute_probabilities(self, samples, query_vars):
        """Compute probabilities from samples"""
        counts = {}
        total = len(samples)
        
        for sample in samples:
            if sample not in counts:
                counts[sample] = 0
            counts[sample] += 1
        
        probabilities = {}
        for sample, count in counts.items():
            probabilities[sample] = count / total
        
        return probabilities

# Example usage
bn = BayesianNetwork()
bn.add_node('A')
bn.add_node('B', ['A'])
bn.add_node('C', ['B'])

bn.set_probability('A', [], 0.3)
bn.set_probability('B', [True], 0.8)
bn.set_probability('B', [False], 0.2)
bn.set_probability('C', [True], 0.9)
bn.set_probability('C', [False], 0.1)

# Exact inference
exact_prob = bn.variable_elimination(['C'], {'A': True})
print(f"Exact P(C|A=true): {exact_prob:.3f}")

# Approximate inference
sampler = MCMCSampler(bn)
approx_probs = sampler.gibbs_sampling(['C'], {'A': True}, 1000)
print(f"Approximate P(C|A=true): {approx_probs}")
```

## Symbolic AI and Knowledge Representation

### 1. Production Rules
```python
class ProductionSystem:
    def __init__(self):
        self.rules = []
        self.working_memory = set()
    
    def add_rule(self, condition, action):
        """Add a production rule"""
        self.rules.append({'condition': condition, 'action': action})
    
    def add_fact(self, fact):
        """Add a fact to working memory"""
        self.working_memory.add(fact)
    
    def match_condition(self, condition):
        """Check if condition matches working memory"""
        if isinstance(condition, str):
            return condition in self.working_memory
        elif condition[0] == 'AND':
            return all(self.match_condition(c) for c in condition[1:])
        elif condition[0] == 'OR':
            return any(self.match_condition(c) for c in condition[1:])
        elif condition[0] == 'NOT':
            return not self.match_condition(condition[1])
    
    def execute_action(self, action):
        """Execute an action"""
        if isinstance(action, str):
            self.working_memory.add(action)
        elif action[0] == 'ADD':
            for fact in action[1:]:
                self.working_memory.add(fact)
        elif action[0] == 'REMOVE':
            for fact in action[1:]:
                self.working_memory.discard(fact)
    
    def forward_chain(self):
        """Forward chaining inference"""
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if self.match_condition(rule['condition']):
                    old_memory = self.working_memory.copy()
                    self.execute_action(rule['action'])
                    if self.working_memory != old_memory:
                        changed = True
    
    def backward_chain(self, goal):
        """Backward chaining inference"""
        if goal in self.working_memory:
            return True
        
        for rule in self.rules:
            if rule['action'] == goal or (isinstance(rule['action'], list) and goal in rule['action']):
                if self.match_condition(rule['condition']):
                    return True
                else:
                    # Try to prove condition
                    if isinstance(rule['condition'], str):
                        if self.backward_chain(rule['condition']):
                            return True
                    elif rule['condition'][0] == 'AND':
                        if all(self.backward_chain(c) for c in rule['condition'][1:]):
                            return True
        
        return False

# Example production system
ps = ProductionSystem()

# Add rules
ps.add_rule('bird', 'can_fly')
ps.add_rule(['AND', 'bird', 'penguin'], 'cannot_fly')
ps.add_rule('penguin', 'bird')

# Add facts
ps.add_fact('penguin')

# Forward chaining
ps.forward_chain()
print(f"Working memory after forward chaining: {ps.working_memory}")

# Backward chaining
result = ps.backward_chain('can_fly')
print(f"Can penguin fly? {result}")
```

### 2. Semantic Networks
```python
class SemanticNetwork:
    def __init__(self):
        self.nodes = {}
        self.relations = {}
    
    def add_node(self, node, properties=None):
        """Add a node to the network"""
        self.nodes[node] = properties or {}
    
    def add_relation(self, source, relation, target):
        """Add a relation between nodes"""
        if source not in self.relations:
            self.relations[source] = {}
        if relation not in self.relations[source]:
            self.relations[source][relation] = []
        self.relations[source][relation].append(target)
    
    def query(self, query):
        """Query the semantic network"""
        if query[0] == 'ISA':
            return self.isa_query(query[1], query[2])
        elif query[0] == 'HAS_PROPERTY':
            return self.property_query(query[1], query[2])
        elif query[0] == 'RELATED':
            return self.relation_query(query[1], query[2], query[3])
    
    def isa_query(self, instance, category):
        """Check if instance is a category"""
        if instance == category:
            return True
        
        if instance in self.relations:
            for relation, targets in self.relations[instance].items():
                if relation == 'ISA':
                    for target in targets:
                        if self.isa_query(target, category):
                            return True
        
        return False
    
    def property_query(self, instance, property_name):
        """Check if instance has property"""
        # Check direct properties
        if instance in self.nodes and property_name in self.nodes[instance]:
            return self.nodes[instance][property_name]
        
        # Check inherited properties
        if instance in self.relations:
            for relation, targets in self.relations[instance].items():
                if relation == 'ISA':
                    for target in targets:
                        if self.property_query(target, property_name):
                            return True
        
        return False
    
    def relation_query(self, source, relation, target):
        """Check if source is related to target"""
        if source in self.relations and relation in self.relations[source]:
            return target in self.relations[source][relation]
        
        return False

# Example semantic network
sn = SemanticNetwork()

# Add nodes
sn.add_node('bird', {'can_fly': True, 'has_wings': True})
sn.add_node('penguin', {'can_fly': False, 'lives_in_antarctica': True})
sn.add_node('animal', {'has_blood': True})

# Add relations
sn.add_relation('penguin', 'ISA', 'bird')
sn.add_relation('bird', 'ISA', 'animal')

# Queries
print(f"Penguin is a bird: {sn.query(['ISA', 'penguin', 'bird'])}")
print(f"Penguin can fly: {sn.query(['HAS_PROPERTY', 'penguin', 'can_fly'])}")
print(f"Penguin has wings: {sn.query(['HAS_PROPERTY', 'penguin', 'has_wings'])}")
```

## Evaluation Metrics

### 1. Logical Reasoning Metrics
```python
class LogicalReasoningEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_soundness(self, system, test_cases):
        """Evaluate soundness of logical reasoning"""
        correct = 0
        total = 0
        
        for premise, conclusion in test_cases:
            result = system.entails(premise, conclusion)
            expected = self.check_logical_entailment(premise, conclusion)
            
            if result == expected:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def evaluate_completeness(self, system, test_cases):
        """Evaluate completeness of logical reasoning"""
        correct = 0
        total = 0
        
        for premise, conclusion in test_cases:
            result = system.entails(premise, conclusion)
            expected = self.check_logical_entailment(premise, conclusion)
            
            if result == expected:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def check_logical_entailment(self, premise, conclusion):
        """Check if premise logically entails conclusion"""
        # Implementation depends on logical system
        return True  # Placeholder
```

### 2. Causal Reasoning Metrics
```python
class CausalReasoningEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_causal_discovery(self, discovered_graph, true_graph):
        """Evaluate causal discovery accuracy"""
        # Compare edge structure
        true_edges = set(true_graph.edges())
        discovered_edges = set(discovered_graph.edges())
        
        precision = len(true_edges & discovered_edges) / len(discovered_edges) if discovered_edges else 0
        recall = len(true_edges & discovered_edges) / len(true_edges) if true_edges else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def evaluate_counterfactual_accuracy(self, model, test_cases):
        """Evaluate counterfactual reasoning accuracy"""
        correct = 0
        total = 0
        
        for factual, intervention, query, expected in test_cases:
            result = model.counterfactual_query(factual, intervention, query)
            
            if abs(result - expected) < 0.1:  # Tolerance
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
```

## Tools and Libraries

- **PyMC**: Probabilistic programming
- **pgmpy**: Bayesian networks and causal inference
- **NetworkX**: Graph algorithms
- **SymPy**: Symbolic mathematics
- **Prover9**: Automated theorem proving

## Best Practices

1. **Soundness**: Ensure logical correctness of reasoning
2. **Completeness**: Cover all valid inferences
3. **Efficiency**: Use appropriate algorithms for scale
4. **Interpretability**: Make reasoning transparent
5. **Robustness**: Handle uncertainty and noise

## Next Steps

1. **Neural-Symbolic Integration**: Combine neural and symbolic approaches
2. **Automated Theorem Proving**: Advanced logical reasoning
3. **Causal Discovery**: Learn causal structure from data
4. **Explainable AI**: Make AI reasoning interpretable
5. **Quantum Reasoning**: Quantum approaches to reasoning

---

*AI Reasoning combines logical rigor with probabilistic methods to enable intelligent systems that can draw conclusions, make inferences, and solve complex problems. From symbolic logic to causal inference, these techniques form the foundation of intelligent decision-making.* 