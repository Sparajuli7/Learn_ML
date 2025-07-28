# Causal AI & Bayesian Methods

## Overview
Causal AI focuses on understanding cause-and-effect relationships in data, going beyond correlation to identify true causal mechanisms. Bayesian methods provide a principled framework for reasoning under uncertainty, combining prior knowledge with observed data.

## Causal Inference Fundamentals

### 1. Causal Graphical Models
```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.variables = set()
        self.interventions = {}
    
    def add_variable(self, variable):
        """Add a variable to the causal graph"""
        self.variables.add(variable)
        self.graph.add_node(variable)
    
    def add_edge(self, cause, effect):
        """Add a causal edge from cause to effect"""
        self.graph.add_edge(cause, effect)
    
    def get_parents(self, variable):
        """Get parents of a variable"""
        return list(self.graph.predecessors(variable))
    
    def get_children(self, variable):
        """Get children of a variable"""
        return list(self.graph.successors(variable))
    
    def get_ancestors(self, variable):
        """Get all ancestors of a variable"""
        return nx.ancestors(self.graph, variable)
    
    def get_descendants(self, variable):
        """Get all descendants of a variable"""
        return nx.descendants(self.graph, variable)
    
    def is_d_separated(self, x, y, z):
        """Check if x and y are d-separated given z"""
        return nx.d_separated(self.graph, {x}, {y}, set(z))
    
    def get_backdoor_paths(self, treatment, outcome):
        """Get backdoor paths between treatment and outcome"""
        paths = []
        for path in nx.all_simple_paths(self.graph, treatment, outcome):
            if self.is_backdoor_path(path, treatment):
                paths.append(path)
        return paths
    
    def is_backdoor_path(self, path, treatment):
        """Check if a path is a backdoor path"""
        if len(path) < 3:
            return False
        
        # Check if path has arrow pointing to treatment
        for i in range(len(path) - 1):
            if path[i+1] == treatment and self.graph.has_edge(path[i], path[i+1]):
                return True
        
        return False
    
    def get_backdoor_adjustment_set(self, treatment, outcome):
        """Find minimal backdoor adjustment set"""
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)
        
        if not backdoor_paths:
            return set()
        
        # Find variables that block all backdoor paths
        blocking_vars = set()
        for path in backdoor_paths:
            # Find variables that can block this path
            for var in path[1:-1]:  # Exclude treatment and outcome
                if var != treatment and var != outcome:
                    blocking_vars.add(var)
        
        return blocking_vars
    
    def do_intervention(self, variable, value):
        """Perform do-intervention on a variable"""
        intervened_graph = self.graph.copy()
        
        # Remove incoming edges to the intervened variable
        for parent in list(intervened_graph.predecessors(variable)):
            intervened_graph.remove_edge(parent, variable)
        
        # Store intervention
        self.interventions[variable] = value
        
        return intervened_graph
    
    def get_causal_effect(self, treatment, outcome, adjustment_set=None):
        """Estimate causal effect using backdoor adjustment"""
        if adjustment_set is None:
            adjustment_set = self.get_backdoor_adjustment_set(treatment, outcome)
        
        # This would typically involve data and statistical estimation
        # Here we return a placeholder
        return f"E[Y|do(X={treatment})] = Σ_z E[Y|X={treatment}, Z=z] P(Z=z)"

# Example: Smoking -> Cancer -> Cough
causal_graph = CausalGraph()
causal_graph.add_variable('Smoking')
causal_graph.add_variable('Cancer')
causal_graph.add_variable('Cough')
causal_graph.add_variable('Age')

causal_graph.add_edge('Age', 'Smoking')
causal_graph.add_edge('Smoking', 'Cancer')
causal_graph.add_edge('Cancer', 'Cough')
causal_graph.add_edge('Age', 'Cancer')

print("Backdoor adjustment set:", causal_graph.get_backdoor_adjustment_set('Smoking', 'Cough'))
```

### 2. Structural Causal Models (SCM)
```python
class StructuralCausalModel:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.noise_distributions = {}
    
    def add_variable(self, variable, parents=None, function=None, noise_dist=None):
        """Add a variable to the SCM"""
        self.variables[variable] = parents or []
        if function:
            self.functions[variable] = function
        if noise_dist:
            self.noise_distributions[variable] = noise_dist
    
    def set_function(self, variable, function):
        """Set the structural function for a variable"""
        self.functions[variable] = function
    
    def set_noise_distribution(self, variable, distribution):
        """Set the noise distribution for a variable"""
        self.noise_distributions[variable] = distribution
    
    def sample(self, size=1000):
        """Sample from the SCM"""
        samples = {}
        
        # Sample in topological order
        order = self.topological_sort()
        
        for variable in order:
            if variable in self.functions:
                # Sample noise
                noise = self.noise_distributions[variable].rvs(size)
                
                # Get parent values
                parent_values = []
                for parent in self.variables[variable]:
                    parent_values.append(samples[parent])
                
                # Apply structural function
                if parent_values:
                    parent_array = np.column_stack(parent_values)
                    samples[variable] = self.functions[variable](parent_array, noise)
                else:
                    samples[variable] = self.functions[variable](None, noise)
            else:
                # Exogenous variable
                samples[variable] = self.noise_distributions[variable].rvs(size)
        
        return samples
    
    def topological_sort(self):
        """Perform topological sort of variables"""
        # Simple implementation - assumes no cycles
        visited = set()
        order = []
        
        def dfs(variable):
            if variable in visited:
                return
            visited.add(variable)
            for parent in self.variables[variable]:
                dfs(parent)
            order.append(variable)
        
        for variable in self.variables:
            dfs(variable)
        
        return order
    
    def do_intervention(self, variable, value):
        """Perform do-intervention"""
        intervened_scm = StructuralCausalModel()
        intervened_scm.variables = self.variables.copy()
        intervened_scm.functions = self.functions.copy()
        intervened_scm.noise_distributions = self.noise_distributions.copy()
        
        # Remove parents of intervened variable
        intervened_scm.variables[variable] = []
        
        # Set function to constant
        intervened_scm.functions[variable] = lambda parents, noise: np.full_like(noise, value)
        
        return intervened_scm
    
    def counterfactual_query(self, factual_evidence, intervention, query_variable):
        """Answer counterfactual queries"""
        # Step 1: Abduction - infer noise values
        noise_values = self.abduce_noise(factual_evidence)
        
        # Step 2: Action - perform intervention
        intervened_scm = self.do_intervention(intervention['variable'], intervention['value'])
        
        # Step 3: Prediction - compute counterfactual
        counterfactual = self.predict_counterfactual(intervened_scm, noise_values, query_variable)
        
        return counterfactual
    
    def abduce_noise(self, evidence):
        """Infer noise values given evidence"""
        # This is a simplified version
        # In practice, this would involve solving the structural equations
        noise_values = {}
        
        for variable, value in evidence.items():
            if variable in self.functions:
                # Solve for noise given the observed value
                parents = self.variables[variable]
                if parents:
                    parent_values = [evidence.get(p, 0) for p in parents]
                    # This is a simplified calculation
                    noise_values[variable] = value - np.mean(parent_values)
                else:
                    noise_values[variable] = value
        
        return noise_values
    
    def predict_counterfactual(self, intervened_scm, noise_values, query_variable):
        """Predict counterfactual value"""
        # Use the intervened SCM with the inferred noise values
        samples = intervened_scm.sample(size=1)
        return samples[query_variable][0]

# Example SCM
from scipy.stats import norm

scm = StructuralCausalModel()

# Add variables
scm.add_variable('X', noise_dist=norm(0, 1))
scm.add_variable('Y', parents=['X'], noise_dist=norm(0, 0.5))
scm.add_variable('Z', parents=['X', 'Y'], noise_dist=norm(0, 0.3))

# Set structural functions
scm.set_function('X', lambda parents, noise: noise)
scm.set_function('Y', lambda parents, noise: 2 * parents[:, 0] + noise)
scm.set_function('Z', lambda parents, noise: parents[:, 0] + 0.5 * parents[:, 1] + noise)

# Sample from SCM
samples = scm.sample(1000)
print("SCM samples shape:", {k: v.shape for k, v in samples.items()})
```

## Bayesian Methods

### 1. Bayesian Networks
```python
class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.cpt = {}  # Conditional Probability Tables
        self.data = None
    
    def add_node(self, node, parents=None):
        """Add a node to the network"""
        self.nodes[node] = parents or []
        self.cpt[node] = {}
    
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
            return 0.5
    
    def fit_from_data(self, data):
        """Learn CPTs from data"""
        self.data = data
        
        for node in self.nodes:
            parents = self.nodes[node]
            self.learn_cpt(node, parents, data)
    
    def learn_cpt(self, node, parents, data):
        """Learn CPT for a node"""
        if not parents:
            # Root node - learn marginal probability
            prob = np.mean(data[node])
            self.set_cpt(node, [], prob)
        else:
            # Learn conditional probabilities
            for parent_combination in self.generate_parent_combinations(parents):
                mask = np.ones(len(data), dtype=bool)
                for i, parent in enumerate(parents):
                    mask &= (data[parent] == parent_combination[i])
                
                if np.sum(mask) > 0:
                    prob = np.mean(data[node][mask])
                    self.set_cpt(node, parent_combination, prob)
    
    def generate_parent_combinations(self, parents):
        """Generate all combinations of parent values"""
        if not parents:
            return [()]
        
        rest = self.generate_parent_combinations(parents[1:])
        return [(True,) + r for r in rest] + [(False,) + r for r in rest]
    
    def variable_elimination(self, query_vars, evidence=None):
        """Variable elimination for exact inference"""
        if evidence is None:
            evidence = {}
        
        # Create factors from CPTs
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
        """Create factors from CPTs"""
        factors = []
        for node in self.nodes:
            parents = self.nodes[node]
            factor = self.create_factor(node, parents)
            factors.append(factor)
        return factors
    
    def create_factor(self, node, parents):
        """Create a factor for a node"""
        factor = {}
        parent_combinations = self.generate_parent_combinations(parents)
        
        for parent_values in parent_combinations:
            for node_val in [True, False]:
                assignment = list(parent_values) + [node_val]
                prob = self.get_probability(node, node_val, list(parent_values))
                factor[tuple(assignment)] = prob
        
        return {'vars': parents + [node], 'table': factor}
    
    def eliminate_variable(self, factors, var):
        """Eliminate a variable from factors"""
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

# Example Bayesian Network
bn = BayesianNetwork()
bn.add_node('A')
bn.add_node('B', ['A'])
bn.add_node('C', ['B'])

# Set CPTs
bn.set_cpt('A', [], 0.3)
bn.set_cpt('B', [True], 0.8)
bn.set_cpt('B', [False], 0.2)
bn.set_cpt('C', [True], 0.9)
bn.set_cpt('C', [False], 0.1)

# Inference
prob = bn.variable_elimination(['C'], {'A': True})
print(f"P(C|A=true) = {prob:.3f}")
```

### 2. Bayesian Parameter Learning
```python
class BayesianParameterLearning:
    def __init__(self, network):
        self.network = network
        self.priors = {}
    
    def set_prior(self, node, prior_distribution):
        """Set prior distribution for a node"""
        self.priors[node] = prior_distribution
    
    def conjugate_posterior(self, node, data, parent_values=None):
        """Compute conjugate posterior for a node"""
        if node not in self.priors:
            # Use uniform prior if no prior specified
            from scipy.stats import beta
            self.priors[node] = beta(1, 1)
        
        prior = self.priors[node]
        
        if parent_values is None:
            # Root node - learn marginal probability
            successes = np.sum(data[node])
            failures = len(data[node]) - successes
            
            # Beta-Binomial conjugate pair
            posterior = beta(prior.args[0] + successes, 
                           prior.args[1] + failures)
        else:
            # Conditional node
            mask = np.ones(len(data), dtype=bool)
            for i, parent in enumerate(self.network.nodes[node]):
                mask &= (data[parent] == parent_values[i])
            
            if np.sum(mask) > 0:
                successes = np.sum(data[node][mask])
                failures = np.sum(mask) - successes
                
                posterior = beta(prior.args[0] + successes, 
                               prior.args[1] + failures)
            else:
                posterior = prior
        
        return posterior
    
    def learn_parameters(self, data):
        """Learn all parameters from data"""
        posteriors = {}
        
        for node in self.network.nodes:
            parents = self.network.nodes[node]
            
            if not parents:
                # Root node
                posteriors[node] = self.conjugate_posterior(node, data)
            else:
                # Conditional node
                posteriors[node] = {}
                for parent_combination in self.generate_parent_combinations(parents):
                    posteriors[node][parent_combination] = self.conjugate_posterior(
                        node, data, parent_combination
                    )
        
        return posteriors
    
    def generate_parent_combinations(self, parents):
        """Generate all combinations of parent values"""
        if not parents:
            return [()]
        
        rest = self.generate_parent_combinations(parents[1:])
        return [(True,) + r for r in rest] + [(False,) + r for r in rest]
    
    def predict_probability(self, posteriors, node, parent_values=None):
        """Predict probability using learned posteriors"""
        if parent_values is None:
            posterior = posteriors[node]
        else:
            posterior = posteriors[node][parent_values]
        
        # Return posterior mean
        return posterior.args[0] / (posterior.args[0] + posterior.args[1])

# Example parameter learning
from scipy.stats import beta

learner = BayesianParameterLearning(bn)

# Set priors
learner.set_prior('A', beta(1, 1))
learner.set_prior('B', beta(1, 1))
learner.set_prior('C', beta(1, 1))

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
data = {
    'A': np.random.binomial(1, 0.3, n_samples),
    'B': np.random.binomial(1, 0.8, n_samples),
    'C': np.random.binomial(1, 0.9, n_samples)
}

# Learn parameters
posteriors = learner.learn_parameters(data)

# Predict probabilities
pred_prob_a = learner.predict_probability(posteriors, 'A')
pred_prob_b = learner.predict_probability(posteriors, 'B', (True,))
pred_prob_c = learner.predict_probability(posteriors, 'C', (True,))

print(f"Learned P(A): {pred_prob_a:.3f}")
print(f"Learned P(B|A=true): {pred_prob_b:.3f}")
print(f"Learned P(C|B=true): {pred_prob_c:.3f}")
```

## Causal Discovery

### 1. PC Algorithm
```python
class PCAlgorithm:
    def __init__(self, data, alpha=0.05):
        self.data = data
        self.alpha = alpha
        self.skeleton = None
        self.directed_edges = set()
        self.undirected_edges = set()
    
    def run(self):
        """Run the PC algorithm"""
        # Step 1: Find skeleton
        self.find_skeleton()
        
        # Step 2: Orient edges
        self.orient_edges()
        
        return self.get_graph()
    
    def find_skeleton(self):
        """Find the skeleton of the graph"""
        n_vars = self.data.shape[1]
        var_names = list(self.data.columns)
        
        # Initialize complete graph
        self.skeleton = set()
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                self.skeleton.add((var_names[i], var_names[j]))
        
        # Test independence
        l = 0
        while True:
            edges_to_remove = set()
            
            for edge in self.skeleton:
                x, y = edge
                neighbors_x = self.get_neighbors(x)
                neighbors_y = self.get_neighbors(y)
                
                # Test all subsets of size l
                for subset in self.generate_subsets(neighbors_x, l):
                    if self.test_independence(x, y, subset):
                        edges_to_remove.add(edge)
                        break
            
            # Remove edges
            self.skeleton -= edges_to_remove
            
            # Check if any edge has fewer than l+1 neighbors
            max_neighbors = max(len(self.get_neighbors(var)) for var in var_names)
            if max_neighbors <= l:
                break
            
            l += 1
    
    def orient_edges(self):
        """Orient edges using orientation rules"""
        # Rule 1: Orient X-Y-Z as X->Y<-Z if X and Z are not adjacent
        for edge1 in self.skeleton:
            for edge2 in self.skeleton:
                if edge1[1] == edge2[0] and edge1[0] != edge2[1]:
                    x, y, z = edge1[0], edge1[1], edge2[1]
                    if (x, z) not in self.skeleton and (z, x) not in self.skeleton:
                        self.directed_edges.add((x, y))
                        self.directed_edges.add((z, y))
                        self.skeleton.discard((x, y))
                        self.skeleton.discard((y, z))
        
        # Rule 2: Avoid cycles
        self.avoid_cycles()
        
        # Remaining edges are undirected
        self.undirected_edges = self.skeleton.copy()
    
    def get_neighbors(self, variable):
        """Get neighbors of a variable in the skeleton"""
        neighbors = set()
        for edge in self.skeleton:
            if edge[0] == variable:
                neighbors.add(edge[1])
            elif edge[1] == variable:
                neighbors.add(edge[0])
        return neighbors
    
    def generate_subsets(self, items, size):
        """Generate all subsets of given size"""
        if size == 0:
            return [()]
        if size > len(items):
            return []
        
        items = list(items)
        from itertools import combinations
        return list(combinations(items, size))
    
    def test_independence(self, x, y, conditioning_set):
        """Test conditional independence using chi-square test"""
        from scipy.stats import chi2_contingency
        
        # Create contingency table
        if not conditioning_set:
            # Marginal independence
            contingency = pd.crosstab(self.data[x], self.data[y])
        else:
            # Conditional independence
            contingency = pd.crosstab(
                self.data[x], 
                self.data[y], 
                self.data[list(conditioning_set)]
            )
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        return p_value > self.alpha
    
    def avoid_cycles(self):
        """Avoid creating cycles in the graph"""
        # Simple cycle detection
        for edge in self.directed_edges:
            if self.would_create_cycle(edge):
                self.directed_edges.discard(edge)
                self.undirected_edges.add(edge)
    
    def would_create_cycle(self, edge):
        """Check if adding edge would create a cycle"""
        # Simplified cycle detection
        # In practice, this would use a proper cycle detection algorithm
        return False
    
    def get_graph(self):
        """Return the discovered graph"""
        return {
            'directed_edges': self.directed_edges,
            'undirected_edges': self.undirected_edges
        }

# Example causal discovery
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# True causal structure: A -> B -> C
A = np.random.binomial(1, 0.3, n_samples)
B = np.random.binomial(1, 0.2 + 0.6 * A, n_samples)
C = np.random.binomial(1, 0.1 + 0.8 * B, n_samples)

data = pd.DataFrame({'A': A, 'B': B, 'C': C})

# Run PC algorithm
pc = PCAlgorithm(data)
discovered_graph = pc.run()

print("Discovered graph:", discovered_graph)
```

### 2. GES Algorithm (Greedy Equivalence Search)
```python
class GESAlgorithm:
    def __init__(self, data, score_function='bic'):
        self.data = data
        self.score_function = score_function
        self.current_graph = None
        self.best_score = float('-inf')
    
    def run(self):
        """Run the GES algorithm"""
        # Phase 1: Forward search
        self.forward_search()
        
        # Phase 2: Backward search
        self.backward_search()
        
        return self.current_graph
    
    def forward_search(self):
        """Forward phase of GES"""
        self.current_graph = self.create_empty_graph()
        self.best_score = self.score_graph(self.current_graph)
        
        while True:
            best_edge = None
            best_score = self.best_score
            
            # Try all possible edge additions
            for edge in self.get_possible_edges():
                if self.is_valid_addition(edge):
                    new_graph = self.add_edge(self.current_graph, edge)
                    score = self.score_graph(new_graph)
                    
                    if score > best_score:
                        best_score = score
                        best_edge = edge
            
            if best_edge is None:
                break
            
            # Add best edge
            self.current_graph = self.add_edge(self.current_graph, best_edge)
            self.best_score = best_score
    
    def backward_search(self):
        """Backward phase of GES"""
        while True:
            best_edge = None
            best_score = self.best_score
            
            # Try all possible edge removals
            for edge in self.get_current_edges():
                new_graph = self.remove_edge(self.current_graph, edge)
                score = self.score_graph(new_graph)
                
                if score > best_score:
                    best_score = score
                    best_edge = edge
            
            if best_edge is None:
                break
            
            # Remove best edge
            self.current_graph = self.remove_edge(self.current_graph, best_edge)
            self.best_score = best_score
    
    def create_empty_graph(self):
        """Create an empty graph"""
        n_vars = self.data.shape[1]
        return np.zeros((n_vars, n_vars))
    
    def get_possible_edges(self):
        """Get all possible edges"""
        n_vars = self.data.shape[1]
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    edges.append((i, j))
        return edges
    
    def get_current_edges(self):
        """Get current edges in the graph"""
        edges = []
        for i in range(self.current_graph.shape[0]):
            for j in range(self.current_graph.shape[1]):
                if self.current_graph[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def is_valid_addition(self, edge):
        """Check if edge addition is valid"""
        # Check for cycles
        temp_graph = self.add_edge(self.current_graph, edge)
        return not self.has_cycle(temp_graph)
    
    def add_edge(self, graph, edge):
        """Add edge to graph"""
        new_graph = graph.copy()
        new_graph[edge[0], edge[1]] = 1
        return new_graph
    
    def remove_edge(self, graph, edge):
        """Remove edge from graph"""
        new_graph = graph.copy()
        new_graph[edge[0], edge[1]] = 0
        return new_graph
    
    def has_cycle(self, graph):
        """Check if graph has cycles"""
        # Simplified cycle detection
        # In practice, this would use DFS or other cycle detection
        return False
    
    def score_graph(self, graph):
        """Score a graph using the specified score function"""
        if self.score_function == 'bic':
            return self.bic_score(graph)
        else:
            return self.log_likelihood_score(graph)
    
    def bic_score(self, graph):
        """Compute BIC score for graph"""
        # Simplified BIC calculation
        # In practice, this would fit the graph and compute BIC
        n_edges = np.sum(graph)
        n_samples = len(self.data)
        n_params = n_edges * 2  # Simplified parameter count
        
        # Simplified log-likelihood
        log_likelihood = -n_edges * 0.1  # Placeholder
        
        bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
        return bic
    
    def log_likelihood_score(self, graph):
        """Compute log-likelihood score for graph"""
        # Simplified log-likelihood calculation
        n_edges = np.sum(graph)
        return -n_edges * 0.1  # Placeholder

# Example GES
ges = GESAlgorithm(data)
discovered_graph_ges = ges.run()
print("GES discovered graph:", discovered_graph_ges)
```

## Causal Effect Estimation

### 1. Propensity Score Matching
```python
class PropensityScoreMatching:
    def __init__(self, data, treatment, outcome, covariates):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
    
    def estimate_propensity_scores(self):
        """Estimate propensity scores using logistic regression"""
        from sklearn.linear_model import LogisticRegression
        
        X = self.data[self.covariates]
        T = self.data[self.treatment]
        
        model = LogisticRegression()
        model.fit(X, T)
        
        propensity_scores = model.predict_proba(X)[:, 1]
        return propensity_scores
    
    def match_treated_control(self, propensity_scores, method='nearest'):
        """Match treated and control units"""
        treated_indices = self.data[self.treatment] == 1
        control_indices = self.data[self.treatment] == 0
        
        treated_scores = propensity_scores[treated_indices]
        control_scores = propensity_scores[control_indices]
        
        matches = {}
        
        if method == 'nearest':
            from sklearn.neighbors import NearestNeighbors
            
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control_scores.reshape(-1, 1))
            
            distances, indices = nn.kneighbors(treated_scores.reshape(-1, 1))
            
            for i, (treated_idx, control_idx) in enumerate(zip(
                np.where(treated_indices)[0], 
                np.where(control_indices)[0][indices.flatten()]
            )):
                matches[treated_idx] = control_idx
        
        return matches
    
    def estimate_ate(self, matches):
        """Estimate Average Treatment Effect"""
        treated_outcomes = []
        control_outcomes = []
        
        for treated_idx, control_idx in matches.items():
            treated_outcomes.append(self.data.iloc[treated_idx][self.outcome])
            control_outcomes.append(self.data.iloc[control_idx][self.outcome])
        
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
        return ate
    
    def run(self):
        """Run propensity score matching"""
        # Estimate propensity scores
        propensity_scores = self.estimate_propensity_scores()
        
        # Match treated and control units
        matches = self.match_treated_control(propensity_scores)
        
        # Estimate ATE
        ate = self.estimate_ate(matches)
        
        return {
            'propensity_scores': propensity_scores,
            'matches': matches,
            'ate': ate
        }

# Example propensity score matching
# Generate synthetic data with confounding
np.random.seed(42)
n_samples = 1000

# Confounder
Z = np.random.normal(0, 1, n_samples)

# Treatment assignment depends on confounder
propensity = 1 / (1 + np.exp(-Z))
T = np.random.binomial(1, propensity, n_samples)

# Outcome depends on treatment and confounder
Y = 2 * T + 1.5 * Z + np.random.normal(0, 0.5, n_samples)

data = pd.DataFrame({'T': T, 'Y': Y, 'Z': Z})

# Run propensity score matching
psm = PropensityScoreMatching(data, 'T', 'Y', ['Z'])
results = psm.run()

print(f"Estimated ATE: {results['ate']:.3f}")
print(f"True ATE: 2.0")
```

### 2. Instrumental Variables
```python
class InstrumentalVariables:
    def __init__(self, data, instrument, treatment, outcome, covariates=None):
        self.data = data
        self.instrument = instrument
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates or []
    
    def two_stage_least_squares(self):
        """Estimate causal effect using 2SLS"""
        from sklearn.linear_model import LinearRegression
        
        # First stage: regress treatment on instrument and covariates
        X1 = self.data[[self.instrument] + self.covariates]
        T = self.data[self.treatment]
        
        stage1_model = LinearRegression()
        stage1_model.fit(X1, T)
        
        # Predicted treatment values
        T_pred = stage1_model.predict(X1)
        
        # Second stage: regress outcome on predicted treatment and covariates
        X2 = pd.DataFrame({'T_pred': T_pred})
        for covariate in self.covariates:
            X2[covariate] = self.data[covariate]
        
        Y = self.data[self.outcome]
        
        stage2_model = LinearRegression()
        stage2_model.fit(X2, Y)
        
        # Extract treatment effect
        treatment_effect = stage2_model.coef_[0]
        
        return {
            'treatment_effect': treatment_effect,
            'stage1_model': stage1_model,
            'stage2_model': stage2_model
        }
    
    def test_exclusion_restriction(self):
        """Test the exclusion restriction assumption"""
        # Regress outcome on instrument directly
        from sklearn.linear_model import LinearRegression
        
        X = self.data[[self.instrument] + self.covariates]
        Y = self.data[self.outcome]
        
        model = LinearRegression()
        model.fit(X, Y)
        
        # If instrument coefficient is close to zero, exclusion restriction holds
        instrument_coef = model.coef_[0]
        
        return {
            'instrument_coefficient': instrument_coef,
            'exclusion_restriction_holds': abs(instrument_coef) < 0.1
        }
    
    def test_relevance(self):
        """Test the relevance assumption"""
        # Regress treatment on instrument
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X = self.data[[self.instrument] + self.covariates]
        T = self.data[self.treatment]
        
        model = LinearRegression()
        model.fit(X, T)
        
        T_pred = model.predict(X)
        r2 = r2_score(T, T_pred)
        
        return {
            'r2': r2,
            'relevance_holds': r2 > 0.1
        }

# Example instrumental variables
# Generate synthetic data with endogeneity
np.random.seed(42)
n_samples = 1000

# Instrument
Z = np.random.normal(0, 1, n_samples)

# Unobserved confounder
U = np.random.normal(0, 1, n_samples)

# Treatment depends on instrument and confounder
T = 0.5 * Z + 0.3 * U + np.random.normal(0, 0.5, n_samples)

# Outcome depends on treatment and confounder
Y = 2 * T + 1.5 * U + np.random.normal(0, 0.5, n_samples)

data = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})

# Run instrumental variables analysis
iv = InstrumentalVariables(data, 'Z', 'T', 'Y')
results = iv.two_stage_least_squares()

print(f"Estimated treatment effect: {results['treatment_effect']:.3f}")
print(f"True treatment effect: 2.0")

# Test assumptions
exclusion_test = iv.test_exclusion_restriction()
relevance_test = iv.test_relevance()

print(f"Exclusion restriction holds: {exclusion_test['exclusion_restriction_holds']}")
print(f"Relevance holds: {relevance_test['relevance_holds']}")
```

## Evaluation Metrics

### 1. Causal Discovery Metrics
```python
class CausalDiscoveryEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_structure(self, discovered_graph, true_graph):
        """Evaluate causal structure discovery"""
        # Convert to edge sets
        discovered_edges = set(discovered_graph['directed_edges'])
        true_edges = set(true_graph['directed_edges'])
        
        # Compute metrics
        tp = len(discovered_edges & true_edges)
        fp = len(discovered_edges - true_edges)
        fn = len(true_edges - discovered_edges)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def evaluate_orientation(self, discovered_graph, true_graph):
        """Evaluate edge orientation accuracy"""
        discovered_edges = set(discovered_graph['directed_edges'])
        true_edges = set(true_graph['directed_edges'])
        
        # Count correctly oriented edges
        correct_orientations = len(discovered_edges & true_edges)
        total_edges = len(true_edges)
        
        orientation_accuracy = correct_orientations / total_edges if total_edges > 0 else 0
        
        return {
            'orientation_accuracy': orientation_accuracy,
            'correct_orientations': correct_orientations,
            'total_edges': total_edges
        }
```

### 2. Causal Effect Estimation Metrics
```python
class CausalEffectEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_ate_estimation(self, estimated_ate, true_ate, method_name):
        """Evaluate ATE estimation accuracy"""
        bias = estimated_ate - true_ate
        relative_bias = bias / true_ate if true_ate != 0 else float('inf')
        
        return {
            'method': method_name,
            'estimated_ate': estimated_ate,
            'true_ate': true_ate,
            'bias': bias,
            'relative_bias': relative_bias,
            'absolute_error': abs(bias)
        }
    
    def compare_methods(self, results_dict, true_ate):
        """Compare different estimation methods"""
        comparisons = {}
        
        for method_name, estimated_ate in results_dict.items():
            comparisons[method_name] = self.evaluate_ate_estimation(
                estimated_ate, true_ate, method_name
            )
        
        return comparisons
```

## Tools and Libraries

- **DoWhy**: Causal inference library
- **CausalDiscoveryToolbox**: Causal discovery algorithms
- **pgmpy**: Bayesian networks and causal inference
- **PyMC**: Probabilistic programming
- **NetworkX**: Graph algorithms

## Best Practices

1. **Assumption Validation**: Always check causal assumptions
2. **Sensitivity Analysis**: Test robustness of conclusions
3. **Domain Knowledge**: Incorporate expert knowledge
4. **Multiple Methods**: Use complementary approaches
5. **Transparency**: Document assumptions and limitations

## Next Steps

1. **Deep Causal Models**: Neural networks for causal inference
2. **Causal Reinforcement Learning**: Causal reasoning in RL
3. **Causal Fairness**: Fairness through causal lens
4. **Causal Interpretability**: Explainable causal models
5. **Causal Transfer Learning**: Causal knowledge transfer

---

*Causal AI and Bayesian methods provide principled approaches to understanding cause-and-effect relationships and reasoning under uncertainty. These techniques are essential for building AI systems that can make reliable predictions and interventions in complex, real-world scenarios.* 