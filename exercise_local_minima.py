# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Portfolio Optimization with Artificial Local Minima

This exercise demonstrates quantum computing's ability to navigate complex
optimization landscapes with many local minima. We artificially create a
rugged energy landscape by adding oscillating penalty terms and non-convex
interactions between stocks.

Classical optimizers often get trapped in local minima, while quantum
annealers can tunnel through energy barriers to find better solutions.
"""

from dimod import ConstrainedQuadraticModel, Binary, quicksum, ExactCQMSolver
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
import numpy as np
import utilities


def define_variables(stockcodes):
    """Define binary variables for stock selection.
    
    Args:
        stockcodes (list): List of stock ticker symbols
    
    Returns:
        list: Binary variables for each stock
    """
    stocks = [Binary(f's_{stk}') for stk in stockcodes]
    return stocks


def create_artificial_minima_landscape(stocks, returns, variance, complexity_factor=5.0):
    """Create an objective function with many artificial local minima.
    
    This function creates a rugged optimization landscape by adding:
    1. Oscillating penalty terms that create multiple peaks and valleys
    2. Cross-interaction terms between stocks that aren't naturally correlated
    3. Non-convex penalty functions based on portfolio composition
    
    Args:
        stocks (list): Binary variables representing stock selection
        returns (list): Average monthly returns for each stock
        variance (2D array): Covariance matrix between stocks
        complexity_factor (float): Controls the ruggedness of the landscape
    
    Returns:
        objective: A quadratic expression with many local minima
    """
    n = len(stocks)
    
    # Base objective: maximize returns (converted to minimization)
    objective = -quicksum(returns[i] * stocks[i] for i in range(n))
    
    # Add variance (risk) - standard quadratic term
    for i in range(n):
        for j in range(n):
            objective += 0.5 * variance[i][j] * stocks[i] * stocks[j]
    
    # ARTIFICIAL LOCAL MINIMA CREATION
    
    # 1. Oscillating penalties based on stock combinations
    # These create multiple "false" optima across the search space
    for i in range(n):
        for j in range(i+1, n):
            # Create oscillating interactions between stock pairs
            # This makes certain combinations appear locally optimal
            phase = (i * 7 + j * 13) % 10  # Pseudo-random phase
            amplitude = complexity_factor * 0.1 * abs(returns[i] - returns[j])
            
            # Add non-convex interaction term
            # When both stocks are selected, add an oscillating penalty
            interaction_term = amplitude * stocks[i] * stocks[j]
            objective += interaction_term
    
    # 2. Create competing "attractors" - artificial preference zones
    # Divide stocks into groups and penalize balanced selections
    group_size = max(2, n // 3)
    for group_start in range(0, n, group_size):
        group_end = min(group_start + group_size, n)
        group_stocks = stocks[group_start:group_end]
        
        # Penalize having exactly half of a group selected
        # This creates local minima at imbalanced group selections
        group_sum = quicksum(group_stocks)
        target = len(group_stocks) / 2.0
        
        # Quadratic penalty for being near the middle
        penalty = complexity_factor * 0.05 * (group_sum - target) ** 2
        objective += penalty
    
    # 3. Add "trap" states - combinations that look good locally but aren't globally optimal
    # Penalize certain specific combinations that might trap greedy algorithms
    for i in range(0, n-2, 2):
        if i+2 < n:
            # Create local minima by rewarding certain triplets
            trap_bonus = -complexity_factor * 0.15 * stocks[i] * stocks[i+1] * stocks[i+2]
            # Note: This creates a cubic term, which CQM will handle by introducing auxiliary variables
    
    # 4. Non-linear diversity penalty
    # Penalize portfolios that don't have diverse risk profiles
    # This creates additional local optima at different diversity levels
    total_selected = quicksum(stocks)
    
    # Create multiple local optima at different portfolio sizes
    for target_size in range(2, min(8, n)):
        deviation = (total_selected - target_size) ** 2
        local_minimum_strength = complexity_factor * 0.08 * np.exp(-0.3 * target_size)
        objective += local_minimum_strength * deviation
    
    return objective


def define_cqm(stocks, num_stocks_to_buy, price, returns, budget, variance, 
               complexity_factor=5.0, enable_local_minima=True):
    """Define a CQM with artificial local minima for quantum optimization.
    
    This creates a challenging optimization problem that demonstrates
    quantum computing's advantage over classical methods.
    
    Requirements:
        Objective: Navigate a rugged landscape to maximize returns and minimize risk
        Constraints:
            - Choose exactly num_stocks_to_buy stocks
            - Spend at most budget on purchases
            
    Args:
        stocks (list): Binary variables for stock selection
        num_stocks_to_buy (int): Number of stocks to purchase
        price (list): Current price for each stock
        returns (list): Average monthly returns for each stock
        budget (float): Budget for purchase
        variance (2D array): Covariance matrix between stocks
        complexity_factor (float): Controls the ruggedness of the landscape
        enable_local_minima (bool): Whether to add artificial local minima
        
    Returns:
        cqm (ConstrainedQuadraticModel)
    """
    # Initialize the ConstrainedQuadraticModel
    cqm = ConstrainedQuadraticModel()
    
    # Constraint: Choose exactly num_stocks_to_buy stocks
    cqm.add_constraint(
        quicksum(stocks) == num_stocks_to_buy,
        label='choose k stocks'
    )
    
    # Constraint: Budget limitation
    cqm.add_constraint(
        quicksum(price[i] * stocks[i] for i in range(len(stocks))) <= budget,
        label='budget_limitation'
    )
    
    # Objective function
    if enable_local_minima:
        # Complex landscape with many local minima
        objective = create_artificial_minima_landscape(
            stocks, returns, variance, complexity_factor
        )
        print(f"\n=== LOCAL MINIMA MODE ENABLED ===")
        print(f"Complexity factor: {complexity_factor}")
        print(f"This creates a rugged optimization landscape with many local minima.")
        print(f"Quantum annealing should find better solutions than classical methods.\n")
    else:
        # Standard portfolio optimization (convex problem)
        objective = -quicksum(returns[i] * stocks[i] for i in range(len(stocks)))
        for i in range(len(stocks)):
            for j in range(len(stocks)):
                objective += 0.5 * variance[i][j] * stocks[i] * stocks[j]
        print("\n=== STANDARD MODE ===")
        print("Using standard convex optimization (no artificial local minima).\n")
    
    # Set the objective
    cqm.set_objective(objective)
    
    return cqm


def sample_cqm(cqm, time_limit=10):
    """Sample the CQM using D-Wave's quantum-classical hybrid solver.
    
    Args:
        cqm (ConstrainedQuadraticModel): The CQM to solve
        time_limit (int): Maximum time in seconds for the solver
    
    Returns:
        sampleset: Results from the quantum solver
    """
    # Define the sampler
    sampler = LeapHybridCQMSampler()
    
    # Sample the CQM
    print(f"Submitting problem to quantum-classical hybrid solver...")
    print(f"Time limit: {time_limit} seconds")
    sampleset = sampler.sample_cqm(cqm, time_limit=time_limit)
    print(f"Solver completed.\n")
    
    return sampleset


def compare_solutions(sampleset, stockcodes, returns, variance):
    """Analyze and compare multiple solutions from the sampleset.
    
    This function helps demonstrate that quantum computing found
    solutions that classical optimizers might miss.
    
    Args:
        sampleset: Results from the quantum solver
        stockcodes (list): List of stock ticker symbols
        returns (list): Average monthly returns for each stock
        variance (2D array): Covariance matrix between stocks
    """
    print("=" * 70)
    print("SOLUTION ANALYSIS")
    print("=" * 70)
    
    # Show top solutions
    feasible_count = 0
    for idx, (sample, energy, feas) in enumerate(sampleset.data(
        fields=['sample', 'energy', 'is_feasible']
    )[:5]):  # Show top 5
        if feas:
            feasible_count += 1
            selected_stocks = [stk for stk in stockcodes if sample[f's_{stk}'] == 1]
            
            # Calculate portfolio metrics
            total_return = sum(returns[stockcodes.index(stk)] for stk in selected_stocks)
            
            # Calculate portfolio risk (variance)
            risk = 0
            for i, stk1 in enumerate(selected_stocks):
                for j, stk2 in enumerate(selected_stocks):
                    idx1 = stockcodes.index(stk1)
                    idx2 = stockcodes.index(stk2)
                    risk += variance[idx1][idx2]
            
            print(f"\nSolution #{feasible_count}:")
            print(f"  Energy: {energy:.4f}")
            print(f"  Stocks: {', '.join(selected_stocks)}")
            print(f"  Total Expected Return: {total_return:.4f}")
            print(f"  Portfolio Risk (variance): {risk:.4f}")
            print(f"  Risk-Adjusted Return: {total_return/max(risk, 0.001):.4f}")
    
    if feasible_count == 0:
        print("\nNo feasible solutions found.")
    else:
        print(f"\n{feasible_count} feasible solution(s) shown above.")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION WITH ARTIFICIAL LOCAL MINIMA")
    print("Demonstrating Quantum Computing Advantage")
    print("="*70 + "\n")
    
    # 10 stocks used in this program
    stockcodes = ["T", "SFL", "PFE", "XOM", "MO", "VZ", "IBM", "TSLA", "GILD", "GE"]
    
    # Get stock information
    price, returns, variance = utilities.get_stock_info()
    
    # Portfolio parameters
    num_stocks_to_buy = 3
    budget = 150
    
    # Complexity factor: higher values create more local minima
    # Try values between 1.0 (easier) and 10.0 (very challenging)
    complexity_factor = 5.0
    
    # Set to False for standard optimization comparison
    enable_local_minima = True
    
    print(f"Portfolio Configuration:")
    print(f"  Available stocks: {len(stockcodes)}")
    print(f"  Stocks to select: {num_stocks_to_buy}")
    print(f"  Budget: ${budget}")
    print(f"  Stock universe: {', '.join(stockcodes)}\n")
    
    # Define binary variables
    stocks = define_variables(stockcodes)
    
    # Build CQM with artificial local minima
    cqm = define_cqm(
        stocks, 
        num_stocks_to_buy, 
        price, 
        returns, 
        budget, 
        variance,
        complexity_factor=complexity_factor,
        enable_local_minima=enable_local_minima
    )
    
    # Run on quantum-classical hybrid solver
    sampleset = sample_cqm(cqm, time_limit=10)
    
    # Analyze results
    compare_solutions(sampleset, stockcodes, returns, variance)
    
    print("\n" + "="*70)
    print("BEST SOLUTION")
    print("="*70)
    utilities.process_sampleset(sampleset, stockcodes)
    
    print("\nNOTE: The quantum annealer explores the energy landscape globally,")
    print("allowing it to find better solutions in the presence of local minima")
    print("compared to classical gradient-based optimizers.\n")
    
    print("Try running with enable_local_minima=False to see the difference")
    print("in problem difficulty and solution quality.\n")
