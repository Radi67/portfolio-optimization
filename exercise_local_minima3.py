# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -------------------------------
# IMPORTS
# -------------------------------

import utilities
import dimod
from dimod import Binary
from dimod import ConstrainedQuadraticModel


# -------------------------------
# DEFINE VARIABLES
# -------------------------------

def define_variables(stockcodes):
    """Create a binary variable for each stock."""
    stocks = [Binary(f"s_{stk}") for stk in stockcodes]
    return stocks


# -------------------------------
# DEFINE CQM WITH MANY MINIMA
# -------------------------------

def define_cqm(stocks, num_stocks_to_buy, price, returns, budget, variance):
    """
    Objective:
        - Maximize returns
        - Minimize variance
    Constraints:
        - Pick exactly num_stocks_to_buy stocks
        - Total cost <= budget
    """

    # 1. Initialize CQM
    cqm = ConstrainedQuadraticModel()

    # 2. Constraint: choose exactly k stocks
    cqm.add_constraint(
        sum(stocks) == num_stocks_to_buy,
        label="choose k stocks"
    )

    # 3. Constraint: stay within budget
    cqm.add_constraint(
        sum(price[i] * stocks[i] for i in range(len(stocks))) <= budget,
        label="budget_limitation"
    )

    # -------------------------------
    # Base objective: maximize returns
    # Convert to minimization by negating
    # -------------------------------

    base_obj = -sum(returns[i] * stocks[i] for i in range(len(stocks)))

    # -------------------------------
    # Add quadratic variance penalty
    # (Risk term)
    # -------------------------------

    variance_term = 0
    n = len(stocks)
    for i in range(n):
        for j in range(n):
            variance_term += variance[i][j] * stocks[i] * stocks[j]

    # -------------------------------
    # Inject frustration to create MANY local minima
    # -------------------------------

    frustration = 0

    # A few direct frustrated edges
    frustration += 3 * stocks[0] * stocks[1]     # penalty
    frustration += -4 * stocks[1] * stocks[3]    # reward
    frustration += 2 * stocks[3] * stocks[5]     # penalty
    frustration += -3 * stocks[2] * stocks[6]    # reward

    # Spin-glass cycle (the real ruggedness)
    for i in range(n):
        frustration += 2 * stocks[i] * stocks[(i+1) % n]         # local penalty
        frustration += -3 * stocks[i] * stocks[(i+2) % n]        # longer-range attraction

    # FINAL OBJECTIVE
    cqm.set_objective(base_obj + variance_term + frustration)

    return cqm


# -------------------------------
# SAMPLING WITH EXACTSOLVER
# -------------------------------

def sample_cqm(cqm):
    """Simulate the CQM using ExactSolver (no cloud needed)."""
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample_cqm(cqm)
    return sampleset


# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":

    # 10 stocks in this example
    stockcodes = ["T", "SFL", "PFE", "XOM", "MO", "VZ", "IBM", "TSLA", "GILD", "GE"]

    price, returns, variance = utilities.get_stock_info()

    num_stocks_to_buy = 2
    budget = 40

    stocks = define_variables(stockcodes)
    cqm = define_cqm(stocks, num_stocks_to_buy, price, returns, budget, variance)

    sampleset = sample_cqm(cqm)

    print("\nPart 3 solution:\n")
    utilities.process_sampleset(sampleset, stockcodes)
