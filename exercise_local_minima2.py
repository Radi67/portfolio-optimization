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

import utilities
import dimod
from dimod import Binary
from dimod import ConstrainedQuadraticModel


def define_variables(stockcodes):
    """Define binary variables for each stock."""
    stocks = [Binary(f"s_{stk}") for stk in stockcodes]
    return stocks


def define_cqm(stocks, num_stocks_to_buy, price, returns, budget):
    """Define a CQM with MANY local minima using frustration terms."""

    # 1. Create CQM
    cqm = ConstrainedQuadraticModel()

    # 2. Choose exactly k stocks
    cqm.add_constraint(
        sum(stocks) == num_stocks_to_buy,
        label="choose k stocks"
    )

    # 3. Budget constraint
    cqm.add_constraint(
        sum(price[i] * stocks[i] for i in range(len(stocks))) <= budget,
        label="budget_limitation"
    )

    # 4. Base objective: maximize returns → minimize negative returns
    base_obj = -sum(returns[i] * stocks[i] for i in range(len(stocks)))

    # -----------------------------
    # Inject frustration to create MULTIPLE LOCAL MINIMA
    # -----------------------------
    frustration = 0

    # Positive = penalty (repulsion)
    frustration += 2 * stocks[0] * stocks[1]
    frustration += 3 * stocks[2] * stocks[3]
    frustration += 2 * stocks[4] * stocks[5]

    # Negative = reward (attraction)
    frustration += -4 * stocks[1] * stocks[4]
    frustration += -3 * stocks[3] * stocks[7]

    # Spin-glass frustration cycle (makes rugged landscape)
    n = len(stocks)
    for i in range(n):
        frustration += 2 * stocks[i] * stocks[(i+1) % n]     # penalize neighbors
        frustration += -3 * stocks[i] * stocks[(i+2) % n]    # encourage next-neighbors

    # Final objective = base returns + frustration landscape
    cqm.set_objective(base_obj + frustration)

    return cqm


def sample_cqm(cqm):
    """Use ExactSolver to simulate D-Wave local minima finding."""

    sampler = dimod.ExactSolver()   # perfect deterministic simulator
    sampleset = sampler.sample_cqm(cqm)
    return sampleset


if __name__ == '__main__':

    # 10 stocks under consideration
    stockcodes = ["T", "SFL", "PFE", "XOM", "MO", "VZ", "IBM", "TSLA", "GILD", "GE"]

    # Get price and returns data
    price, returns, variance = utilities.get_stock_info()

    # Number of stocks to pick
    num_stocks_to_buy = 2

    # Max budget
    budget = 80

    # Variables
    stocks = define_variables(stockcodes)

    # Build the CQM
    cqm = define_cqm(stocks, num_stocks_to_buy, price, returns, budget)

    # Solve using D-Wave simulator
    sampleset = sample_cqm(cqm)

    print("\nPart 2 solution:\n")
    utilities.process_sampleset(sampleset, stockcodes)
