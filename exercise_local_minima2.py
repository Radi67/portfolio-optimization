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

# TODO: Import any required packages here


import utilities
import dimod
from dimod import Binary
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel

def define_variables(stockcodes):
    """Define the variables to be used for the CQM.
    Args:
        stockcodes (list): List of stocks under consideration
    
    Returns:
        stocks (list): 
            List of variables named 's_{stk}' for each stock stk in stockcodes, where stk is replaced by the stock code.
    """

    # TODO: Define your list of variables and call it stocks
     ## Hint: Remember to import the required package at the top of the file for Binary variables
        stocks = [Binary(f"s_{stk}") for stk in stockcodes]
    return stocks

def define_cqm(stocks, num_stocks_to_buy, price, returns, budget):
    """Define a CQM for the exercise. 
    Requirements:
        Objective: Maximize returns
        Constraints:
            - Choose exactly num_stocks_to_buy stocks
            - Spend at most budget on purchase
            
    Args:
        stocks (list):
            List of variables named 's_{stk}' for each stock in stockcodes
        num_stocks_to_buy (int): Number of stocks to purchase
        price (list):
            List of current price for each stock in stocks
                where price[i] is the price for stocks[i]
        returns (list):
            List of average monthly returns for each stock in stocks
                where returns[i] is the average returns for stocks[i]
        budget (float):
            Budget for purchase
        
    Returns:
        cqm (ConstrainedQuadraticModel)
    """

    # TODO: Initialize the ConstrainedQuadraticModel called cqm
    ## Hint: Remember to import the required package at the top of the file for ConstrainedQuadraticModels
        # 1. Initialize CQM
    cqm = ConstrainedQuadraticModel()

    # 2. Constraint: choose exactly num_stocks_to_buy stocks
    cqm.add_constraint(
        sum(stocks) == num_stocks_to_buy,
        label='choose k stocks'
    )

    # 3. Constraint: budget
    cqm.add_constraint(
        sum(price[i] * stocks[i] for i in range(len(stocks))) <= budget,
        label='budget_limitation'
    )

    # --- Inject frustration to create many local minima ---
    # Penalize selecting some pairs together (positive weights)
    frustration = 0
    frustration += 2*stocks[0]*stocks[1]
    frustration += 3*stocks[2]*stocks[3]
    frustration += 2*stocks[4]*stocks[5]

    # Encourage incompatible behavior in others (negative weights)
    frustration += -4*stocks[1]*stocks[4]
    frustration += -3*stocks[3]*stocks[7]

    # Add a frustration loop (like a spin-glass cycle)
    n = len(stocks)
    for i in range(n):
        frustration += 2 * stocks[i] * stocks[(i+1) % n]
        frustration += -3 * stocks[i] * stocks[(i+2) % n]

    # 4. Objective: maximize returns   (we MINIMIZE NEGATIVE returns)
    base_obj = -sum(returns[i] * stocks[i] for i in range(len(stocks)))

    # TODO: Add a constraint to choose exactly num_stocks_to_buy stocks
    ## Important: Use the label 'choose k stocks', this label is case sensitive
    

    # TODO: Add an objective function maximize returns
    ## Hint: Use the information in returns, and remember to convert to minimization
    

    # TODO: Add a constraint that the cost of the purchased stocks is less than or equal to the budget
    ## Important: Use the label 'budget_limitation', this label is case sensitive and uses an underscore

    cqm.set_objective(base_obj + frustration)

    return cqm

def sample_cqm(cqm):

    # TODO: Define your sampler as LeapHybridCQMSampler
    ## Hint: Remember to import the required package at the top of the file
    

    # TODO: Sample the ConstrainedQuadraticModel cqm and store the result in sampleset
    

    return sampleset


if __name__ == '__main__':

    # 10 stocks used in this program
    stockcodes=["T", "SFL", "PFE", "XOM", "MO", "VZ", "IBM", "TSLA", "GILD", "GE"]

    # Compute relevant statistics like price, average returns, and covariance
    price, returns, variance = utilities.get_stock_info()

    # Number of stocks to select
    num_stocks_to_buy = 2

    # Set the budget for the purchase
    budget = 80

    # Add binary variables for stocks
    stocks = define_variables(stockcodes)

    # Build CQM
    cqm = define_cqm(stocks, num_stocks_to_buy, price, returns, budget)

    # Run CQM on hybrid solver
    sampleset = sample_cqm(cqm)
    
    # Process and print solution
    print("\nPart 2 solution:\n")
    utilities.process_sampleset(sampleset, stockcodes)
