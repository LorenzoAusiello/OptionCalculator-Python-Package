# Import packages needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf

# Building OptionCalculator
class OptionCalculator:
    # Initialize the option calculator with stock price (S), strike price (K), risk-free rate (r), and time to maturity (tau)
    def __init__(self, S, K, r, tau):
        self.S = S
        self.K = K
        self.r = r
        self.tau = tau
        
    # Calculate Black-Scholes option prices given a volatility (sigma)
    def BS_price(self, sigma):   
        try:
            # Check if sigma is a float or integer, raise a TypeError if not
            if type(sigma) != int and type(sigma) != float:
                raise TypeError("Volatility must be float/integer")
        except TypeError as e:
            return e
        else:
            # Calculate call and put prices using Black-Scholes formula
            d2 = (math.log(self.S / self.K) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
            d1 = d2 + sigma * math.sqrt(self.tau)
            call_price = self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.tau) * norm.cdf(d2)
            put_price = self.K * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            return call_price, put_price
            
    # Calculate implied volatility given option price and type (call/put)  
    def BS_implied_vol(self, price, optype):
        try:
            # Check if optype is 'call' or 'put', raise a TypeError if not
            if optype.lower() != "call" and optype.lower() != "put" :
                raise TypeError("Option type should be Call or Put")
             # Check if price is a float or integer, raise a TypeError if not    
            if type(price) not in [int, float]:
                raise TypeError("Price should be an integer or a float")   
        except TypeError as TE:
            return TE
        except AttributeError as TE:
            print("TypeError(Option type should be a string)")
        else:
            def black_scholes_call(sigma):
                d2 = (math.log(self.S / self.K) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
                d1 = d2 + sigma * math.sqrt(self.tau)
                call_price = self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.tau) * norm.cdf(d2)-price
                return call_price
            def black_scholes_put(sigma):
                d2 = (math.log(self.S / self.K) + (self.r - 0.5 * sigma**2) * self.tau) / (sigma * math.sqrt(self.tau))
                d1 = d2 + sigma * math.sqrt(self.tau)
                put_price = self.K * math.exp(-self.r * self.tau) * norm.cdf(-d2) - self.S * norm.cdf(-d1)-price
                return put_price
            if optype.lower()=="call":
                implied_volatility = brentq(black_scholes_call, -10.0, 10.0)
            elif optype.lower()=="put":
                implied_volatility = brentq(black_scholes_put, -10.0, 10.0)
            return implied_volatility
            print(implied_volatility)
    
   # Implement put-call parity to calculate the price of the other option given one option's price
    def PC_parity(self, price, optype):
        try:
            # Check if optype is 'call' or 'put', raise a TypeError if not
            if optype.lower() != "call" and optype.lower() != "put" :
                raise TypeError("Option type should be Call or Put")
            # Check if price is a float or integer, raise a TypeError if not    
            if type(price) not in [int, float]:
                raise TypeError("Price should be an integer or a float")   
        except TypeError as TE:
            return TE
        except AttributeError as TE:
            print("TypeError(Option type should be a string)")
        else:
            put_price = max(price+math.exp(-self.r * self.tau)*self.K-self.S, 0)
            call_price = max(price-math.exp(-self.r * self.tau)*self.K+self.S, 0)
            if optype.lower()=="call":
                return put_price
                print(f"Put Option Price: {put_price}")
            elif optype.lower()=="put":
                return call_price
                print(f"Call Option Price: {call_price}")
    
    # Check for arbitrage opportunities based on call and put prices
    def arbitrage_opportunity(self, call_price, put_price):
        try:
            # Check if call_price and put_price are floats or integers, raise a TypeError if not
            if type(call_price) not in [int, float] or type(put_price) not in [int, float]:
                raise TypeError("Prices should be integers or floats")
        except TypeError as TE:
            return TE
        else:
            # Check for arbitrage opportunities and print the respective message
            if round(call_price-put_price, 4) == round(self.S-math.exp(-self.r * self.tau)*self.K, 4):
                print('There are no arbitrage opportunities')
            elif round(call_price-put_price, 4) < round(self.S-math.exp(-self.r * self.tau)*self.K, 4):
                print(f'Arbitrage opportunity! Long call, short put, short underlying and lend {self.K} at {self.r*100}%')
            elif round(call_price-put_price, 4) > round(self.S-math.exp(-self.r * self.tau)*self.K, 4):
                print(f'Arbitrage opportunity! Short call, long put, long underlying and borrow {self.K} at {self.r*100}%')
    
    # Implement the long straddle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss
    def long_straddle(self, sigma):
        call_price, put_price = self.BS_price(sigma)
        total_cost = call_price + put_price

        # Generate a range of stock prices for the graph
        stock_prices = np.linspace(0, self.K*2, 100)
        
        # Calculate profit/loss for each stock price
        profits_losses = -(call_price - np.maximum(self.K - stock_prices, 0)) - (put_price - np.maximum(stock_prices - self.K, 0))

        # Plot the profit/loss graph
        plt.figure(figsize=(8, 5))
        plt.plot(stock_prices, profits_losses)
        x1 = self.K + total_cost
        x2 = self.K - total_cost
        plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
        plt.axvline(x=x2, color='g', linestyle='--')
        plt.title('Straddle Strategy Profit/Loss')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(stock_prices), max(stock_prices))
        plt.show()
        print(f"Profit if price will be lower than {x2} or higher than {x1}")
        print(f"Loss if price will be between {x2} and {x1}")
        
    # Implement the short straddle strategy by calculating option prices and plotting profit/loss
    # Calculate total cost, generate stock price range, compute profit/loss, and plot the graph
    # Identify breakeven points and display information about profit/loss    
    def short_straddle(self, sigma):
        call_price, put_price = self.BS_price(sigma)
            # Since it's a short straddle, we are selling both call and put
        total_credit = call_price + put_price

            # Generate a range of stock prices for the graph
        stock_prices = np.linspace(0, self.K * 2, 100)
        
            # Calculate profit/loss for each stock price
        profits_losses = (call_price - np.maximum(self.K - stock_prices, 0)) + (put_price - np.maximum(stock_prices - self.K, 0))

            # Plot the profit/loss graph
        plt.figure(figsize=(8, 5))
        plt.plot(stock_prices, profits_losses)
        x1 = self.K + total_credit
        x2 = self.K - total_credit
        plt.axvline(x=x1, color='g', linestyle='--', label='Breakeven')
        plt.axvline(x=x2, color='g', linestyle='--')
        plt.title('Short Straddle Strategy Profit/Loss')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(stock_prices), max(stock_prices))
        plt.show()
        print(f"Profit if price will be between {x2} and {x1}")
        print(f"Loss if price will be lower than {x2} or higher than {x1} ")
    
    # Implement long call and short put strategy (synthetic long position)
    def synthetic_long_position(self, sigma):
        call_price, _ = self.BS_price(sigma)  # Get the price of the call option
        
        # Let's assume the put option is shorted at the same strike price and expiration
        put_price = -self.BS_price(sigma)[1]  # Negative value to represent shorting the put
        
        # Net cost or credit of the position
        net_cost_credit = call_price + put_price

        # Generate a range of stock prices for the graph
        stock_prices = np.linspace(0, self.K * 2, 100)
        
        # Calculate profit/loss for each stock price
        profits_losses = stock_prices - self.K - (net_cost_credit)

        # Plot the profit/loss graph
        plt.figure(figsize=(8, 5))
        plt.plot(stock_prices, profits_losses)
        plt.axvline(x=self.K+net_cost_credit, color='g', linestyle='-', label='Breakeven')
        plt.title('Synthetic Long Position Profit/Loss')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(stock_prices), max(stock_prices))
        plt.show()
        print(f"Net Cost/Credit: {net_cost_credit}")
        print(f"Profit if price will higher than {self.K+net_cost_credit}")
        print(f"Loss if price will be lower than {self.K+net_cost_credit}")
        
    # Implement short call and long put strategy (synthetic short position)
    def synthetic_short_position(self, sigma):
        call_price = -self.BS_price(sigma)[0]  # Get the price of the call option
        
        # Let's assume the put option is shorted at the same strike price and expiration
        put_price = self.BS_price(sigma)[1]  # Negative value to represent shorting the put
        
        # Net cost or credit of the position
        net_cost_credit = call_price + put_price

        # Generate a range of stock prices for the graph
        stock_prices = np.linspace(0, self.K * 2, 100)
        
        # Calculate profit/loss for each stock price
        profits_losses = self.K - stock_prices - (net_cost_credit)

        # Plot the profit/loss graph
        plt.figure(figsize=(8, 5))
        plt.plot(stock_prices, profits_losses)
        plt.axvline(x=self.K - net_cost_credit, color='g', linestyle='-', label='Breakeven')
        plt.title('Synthetic Short Position Profit/Loss')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.legend()
        plt.grid(True)
        plt.xlim(min(stock_prices), max(stock_prices))
        plt.show()
        print(f"Net Cost/Credit: {net_cost_credit}")
        print(f"Loss if price will higher than {self.K - net_cost_credit}")
        print(f"Profit if price will be lower than {self.K - net_cost_credit}")

    # This is a function that creates a stock price evolution binomial tree using Cox-Ross-Rubinestein scheme
    def CRR(self, sigma, N):
        try:
            if type(sigma) != float and type(sigma) != int:
                raise TypeError("Volatility must be float/integer")
            if type(N) != int:
                raise TypeError("Number of steps must be an integer")
        except TypeError as e:
            return e
        else:
            # Initialize matrix with columns = N + 1, rows = (N * 2) + 1
            CRR = np.zeros(((N * 2 + 1), (N + 1)))
            # Center starting node = S0
            center = math.floor(len(CRR) / 2)
            CRR[center, 0] = self.S
            # CRR parameters
            dt = self.tau / N
            u = math.exp(sigma * math.sqrt(dt))
            d = math.exp(-sigma * math.sqrt(dt))
            q = (math.exp(self.r * dt) - d) / (u - d)
            # Looping through one column at a time
            for i in range(1, np.shape(CRR)[1] + 1):
                # The following lines indexes what row range to look at in every column skipping cells that should = 0
                # Note this could be made more efficient as the run time complexity is approx O(n^2) given the nested for-loops
                # Luckily, N will not have to be too large given the tree method price and BS converge around N = 200
                x = list(range(center - (i - 1), center + i, 2))
                y = i - 1
                CRR[x,y] = self.S
                start = center - (i - 1)
                end = center + (i - 1)
                # Amount of "up-moves" for each node
                x = list(range((i-1), -1, -1))
                temp = []
                for z in x:
                    temp.extend([z, 0])
                x = temp
                count = -1
                # Asjusting price for each indexed node with the corresponding up-move value
                for j in range(start, end + 1):
                    count = count + 1
                    CRR[j, (i - 1)] = CRR[j, (i - 1)] * u**(x[count]) * d**((i - 1) - x[count])
            # Returning tree in matrix form
            return(CRR)

    # This function calculates "today's" European call option price given some parameters
    # Backwards dynamic (recursive) programming (starting at CRR terminal nodes [ST - K]+ --> first node V0)
    def Euro_Call_CRR(self, tree, sigma):
        try:
            if type(sigma) != int and type(sigma) != float:
                raise TypeError("Volatility must be float/integer")
        except TypeError as e:
            return e
        else:
            # CRR parameters
            N = np.shape(tree)[1] - 1
            dt = self.tau / N
            u = math.exp(sigma * math.sqrt(dt))
            d = math.exp(-sigma * math.sqrt(dt))
            q = (math.exp(self.r * dt) - d) / (u - d)
            V = np.zeros(((N * 2 + 1), (N + 1)))
            center = math.floor(len(tree) / 2)
            # Payoff = [ST - K]+
            for i in range(0, np.shape(tree)[0], 1):
                payoff = tree[i, (np.shape(tree)[1] - 1)] - self.K
                V[i, (np.shape(tree)[1] - 1)] = payoff if payoff > 0 else 0
            # Same procedure as CRR
            for j in range(np.shape(tree)[1] - 2, -1, -1):  
                start = center - j
                end = center + j
                x = [1] * (j + 1)
                temp = []
                for z in x:
                    temp.extend([z, 0])
                x = temp
                count = -1
                # V = PV(E[V]) = e^-rt * (V("up") * q + V("down") * (1 - q))
                for k in range(start, end + 1):
                    count = count + 1
                    V[k, j] = x[count] * math.exp(-self.r * dt) * ((q * V[(k - 1), (j + 1)]) + (1 - q) * V[(k + 1), (j + 1)])
            # Returns both fair price of call option (starting node) and the tree evolution
            return(V, V[center, 0])

    # This function calculates "today's" European put option price given some parameters
    # Same procedure as Euro_Call_CRR
    def Euro_Put_CRR(self, tree, sigma):
        try:
            if type(sigma) != int and type(sigma) != float:
                raise TypeError("Volatility must be float/integer")
        except TypeError as e:
            return e
        else:
            N = np.shape(tree)[1] - 1
            dt = self.tau / N
            u = math.exp(sigma * math.sqrt(dt))
            d = math.exp(-sigma * math.sqrt(dt))
            q = (math.exp(self.r * dt) - d) / (u - d)
            V = np.zeros(((N * 2 + 1), (N + 1)))
            center = math.floor(len(tree) / 2)
            for i in range(0, np.shape(tree)[0], 1):
                # Payoff = [K - ST]+
                # Only performs payoff function if CRR terminal node (ST) is not 0
                if tree[i, (np.shape(tree)[1] - 1)] != 0:
                    payoff = self.K - tree[i, (np.shape(tree)[1] - 1)]
                    V[i, (np.shape(tree)[1] - 1)] = payoff if payoff > 0 else 0
            for j in range(np.shape(tree)[1] - 2, -1, -1):  
                start = center - j
                end = center + j
                x = [1] * (j + 1)
                temp = []
                for z in x:
                    temp.extend([z, 0])
                    x = temp
                count = -1
                for k in range(start, end + 1):
                    count = count + 1
                    V[k, j] = x[count] * math.exp(-self.r * dt) * ((q * V[(k - 1), (j + 1)]) + (1 - q) * V[(k + 1), (j + 1)])
            return(V, V[center, 0])

    # Style feature for dataframe output (removes all zeros so the CRR tree evolution is more clear)
    def remove_zeros(self, cell):
        cell = '' if cell == 0 else cell
        return cell

    # Function demonstrating the convergence of call option price derived from binomial tree method with Black Scholes price
    # Takes input equal to the number of maximum tree steps you want
    # Recommendation: use at least 200 steps to clearly see the convergence dynamic
    def plot_convergence_call(self, N, sigma):
        try:
            if type(sigma) != float and type(sigma) != int:
                raise TypeError("Volatility must be float/integer")
            if type(N) != int:
                raise TypeError("Number of steps must be an integer")
        except TypeError as e:
            return e
        else:
            steps = list(range(1, N, 5))
            call_price = [0] * len(steps)
            error = [0] * len(steps)
            BS_call, BS_put = self.BS_price(sigma)
            for i in range(len(steps)):
                binomial_tree = self.CRR(sigma, steps[i])
                call_price[i] = float(self.Euro_Call_CRR(binomial_tree, sigma)[1])
                error[i] = abs(call_price[i] - BS_call) / BS_call
            fig, graphs = plt.subplots(1,2,sharey = False, sharex = False, figsize = (15,5))
            graphs[0].plot(steps, call_price, color = "green")
            graphs[0].axhline(y=BS_call, color='red', linestyle='--', label='Black Scholes Price')
            graphs[0].set_title("Binomial Tree European Call Option Price Convergence with Black Scholes")
            graphs[0].set_xlabel("Number of Tree Steps")
            graphs[0].set_ylabel("Option Price")
            graphs[0].legend()
            graphs[1].plot(steps, error)
            graphs[1].axhline(y=0, color='red', linestyle='--', label='0 Error')
            graphs[1].set_title("Absolute Relative Pricing Error")
            graphs[1].set_xlabel("Number of Tree Steps")
            graphs[1].set_ylabel("Error (%)")
            graphs[1].legend()
            plt.tight_layout()

    # Same as plot_convergence_call but for a European put option
    def plot_convergence_put(self, N, sigma):
        try:
            if type(sigma) != float and type(sigma) != int:
                raise TypeError("Volatility must be float/integer")
            if type(N) != int:
                raise TypeError("Number of steps must be an integer")
        except TypeError as e:
            return e
        else:
            steps = list(range(1, N, 5))
            put_price = [0] * len(steps)
            error = [0] * len(steps)
            BS_call, BS_put = self.BS_price(sigma)
            for i in range(len(steps)):
                binomial_tree = self.CRR(sigma, steps[i])
                put_price[i] = float(self.Euro_Put_CRR(binomial_tree, sigma)[1])
                error[i] = abs(put_price[i] - BS_put) / BS_put
            fig, graphs = plt.subplots(1,2,sharey = False, sharex = False, figsize = (15,5))
            graphs[0].plot(steps, put_price, color = "green")
            graphs[0].axhline(y=BS_put, color='red', linestyle='--', label='Black Scholes Price')
            graphs[0].set_title("Binomial Tree European Put Option Price Convergence with Black Scholes")
            graphs[0].set_xlabel("Number of Tree Steps")
            graphs[0].set_ylabel("Option Price")
            graphs[0].legend()
            graphs[1].plot(steps, error)
            graphs[1].axhline(y=0, color='red', linestyle='--', label='0 Error')
            graphs[1].set_title("Absolute Relative Pricing Error")
            graphs[1].set_xlabel("Number of Tree Steps")
            graphs[1].set_ylabel("Error (%)")
            graphs[1].legend()
            plt.tight_layout()

    # Defining a function to calculate call and put price using Monte-Carlo Simulation
    def mcs_price(self, sigma, no_of_intervals, no_of_paths, display_plot:bool):
        
        # Using try-except block to raise an error if the input parameters are not integers
        try:
            if type(no_of_intervals) != int or type(no_of_paths) != int:
                raise TypeError
        except TypeError:
            print("Number of intervals and number of paths should be integer")
            
        else:    
                
            # Initiating a dictionary to store pay-offs for every simulated path
            pay_offs = {'call': [] , 'put': []}
            
            if display_plot == True:
                # Initialising a graph figure
                plt.figure(figsize=(8, 5))
                
            for path in range(no_of_paths):
                # Initialising a list to store the prices at each interval
                interval_prices = [self.S]
            
                # Using a nested for loop to iterate over every interval in each path
                for i in range(no_of_intervals):
                    interval_prices.append(interval_prices[-1]*math.exp((self.r - (sigma**2)/2)*(self.tau/no_of_intervals) + sigma*np.random.normal()*((self.tau/no_of_intervals)**0.5)))
                
                if display_plot == True:    
                    # Plotting the line graph for each path
                    plt.plot(interval_prices)
                     
                # Using the stock price at the end of a path, calculating the pay-off values and storing them in a dictionary 
                maturity_stock_price = interval_prices[-1]
                pay_offs['call'].append(max(maturity_stock_price - self.K, 0))
                pay_offs['put'].append(max(self.K - maturity_stock_price, 0))
            
            # Calculating the option price by taking the average of all simulated pay-offs and discounting it
            call_price = np.mean(pay_offs['call'])*math.exp(-self.r * self.tau)
            put_price = np.mean(pay_offs['put'])*math.exp(-self.r * self.tau)
            
            if display_plot == True:
                # Specifying the details of graph figure
                plt.title('Monte-Carlo simulations')
                plt.xlabel('Intervals')
                plt.ylabel('Stock Price')
                plt.show()
            
            return call_price, put_price
        
    # Defining a function to do a sensitivity analysis for a given array of volatility values
    def sensitivity_analysis(self, option_type, calculation_method, volatility_values, no_of_intervals=None, no_of_paths=None):
        
        # Initiating a list to store the different prices of an option
        option_prices = []

        # Iterating over each of the volatility value
        for value in volatility_values:
            
            # Specifying the if condition to use the appropriate function for calculating the price and raising errors if the inputs are not proper
            if calculation_method == "mcs":
                if no_of_intervals is None or no_of_paths is None:
                    raise ValueError("For monte-carlo based calculations, please specify no_of_intervals and no_of_paths.")
                
                # Setting the display_plot parameter to "False" to avoid multiple graphs of MCS based prices
                call_price, put_price = self.mcs_price(value, no_of_intervals, no_of_paths, False)
                
            elif calculation_method == "bsm":
                call_price, put_price = self.BS_price(value)
                
            else:
                raise ValueError("Please specify the pricing method as either 'mcs' or 'bsm'.")

            # Updating the option price list as per the type of option specified
            if option_type == 'call':
                option_prices.append(call_price)
                
            if option_type == 'put': 
                option_prices.append(put_price)

        # Plotting the graph for sensitivity analysis
        plt.figure(figsize=(8, 5))
        plt.plot(volatility_values, option_prices, marker='o')
        plt.title(f'Volatility based Sensitivity Analysis for {option_type.capitalize()} Option Price')
        plt.xlabel("Volatility")
        plt.ylabel(f'{option_type.capitalize()} Option Price')
        plt.show()



# Extra function not part of class for stylistic purposes for dataframe output:
def in_the_money(cell):
    cell_color = 'background-color: green' if cell > 0 else ''
    num_color = 'color: white' if cell > 0 else ''
    return f'{cell_color}; {num_color}'
    
    