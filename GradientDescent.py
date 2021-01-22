#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:13:22 2021

@author: romanschulze
"""

# ----------------------------------------------------------------------------
# Gradient Descent - Deriving optimal set of parameters for Linear Regression
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Overview of Script:
# ----------------------------------------------------------------------------
# 1. Generate sample data
# 2. Gradient Descent Algorithm 
# 3. Comparison with sklearn linear Regression
# ----------------------------------------------------------------------------



# import libraries
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.offline import plot



# ----------------------------------------------------------------------------
# 1. Generate sample data
# ----------------------------------------------------------------------------


# seed for reproducibility
np.random.seed(123)

# create X
X = np.random.normal(0, 0.9, 1000).reshape(1000, -1).round(2)

# create y
y = X + np.random.normal(0, 0.4, 1000).reshape(1000, -1).round(2)


# show distribution of X and y
plt.scatter(X, y, alpha=0.7, edgecolors="black")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Distribution of X and y")
plt.show()



# ----------------------------------------------------------------------------
# 2. Gradient Descent Algorithm 
# ----------------------------------------------------------------------------



# 1. Sum of squared residuals
def getResiduals(beta, X, y):
    """
    Derive array of residuals:
   
    Parameters
    ----------
    X : array of independent variables, shape: (n_samples, n_features).    
    y : array containing dependent variable, shape: (n_samples).
    beta : array of coefficients, shape: (n_features).
    
    Returns
    -------
    residuals : array containing difference of observed and predicted values.
    """
    # number of samples
    N = len(y)
    # predict y
    y_hat = np.dot(X, beta)
    # Derive difference between predictions and acutal realizations of y
    residuals = (1/2 * N) * np.sum(np.square(y_hat - y))
    # return residuals
    return residuals


# 2. plot optimization process
def plotOptimizationProcess(X, y, y_hat):
    """
    Visualize gradient descent:
   
    Parameters
    ----------
    X : array of independent variables, shape: (n_samples, n_features).    
    y : array containing dependent variable, shape: (n_samples).
    beta : array of coefficients, shape: (n_features).
    
    Returns
    -------
    plot : a plot with the rotating regression line.
    """
    # Plot X and y values distribution
    plt.scatter(X[:, 1], y, alpha=0.7, edgecolors="black")
    # plot regression line
    plt.plot(X[:, 1], y_hat, color="magenta")
    # Add xlabel
    plt.xlabel("X")
    # Add ylabel
    plt.ylabel("y")
    # Add title
    plt.title("Optimization of beta")
    # show plot
    plt.show()


# 3. Gradient Descent
def gradientDescent(X, y, learning_rate=0.05, max_iter=200, visualize=False):
    """
    Derive optimal paramaters for linear regression model:
   
    Parameters
    ----------
    X : array of independent variables, shape: (n_samples, n_features).    
    y : array containing dependent variable, shape: (n_samples).
    learning_rate : hyperparameter that controls the change of model by current error.
    max_iter: maximum number of iterations to find optimal set of parameters.
    visualize: plot rotating line (default=False).
    
    Returns
    -------
    beta : array of coefficients, shape: (n_features).
    residuals : array containing difference of observed and predicted values.
    """
    # number of instances 
    N = len(y)
    # Add constant to X matrix 
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))  
    # initialize iteration
    iter = 1
    # initialize beta with zero for all parameters
    beta = np.zeros((X.shape[1], 1))
    # empty list to store ssr 
    ssr = []
  
    while (iter < max_iter):
        # prediction of y 
        y_hat = np.dot(X, beta)
        # SSR
        ssr.append(getResiduals(beta, X, y))
        # derive gradient 
        gradient = 1/N * (X.T).dot((y_hat - y))
        # update beta
        beta -= learning_rate * gradient
        # update iteration counter
        iter += 1
        # plot rotating line
        if visualize is True:
            plotOptimizationProcess(X, y, y_hat)
    # return beta
    return beta.round(4).T, ssr


# derive optimal parameters
beta, ssr = gradientDescent(X, y, visualize = False)


print(70 * ("-"))
print("Linear Regression Model using my own Gradient Descent")
print(70 * ("-"))
print(f"Intercept: {beta[0][0]}     coefficients:{beta[0][1]}")


# 4. plot loss 
def plotLoss():
    """
    Plot reduction of SSR vs number of iterations using Matplotlib.
    
    Returns
    -------
    plot : A plot visualizing the relationship of SSR and iterations.
    """
    # ssr
    ssr = np.log(gradientDescent(X, y)[1])
    # number of iterations 
    iterations = np.log(np.arange(1, len(ssr) + 1, 1))
    # plot reduction of ssr
    plt.plot(iterations, ssr)
    # xlabel
    plt.xlabel("Iteration")
    # ylabel
    plt.ylabel("SSR")
    # title
    plt.title("Reduction of SSR by number of Iterations")
    # show plot 
    plt.show()
    
# call plotLoss function
plotLoss()


# 5. plot loss using plotly
def plotlyPlot():
    """
    Plot reduction of SSR vs number of iterations using Ploty Express.
    
    Returns
    -------
    plot : A plot visualizing the relationship of SSR and iterations.
    """
    # ssr
    ssr = gradientDescent(X, y)[1]
    # number of iterations 
    iterations = np.arange(0, len(ssr), 1)

    # use plotly to show graph
    fig = px.line(x=iterations, y=ssr, template="plotly_dark", 
                  labels=dict(x="Iterations", y="Sum of squared reiduals"),
                  title="Evolution of Sum of squared residuals by iteration")
    
    # update xticks
    fig.update_layout(
        xaxis = dict(
            tickmode = "linear",
            tick0 = 0,
            dtick = 5
        )
    )
    # show plot
    plot(fig)

# call plotlyPlot function
plotlyPlot()


# ----------------------------------------------------------------------------
# 3. Comparison with sklearn linear Regression
# ----------------------------------------------------------------------------

# Use sklearn LinearRegression for camparison
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.linear_model import LinearRegression

print(70 * ("-"))
print("Linear Regression Model using sklearn and the same sample data")
print(70 * ("-"))
# train model
reg = LinearRegression().fit(X, y)
print(f"Intercept:{np.round(reg.intercept_, 4)}",
      f"coefficients: {np.round(reg.coef_, 4)}")
