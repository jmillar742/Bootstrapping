#%%
#Packages used
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math

# set seed for reproducibility
np.random.seed(123)
# Reads in all data from 3 csv files
df = pd.read_csv('IT Salary Cleaned.csv')
mdf = pd.read_csv('Male IT Salary.csv')
fdf = pd.read_csv('Female IT Salary.csv')
# Sets names of columns
df.columns = ['Age','Gender','City','Position','Total Years', 'Years in Germany', 'Seniority Level', 'Annual brutto salary', 'Annual bonuses', 'Annual Total Salary', 'One year ago Brutto Salary', 'One year ago bonuses', 'One year ago Total Salary', 'Number of vacation days', 'Company type']
mdf.columns = ['Age','Gender','City','Position','Total Years', 'Years in Germany', 'Seniority Level', 'Annual brutto salary', 'Annual bonuses', 'Annual Total Salary', 'One year ago Brutto Salary', 'One year ago bonuses', 'One year ago Total Salary', 'Number of vacation days', 'Company type']
fdf.columns = ['Age','Gender','City','Position','Total Years', 'Years in Germany', 'Seniority Level', 'Annual brutto salary', 'Annual bonuses', 'Annual Total Salary', 'One year ago Brutto Salary', 'One year ago bonuses', 'One year ago Total Salary', 'Number of vacation days', 'Company type']

# Samples 50 rows from male and female samples
male_sample = mdf.sample(50, random_state = 7)
female_sample = fdf.sample(50, random_state = 7)

# Bootstraps for t-test
male_bootstrap = pd.DataFrame([male_sample.sample(50,replace=True)['Annual Total Salary'].mean() for i in range(0,51)])
female_bootstrap = pd.DataFrame([female_sample.sample(50,replace=True)['Annual Total Salary'].mean() for i in range(0,51)])

# Calculation for t-value
s_p = math.sqrt(((male_bootstrap.var()/51)+(female_bootstrap.var()/51)))
t = (male_bootstrap.mean() - female_bootstrap.mean())/(s_p)
 
# Prints t-value
print("t-value =", t)

# Sets sample size used in F-test
samplesize = 20

# F-test functions
def SumSquareMean(sampleY):
    ss = 0
    mean = sampleY.mean()
    for i in range(len(sampleY)):
        ss += (sampleY[i] - mean)*(sampleY[i] - mean)
    return ss
def fit(sampleX, sampleY, x):
    m,c = np.polyfit(sampleX, sampleY, deg = 1)
    y = m*x+c
    return y
def SumSquareFit(sampleX, sampleY):
    ss = 0
    for i in range(len(sampleY)):
        ss += (sampleY[i]-fit(sampleX, sampleY, sampleX[i]))*(sampleY[i]-fit(sampleX, sampleY, sampleX[i]))
    return ss
def CalculateFValue(sampleX, sampleY):
    Fval = (SumSquareMean(sampleY) - SumSquareFit(sampleX, sampleY))/(SumSquareFit(sampleX, sampleY)/(samplesize-2))
    return Fval

# Bootstrap function for the f-test
def bootstrap_p_value_for_f_test(x,y, NBoot):
    n = len(ageArray)
    observed_t = CalculateFValue(x, y)
    # Creates an array of zeros
    ts = np.zeros(NBoot)
    for i in range(NBoot):
        x_boot = np.random.choice(x, n, replace=True)
        y_boot = np.random.choice(y, n, replace=True)
        ts[i] = CalculateFValue(x_boot,y_boot)
    p_value = np.mean(ts > observed_t)
    return p_value, x_boot, y_boot

# Generate observations of age and salary
s=df.sample(samplesize,random_state=8)
s.head()
ageArray = s['Age'].to_numpy()
salaryArray = s['Annual Total Salary'].to_numpy()

# Use this line to change number of bootstrap samples (used for the table)
Nboot = 1000

# Calls functions to calculate values
p_value,age_boot,salary_boot = bootstrap_p_value_for_f_test(ageArray, salaryArray,Nboot)
F_value = CalculateFValue(ageArray, salaryArray)
m,c = np.polyfit(ageArray, salaryArray, deg = 1)
# Prints out values for F-test
print("Equation of fit line: y = ", m,"x+", c )
print("Sum squared mean = ", SumSquareMean(salaryArray))
print("Sum squared fit =", SumSquareFit(ageArray, salaryArray))
print("Sample mean of salaries = ", salaryArray.mean())
print("F-value = ", F_value)
print("p-value = ", p_value)

# Prints sample points
print(ageArray, salaryArray)
# Displays a scatter plot
plt.scatter(ageArray, salaryArray)
m,c = np.polyfit(ageArray, salaryArray, deg = 1)
x = np.linspace(ageArray.min(), ageArray.max(), 100)
y = m*x+c
plt.plot(x,y, '-r')
plt.show()
# %%
