# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:55:47 2015

@author: patrickkennedy
"""





"""
import pandas as pd
recent_grads = pd.read_csv("recent-grads.csv")

def calculate_low_wage_proportion(df):
    low_wage = df["Low_wage_jobs"]
    total = df["Total"]
    low_wage_proportion = low_wage / total
    return low_wage_proportion


majors = recent_grads['Major'].value_counts().index

recent_grads_lower_emp_count = 0
all_ages_lower_emp_count = 0

for m in majors:
    recent_grad_row = recent_grads[recent_grads['Major'] == m]
    all_ages_row = all_ages[all_ages['Major'] == m]
    
    recent_grad_emp_rate = recent_grad_row['Unemployment_rate'].values
    all_ages_emp_rate = all_ages_row['Unemployment_rate'].values
    
    if recent_grad_emp_rate < all_ages_emp_rate:
        recent_grads_lower_emp_count += 1
    else:
        all_ages_lower_emp_count += 1
        
        
        


import matplotlib.pyplot as plt
avengers = pd.read_csv('avengers.csv')

true_avengers = pd.DataFrame()
avengers['Year'].hist()

true_avengers = avengers[avengers['Year'] >= 1960]


columns = ['Death1', 'Death2', 'Death3', 'Death4', 'Death5']
true_avengers[columns]

def clean_deaths(row):
    death_counter = 0
    for column in columns:
        death = row[column]
        if pd.isnull(death) or death == 'NO':
            continue
        elif death == 'YES':
            death_counter+=1
    return death_counter
    
true_avengers['Deaths'] = true_avengers.apply(lambda row: clean_deaths(row), axis=1)
    
    
    
joined_accuracy_count = int()

accurate = 2015 - true_avengers['Year']
reported = true_avengers['Years since joining']

joined_accuracy_count_list = accurate - reported
joined_accuracy_count = sum(joined_accuracy_count_list == 0)


years_since_joining_acc = true_avengers[true_avengers['Years since joining'] == true_avengers[reported])]
joined_accuracy_count = len(true_avengers[years_since_joining_acc]) - len(true_avengers['Year'])
    

"""

"""
import numpy as np

t_num = .299 - .307
t_den = ((.05/150)+(.08/165))


v_num = ((.05/150)+(.08/165))**2
v_den = (.05**2)/((150**2)*149) + (.08**2)/((165**2)*164)
"""


"""
def performOps(A):
    m = len(A)
    n = len(A[0])
    B = []
    for i in xrange(len(A)):
        B.append([0] * n)
        for j in xrange(len(A[i])):
            B[i][n - 1 - j] = A[i][j]
    return B
    
A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


B = performOps(A)
for i in xrange(len(B)):
    for j in xrange(len(B[i])):
        print B[i][j],

"""

"""
A = [ 14, 5, 14, 34, 42, 63, 17, 25, 39, 61, 97, 55, 33, 96, 62, 32, 98, 77, 35 ]
B = 56

def rotateArray(self, A, B):
        ret = []
        for i in xrange(len(A)):
            
            if i < len(A)-B:
                ret.append(A[i+B])
            else:
                for j in xrange(B):
                    ret.append(A[j])
        return ret
"""

"""
def spiralOrder(A):
    result = []
        
    top = 0
    bottom = len(A)-1
    left = 0
    right = len(A[0])-1
    direction = 0
        
    if len(A) == 1:
        return A[0]
                
    while(top <= bottom and left <= right):
        print(str(top) + ", " + str(bottom) + ", " + str(left) + ", " + str(right))            
            
        
        if direction == 0:
            print("top going right")
            for i in range(left, right+1):
                print(A[top][i])                
                result.append(A[top][i])
            top += 1
            direction = 1
            
            
        elif direction == 1:
            print("top going down")
            for i in range(top, bottom+1):
                print(A[i][right])                
                result.append(A[i][right])
            right -= 1
            direction = 2
            

        elif direction == 2:
            print("bottom going left")
            for i in range(right, left-1, -1):
                print(A[bottom][i])                
                result.append(A[bottom][i])
            bottom -= 1
            direction = 3
            

        elif direction == 3:
            print("bottom going top")
            for i in range(bottom, top-1, -1):
                print(A[i][left])                
                result.append(A[i][left])
            left += 1
            direction = 0
            
        
        
    return result
    
    
    
    """
    
    

"""
def maxSubArray(A):
    
    max_ending_here = max_so_far = A[0]
    for x in A[1:]:
        max_ending_here = max(x, max_ending_here+x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
"""

"""
def getRow(A):
    row_list = [1]
    
    if A == 0:
        return row_list
    else:
        for i in range(A):
            row_list.append(row_list[-1] * (A-i)/(i+1))
    
    return row_list
"""

"""
def generate(A):
    list_row_list = []
    counter = A-1    
    
    if counter == 0:
        row_list = [1]
        list_row_list.append(row_list)
        return list_row_list
    else:
        while counter > -1:        
            row_list = [1]            
            for i in range(counter):
                row_list.append(row_list[-1] * (counter-i)/(i+1))
            list_row_list.append(row_list)   
            counter -= 1
        #list_row_list.append([1])


    return list(reversed(list_row_list))
"""


"""
def coverPoints(X, Y):
    import math
    steps = 0    
    if len(X) == 1 or len(Y) == 1:
        return 0
    else:
        for i in range(len(X)):
            X_dist = X[i] - X[i-1]
            Y_dist = Y[i] - Y[i-1]
        
            if X_dist == Y_dist:
                steps += X_dist
            else:
                total_dist = math.trunc(math.sqrt(X_dist**2 + Y_dist**2))
                steps += total_dist
    
    return steps
"""

"""
def largestNumber(A):
    A = [str(x) for x in A]
    A.sort(reverse=True)
    
    B = [int(x) for x in A]
    if sum(B) == 0:
        return "0"
    else:
        count = 1
        while count > 0:
            count = 0            
            for i in range(len(A)-1):
                test1 = A[i]+A[i+1]
                test2 = A[i+1]+A[i]
            
                if test1 >= test2:
                    pass
                else:
                    temp = A[i+1]
                    A[i+1] = A[i]
                    A[i] = temp
                    count = 1
                    
    
        return "".join(A)
            
"""

"""
def wave(A):
    #dummyBool = True    
    A.sort()
    for i in range(len(A)-1):
        if A[i] < A[i+1] and i%2==0:
            temp = A[i+1]
            A[i+1] = A[i]
            A[i] = temp
            #dummyBool = False
        else:
            pass            
            #dummyBool = True
    return A
        
"""

"""
def merge(intervals):
       intervals.sort(key=lambda interval: interval.start)
       retIntervals = []
       first = intervals[0].start
       last = intervals[0].end
       for i in range(1,len(intervals)):
           if last > intervals[i].end:
               #current interval overlaps new one
               continue
           elif last < intervals[i].start:
               #current interval ended
               retIntervals.append(Interval(first,last))
               first = intervals[i].start
               last = intervals[i].end
           else:
               # New interval last detected
               last = intervals[i].end
       retIntervals.append(Interval(first,last))
       return retIntervals
"""

"""
# @param A : integer
# @return an integer
def isPrime(A):
    upperLimit = int(A**0.5)
    for i in xrange(2, upperLimit + 1):
        if i < A and A % i == 0:
            return 0
    return 1

     
        
def squareSum(A):
	ans = []
	a = 0
	while a * a < A:
		b = 0
		while b * b < A:
			if a * a + b * b == A:
				newEntry = [a, b]
				ans.append(newEntry)
			b += 1
		a += 1
	return ans
        

def findDigitsInBinary(A):
    n = A    
    if n == 0:
        return 0
        
    binary_list = []
    while n > 0:
        remainder = n%2
        binary_list.append(str(remainder))        
        n = n/2
    
    binary_string = "".join(binary_list)
    #binary_string = reversed(binary_string)    
    return int("".join(list(reversed(binary_string))))
    
    
    
def allFactors(A):
    factors = []
    for i in xrange(1, int(A**0.5)+1):
        if A%i == 0:
            factors.append(i)
            if i != A**0.5:
                factors.append(A/i)
    factors.sort()
    return factors
    
def sieve(A):
    primes = []
    for i in xrange(0, A+1):
        primes.append(1)
    primes[0] = 0
    primes[1] = 0
    
    
    for i in xrange(2, A):
        if primes[i] == 1:
            for j in xrange(2, A):
                if i*j > A:
                    pass
                else:
                    primes[i*j] = 0
    
    prime_list = []
    for i in xrange(len(primes)):
        if primes[i] == 1:
            prime_list.append(i)
    
    
    return prime_list
    
    
    
def isPrime(A):
    for i in range(2, int(A**0.5)+1):
        if A%i == 0:
            return 0
    return 1
    
    
    
def isPalindrome(A):    
    A = str(A)    
    if A[::] == A[::-1]:
        for i in range(len(A)):
            if i < 0:
                return False
                
        return True
    
    return False
    
    
def reverse(A):
    B = []
    
    A_string = str(A)        

    if A >= 0:
        B_string = A_string[::-1]
        B = int(B_string)
    else:
        B_string = A_string[:0:-1]
        print(B_string)        
        B = int(B_string)
        B = B*-1
    
    if B.bit_length() < 32:
        return B
    else:
        return 0
    

def gcd(A, B):
    X = max(A, B)
    Y = min(A, B)    
    if Y == 0:
        return X
        
    while Y:
        X, Y = Y, X%Y
    return X
    
    #return gcd(X-Y,Y)
    


def uniquePaths(A, B):
    X = A-1
    Y = B-1
    numerator = X+Y
    
    for i in range(1, (X+Y)):
        numerator = numerator * ((X+Y)-i)
    #print(numerator)        
    denom1 = X
    denom2 = Y
    
    if denom1 == 0:
        denom1 = 1
    if denom2 == 0:
        denom2 = 1
        
    for i in range(1, X):
        denom1*=(X-i)
    for i in range(1, Y):
        denom2*=(Y-i)
    #print(denom1*denom2)
    return numerator / (denom1*denom2)
    
    
    
    





def trailingZeroes(A):
    
    if A <= 0:
        return 0
    count = 1
    zeroes = 0
    while A/(5**count) >= 1:
        zeroes += round(A/(5**count))
        count += 1
        
    return int(zeroes)



def titleToNumber(A):
    A = A.lower()  
    A = A[::-1]
    column = dict()
    count = 1
    value = 0    
    for letter in "abcdefghijklmnopqrstuvwxyz":
        column[count] = letter
        count+=1    
    for n in range(len(A)):
        value += 26**n*column[A[n]]
    return value
    
    
def convertToTitle(N):
    column = ""
    col_dict = {}
    count = 1
    for letter in "abcdefghijklmnopqrstuvwxyz":
        col_dict[count] = letter
        count+=1
    
    while N > 0:
        if N%26 == 0:
            column += (col_dict[26])
        else:
            column += (col_dict[N%26])
        
        if N%26 == 0:
            N = (N / 26)-1
        else:
            N /= 26
    
    column = column[::-1]
    return column.upper()
        
"""        

"""
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt

def normalized_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features
    
def recover_params(means, std_devs, norm_intercept, norm_params):
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params
    
def linear_regression_GD(features, values):
    means, std_devs, features = normalized_features(features)
    model = SGDRegressor(eta0=0.001)
    results = model.fit(features, values)
    intercept = results.intercept_
    params = results.coef_
    return intercept, params
    
    
def predictionsGD(dataframe):
    features = dataframe[['rain', 'hour']]
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_units = pd.get_dummies(dataframe['DATEn'])
    features = features.join(dummy_units)
    
    values = dataframe['ENTRIESn_hourly']
    
    features_array = features.values
    values_array = values.values
    
    means, std_devs, normalized_features_array = normalized_features(features_array)
    
    norm_intercept, norm_params = linear_regression_GD(normalized_features_array, values_array)
    
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)
    
    predictions = intercept + np.dot(features_array, params)
    print(params)
    return predictions
    
    

def mann_whitney_plus_means(turnstile_weather):
    with_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==1]
    without_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==0]
    with_rain_mean = np.mean(with_rain)
    without_rain_mean = np.mean(without_rain)
    U, p = scipy.stats.mannwhitneyu(with_rain, without_rain)
    return with_rain_mean, without_rain_mean, U, p
    

def compute_r_squared(data, predictions):
    mean_data = np.mean(data)
    SST = np.sum((data - mean_data)**2)
    SSres = np.sum((predictions - mean_data)**2)
    r_squared = SSres / SST
    return r_squared




def linear_regression_OLS(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    params = results.params[1:]
    intercept = results.params[0]
    return intercept, params

def predictionsLR(dataframe):
    features = dataframe[['rain']]
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    values = dataframe['ENTRIESn_hourly']
    
    intercept, params = linear_regression_OLS(features, values)
    predictions = intercept + np.dot(features, params)
    return predictions
    

def stationMeans(dataframe):    
    
    rain = []
    no_rain = []
    station_list = []
    entriesn_hourly_station = pd.DataFrame()
    
    
    for station in dataframe['station'].unique():
        rain.append(np.mean(dataframe['ENTRIESn_hourly'][dataframe['rain']==1][dataframe['station']==station]))
        no_rain.append(np.mean(dataframe['ENTRIESn_hourly'][dataframe['rain']==0][dataframe['station']==station]))
        station_list.append(station)
    
    entriesn_hourly_station["station"] = station_list
    entriesn_hourly_station["rain"] = rain
    entriesn_hourly_station["no_rain"] = no_rain
    entriesn_hourly_station["diff"] = entriesn_hourly_station["rain"] - entriesn_hourly_station["no_rain"]
    
    
    
    return entriesn_hourly_station





def sortDictValues(dictionary):
    import operator
    sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
    return sorted_dictionary

"""
"""
What I am trying to do here is create a visualization for the stations and mean entries
per hour.  Something to show how 'used' the stations are.  A bar graph is fine for this.
The trouble I am having is actually plotting the values in a way that makes sense.
"""
"""


def rain_hists(rain, no_rain):
    
    plt.hist(no_rain.values, bins=20, range=[26000, 50000], color='g', label='No Rain')
    plt.hist(rain.values, bins=20, range=[26000, 50000], color='b', label='Rain')
    plt.title("Turnstile Entries per Hour Given Presence of Rain")
    plt.xlabel("Volume of Ridership in Entries per Hour")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    

def station_barchart(dataFrame):
    
    #dataFrame = stationMeans(data)
    #count = dataFrame['count']
    #entries = dataFrame['diff']
    #stas = dataFrame['station']        
    
    #index = index[::-1]
    #entries = entries[::-1]
    #stations = stations[::-1]    
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    
    #ax.barh(dataFrame['diff'], dataFrame['count'], color='b')
    #ax.set_title("NYC Subway Stations with the Biggest Difference in Ridership by Rain Conditions")
    #ax.set_xlabel("Mean Entries per Hour Difference with Rain")
    #ax.set_ylim(0,dataFrame.shape[0])
    #ax.set_xlim(-500,2000)    
    #yTickMarks = dataFrame['station']
    #ax.set_yticks(dataFrame['count'][::10])    
    #yTickNames = ax.set_yticklabels(yTickMarks)
    #plt.setp(yTickNames)    
    
    plt.barh(dataFrame['count'][::12], dataFrame['diff'][::12], height=7)
    plt.yticks(dataFrame['count'][::12],dataFrame['station'])
    plt.xlabel('Difference in Mean Entries per Hour')
    plt.title('Station Ridership Difference Given Presence of Rain')    
    
    plt.tight_layout()
    
    plt.show()
    """
"""
    Ok what do I want to do here... show subway stations with the biggest difference in ridership
    under rain conditions... but maybe i need context here too so i show the general pattern of differences
    with the tope 50 or whatever looks ok with the x axis labels
"""
"""
    
    
def ridershipTOD(dataframe, hours):
    #for hour in dataframe['hour']:
    #    dataframe['Mean_hourly'] = np.mean(dataframe['ENTRIESn_hourly'][dataframe['hour']==hour])
    
    
    plt.bar([0, 4, 8, 12, 16, 20],hours, align='center')
    plt.title("Turnstile Entries per Hour by Time of Day")
    plt.xticks(["12am", "4am", "8am", "12pm", "4pm", "8pm"])
    plt.ylabel("Entries per Hour")
    plt.show()
    
    
    
    
    
def primesum1(A):
    
    if A < 3:
        return 0
    
    primes = []
    for i in xrange(0, A+1):
        primes.append(1)
    primes[0] = 0
    primes[1] = 0    
    
    
    for i in xrange(2, A):
        if primes[i] == 1:
            for j in xrange(2, A):
                if i*j > A:
                    pass
                else:
                    primes[i*j] = 0
    
        if primes[A-i] == 1:
            for j in xrange(2, A):
                if (A-i)%j == 0 and j < (A-i):
                    primes[(A-i)] = 0
                    break
        
        if primes[i] == 1 and primes[A-i] == 1:
            return (i, A-i)




def primesum(n):
    for i in xrange(2, n):
        if is_prime(i) and is_prime(n - i):
            return i, n - i

def is_prime(n):
    if n < 2:
        return False

    for i in xrange(2, int(n**0.5) + 1):
        if n % i == 0:
            return False

    return True




def isPower(A):
    if A <= 0:
        return False
    divisor = float()
    count = 2.0

    while count < (A**0.5)+2:    
        divisor = 1/count        
        for p in xrange(2,int(A**divisor)+2):
            if p**count > A:
                break
            elif p**count == A:
                return True
        count += 1.0
    return False
            
def isSquare(A):
    div = float()
    count = 2.0    
    while count < A:
        div = 1/count    
        for i in xrange(2, A+1):
            if i**count == A:
                return i, count
        count += 1.0
    return False
                
    
    
def arrange(A):
    A[:] = [(((A[A[i]]%len(A))*len(A))+A[i]) / len(A) for i in range(len(A))]
    return A
    
    
"""
"""
import numpy as np
def plusOne(n):       
       
    for i in range(1, len(n)+1):
        n[-i] = n[-i]+1        
        if n[-i] == 10:
            n[-i] = 0
        else:
            return np.trim_zeros(n, trim='f')

    n.insert(0,1)    
    return np.trim_zeros(n, trim='f')
    
    
    
    
     
    n_rev = list(reversed(n))
    for i in range(len(n)):
        n_rev[i] = n_rev[i]+1
        if n_rev[i] == 10:
            n_rev[i] = 0
        else:
            n_new = list(reversed(n_rev))
            for i in range(len(n)):
                if n_new[i] != 0:
                    n_new = n_new[i:]
                    return n_new       
            
    n_new = list(reversed(n_rev))
    for i in range(len(n)):
        if n_new[i] != 0:
            n_new = n_new[i:]
            n_new.insert(0,1)
            return n_new
        elif sum(n_new) == 0:
            n_new.insert(0,1)
            return n_new
            
            
def repeatedNumber(A):
    
    sumOfA = 0
    sumOfA2 = 0
    n = 0
    for a in A:
        sumOfA2 += a*a
        sumOfA += a
        n += 1
    sumOfN = n*(n+1)/2
    print(sumOfN)
    retA = sumOfN - sumOfA
    print(retA)
    retB = (sumOfN*(2*n+1)/3 - sumOfA2)/retA
    print(retB)    
    x = (retB-retA)/2
    return [x,x + retA]


"""



def findRank(A):
    rank = 0 
    if A == "":
        return 0
    for i in range(len(A)):
        letter = A[i]
        count = 0
        factorial = 1
        for j in A[i:]:
            if j < letter:
                count += 1
        for k in range(1, len(A)-i):
            factorial = factorial*k
        rank += count*factorial
        print(rank)
    return (rank+1)%1000003
            

def comb(m, lst):
    S = set(lst)

    collect = set()
    step = set([''])
    while step:
        step = set(a+b for a in step for b in S if len(a+b) == m)
        collect |= step

    print sorted(collect)
    
    
def permutations(string):
     if not string:
             return ['']
     ret = []
     for i, d in enumerate(string):
             perms = permutations(string[:i] + string[i+1:])
             for perm in perms:
                     ret.append(d + perm)
     return ret




