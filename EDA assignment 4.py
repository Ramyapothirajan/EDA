import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/2a.EDA/Q1_a.csv")

cars.shape
speed_skew = cars.speed.skew()
speed_kurtosis = cars.speed.kurt()
distance_skew = cars.dist.skew()
distance_kurtosis = cars.dist.kurt()

print("Skewness of speed", speed_skew)
print("Kurtosis of speed", speed_kurtosis)

print("Skewness of distance", distance_skew)
print("Kurtosis of distance", distance_kurtosis)

#Interpret the results
if speed_skew < 0:
    print("The data distribution of speed is Negatively Skewed")
elif speed_skew > 0:
    print("The data distribution of speed is Positively Skewed")
else:
    print("The data distribution of speed is Symmetric")
    
if distance_skew < 0:
    print("The data distribution of distance is Negatively Skewed")
elif distance_skew > 0:
    print("The data distribution of distance is Positively Skewed")
else:
    print("The data distribution of distance is Symmetric")
    
if speed_kurtosis < 0:
    print("The distribution of speed has light tails (platykurtic).")
elif speed_kurtosis > 0:
    print("The distribution of speed has heavy tails (leptokurtic).")
else:
    print("The distribution of speed has tails similar to a normal distribution (mesokurtic).")

if distance_kurtosis < 0:
    print("The distribution of distance has light tails (platykurtic).")
elif distance_kurtosis > 0:
    print("The distribution of distance has heavy tails (leptokurtic).")
else:
    print("The distribution of distance has tails similar to a normal distribution (mesokurtic).")


# Histogram plot to visualize the shape of the distribution of data 
plt.figure(figsize = (10, 6))
plt.subplot(1, 2, 1)
plt.hist(cars.speed, bins = 5, edgecolor = 'black', color = 'blue')
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.title("Speed of the car (kmph)")

plt.subplot(1, 2, 2)
plt.hist(cars.dist, bins = 10, edgecolor = 'black', color = 'pink')
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distance travelled in km")

plt.tight_layout()
plt.show()

# Boxplot to visualize outliers in data
plt.figure()  
plt.subplot(1, 2, 1)
plt.boxplot(cars.speed)
plt.ylabel("Speed")

plt.subplot(1, 2, 2)
plt.boxplot(cars.dist)
plt.ylabel("Distance")

plt.tight_layout()
plt.show()



import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/2a.EDA/Cars.csv")

speed_skew = skew(df['SP'])
speed_kurtosis = skew(df['WT'])
weight_skew = kurtosis(df['SP'])
weight_kurtosis = kurtosis(df['WT'])

print("Skewness of speed", speed_skew)
print("Kurtosis of speed", speed_kurtosis)

print("Skewness of Weight", weight_skew)
print("Kurtosis of weight", weight_kurtosis)


#Interpret the results
if speed_skew < 0:
    print("The data distribution of top speed is Negatively Skewed")
elif speed_skew > 0:
    print("The data distribution of top speed is Positively Skewed")
else:
    print("The data distribution of top speed is Symmetric")
    
if weight_skew < 0:
    print("The data distribution of weight is Negatively Skewed")
elif weight_skew > 0:
    print("The data distribution of weight is Positively Skewed")
else:
    print("The data distribution of weight is Symmetric")
    
if speed_kurtosis < 0:
    print("The distribution of top speed has light tails (platykurtic).")
elif speed_kurtosis > 0:
    print("The distribution of top speed has heavy tails (leptokurtic).")
else:
    print("The distribution of top speed has tails similar to a normal distribution (mesokurtic).")

if weight_kurtosis < 0:
    print("The distribution of weight has light tails (platykurtic).")
elif weight_kurtosis > 0:
    print("The distribution of weight has heavy tails (leptokurtic).")
else:
    print("The distribution of weight has tails similar to a normal distribution (mesokurtic).")


# Histogram plot to visualize the shape of the distribution of data 

plt.figure(figsize = (10,6))
plt.subplot(1,2,1)
plt.hist(df.SP, bins = 'auto', edgecolor = 'yellow', color = 'green')
plt.xlabel("Top Speed in km/h")
plt.ylabel("Frequency")
plt.title("Top Speed of the car")

plt.subplot(1,2,2)
plt.hist(df.WT, bins = 'auto', edgecolor = 'green', color = 'yellow')
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.title("Weight of the car")

plt.tight_layout()
plt.show()

# Boxplot to visualize outliers in data
    
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.boxplot(df.SP)
plt.title("Top Speed of the car")

plt.subplot(1,2,2)
plt.boxplot(df.WT)
plt.title("Weight of the car")

plt.tight_layout()
plt.show() 



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

scores = [34, 36, 36, 38, 38, 39, 39, 40, 40, 41, 41, 41, 41, 42, 42, 45, 49, 56]
mean = np.mean(scores)
median =  np.median(scores)
variance = np.var(scores)
standard_deviation = np.std(scores)
print("The mean of given scores:", mean)
print("The median of given scores:",median)
print("The variance of given scores:", variance)
print("The standard deviation of given scores:", standard_deviation)

print("The student scores range from %d to %d"%(min(scores), max(scores)))


# The mean & median are close in value, indicating that the data is roughly symmetric or normally distributed
# The variance and standard deviation values are relatively moderate, suggesting a moderate spread or dispersion of the scores around the mean.
# Based on the mean, median, variance, and standard deviation, we can conclude that the student marks appear to be relatively normally distributed, 
# with moderate variation and no significant outliers.

measures = ['Mean', 'Median', 'Variance', 'Standard Deviation']
values = [mean, median, variance, standard_deviation]
plt.bar(measures, values)
plt.xlabel('Measures')
plt.ylabel('Values')
plt.title('Measures of Student Marks')
plt.show()

sns.boxplot(scores, orient = 'horizontal', showfliers = True)
plt.xlabel('Scores')
plt.title('Boxplot of Student Marks')
plt.show()
# There are no extreme outliers in the dataset that significantly influence the measures of central tendency or spread.

When the mean and median are equal, it implies that the data has a balanced distribution around the center.

A skewness value of zero indicates perfect symmetry, meaning that the data is evenly distributed around the mean.

When the mean of a dataset is greater than the median, it indicates that the data is positively skewed or right-skewed.
In a positively skewed distribution, the majority of the data points are concentrated towards the left side, with a few larger values stretching the right tail of the distribution. 
This implies that the data has a longer right tail and is positively skewed.

When the median of a dataset is greater than the mean, it indicates that the data is negatively skewed or left-skewed.
In a negatively skewed distribution, the majority of the data points are concentrated towards the right side, with a few smaller values stretching the left tail of the distribution.  
This implies that the data has a longer left tail and is negatively skewed.

A positive kurtosis value indicates that the data has heavier tails and a more peaked or leptokurtic distribution compared to a normal distribution.
A positive kurtosis value indicates a more peaked or concentrated distribution in the center. This means that the data points are more concentrated around the mean, resulting in a taller and narrower peak compared to a normal distribution.

A negative kurtosis value indicates that the data has lighter tails and a less peaked or platykurtic distribution compared to a normal distribution.
A negative kurtosis value indicates a less peaked or flatter distribution in the center. This means that the data points are more spread out and have a wider peak compared to a normal distribution.

