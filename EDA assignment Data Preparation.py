# Discretization 
import pandas as pd

iris = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/iris.csv")
iris.dtypes
# The "Unnamed" identifiers have been removed from the DataFrame since it's indicates the indexing which doesn't provide any useful information. 
iris = iris.drop(iris.columns[iris.columns.str.contains('Unnamed')], axis = 1)
iris['Species'].value_counts()
# For each category in the 'Species' column are the same (50 occurrences for each category). In this case, the variance is indeed zero.
# Drop the 'Species' column since it does not provide any useful information for analysis or modeling because it lacks variability
iris = iris.drop('Species', axis = 1)
# iris.isna().sum()

iris['Sepal.Len_1'] = pd.cut(iris['Sepal.Length'], 
                             bins = [min(iris['Sepal.Length']), iris['Sepal.Length'].median(), max(iris['Sepal.Length'])],
                             include_lowest = True, labels = ['Short', 'Long'])

iris['Sepal.Wid_1'] = pd.cut(iris['Sepal.Width'],
                             bins = [min(iris['Sepal.Width']), iris['Sepal.Width'].median(), max(iris['Sepal.Width'])],
                             include_lowest = True, labels = ['Narrow', 'Wide'])

iris['Petal.Len_1'] = pd.cut(iris['Petal.Length'], 
                             bins = [min(iris['Petal.Length']), iris['Petal.Length'].median(), max(iris['Petal.Length'])],
                             include_lowest = True, labels = ['Short', 'Long'])

iris['Petal.Wid_1'] = pd.cut(iris['Petal.Width'],
                             bins = [min(iris['Petal.Width']), iris['Petal.Width'].median(), max(iris['Petal.Width'])],
                             include_lowest = True, labels = ['Narrow', 'Wide'])

# Duplication & TypeCasting
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

retail = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Online Retail.csv", encoding = 'latin1')
retail.dtypes
retail.info()
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])

duplicate = retail.duplicated()  # Returns Boolean Series denoting duplicate rows
sum(duplicate)
retail = retail.drop_duplicates(keep = 'last')
print(retail.isna().sum())

# Unable to convert the datatype since CustomerID & Description features contains NA value, using imputation methods filling the values
from feature_engine.imputation import RandomSampleImputer
random_imp = RandomSampleImputer()
retail[['CustomerID', 'Description']] = pd.DataFrame(random_imp.fit_transform(retail[['CustomerID', 'Description']]))
retail.isna().sum()

retail[['CustomerID']] = retail[['CustomerID']].astype(int)
retail.dtypes

plt.hist(retail['UnitPrice'], bins = 5)
plt.boxplot(retail['UnitPrice'])

#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
#retail[['CustomerID', 'Description']] = pd.DataFrame(imp.fit_transform(retail[['CustomerID', 'Description']]))

# Missing Values
import pandas as pd 
import numpy as np

claimants = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/claimants.csv")
claimants.dtypes
# Check for count of NA's in each column
claimants.isna().sum()

from feature_engine.imputation import RandomSampleImputer
random_imputer = RandomSampleImputer()
claimants['CLMAGE'] = pd.DataFrame(random_imputer.fit_transform(claimants[['CLMAGE']]))
claimants['CLMAGE'].isnull().sum()
#replacing the 0 values with median value
claimants['CLMAGE'].replace(0, np.nan, inplace = True)
median = claimants['CLMAGE'].median()
claimants['CLMAGE'].fillna(median, inplace = True)

from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
claimants['CLMSEX'] = pd.DataFrame(mode_imputer.fit_transform(claimants[['CLMSEX']]))
claimants['CLMINSUR'] = pd.DataFrame(mode_imputer.fit_transform(claimants[['CLMINSUR']]))
claimants['SEATBELT'].fillna(claimants['SEATBELT'].mode().iloc[0], inplace = True)
claimants.isnull().sum()

claimants[['CLMAGE', 'CLMSEX', 'CLMINSUR', 'SEATBELT']] = claimants[['CLMAGE', 'CLMSEX', 'CLMINSUR', 'SEATBELT']].astype('int64')
claimants.dtypes

# Outliers 

import pandas as pd 
import seaborn as sns 

bt = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Boston.csv")
bt.dtypes

# convert the float datatype which has discrete values into discrete datatype
bt[['zn', 'age', 'chas', 'rad', 'tax']] = bt[['zn', 'age', 'chas', 'rad', 'tax']].astype('int64')
bt.dtypes

# There are no outliers found in the features 'indus', 'nox', 'age', 'dis', 'rad', 'tax'
# 'chas' since it contains boolean values so no need to find outliers for this feature
sns.boxplot(bt.crim) # right tail
sns.boxplot(bt.zn) # right tail
sns.boxplot(bt.indus) # no outliers
sns.boxplot(bt.nox) # no outliers
sns.boxplot(bt.rm) # both tails
sns.boxplot(bt.age) # no outliers
sns.boxplot(bt.dis) # right tail
sns.boxplot(bt.rad) # no outliers
sns.boxplot(bt.tax) # no outliers
sns.boxplot(bt.ptratio) # left tail
sns.boxplot(bt.black) # left tail
sns.boxplot(bt.lstat) # right tail
sns.boxplot(bt.medv) # both tails

# Winsorization is the effective technique to treat outliers
from feature_engine.outliers import Winsorizer

# replace the outliers with maximun and minimum values
winsor_iqr = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, 
                        variables = ['rm', 'medv', 'black', 'crim', 'zn', 'lstat', 'ptratio', 'dis'])
bt_iqr = winsor_iqr.fit_transform(bt)

sns.boxplot(bt_iqr.rm)
sns.boxplot(bt_iqr.medv)
sns.boxplot(bt_iqr.black) # not able to treat outlier using quantile approach
sns.boxplot(bt_iqr.crim) # not able to treat outlier using quantile & gaussian approach
sns.boxplot(bt_iqr.zn) # not able to treat outlier using quantile & gaussian approach
sns.boxplot(bt_iqr.lstat)
sns.boxplot(bt_iqr.ptratio)
sns.boxplot(bt_iqr.dis)


winsor_percentile = Winsorizer(capping_method = 'quantiles', tail = 'both', 
                               fold = 0.05, variables = ['ptratio', 'dis'])
bt_percentile = winsor_percentile.fit_transform(bt)

sns.boxplot(bt_percentile.ptratio)
sns.boxplot(bt_percentile.dis)


bt = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Boston.csv")
bt.dtypes

# convert the float datatype which has discrete values into discrete datatype
bt[['zn', 'age', 'chas', 'rad', 'tax']] = bt[['zn', 'age', 'chas', 'rad', 'tax']].astype('int64')
bt.dtypes



# Zero Variance

import pandas as pd
df_z = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Z_dataset.csv")

df_z.dtypes

# 'Id' column in the dataset serves as an index or identifier for the records and does not provide any meaningful information for analysis, 
# It is not necessary to include it as a feature in analysis

df_z['colour'].value_counts()
# Variance is a measure of variability, and if all values are the same, there is no variability, resulting in a variance of zero
# For each category in the 'colour' column are the same (50 occurrences for each category). In this case, the variance is indeed zero.
# The column does not provide any useful information for analysis or modeling because it lacks variability

df_z = df_z.drop(['colour', 'Id'], axis = 1)

df_z[['square.length', 'square.breadth', 'rec.Length', 'rec.breadth']].var()
# The columns 'square.length', 'square.breadth', 'rec.Length', 'rec.breadth' have high variance and they are valuable for analysis, it is indeed not recommended to remove them from the dataset.
# High variance indicates that these columns exhibit significant variability in their values, which can provide important information for further analysis or modeling tasks.
# so we can retain these columns in dataset as they contain meaningful and informative features.

# Dummy Variables

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Animal_category.csv")
df.columns # provide column names
df.shape # will give the shape of the dataframe
df.dtypes
df.info()

# 'Index' column in the dataset serves as an index or identifier for the records and does not provide any meaningful information for analysis, 
# It is not necessary to include it as a feature in analysis
df = df.drop('Index', axis = 1)

# Perform label encoding for 'Animals', 'Types' columns
label_encoder = LabelEncoder()
df[['Animals', 'Types']] = df[['Animals', 'Types']].apply(label_encoder.fit_transform)

# Perform one-hot encoding for the 'Homly', 'Gender' feature
df = pd.get_dummies(df, columns = ['Homly', 'Gender'], drop_first = True)
print("Encoded DataFrame: \n", df)

# Standardization

import pandas as pd
from sklearn.preprocessing import StandardScaler

seeds_data = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/Seeds_data.csv")
seeds_data.info()
seeds_data.Type.value_counts()
# For each category in the 'Type' column are the same (70 occurrences for each category). In this case, the variance is indeed zero.
# The column does not provide any useful information for analysis or modeling because it lacks variability
seeds_data = seeds_data.drop(['Type'], axis = 1)
s = seeds_data.describe()

# To scale data
std_scaler = StandardScaler()
s_new = std_scaler.fit_transform(seeds_data)
# Convert the array back to a dataframe
data = pd.DataFrame(s_new)
# print the pre-processed Dataframe 
print(data)
result = data.describe()
print(result)

# Transformation

import pandas as pd
import scipy.stats as stats 
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Learning/EDA (Exploratory Data Analytics)/Data Preprocessing/InClass_DataPreprocessing_datasets/calories_consumed.csv")
df.dtypes
df.isna().sum()

stats.probplot(df['Calories Consumed'], dist = stats.norm, plot = pylab)
stats.probplot(np.log(df['Calories Consumed']), dist = stats.norm, plot = pylab)

from feature_engine import transformation
tf = transformation.YeoJohnsonTransformer(variables = 'Calories Consumed')
df_tf = tf.fit_transform(df)
stats.probplot(df_tf['Calories Consumed'], dist = stats.norm, plot = pylab)

stats.probplot(df['Weight gained (grams)'], dist = stats.norm, plot = pylab)
fitted_data, fitted_lambda = stats.boxcox(df['Weight gained (grams)'])
# the code performs the Box-Cox transformation on the education.workex data, 
#saves the lambda value, and then plots the original data and the transformed data side by side using kernel density estimation (KDE) plots.
stats.probplot(fitted_data, dist = stats.norm, plot = pylab)

fig, ax = plt.subplots(1,2)
#df['Weight gained (grams)'] = df['Weight gained (grams)'].astype('float')#
#valid_data = df['Weight gained (grams)'].dropna().astype(float)
sns.distplot(df['Weight gained (grams)'],hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 2}, label = "Non-Normal", ax = ax[0])

sns.distplot(fitted_data, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 2}, label = "Normal", ax = ax[1])  

ax[0].legend(loc = 'upper right')                     
ax[1].legend(loc = 'upper right') 
plt.tight_layout()


#String Manipulstions

word = "Grow Gratitude"
print(word[0])
print(len(word))
print(word.count('G'))

string = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."
count = 0
for char in string:
    count+=1
print(count)
#len(string)

s = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
word = s.split()[5]
print(word[5])
print(s[:3])
#print(s[3:])
#print(s[:-3])
print(s[-3:])

w = "stay positive and optimistic"
split = w.split()
output = "".join(split)
print(output)

w = "stay positive and optimistic"
words = w.split()
char_1 = "H"
char_2 = "d"
char_3 = "c"
#1
for i in words:
    if i.startswith(char_1):
        print(i)
    if i.endswith(char_2) or i.endswith(char_3):
        print(i)

print("\U0001F728")
string = "Grow Gratitude"
str_new = string.replace("Grow", "Growth of")
print(str_new)


code = 129680
symbol = chr(code)
print(symbol * 108)

print("\U0001FA90" * 108)
symbol = "\U0001FA90"
code_point = ord(symbol)
print(code_point)


sentence = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS.repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT.mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“.eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"
reversed_sentence = sentence[::-1]
print(reversed_sentence)
