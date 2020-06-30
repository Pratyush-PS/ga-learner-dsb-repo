# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset = dataset.drop('Id',1)
print(dataset.describe())
# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns.tolist()


#number of attributes (exclude target)
size = len(cols)-1

#x-axis has target attribute to distinguish between classes
x = cols[-1]
y = cols[:-1]

#y-axis shows values of an attribute
for i in range(size):
    sns.violinplot(x=x, y = cols[i], data=dataset)

plt.show()

#Plot violin for all attributes



# --------------
import numpy

upper_threshold = 0.5
lower_threshold = -0.5



# Code Starts Here

# create a subset of dataframe with only the first 10 features
subset_train = dataset.iloc[:, :10]

# Calculate the Pearson correlation
data_corr = subset_train.corr()

# Plot a heatmap
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(data_corr,vmax=0.8,square=True);

# List the correlation pairs
correlation = data_corr.unstack().sort_values(kind='quicksort')

# Select the highest correlation pairs using slicing
corr_var_list = correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation!=1)]
print(corr_var_list)
# Code ends here



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

r,c = dataset.shape
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)



#Standardized

scaler = StandardScaler()

#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp = scaler.transform(X_test.iloc[:,:10])

#Concatenate scaled continuous data and categorical
X_train1 = numpy.concatenate((X_train_temp,X_train.iloc[:,10:c-1]),axis=1)
X_test1 = numpy.concatenate((X_test_temp,X_test.iloc[:,10:c-1]),axis=1)

scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

# Code starts here
skb = SelectPercentile(score_func=f_classif,percentile=90)
predictors = skb.fit_transform(X_train1, Y_train)
scores = list(skb.scores_)

Features = scaled_features_train_df.columns

dataframe = pd.DataFrame({'Features':Features,'Scores':scores})

dataframe=dataframe.sort_values(by='Scores',ascending=False)

top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])

print(top_k_predictors)

# Code Ends here


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


clf  = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features = clf1.fit(X_train, Y_train)
prediction_all_features = model_fit_all_features.predict(X_test)
score_all_features = accuracy_score(Y_test, prediction_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
prediction_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test,prediction_top_features)
print(score_all_features,score_top_features)



