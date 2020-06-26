# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
sns.distplot(data['Rating'].dropna())
plt.show()

data = data[data['Rating']<=5]
sns.distplot(data['Rating'].dropna())
plt.show()
#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()

percent_null = total_null/data.isnull().count()

missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)
# code ends here
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)


# --------------

#Code starts here
sns.catplot(x='Category', y='Rating',data=data, kind='box', height=10)
plt.xticks(rotation=90)
plt.show()

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].str.replace(',','').astype(int)

le = LabelEncoder()

data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x='Installs', y='Rating',data=data)
plt.show()



#Code ends here



# --------------
#Code starts here

print(data['Price'].value_counts())

data['Price'] = data['Price'].str.replace('$','').astype(float)
sns.regplot(x='Price',y='Rating',data=data)
plt.show()

#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())
ix = data.columns.get_loc("Genres")
for i in range(len(data['Genres'])):
	if isinstance(data.iloc[i,ix],str):
		data.iloc[i,ix] = data.iloc[i,ix].split(';')[0]


gr_mean = data.groupby('Genres', as_index=False)['Rating'].mean()
gr_mean.describe()

gr_mean = gr_mean.sort_values(by='Rating')
print(gr_mean.iloc[0,])
print(gr_mean.iloc[-1,])
#Code ends here


# --------------

#Code starts here
print(data['Last Updated'].describe())

data['Last Updated'] = data['Last Updated'].astype('datetime64')
print(data.dtypes)

max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
print(data['Last Updated Days'])

sns.regplot(x='Last Updated Days', y='Rating', data=data)
plt.title('Rating vs Last Updated [Regplot]')
plt.show()
#Code ends here


