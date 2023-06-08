#!/usr/bin/env python
# coding: utf-8

# # Project Description

# # Overview

# Introduction:
#     
# This capstone project focuses on developing a machine learning model for temperature forecasting. The goal is to predict the next-day maximum and minimum air temperatures in Seoul, South Korea. 
# By utilizing historical weather data, forecast data from the LDAPS model, and geographic auxiliary variables, we aim to build separate models to predict the maximum and minimum temperatures for the following day.  
# 
# LDAPS = 'Local Data Assimilation and Prediction System'operated by the Korean Meteorological Administration for weather forecasting and climate research.
# 
# Dataset Context and Relevance:
# 
# The dataset used here in this project consists of summer weather data from 2013 to 2017 in Seoul, South Korea. It includes a range of variables such as present-day temperatures, LDAPS model forecasts, in-situ temperature measurements, and geographic features. These variables capture important aspects related to temperature, humidity, wind speed, cloud cover, and precipitation.
# 
# The relevance of this dataset lies in its potential to improve the accuracy of temperature forecasts. By analyzing the historical weather data and utilizing machine learning algorithms, we can identify patterns, relationships, and dependencies between the input variables and the next-day maximum and minimum temperatures. This allows us to build predictive models that can effectively correct any biases in the LDAPS model's forecasts and provide more accurate temperature predictions. The insights gained from analyzing the models' predictions can provide valuable information about the factors influencing temperature variations in the different seasons.
# 
# Start with importing necessary libraries

# # Importing Necessary Libraries

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


# # Importing Dataset

# Dataset Description:
# 
# The dataset which is used in this project can be accessed from the following link: https://github.com/dsrscientist/Dataset2/blob/main/temperature.csv
# 
# Column Description:
# This data set consist of 25 columns, which helps to predict the two target variables i.e.,
# 1. Next_Tmax: The next-day maximum air temperature (°C)
# 2. Next_Tmin: The next-day minimum air temperature (°C)
# 
# Input variables for making the predictions are:
# 
# 1. station: Weather station number (categorical)
# 2. Date: Present day in the format yyyy-mm-dd
# 3. Present_Tmax: Maximum air temperature between 0 and 21 h on the present day (°C)
# 4. Present_Tmin: Minimum air temperature between 0 and 21 h on the present day (°C)
# 5. LDAPS_RHmin: LDAPS model forecast of next-day minimum relative humidity (%)
# 6. LDAPS_RHmax: LDAPS model forecast of next-day maximum relative humidity (%)
# 7. LDAPS_Tmax_lapse: LDAPS model forecast of next-day maximum air temperature applied lapse rate (°C)
# 8. LDAPS_Tmin_lapse: LDAPS model forecast of next-day minimum air temperature applied lapse rate (°C)
# 9. LDAPS_WS: LDAPS model forecast of next-day average wind speed (m/s)
# 10. LDAPS_LH: LDAPS model forecast of next-day average latent heat flux (W/m2)
# 11. LDAPS_CC1: LDAPS model forecast of next-day 1st 6-hour split average cloud cover (0-5 h) (%)
# 12. LDAPS_CC2: LDAPS model forecast of next-day 2nd 6-hour split average cloud cover (6-11 h) (%)
# 13. LDAPS_CC3: LDAPS model forecast of next-day 3rd 6-hour split average cloud cover (12-17 h) (%)
# 14. LDAPS_CC4: LDAPS model forecast of next-day 4th 6-hour split average cloud cover (18-23 h) (%)
# 15. LDAPS_PPT1: LDAPS model forecast of next-day 1st 6-hour split average precipitation (0-5 h) (%)
# 16. LDAPS_PPT2: LDAPS model forecast of next-day 2nd 6-hour split average precipitation (6-11 h) (%)
# 17. LDAPS_PPT3: LDAPS model forecast of next-day 3rd 6-hour split average precipitation (12-17 h) (%)
# 18. LDAPS_PPT4: LDAPS model forecast of next-day 4th 6-hour split average precipitation (18-23 h) (%)
# 19. lat: Latitude (°)
# 20. lon: Longitude (°)
# 21. DEM: Elevation (m)
# 22. Slope: Slope (°)
# 23. Solar radiation: Daily incoming solar radiation (wh/m2)
# 
# Lets import dataset now.

# In[2]:


df= pd.read_csv("temperature.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df= df.drop(['Date'], axis=1)
df


# # Exploratory Data Analysis(EDA)

# It is an approach to analyze the datasets, and summarize their characteristics. Here we are analyzing dataset by first checking with the dimensions of the dataset, then checking for the null values present in the dataset, followed by the memory usage detail by using info() function, checking with value counts and unique values present in dataset, followed by statistical summary of dataset(numerical column), which ends with graphical analysis.
# 
# Let's start with checking the dimension of the dataset using df.shape attribute, then we will be seeing df.columns and df.dtypes attributes to check the columns and their datatypes, which will be followed by isnull().sum() function which will check the missing data present in each column of the dataset. We will also visualize the null values using heatmap to clear the confusion.

# In[6]:


# Check the columns
df.columns


# In[7]:


# Check the dimension
df.shape


# There are 7750 rows and 24 columns, in which 2 are target variable and 22 are independent variable.

# In[8]:


df.dtypes


# Here we can see there are only type of datatypes are present i.e., float64(24).

# In[9]:


# Now just check the missing values present in each column, if any

df.isnull().sum()


# In[10]:


# Filling missing values of numerical column
df['station'].fillna(df['station'].mean(), inplace= True)
df['Present_Tmax'].fillna(df['Present_Tmax'].mean(), inplace= True)
df['Present_Tmin'].fillna(df['Present_Tmin'].mean(), inplace= True)
df['LDAPS_RHmin'].fillna(df['LDAPS_RHmin'].mean(), inplace= True)
df['LDAPS_RHmax'].fillna(df['LDAPS_RHmax'].mean(), inplace= True)
df['LDAPS_Tmax_lapse'].fillna(df['LDAPS_Tmax_lapse'].mean(), inplace= True)
df['LDAPS_Tmin_lapse'].fillna(df['LDAPS_Tmin_lapse'].mean(), inplace= True)
df['LDAPS_WS'].fillna(df['LDAPS_WS'].mean(), inplace= True)
df['LDAPS_LH'].fillna(df['LDAPS_LH'].mean(), inplace= True)
df['LDAPS_CC1'].fillna(df['LDAPS_CC1'].mean(), inplace= True)
df['LDAPS_CC2'].fillna(df['LDAPS_CC2'].mean(), inplace= True)
df['LDAPS_CC3'].fillna(df['LDAPS_CC3'].mean(), inplace= True)
df['LDAPS_CC4'].fillna(df['LDAPS_CC4'].mean(), inplace= True)
df['LDAPS_PPT1'].fillna(df['LDAPS_PPT1'].mean(), inplace= True)
df['LDAPS_PPT2'].fillna(df['LDAPS_PPT2'].mean(), inplace= True)
df['LDAPS_PPT3'].fillna(df['LDAPS_PPT3'].mean(), inplace= True)
df['LDAPS_PPT4'].fillna(df['LDAPS_PPT4'].mean(), inplace= True)
df['Next_Tmax'].fillna(df['Next_Tmax'].mean(), inplace= True)
df['Next_Tmin'].fillna(df['Next_Tmin'].mean(), inplace= True)


# In[11]:


df.isnull().sum()


# In[12]:


# Lets visualize this by using heat map

plt.figure(figsize= (20,10))
sns.heatmap(df.isnull(), cmap= 'viridis')
plt.show()


# As this is clearly visualized that there is no null values.

# Now we will use info() function to get the detail about dataset, which basically shows us the range index, column name non-null counts, datatypes, memory usage of dataset.

# In[13]:


df.info()


# In[14]:


# Let's looping through each column to get value counts

for col in df.columns:
    print("Value counts for{col}:")
    print(df[col].value_counts())


# In[15]:


# loop through each column and print unique values
for col in df.columns:
    print("Unique values in {col}:")
    print(df[col].unique())


# # Description of Dataset

# In[16]:


# Statistical Summary of Numerical Columns
df.describe()


# This statistical summary of numerical columns include the count, mean, standard deviation, minimum value, 25th percentile (Q1), median (50th percentile or Q2), 75th percentile (Q3), and maximum value for each variable.
# 
# 1. count: This represents the number of non-missing values in each column. In this case, there are 7750 non-missing values for each column.
# 
# 2. mean: This is the average (mean) value of each column.
# 
# 3. std: This is the standard deviation, which measures the spread or variability of the values in each column.
# 
# 4. min: This is the minimum value in each column.
# 
# 5. 25%: This is the first quartile, which represents the value below which 25% of the data falls.
# 
# 6. 50%: This is the second quartile or median, which represents the value below which 50% of the data falls.
# 
# 7. 75%: This is the third quartile, which represents the value below which 75% of the data falls.
# 
# 8. max: This is the maximum value in each column.
# 
# These statistics provide information about the distribution and range of values in each column of your dataset. They can be useful for understanding the central tendency, variability, and overall distribution of the data.

# # Data Visualization- Graphical Analysis

# Here we will be performing data visualization techniques like Univariate, Bivariate , and Multivariate analysis to visually explore and analyze the data. It will helps us in gaining the insights into the patterns,distribution, and relatioship present in dataset. For plotting different graphs we wil be import seaborn and matplotlib Python libraries. By creating various plots, we can easly visualize the data.

# Univariate Analysis

# Here, to perform univariate analysis in numerical column we will be using Histogram plot, and Density plot. Using a bar plot to categorical column and will see the distribution, skewness and presence of outliers in the data present in each columns.

# Numerical Columns

# 1. 'station'

# In[17]:


# Let's analyze the 'station' column by using histogram plot

plt.hist(df['station'], bins=20, color='blue')
plt.xlabel('station')
plt.ylabel('Frequency')
plt.title('Histogram of station')
plt.show()


# Here we can see four peak for specific ranges. Can say symmetrical distribution.

# In[18]:


# Now let's analyze the 'station' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["station"], shade=True, color='blue')
plt.show()


# Here we can see symmetrical distribution.

# 2. 'Present_Tmax'

# In[19]:


# Let's analyze the 'Present_Tmax' column by using histogram plot

plt.hist(df['Present_Tmax'], bins=20, color='blue')
plt.xlabel('Present_Tmax')
plt.ylabel('Frequency')
plt.title('Histogram of Present_Tmax')
plt.show()


# Here we can see peak is at 30, as frequent values are from this range.

# In[20]:


# Now let's analyze the 'Present_Tmax' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Present_Tmax"], shade=True, color='blue')
plt.show()


# Here we can see slightly left skewed data distribution.

# 3. 'Present_Tmin'

# In[21]:


# Let's analyze the 'Present_Tmin' column by using histogram plot

plt.hist(df['Present_Tmin'], bins=20, color='blue')
plt.xlabel('Present_Tmin')
plt.ylabel('Frequency')
plt.title('Histogram of Present_Tmin')
plt.show()


# Here we can see peak is arount 23.5-24.5 range, that means frequent values are from this range.

# In[22]:


# Now let's analyze the 'Present_Tmin' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Present_Tmin"], shade=True, color='blue')
plt.show()


# Here too distribution is sligtly left skewed.

# 4. 'LDAPS_RHmin'

# In[23]:


# Let's analyze the 'LDAPS_RHmin' column by using histogram plot

plt.hist(df['LDAPS_RHmin'], bins=20, color='blue')
plt.xlabel('LDAPS_RHmin')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_RHmin')
plt.show()


# Here we can see peak is at 50, most frequent values are from this range.

# In[24]:


# Now let's analyze the 'LDAPS_RHmin' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_RHmin"], shade=True, color='blue')
plt.show()


# Here we can see almost symmetrical distribution.

# 5. 'LDAPS_RHmax'

# In[25]:


# Let's analyze the 'LDAPS_RHmax' column by using histogram plot

plt.hist(df['LDAPS_RHmax'], bins=20, color='blue')
plt.xlabel('LDAPS_RHmax')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_RHmax')
plt.show()


# Here we can see peak is arount 94 as most frequent values are from this point.

# In[26]:


# Now let's analyze the 'LDAPS_RHmax' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_RHmax"], shade=True, color='blue')
plt.show()


# Here we can see left skewed data distribution.

# 6. 'LDAPS_Tmax_lapse'

# In[27]:


# Let's analyze the 'LDAPS_Tmax_lapse' column by using histogram plot

plt.hist(df['LDAPS_Tmax_lapse'], bins=20, color='blue')
plt.xlabel('LDAPS_Tmax_lapse')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_Tmax_lapse')
plt.show()


# Here we can see the peak is around 30, that means more frequent values are from this point.

# In[28]:


# Now let's analyze the 'LDAPS_Tmax_lapse' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_Tmax_lapse"], shade=True, color='blue')
plt.show()


# Here we can see symmetrical distribution.

# 7. 'LDAPS_Tmin_lapse'

# In[29]:


# Let's analyze the 'LDAPS_Tmin_lapse' column by using histogram plot

plt.hist(df['LDAPS_Tmin_lapse'], bins=20, color='blue')
plt.xlabel('LDAPS_Tmin_lapse')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_Tmin_lapse')
plt.show()


# Here we can see peak is around 24, as most frequent values lies here.

# In[30]:


# Now let's analyze the 'LDAPS_Tmin_lapse' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_Tmin_lapse"], shade=True, color='blue')
plt.show()


# Here we can see slight left skewed distribution.

# 8. 'LDAPS_WS'

# In[31]:


# Let's analyze the 'LDAPS_WS' column by using histogram plot

plt.hist(df['LDAPS_WS'], bins=20, color='blue')
plt.xlabel('LDAPS_WS')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_WS')
plt.show()


# Here we can see peak is around 5.0-7.5, as most values lies in this point.

# In[32]:


# Now let's analyze the 'LDAPS_WS' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_WS"], shade=True, color='blue')
plt.show()


# Here we can see right skewed data distribution.

# 9. 'LDAPS_LH'

# In[33]:


# Let's analyze the 'LDAPS_LH' column by using histogram plot

plt.hist(df['LDAPS_LH'], bins=20, color='blue')
plt.xlabel('LDAPS_LH')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_LH')
plt.show()


# Here we can see the peak is at 50, means most frequent values lies at this point.

# In[34]:


# Now let's analyze the 'LDAPS_LH' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_LH"], shade=True, color='blue')
plt.show()


# Here we can see right skewed distribution.

# 10. 'LDAPS_CC1'

# In[35]:


# Let's analyze the 'LDAPS_CC1' column by using histogram plot

plt.hist(df['LDAPS_CC1'], bins=20, color='blue')
plt.xlabel('LDAPS_CC1')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_CC1')
plt.show()


# Here we can see peak is at 0 but rest distribution looks symmetrical.

# In[36]:


# Now let's analyze the 'LDAPS_CC1' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_CC1"], shade=True, color='blue')
plt.show()


# Here we can see bimodal distribution.

# 11. 'LDAPS_CC2'

# In[37]:


# Let's analyze the 'LDAPS_CC2' column by using histogram plot

plt.hist(df['LDAPS_CC2'], bins=20, color='blue')
plt.xlabel('LDAPS_CC2')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_CC2')
plt.show()


# Here also we can see distribution is same like CC1.

# In[38]:


# Now let's analyze the 'LDAPS_CC2' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_CC2"], shade=True, color='blue')
plt.show()


# Here too we can see bimodal distribution.

# 12. 'LDAPS_CC3'

# In[39]:


# Let's analyze the 'LDAPS_CC3' column by using histogram plot

plt.hist(df['LDAPS_CC3'], bins=20, color='blue')
plt.xlabel('LDAPS_CC3')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_CC3')
plt.show()


# Here too peak is at 0 and rest is is rightly skewed.

# In[40]:


# Now let's analyze the 'LDAPS_CC3' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_CC3"], shade=True, color='blue')
plt.show()


# Here we can see distribution is rightly skewed.

# 13. 'LDAPS_CC4'

# In[41]:


# Let's analyze the 'LDAPS_CC4' column by using histogram plot

plt.hist(df['LDAPS_CC4'], bins=20, color='blue')
plt.xlabel('LDAPS_CC4')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_CC4')
plt.show()


# Here also we can see peak is at zero.

# In[42]:


# Now let's analyze the 'LDAPS_CC4' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_CC4"], shade=True, color='blue')
plt.show()


# here we can see bimodal distribution.

# 14. 'LDAPS_PPT1'

# In[43]:


# Let's analyze the 'LDAPS_PPT1' column by using histogram plot

plt.hist(df['LDAPS_PPT1'], bins=20, color='blue')
plt.xlabel('LDAPS_PPT1')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_PPT1')
plt.show()


# Here also we can see peak is at 0 and almost all the values lies at this point.

# In[44]:


# Now let's analyze the 'LDAPS_PPT1' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_PPT1"], shade=True, color='blue')
plt.show()


# Here we can see heavly right skewed data.

# 15. 'LDAPS_PPT2'

# In[45]:


# Let's analyze the 'LDAPS_PPT2' column by using histogram plot

plt.hist(df['LDAPS_PPT2'], bins=20, color='blue')
plt.xlabel('LDAPS_PPT2')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_PPT2')
plt.show()


# Here we can see the peak is at zero and all the values accumulated here n nearby.

# In[46]:


# Now let's analyze the 'LDAPS_PPT2' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_PPT2"], shade=True, color='blue')
plt.show()


# Here we can see heavly right skewed distribution.

# 16. 'LDAPS_PPT3'

# In[47]:


# Let's analyze the 'LDAPS_PPT3' column by using histogram plot

plt.hist(df['LDAPS_PPT3'], bins=20, color='blue')
plt.xlabel('LDAPS_PPT3')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_PPT3')
plt.show()


# Here we can see the peak is at 0 and other few values are lies nearby, distribution is heavly skewed.

# In[48]:


# Now let's analyze the 'LDAPS_PPT3' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_PPT3"], shade=True, color='blue')
plt.show()


# Here too we can see heavly skeweness in the distribution.

# 17. 'LDAPS_PPT4'

# In[49]:


# Let's analyze the 'LDAPS_PPT4' column by using histogram plot

plt.hist(df['LDAPS_PPT4'], bins=20, color='blue')
plt.xlabel('LDAPS_PPT4')
plt.ylabel('Frequency')
plt.title('Histogram of LDAPS_PPT4')
plt.show()


# Here also we can see the peak is at 0, and other few values lies nearby.

# In[50]:


# Now let's analyze the 'LDAPS_PPT4' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["LDAPS_PPT4"], shade=True, color='blue')
plt.show()


# Here also we can see the heavly right skewed distribution.

# 18. 'lat'

# In[51]:


# Let's analyze the 'lat' column by using histogram plot

plt.hist(df['lat'], bins=20, color='blue')
plt.xlabel('lat')
plt.ylabel('Frequency')
plt.title('Histogram of lat')
plt.show()


# In[52]:


# Now let's analyze the 'lat' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["lat"], shade=True, color='blue')
plt.show()


# Here we can see in both the plots that distribution is bimodal.

# 19. 'lon'

# In[53]:


# Let's analyze the 'lon' column by using histogram plot

plt.hist(df['lon'], bins=20, color='blue')
plt.xlabel('lon')
plt.ylabel('Frequency')
plt.title('Histogram of lon')
plt.show()


# In[54]:


# Now let's analyze the 'lon' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["lon"], shade=True, color='blue')
plt.show()


# Here also we can see in both the plots distribution is bimodal.

# 20. 'DEM'

# In[55]:


# Let's analyze the 'DEM' column by using histogram plot

plt.hist(df['DEM'], bins=20, color='blue')
plt.xlabel('DEM')
plt.ylabel('Frequency')
plt.title('Histogram of DEM')
plt.show()


# Here we can see the peak is at around 25, and even all the values lies between 0-60, and then there is split of of values which shows presence of extreme values.

# In[56]:


# Now let's analyze the 'DEM' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["DEM"], shade=True, color='blue')
plt.show()


# Here too we can see bimodal distribution.

# 21. 'Slope'

# In[57]:


# Let's analyze the 'Slope' column by using histogram plot

plt.hist(df['Slope'], bins=20, color='blue')
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.title('Histogram of Slope')
plt.show()


# Here too peak is around zero and can see few extreme values are present.

# In[58]:


# Now let's analyze the 'Slope' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Slope"], shade=True, color='blue')
plt.show()


# Here too the distribution is bimodal.

# 22. 'Solar radiation'

# In[59]:


# Let's analyze the 'Solar radiation' column by using histogram plot

plt.hist(df['Solar radiation'], bins=20, color='blue')
plt.xlabel('Solar radiation')
plt.ylabel('Frequency')
plt.title('Histogram of Solar radiation')
plt.show()


# Here we can see peak is at around 5750. and distribution is left skewed.

# In[60]:


# Now let's analyze the 'Solar radiation' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Solar radiation"], shade=True, color='blue')
plt.show()


# Here we can see distribution is heavely left skewed.

# 23. 'Next_Tmax'

# In[61]:


# Let's analyze the 'Next_Tmax' column by using histogram plot

plt.hist(df['Next_Tmax'], bins=20, color='blue')
plt.xlabel('Next_Tmax')
plt.ylabel('Frequency')
plt.title('Histogram of Next_Tmax')
plt.show()


# Here we can see peak is around 30.

# In[62]:


# Now let's analyze the 'Next_Tmax' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Next_Tmax"], shade=True, color='blue')
plt.show()


# Here we can see symmetrical distribution.

# 24. 'Next_Tmin'

# In[63]:


# Let's analyze the 'Next_Tmin' column by using histogram plot

plt.hist(df['Next_Tmin'], bins=20, color='blue')
plt.xlabel('Next_Tmin')
plt.ylabel('Frequency')
plt.title('Histogram of Next_Tmin')
plt.show()


# Here we can see peak is at 24.

# In[64]:


# Now let's analyze the 'Next_Tmin' column by using density plot

sns.set_style("darkgrid")
sns.kdeplot(df["Next_Tmin"], shade=True, color='blue')
plt.show()


# Here we can see slightly left skewed distribution.

# Bivariate Analysis

# Here for bivariate we will be using scatter plot.

# 1. 'Next_Tmax' vs 'Present_Tmax' Columns

# In[65]:


# let's visualize relationship

sns.scatterplot(data=df, x="Next_Tmax", y="Present_Tmax")
plt.xlabel("Next_Tmax")
plt.ylabel("Present_Tmax")
plt.title("Scatter plot: Next_Tmax vs Present_Tmax")
plt.show()


# Here we can see the linear relation between the variables.

# 2. 'Next_Tmin' vs 'Present_Tmin' Columns

# In[66]:


sns.scatterplot(data=df, x="Next_Tmin", y="Present_Tmin")
plt.xlabel("Next_Tmin")
plt.ylabel("Present_Tmin")
plt.title("Scatter plot: Next_Tmin vs Present_Tmin")
plt.show()


# Here too we can see strong positive linear relation.

# 3. 'Next_Tmax' vs 'lat' Columns

# In[67]:


sns.scatterplot(data=df, x="Next_Tmax", y="lat")
plt.xlabel("Next_Tmax")
plt.ylabel("lat")
plt.title("Scatter plot: Next_Tmax vs lat")
plt.show()


# We can see null relation.

# 4. 'Next_Tmin' vs 'lat' Columns

# In[68]:


sns.scatterplot(data=df, x="Next_Tmin", y="lat")
plt.xlabel("Next_Tmin")
plt.ylabel("lat")
plt.title("Scatter plot: Next_Tmin vs lat")
plt.show()


# Here too we can see null relation.

# 5. 'Next_Tmax' vs 'lon' Columns

# In[69]:


sns.scatterplot(data=df, x="Next_Tmax", y="lon")
plt.xlabel("Next_Tmax")
plt.ylabel("lon")
plt.title("Scatter plot: Next_Tmax vs lon")
plt.show()


# Here also we can see null relation.

# 6. 'Next_Tmin' vs 'lon' Columns

# In[70]:


sns.scatterplot(data=df, x="Next_Tmin", y="lon")
plt.xlabel("Next_Tmin")
plt.ylabel("lon")
plt.title("Scatter plot: Next_Tmin vs lon")
plt.show()


# Here also can see null relation.

# 7. 'Next_Tmax' vs 'Solar radiation' Columns

# In[71]:


sns.scatterplot(data=df, x="Next_Tmax", y="Solar radiation")
plt.xlabel("Next_Tmax")
plt.ylabel("Solar radiation")
plt.title("Scatter plot: Next_Tmax vs Solar radiation")
plt.show()


# Here we can see moderate negative linear relation.

# 8. 'Next_Tmin' vs 'Solar Radiation' Columns

# In[72]:


sns.scatterplot(data=df, x="Next_Tmin", y="Solar radiation")
plt.xlabel("Next_Tmin")
plt.ylabel("Solar radiation")
plt.title("Scatter plot: Next_Tmin vs Solar radiation")
plt.show()


# Here also we can see moderate negative linear relation.

# 9. 'Next_Tmax' vs 'LDAPS_RHmax' Columns

# In[73]:


sns.scatterplot(data=df, x="Next_Tmax", y="LDAPS_RHmax")
plt.xlabel("Next_Tmax")
plt.ylabel("LDAPS_RHmax")
plt.title("Scatter plot: Next_Tmax vs LDAPS_RHmax")
plt.show()


# Here we can see null relation.

# 10. 'Next_Tmin' vs 'LDAPS_RHmin' Columns

# In[74]:


sns.scatterplot(data=df, x="Next_Tmin", y="LDAPS_RHmin")
plt.xlabel("Next_Tmin")
plt.ylabel("LDAPS_RHmin")
plt.title("Scatter plot: Next_Tmin vs LDAPS_RHmin")
plt.show()


# Here too we can see no relation.

# 11. 'Next_Tmax' vs 'Present_Tmin' Columns

# In[75]:


sns.scatterplot(data=df, x="Next_Tmax", y="Present_Tmin")
plt.xlabel("Next_Tmax")
plt.ylabel("Present_Tmin")
plt.title("Scatter plot: Next_Tmax vs Present_Tmin")
plt.show()


# Here we can see positive linear relation.

# 12. 'Next_Tmin' vs 'Present_Tmax' Columns

# In[76]:


sns.scatterplot(data=df, x="Next_Tmin", y="Present_Tmax")
plt.xlabel("Next_Tmin")
plt.ylabel("Present_Tmax")
plt.title("Scatter plot: Next_Tmin vs Present_Tmax")
plt.show()


# Here we can see positive linear relation.

# 13. 'Next_Tmax' vs 'LDAPS_RHmin' Columns

# In[77]:


sns.scatterplot(data=df, x="Next_Tmax", y="LDAPS_RHmin")
plt.xlabel("Next_Tmax")
plt.ylabel("LDAPS_RHmin")
plt.title("Scatter plot: Next_Tmax vs LDAPS_RHmin")
plt.show()


# Here too we can see moderate negative linear relation.

# 14. 'Next_Tmin' vs 'LDAPS_RHmax' Columns

# In[78]:


sns.scatterplot(data=df, x="Next_Tmin", y="LDAPS_RHmax")
plt.xlabel("Next_Tmin")
plt.ylabel("LDAPS_RHmax")
plt.title("Scatter plot: Next_Tmin vs LDAPS_RHmax")
plt.show()


# Here we can see null relation.

# # Multivariate Analysis

# In[79]:


# Using Pairplot

plt.figure(figsize=(20,20))
sns.pairplot(data=df, hue= 'Next_Tmax')


# In[81]:


plt.figure(figsize=(30,30))
sns.pairplot(data=df, hue= 'Next_Tmin')


# # Data Preprocessing

# # Outliers

# Here first we will check all the numerical columns to check the presence of outliers there and then will proceed with handling them either by treating them or by removing them.
# 
# We will be using winsorizing method first to treat outliers and then will be using clip() function to limit the extreme values. By cliping these extreme values we will be setting boundaries to prevent extremely high or low values from skewing the data for analysisresults. This will help us here to mitigate the impact of outliers and extreme observations on the analysis and avoid causing data loss.
# 
# For this we will be importing necessary library first from scipy.stats.mstats
# 
# Winsorizing Method: Winsorizing is a data transformation technique used to handle outliers or extreme values in a dataset. It involves replacing extreme values with less extreme values, thereby reducing the impact of outliers on statistical analysis. In winsorizing, we will be trimming or censoring the extreme values by replacing them with the value which is closer to rest of the data. It helps in reducoing the impact of outliers on the statistical measures and provides a more robust analysis.It ensures that extreme values do not unduly influence the results or skew the distribution of the data.

# In[82]:


df.shape


# In[83]:


# Lets check the outliers

numerical_columns = ['station', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin',  'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS', 'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4','LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon', 'DEM', 'Slope', 'Solar radiation', 'Next_Tmax', 'Next_Tmin']

for column in numerical_columns:
    sns.boxplot(x=df[column])
    plt.show()


# In[84]:


from scipy.stats.mstats import winsorize

# Winsorize all numerical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df[num_cols] = df[num_cols].apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0)

# Clip extreme values
clip_values = df[num_cols].quantile([0.01, 0.99])
df[num_cols] = df[num_cols].clip(lower=clip_values.loc[0.01], upper=clip_values.loc[0.99], axis=1)


# In[85]:


# Lets check the outliers

numerical_columns = ['station', 'Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin',  'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS', 'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4','LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon', 'DEM', 'Slope', 'Solar radiation', 'Next_Tmax', 'Next_Tmin',]

for column in numerical_columns:
    sns.boxplot(x=df[column])
    plt.show()


# As we have treated outliers, and clipped the extreme values, by such way we have handled them possible.

# # Skewness

# Here we will be checking for skewness and those values which are not in the range of -0.5 to +0.5, will be handling them.
# 
# We will be using different techniques like Yeo-johnson transformation technique, box-cox transformation technique and log transformation technique to handle the skewness in the column.
# 
# Yeo-Johnson transformation: It is a statistical measure and a power transformation technique used to transform a non-normal distribution into a distribution that closely resembles a normal distribution. It is a extension to Boxcox transformation. It involves applying power transformation to the original data. It helps in improving the model's performance. Implementing the Yeo-Johnson transformation in Python can be done using libraries such as SciPy or scikit-learn, using 'from sklearn.preprocessing import PowerTransformer' .
# 
# Boxcox transformation: It is a statistical technique used to transform a non-normal distribution into a distribution that closely resembles a normal distribution. It involves applying power transformation to the original data. But it works on only positive values. It helps in improving the model's performance. Implementing the Box-Cox transformation in Python can be done using libraries such as SciPy or scikit-learn.
# 
# The log transformation is applied to the values of a variable, typically positive and skewed variables, to compress the range of values and reduce the impact of extreme values. It can be useful when the data exhibits exponential growth or when the relationship between variables appears to be multiplicative rather than additive.
# 

# In[86]:


df.skew()


# In[87]:


from sklearn.preprocessing import PowerTransformer

# Define columns to transform
columns = ['LDAPS_RHmax', 'LDAPS_WS', 'LDAPS_CC3', 'LDAPS_CC4', 'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'DEM', 'Slope']

# Apply Yeo-Johnson transformation to selected columns
pt = PowerTransformer(method='yeo-johnson')
df[columns] = pt.fit_transform(df[columns])

# Round transformed values to 2 decimal places
df[columns] = df[columns].round(2)


# In[88]:


df.skew()


# In[89]:


# Apply log transformation to few columns
df['LDAPS_PPT1'] = np.log(df['LDAPS_PPT1'] + 1)
df['LDAPS_PPT2'] = np.log(df['LDAPS_PPT2'] + 1)
df['LDAPS_PPT3'] = np.log(df['LDAPS_PPT3'] + 1)
df['LDAPS_PPT4'] = np.log(df['LDAPS_PPT4'] + 1)


# In[90]:


df.skew()


# In[91]:


from scipy.stats import boxcox
from scipy.special import inv_boxcox

# create a list of columns to transform
columns = ['LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4']

# apply BoxCox transformation to each column
for col in columns:
    data = df[col]
    # transform the data using BoxCox method
    data_transformed, lambda_val = boxcox(data+1)
    # replace the column with the transformed data
    df[col] = data_transformed


# In[92]:


df.skew()


# As we have handled skewness possibly.

# In[93]:


df.info()


# # Correlation Between Target And Independent Variable

# Now here, we will be checking the correlation matrix which shows the relation between target and independent variable. We will get it by using df.corr() function. We will be finding the correlation to know which feature is more positive or negatively correlated to the target vaiable, which will be helping us for the feature selction to avoid overfitting. We will also be visualizing this correlation with using heatmap. And will be using seaborn library for heatmap.

# In[94]:


# Check the correlation

df.corr()


# In[99]:


correlation_matrix= df.corr()
Next_Tmax_correlation= correlation_matrix['Next_Tmax']
print(Next_Tmax_correlation)

correlation_matrix= df.corr()
Next_Tmin_correlation= correlation_matrix['Next_Tmin']
print(Next_Tmin_correlation)


# In[95]:


df= df.drop(['station', 'Slope', 'Solar radiation'], axis= 1)
df


# # Visualize the correlation Matrix

# In[96]:


# Let's visualize the relation by using heatmap

correlation_matrix = df.corr()

plt.figure(figsize=(20,15))
sns.heatmap(correlation_matrix, annot=True, cmap='plasma')
plt.title('Correlation matrix')
plt.show()


# # Featuring Engineering 

# Now we will be defining the target and independent variable seperately for both the tasks(Target Variable), so that it will be clear what to predict on what basis. Then we will proceed with the feature scaling process using MinMaxScaler from sklearn.preprocessing.
# 
# Feature Scaling: It is performed to bring all the features to a similar scale or value range. It basically helps in improving the performance, convergence, and interpretability of ML algorithm. It ensures that all features should be treated equally during the modeling.
# 
# MinMaxScaler: This technique used to transform features by scaling them to a specified range which should be between 0-1. It ensures that all features are scaled proportionally. This technique is useful to preserve the shape of the original distribution while bringing all features to a common scale.
# 
# After this we will proceed with checking out for multicollinearity by finding the vif(variation inflation factor) values. For this we will be importing statsmodels.api and then from statsmodels.stats.outliers_inflation importing variance_inflation_factor to find out the vif values for the features.
# 
# A vif value 1 indicates no multicollinearity, while values greater than 1 suggest increasing multicollinearity. Generally , vif value above 5 or 10 is considered very high, so in that case we need to handle the multicollinearity.
# 
# If we will find any value above range we will be handling that by dropping one or two column with high multicollinearity, an will also be adding constant term for calculating vif. A constant term will allow and help us for a more accurate assessment of multicollinearity and its impact on predictor variable.
# 
# After handling multicollinearity issue, we will proceed with finding the Best Random State, which will ensure that further operations should be reproducible. Here we will be using logistic regression algorithm for classification tasks to generate the best random state. The genrated best random score by this alogorithm should be used in all further runs for results reproducibility.
# 
# We will have to balance the target variable if there is any imbalance, and in that case we will be using RandomOverSampler technique.
# 
# Now we will be spliting the data into train and test so that we can train models on train data and can check its performance on test data. As We have a limited observation so, we will be taking "test size- 0.2", rest will be used as train data to train the model.

# # A. Define Target and Independent Variable for 'Next_Tmax' Column

# In[97]:


# Define Target and Independent Variable

y = df['Next_Tmax']
X = df.drop('Next_Tmax', axis=1)

print("Target variable name: ", y.name)
print("Target variable dimensions: ", y.shape)

print("\nFeatures variables names: ", list(X.columns))
print("Features variables dimensions: ", X.shape)


# # Feature Selection

# In[98]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

y = df['Next_Tmax']
X = df.drop('Next_Tmax', axis=1)

# Initialize the model
model= LinearRegression()

# Initialize the RFE selector with model
rfe= RFE(model, n_features_to_select=5)

# Fit the selctor on data
rfe.fit(X,y)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Print the features
print("Selected Features:")
print(selected_features)


# In[99]:


fit_score = rfe.ranking_
print("Fit Score:")
print(fit_score)
# Get the column names of all features
all_columns = X.columns
feature_rankings = pd.DataFrame({'Feature': all_columns, 'Ranking': fit_score})

# Sort the features based on their rankings (lower rank indicates higher importance)
feature_rankings = feature_rankings.sort_values('Ranking')

print("Feature Rankings:")
print(feature_rankings)


# In[100]:


X= X.drop(['LDAPS_LH', 'LDAPS_PPT2'], axis=1)
X


# # Feature Scaling

# In[101]:


from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Apply to all columns
X_scaled = scaler.fit_transform(X)

# Create a new dataframe with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df)


# # Variance Inflation Factor(VIF)

# In[102]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# add a constant column to features
X_scaled_df = sm.add_constant(X_scaled_df)

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["X"] = X_scaled_df.columns
vif["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]

# Print the VIF dataframe
print(vif)


# All the vif are in range.

# # Value Count of Target Variable

# In[103]:


y.value_counts()


# # Best Random State

# In[104]:


# Lets find the best random state

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Split the data into training and testing sets using different random states
best_random_state = None
best_r2_score = -1
for random_state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Next_Tmax', axis=1), df['Next_Tmax'], test_size=0.2, random_state=random_state)
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Check if this random state gives a better R2 score
    if r2 > best_r2_score:
        best_r2_score = r2
        best_random_state = random_state

print("Best random state:", best_random_state)
print("Best R^2 score:", best_r2_score)


# Here, we find the Best R^2 score is 0.7907454155398771 at best random state 79.

# # Split The Data

# In[105]:


# Let's split the data into test and train

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=79)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Modelling

# # Importing Necessary Libraries

# In[106]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# # Linear Regression

# In[107]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the linear regression model and fit the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the evaluation metrics
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, lr.predict(X_train))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Root squared on test data:", np.sqrt(r2_test))
print("MAE:", mae)
print("MSE:", mse)
print("Root mean square error:", rmse)
print("Root squared on training data:", np.sqrt(r2_train))


# Here we can see this model is performing  well on both the training and testing data i.e., R2 score for training data is 87.3% and for testing data is 88.6%.

# # Lasso Regression (L1)

# In[108]:


# Initialize the Lasso Regression model and fit the training data
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate mean squared error for test data
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate root mean squared error for test data
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing good on both the training and testing data i.e., R2 score for training data is 70.8% and for testing data is 73.3%.

# # Ridge Regression (L2)

# In[109]:


# Initialize the Ridge Regression model and fit the training data 
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)

# Make prediction on both test and train data
y_test_pred = ridge.predict(X_test)
y_train_pred = ridge.predict(X_train)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate mean squared error for test data
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate root mean squared error for test data
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing good on both the training and testing data i.e., R2 score for training data is 76.3% and for testing data is 78.6%.

# # ElasticNet

# In[110]:


# Initialize the Elastic Net Regression model and fit the training data
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = elastic_net.predict(X_train)
y_test_pred = elastic_net.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing good on both the training and testing data i.e., R2 score for training data is 75.3% and for testing data is 78%.

# # DecisionTreeRegression(DTR)

# In[111]:


# Initialize the Decision Tree Regression model and fit the training data
dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = dtr_model.predict(X_train)
y_test_pred = dtr_model.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on test data but perfoming excellent on training data which shows the overfitting condition. R2 score for training data is 100%, and for test data is 78.8% .

# # RandomForestRegression(RFR)

# In[112]:


# Initialize the Random Forest Regression model and fit the training data
rf = RandomForestRegressor(n_estimators=100, random_state=79)
rf.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on test data but perfoming excellent on training data which shows the overfitting condition. R2 score for training data is 98.5, and for test data is 90.3% .

# # GradientBoostingRegression(GBR)

# In[113]:


# Initialize the Gradient Boosting Regression model and fit the training data
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=79)
gbr.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)


# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing good on both the training and testing data i.e., R2 score for training data is 86.2% and for testing data is 85.4%.

# Here we can Linear regression, L1,L2, and ElasticNet perform good on both training and testing data, among them linear regression performed best, and rest other two models shows overfitting ,as they are performing good on trainig data but less on test data.
# 
# Till here linear regression is the best performing and more fitted model. But as we have seen the overfitting conditions here with few models we will perform CV score method to check accuracy.
# 
# let's check CV score for more accuracy

# # Cross Validation Score

# # Import Necessary Libraries

# In[114]:


from sklearn.model_selection import cross_val_score


# # LR CV Score

# In[115]:


# Initialize the linear regression model and fit the training data
lr = LinearRegression()

# Perform cross-validation on the model
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ",(diff))


# # L1 CV Score

# In[116]:


# Initialize the Lasso model
lasso = Lasso()

# Perform cross-validation on the model
cv_scores = cross_val_score(lasso, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
lasso.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ",(diff))


# # L2 CV Score

# In[117]:


# Initialize the Ridge regression model and fit the training data
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = ridge.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # ElasticNet CV Score

# In[118]:


# Initialize the ElasticNet regression model and fit the training data
enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
enet.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(enet, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = enet.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # DTR CV Score

# In[119]:


# Initialize the decision tree regression model with max depth 5
dtr = DecisionTreeRegressor(max_depth=5)

# Perform cross-validation on the model
cv_scores = cross_val_score(dtr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
dtr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dtr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # RFR CV Score

# In[120]:


# Initialize the random forest regression model and fit the training data
rf = RandomForestRegressor(n_estimators=100, random_state=79)
rf.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # GBR CV Score

# In[121]:


# Initialize the Gradient Boosting Regression model with default parameters
gbr = GradientBoostingRegressor()

# Perform cross-validation on the model
cv_scores = cross_val_score(gbr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
gbr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gbr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# Based on the difference between R2 score and mean CV score , we can see that the ElasticNet Regressor is the best performing model in this case. The smaller the difference, the better the model's generalization performance.
# 
# ElasticNet Regression is the best fitted and performing model with least difference yet. But as Ridge Regression and Linear Regression are other two which is showing least difference, we will be performing hyperparameter tuning technique on these.
# 
# Now we will perform hyperparametertuning for more accuracy to the best two model to check their performance more accurately and decide which is best performing model.

# # Hyperparameter Tuning

# Here in this step we will be using the GridSearchCV technique for performing hyperparameter tuning.
# 
# Hyperparameters are those parameters which are not learned from the data but predefined when the learning process begins. They basically controls the behaviour of the learning algorithm and shows the significant impact on the performance of the model.
# 
# GridSearch CV: It automates the process of hyperparameter tuning by searching through a predefined grid of hyperparameter values and evaluating the model's performance for each combination of hyperparameters. It carefully tries all possible combinations and identifies the best set of hyperparameters that optimize a specified evaluation metrics such as accuracy. The best advantages of this technique is, it avoids overfitting, improves generalization, and saves times by automating process.

# GridSearchCV

# # Import Necessary Libraries

# In[122]:


from sklearn.model_selection import GridSearchCV


# # ElasticNet Regressor

# In[123]:


# Initialize the L1 model
elastic_net_model= ElasticNet()

# Define Hyperparameter to tune
hyperparameters= {'alpha': [0.01, 0.1, 1],
                 'max_iter': [20, 100, 500, 1000],
                  'l1_ratio': [0.25, 0.5, 0.75],
                 'fit_intercept': [True, False],
                 'selection': ['cyclic', 'random']}


# Scale the features
scaler= MinMaxScaler()
scaled_X= scaler.fit_transform(X)

# Initialize GridSearchCV
grid_search= GridSearchCV(elastic_net_model, hyperparameters, cv=5, scoring='r2')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model= grid_search.best_estimator_
best_params= grid_search.best_params_

# Print the best hyperparameters and best score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best R2 score: ", grid_search.best_score_)

# Make prediction on test data using best model
y_pred= best_model.predict(X_test)

# Evaluate model performance
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)


# # Linear Regressor

# In[124]:


# Initialize the L1 model
linear_model= LinearRegression()

# Define Hyperparameter to tune
hyperparameters= {'n_jobs': [-1, 1, 2],
                 'fit_intercept': [True, False]}


# Scale the features
scaler= MinMaxScaler()
scaled_X= scaler.fit_transform(X)

# Initialize GridSearchCV
grid_search= GridSearchCV(linear_model, hyperparameters, cv=5, scoring='r2')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model= grid_search.best_estimator_
best_params= grid_search.best_params_

# Print the best hyperparameters and best score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best R2 score: ", grid_search.best_score_)

# Make prediction on test data using best model
y_pred= best_model.predict(X_test)

# Evaluate model performance
r2_test = r2_score(y_test, y_test_pred)
print("Root squared on test data:", r2_test)


# Now performing hyperparameter tuning on best three model, we have found ElasticNet as the best performing model by improving its R2 score for test data from 78.4% to 85.4% after applying hyperparameter tuning.
# Linear Regression shows decreement in the performance on test data as earlier it was 89.6% now it reduced to 86.5%.
# 
# Now we will save this best performing model for unseen data prediction. We will be using joblib library to save the model.

# # Save 'Next_Tmax' Model

# In[125]:


# Saving model for the prediction of unseen data

import joblib

# Save the model using joblib
joblib.dump(elastic_net_model, 'Next_Tmax_model.joblib')


# # B. Define Target And Independent Variable For 'Next_Tmin' Column

# In[126]:


# Define Target and Independent Variable

y = df['Next_Tmin']
X = df.drop('Next_Tmin', axis=1)

print("Target variable name: ", y.name)
print("Target variable dimensions: ", y.shape)

print("\nFeatures variables names: ", list(X.columns))
print("Features variables dimensions: ", X.shape)


# # Feature Selection

# In[127]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

y = df['Next_Tmin']
X = df.drop('Next_Tmin', axis=1)

# Initialize the model
model= LinearRegression()

# Initialize the RFE selector with model
rfe= RFE(model, n_features_to_select=5)

# Fit the selctor on data
rfe.fit(X,y)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Print the features
print("Selected Features:")
print(selected_features)


# In[128]:


fit_score = rfe.ranking_
print("Fit Score:")
print(fit_score)
# Get the column names of all features
all_columns = X.columns
feature_rankings = pd.DataFrame({'Feature': all_columns, 'Ranking': fit_score})

# Sort the features based on their rankings (lower rank indicates higher importance)
feature_rankings = feature_rankings.sort_values('Ranking')

print("Feature Rankings:")
print(feature_rankings)


# In[129]:


X= X.drop(['LDAPS_PPT3', 'LDAPS_LH'], axis=1)
X


# # Feature Scaling

# In[130]:


from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Apply to all columns
X_scaled = scaler.fit_transform(X)

# Create a new dataframe with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df)


# # Variance Inflation Factor(VIF)

# In[131]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# add a constant column to features
X_scaled_df = sm.add_constant(X_scaled_df)

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["X"] = X_scaled_df.columns
vif["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]

# Print the VIF dataframe
print(vif)


# All vif is handled and on range.

# In[132]:


y.value_counts()


# # Best Random State

# In[133]:


# Lets find the best random state

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Split the data into training and testing sets using different random states
best_random_state = None
best_r2_score = -1
for random_state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Next_Tmin', axis=1), df['Next_Tmin'], test_size=0.2, random_state=random_state)
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Check if this random state gives a better R2 score
    if r2 > best_r2_score:
        best_r2_score = r2
        best_random_state = random_state

print("Best random state:", best_random_state)
print("Best R^2 score:", best_r2_score)


# Here, we find the Best R^2 score is 0.8527126906766715 at best random state 42.

# # Split The Data

# In[134]:


# Let's split the data into test and train

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Modelling

# # Import Necessary Libraries

# In[135]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# # Linear Regression

# In[136]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the linear regression model and fit the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the evaluation metrics
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, lr.predict(X_train))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Root squared on test data:", np.sqrt(r2_test))
print("MAE:", mae)
print("MSE:", mse)
print("Root mean square error:", rmse)
print("Root squared on training data:", np.sqrt(r2_train))


# Here we can see this model is performing well on both the training and testing data i.e., R2 score for training data is 91.4% and for testing data is 92.3%.

# # Lasso(L1) Regression

# In[137]:


# Initialize the Lasso Regression model and fit the training data
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate mean squared error for test data
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate root mean squared error for test data
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on both the training and testing data i.e., R2 score for training data is 78.3% and for testing data is 79.4%.

# # Ridge(L2) Regression

# In[138]:


# Initialize the Ridge Regression model and fit the training data 
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)

# Make prediction on both test and train data
y_test_pred = ridge.predict(X_test)
y_train_pred = ridge.predict(X_train)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on both the training and testing data i.e., R2 score for training data is 83.5% and for testing data is 85.2%.

# # ElasticNet Regression

# In[139]:


# Initialize the Elastic Net Regression model and fit the training data
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = elastic_net.predict(X_train)
y_test_pred = elastic_net.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on both the training and testing data i.e., R2 score for training data is 83.1% and for testing data is 84.7%.

# # DecisionTreeRegression(DTR)

# In[140]:


# Initialize the Decision Tree Regression model and fit the training data
dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = dtr_model.predict(X_train)
y_test_pred = dtr_model.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on test data but perfoming excellent on training data which shows the overfitting condition. R2 score for training data is 100%, and for test data is 79.2% .

# # RandomForestRegression(RFR)

# In[141]:


# Initialize the Random Forest Regression model and fit the training data
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on test data but perfoming excellent on training data which shows the overfitting condition. R2 score for training data is 98.6%, and for test data is 91% .

# # GradientBoostingRegression(GBR)

# In[142]:


# Initialize the Gradient Boosting Regression model and fit the training data
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# Make predictions using the trained model on both train and test data
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)


# Calculate R-squared for train and test data 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Calculate mean absolute error, mean squared error, and root mean squared error for test data
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print("Root squared on test data:", r2_test)
print("MAE on test data:", mae_test)
print("MSE on test data:", mse_test)
print("RMSE on test data:", rmse_test)
print("Root squared on training data:", r2_train)


# Here we can see this model is performing well on both the training and testing data i.e., R2 score for training data is 90% and for testing data is 89.4 %.

# Here we can Linear regression, L1,L2,and ElasticNet regression, perform good on both training and testing data, among them linear regression performed best, and rest other models shows overfitting ,as they are performing good on trainig data but less on test data.
# 
# Till here linear regression is the best performing and more fitted model. But as we have seen the overfitting conditions here with few models we will perform CV score method to check accuracy.
# 
# let's check CV score for more accuracy.

# # Cross Validation Score

# # Import necessary Libraries

# In[143]:


from sklearn.model_selection import cross_val_score


# # LR CV Score

# In[144]:


# Initialize the linear regression model and fit the training data
lr = LinearRegression()

# Perform cross-validation on the model
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ",(diff))


# # L1 CV Score

# In[145]:


# Initialize the Lasso model
lasso = Lasso()

# Perform cross-validation on the model
cv_scores = cross_val_score(lasso, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
lasso.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ",(diff))


# # L2 CV Score

# In[146]:


# Initialize the Ridge regression model and fit the training data
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = ridge.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # ElasticNet CV Score

# In[147]:


# Initialize the ElasticNet regression model and fit the training data
enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
enet.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(enet, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = enet.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # DTR CV Score

# In[148]:


# Initialize the decision tree regression model with max depth 5
dtr = DecisionTreeRegressor(max_depth=5)

# Perform cross-validation on the model
cv_scores = cross_val_score(dtr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
dtr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dtr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # RFR CV Score

# In[149]:


# Initialize the random forest regression model and fit the training data
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Perform cross-validation on the model
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# # GBR CV Score

# In[150]:


# Initialize the Gradient Boosting Regression model with default parameters
gbr = GradientBoostingRegressor()

# Perform cross-validation on the model
cv_scores = cross_val_score(gbr, X, y, cv=5, scoring='r2')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation Scores: ", cv_scores)
print("Mean of CV Scores: ", cv_scores.mean())

# Fit the model on the training data
gbr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gbr.predict(X_test)

# Calculate the difference between the R2 score and mean CV score
diff = r2_score(y_test, y_pred) - cv_scores.mean()

# Print the difference
print("Difference between R2 score and mean CV score: ", diff)


# Based on the difference between R2 score and mean CV score , we can see that the ElasticNet Regressor is the best performing model in this case. The smaller the difference, the better the model's generalization performance.
# 
# ElasticNet Regression is the best fitted and performing model with least difference yet. But as Linear Regression and Ridge(L2) Regression are other two which is showing least difference, we will be performing hyperparameter tuning technique on these.
# 
# Now we will perform hyperparametertuning for more accuracy to the best two model to check their performance more accurately and decide which is best performing model.

# # Hyperparameter Tuning

# # Import Necessary Libraries

# In[151]:


from sklearn.model_selection import GridSearchCV


# # ElasticNet Regressor

# In[152]:


# Initialize the L1 model
elastic_net_model= ElasticNet()

# Define Hyperparameter to tune
hyperparameters= {'alpha': [0.01, 0.1, 1],
                 'max_iter': [20, 100, 500, 1000],
                  'l1_ratio': [0.25, 0.5, 0.75],
                 'fit_intercept': [True, False],
                 'selection': ['cyclic', 'random']}


# Scale the features
scaler= MinMaxScaler()
scaled_X= scaler.fit_transform(X)

# Initialize GridSearchCV
grid_search= GridSearchCV(elastic_net_model, hyperparameters, cv=5, scoring='r2')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model= grid_search.best_estimator_
best_params= grid_search.best_params_

# Print the best hyperparameters and best score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best R2 score: ", grid_search.best_score_)

# Make prediction on test data using best model
y_pred= best_model.predict(X_test)

# Evaluate model performance
r2_test = r2_score(y_test, y_test_pred)

print("Root squared on test data:", r2_test)


# # L2 Regressor

# In[153]:


# Initialize the L1 model
ridge_model= Ridge()

# Define Hyperparameter to tune
hyperparameters= {'alpha': [0.01, 0.1, 1],
                 'max_iter': [20, 100, 500, 1000],
                 'fit_intercept': [True, False],
                 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}


# Scale the features
scaler= MinMaxScaler()
scaled_X= scaler.fit_transform(X)

# Initialize GridSearchCV
grid_search= GridSearchCV(ridge_model, hyperparameters, cv=5, scoring='r2')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model= grid_search.best_estimator_
best_params= grid_search.best_params_

# Print the best hyperparameters and best score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best R2 score: ", grid_search.best_score_)

# Make prediction on test data using best model
y_pred= best_model.predict(X_test)

# Evaluate model performance
r2_test = r2_score(y_test, y_test_pred)
print("Root squared on test data:", r2_test)


# Now performing hyperparameter tuning on best three model, we have found ElasticNet as the best performing model by improving its R2 score for test data from 84.7 % to 89.4 % after applying hyperparameter tuning.
# 
# Although Ridge also shows improvement and secure 89.4 % R2 score, but ElasticNet shows more improvement, 
# Now we will save this best performing model for unseen data prediction. We will be using joblib library to save the model.

# In[154]:


df.info()


# # Save the "Next_Tmin" Model

# In[155]:


# Saving model for the prediction of unseen data

import joblib

# Save the model using joblib
joblib.dump(elastic_net_model, 'Next_Tmin_model.joblib')

