# %% [markdown]
# 
# # <p style="background-color:#8b7a5e;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:30px 30px;">Life expectancy Regression Using ANN</p>

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Import Needed Libraries</p>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model #for model visualization


from warnings import filterwarnings
filterwarnings('ignore')

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Read dataset into DataFrame</p>

# %%
df=pd.read_csv('/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv')
df

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Features Metadata</p>

# %%
df.columns

# %% [markdown]
# |Feature | Description
# |------|------------
# |**Country** | countries has been collected from the same WHO data repository website 
# |**Year**|year 2013-2000
# |**Status**|Status of country **Developing** or **Developed**
# |**Life expectancy**|Life Expectancy in age **our target**
# |**Adult Mortality**|Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
# |**infant deaths**|Number of Infant Deaths per 1000 population
# |**Alcohol**|Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
# |**percentage expenditure**|Expenditure on health as a percentage of Gross Domestic Product per capita(%)
# |**Hepatitis B**|Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
# |**Measles**|Measles - number of reported cases per 1000 population
# |**BMI**|Average Body Mass Index of entire population
# |**under-five deaths**|Number of under-five deaths per 1000 population
# |**Polio**|Polio (Pol3) immunization coverage among 1-year-olds (%)
# |**Total expenditure**|General government expenditure on health as a percentage of total government expenditure (%)
# |**Diphtheria**|Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
# |**HIV/AIDS**|Deaths per 1 000 live births HIV/AIDS (0-4 years)
# |**GDP**|Gross Domestic Product per capita (in USD)
# |**Population**|Population of the country
# |**thinness  1-19 years**|Prevalence of thinness among children and adolescents for Age 10 to 19 (% )
# |**thinness 5-9 years**|Prevalence of thinness among children for Age 5 to 9(%)
# |**Income composition of resources**|Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
# |**Schooling**|Number of years of Schooling(years)

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Exploratory Data Analysis</p>

# %% [markdown]
# # **<font color = #208AAE>DataFrame Shape</font>**

# %%
#print number of rows and columns in the dataset

print("Number of Rows:",df.shape[0])
print("Number of Features:",df.shape[1])

# %% [markdown]
# # **<font color = #208AAE>DataFrame Info</font>**

# %%
df.info()

# %% [markdown]
# ### **<font color = #208AAE>From previous result we can put in our mind that:</font>**
# 
# #### **<font color = #8b7a5e>Numerical Features are:</font>**
# 
# - 'Year'
# - 'Life expectancy '
# - 'Adult Mortality'
# - 'infant deaths'
# - 'Alcohol'
# - 'percentage expenditure'
# - 'Hepatitis B'
# - 'Measles '
# - ' BMI '
# - 'under-five deaths '
# - 'Polio'
# - 'Total expenditure'
# - 'Diphtheria '
# - ' HIV/AIDS'
# - 'GDP'
# - 'Population'
# - ' thinness  1-19 years'
# - ' thinness 5-9 years'
# - 'Income composition of resources'
# - 'Schooling'
# 
# #### **<font color = #8b7a5e>Categorical Features (which need encoding later) are:</font>**
# 
# - 'Country'
# - 'Status'
# 
# #### **<font color = #8b7a5e>Columns that have null values (which need handling later):</font>**
# 
# - 'Life expectancy '
# - 'Adult Mortality'
# - 'Alcohol'
# - 'Hepatitis B'
# - ' BMI '
# - 'Polio'
# - 'Total expenditure'
# - 'Diphtheria '
# - 'GDP'
# - 'Population'
# - ' thinness  1-19 years'
# - ' thinness 5-9 years'
# - 'Income composition of resources'
# - 'Schooling'
# 
# #### **<font color = #8b7a5e>Target Feature:</font>**
# 
# - 'Life expectancy '

# %%
df.isnull().sum()

# %% [markdown]
# # **<font color = #208AAE>Statistical Info for Numerical Features</font>**

# %%
df.describe().T

# %% [markdown]
# # **<font color = #208AAE>Exploring Numerical Features</font>**
# 
# - 'Year'
# - 'Life expectancy '
# - 'Adult Mortality'
# - 'infant deaths'
# - 'Alcohol'
# - 'percentage expenditure'
# - 'Hepatitis B'
# - 'Measles '
# - ' BMI '
# - 'under-five deaths '
# - 'Polio'
# - 'Total expenditure'
# - 'Diphtheria '
# - ' HIV/AIDS'
# - 'GDP'
# - 'Population'
# - ' thinness  1-19 years'
# - ' thinness 5-9 years'
# - 'Income composition of resources'
# - 'Schooling'

# %%
df.head()

# %%
# col =[- 'Year', 'Life expectancy ' ,'Adult Mortality','infant deaths', 'Alcohol','percentage expenditure','Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria '
# , ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

cols = df.select_dtypes(include=['float64', 'int64']).columns

for i in cols:
    print(i, df[i].value_counts())
    print('-----------------------------------')

# %% [markdown]
# # **<font color = #208AAE>Exploring Categorical Features</font>**
# 
# #### **<font color = #8b7a5e>'Country' Feature</font>**

# %%
df['Country'].value_counts()

# %% [markdown]
# #### **<font color = #8b7a5e>'Status' Feature</font>**

# %%
df['Status'].value_counts()

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Data Cleaning</p>

# %% [markdown]
# # **<font color = #208AAE>Handling Missing Values</font>**
# - 'Life expectancy '
# - 'Adult Mortality'
# - 'Alcohol'
# - 'Hepatitis B'
# - ' BMI '
# - 'Polio'
# - 'Total expenditure'
# - 'Diphtheria '
# - 'GDP'
# - 'Population'
# - ' thinness  1-19 years'
# - ' thinness 5-9 years'
# - 'Income composition of resources'
# - 'Schooling'

# %%
df.isnull().sum()

# %% [markdown]
# ## **As the number of Missing Values is large I will use Imputing Method to fill null values with mean Value**

# %%
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)

df['Life expectancy']=imputer.fit_transform(df[['Life expectancy']])
df['Adult Mortality']=imputer.fit_transform(df[['Adult Mortality']])
df['Alcohol']=imputer.fit_transform(df[['Alcohol']])
df['Hepatitis B']=imputer.fit_transform(df[['Hepatitis B']])
df['BMI']=imputer.fit_transform(df[['BMI']])
df['Polio']=imputer.fit_transform(df[['Polio']])
df['Total expenditure']=imputer.fit_transform(df[['Total expenditure']])
df['Diphtheria']=imputer.fit_transform(df[['Diphtheria']])
df['GDP']=imputer.fit_transform(df[['GDP']])
df['Population']=imputer.fit_transform(df[['Population']])
df['thinness  1-19 years']=imputer.fit_transform(df[['thinness  1-19 years']])
df['thinness 5-9 years']=imputer.fit_transform(df[['thinness 5-9 years']])
df['Income composition of resources']=imputer.fit_transform(df[['Income composition of resources']])
df['Schooling']=imputer.fit_transform(df[['Schooling']])

# %%
df.isnull().sum()

# %% [markdown]
# # **<font color = #208AAE>Handling Outliers</font>**
# 
# #### **<font color = #8b7a5e>First I will draw boxplot to check outliers</font>**

# %%
# Loop through each column and create a box plot
for column in df.columns:
    fig = px.box(df, y=column, title=f'Box Plot for {column}')
    
    # Update layout to center the title and make it bold
    fig.update_layout(
        title=dict(text=f'<b>Box Plot for {column}</b>', x=0.5),
        boxmode='group'  
    )
    
    fig.show()

# %% [markdown]
# #### **<font color = #8b7a5e>Second, dealing with outliers</font>**

# %%
# Specify the list of columns you want to handle outliers for
cols_to_handle_outliers = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
    'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
    'thinness  1-19 years', 'thinness 5-9 years',
    'Income composition of resources', 'Schooling'
]

# Perform outlier handling for each specified column
for col_name in cols_to_handle_outliers:
    # Calculate quartiles and IQR
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with the mean value of the column
    df[col_name] = np.where((df[col_name] > upper_bound) | (df[col_name] < lower_bound), np.mean(df[col_name]), df[col_name])


# %% [markdown]
# #### **<font color = #8b7a5e>Thirdly I will draw boxplot to check outliers after handling it</font>**

# %%
for column in df.columns:
    fig = px.box(df, y=column, title=f'Box Plot for {column}')
    
    # Update layout to center the title and make it bold
    fig.update_layout(
        title=dict(text=f'<b>Box Plot for {column}</b>', x=0.5),
        boxmode='group'  
    )
    
    fig.show()

# %% [markdown]
# #### **<font color = #8b7a5e>Much better 🥰</font>**

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Data Visualization</p>

# %%
#using plotly draw count plot for df['Year'] 
fig = px.histogram(df, x='Year', color='Year', title='Count Plot for Year')

#update layout to center the title and make it bold
fig.update_layout(
    title=dict(text='<b>Count Plot for Year</b>', x=0.5)
)

fig.show()

# %%
#using plotly draw line plot to show the trend of life expectancy over the years
fig = px.line(df.sort_values(by='Year'), x='Year', y='Life expectancy',animation_frame='Country',animation_group='Year',color='Country', title='Trend of Life Expectancy Over the Years')

#update layout to center the title and make it bold
fig.update_layout(
    title=dict(text='<b>Trend of Life Expectancy Over the Years</b>', x=0.5)
)

fig.show()

# %%
#using plotly draw count plot for df['Status'] and color each bar with different color
fig = px.histogram(df, x='Status', color='Status', title='Count Plot for Status of Country')

#update layout to center the title and make it bold
fig.update_layout(
    title=dict(text='<b>Count Plot for Status of Country</b>', x=0.5)
)

fig.show()

# %% [markdown]
# **Most of the data was collected in 2013**

# %% [markdown]
# ## **Let's see range of Life expectancy for developing and developed Countries**

# %% [markdown]
# ### **<font color = #8b7a5e>Developing</font>**

# %%
# Filter DataFrame for 'Developing' status
developing_df = df[df['Status'] == 'Developing']
# Create a histogram 
fig = px.histogram(developing_df, x='Life expectancy', title="Life Expectancy of Developing Nations")
fig.update_layout(
    xaxis_title='Ages',
    yaxis_title='Number of Countries',
    title_text='<b>Life Expectancy of Developing Countries</b>',
    title_x=0.5,  # Center title
)
fig.show()

# %% [markdown]
# > **We can say the range is from like 41 to 90 in Developing Countries**

# %% [markdown]
# ### **<font color = #8b7a5e>Developed</font>**

# %%
# Filter DataFrame for 'Developing' status
developing_df = df[df['Status'] == 'Developed']

# Create a histogram 
fig = px.histogram(developing_df, x='Life expectancy', title="Life Expectancy of Developing Nations")
fig.update_layout(
    xaxis_title='Ages',
    yaxis_title='Number of Countries',
    title_text='<b>Life Expectancy of Developed Countries</b>',
    title_x=0.5,  # Center title
)
fig.show()

# %% [markdown]
# > **We can say the range is from like 70 to 90 in Developed Countries**

# %%
#using plotly to visualize Average Adult Mortality of Developing and Developed Countries
fig = px.bar(df.groupby('Status', as_index=False).agg({'Adult Mortality':'mean'}), 
             x='Status', 
             y='Adult Mortality', 
             color='Status',
             title='Average Adult Mortality of Developing and Developed Countries')

# Update layout to center the title
fig.update_layout(title_text='<b>Average Adult Mortality of Developing and Developed Countries</b>', title_x=0.5)

# Show the plot
fig.show()  

# %%
#using plotly to visualize Average Infant deaths of Developing and Developed Countries
fig = px.bar(df.groupby('Status', as_index=False).agg({'infant deaths':'mean'}), 
             x='Status', 
             y='infant deaths', 
             color='Status',
             title='Average Infant deaths of Developing and Developed Countries')

# Update layout to center the title
fig.update_layout(title_text='<b>Average Infant deaths of Developing and Developed Countries</b>', title_x=0.5)

# Show the plot
fig.show()  

# %% [markdown]
# > **Developing Countries have highest Adult Mortality and Infant deaths**

# %%
#using plotly to visualize Average Alcohol consumption of Developing and Developed Countries
fig = px.bar(df.groupby('Status', as_index=False).agg({'Alcohol':'mean'}), 
             x='Status', 
             y='Alcohol', 
             color='Status',
             title='Average Alcohol consumption of Developing and Developed Countries')

# Update layout to center the title
fig.update_layout(title_text='<b>Average Alcohol consumption of Developing and Developed Countries</b>', title_x=0.5)

# Show the plot
fig.show()

# %%
#using plotly to visualize scatter ploy of Life expectancy vs Adult Mortality for countries over years
fig = px.scatter(df.sort_values(by='Year'), x='Life expectancy', y='Adult Mortality',color='Country', size='Year', title='Life expectancy vs Adult Mortality for Countries over Years')

# Update layout to center the title
fig.update_layout(title_text='<b>Life expectancy vs Adult Mortality for Countries over Years</b>', title_x=0.5)

# Show the plot
fig.show()

# %%
#using plotly to visualize scatter ploy of Life expectancy vs Infant deaths for Countries over Years
fig = px.scatter(df.sort_values(by='Year'), x='Life expectancy', y='infant deaths',color='Country', size='Year', title='Life expectancy vs Infant deaths for Countries over Years')

# Update layout to center the title
fig.update_layout(title_text='<b>Life expectancy vs Infant deaths for Countries over Years</b>', title_x=0.5)

# Show the plot
fig.show()

# %%
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create a correlation matrix for selected numeric columns
correlation_matrix = df[numeric_columns].corr()

# Plot heatmap using Plotly Express
fig = px.imshow(correlation_matrix,
                labels=dict(x='Columns', y='Columns', color='Correlation'),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='blues',
                title='Heatmap: Correlation Matrix of Numeric Columns')


fig.update_layout(title_text='<b> Heatmap: Correlation Matrix of Numeric Columns </b>', title_x=0.5 ,width=1200 ,height=1200)
fig.show()

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Data Preprocessing</p>

# %% [markdown]
# # **<font color = #208AAE>Handling Categorical Features (encoding)</font>**
# 
# - 'Country'
# - 'Status'

# %% [markdown]
# #### **<font color = #8b7a5e>Values Before Handling</font>**

# %%
df['Country'].unique()

# %%
df['Status'].unique()

# %%
# Columns to apply label encoding
cols_to_encode = ['Country', 'Status']

# Apply label encoding to X
label_encoder_df = LabelEncoder()
for col in cols_to_encode:
    df[col] = label_encoder_df.fit_transform(df[col])

# %% [markdown]
# #### **<font color = #8b7a5e>Values After Handling</font>**

# %%
df['Country'].unique()

# %%
df['Status'].unique()

# %% [markdown]
# # **<font color = #208AAE>Splitting Features from Target</font>**

# %%
X = df.drop('Life expectancy', axis=1)
y = df['Life expectancy']

# %%
X

# %% [markdown]
# # **<font color = #208AAE>Data Scaling</font>**

# %%
# Columns to scale
cols_to_scale = ['Country', 'Year', 'Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure',
       'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
       'thinness  1-19 years', 'thinness 5-9 years',
       'Income composition of resources', 'Schooling']

# Apply Min-Max scaling to the specified columns
scaler = MinMaxScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# %%
X

# %% [markdown]
# # **<font color = #208AAE>Splitting data into Train Test</font>**

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print(f"Shape of X_train is: {X_train.shape}")
print(f"Shape of Y_train is: {y_train.shape}\n")
print(f"Shape of X_test is: {X_test.shape}")
print(f"Shape of Y_test is: {y_test.shape}")

# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Geospatial Analysis of Life Expectancy by Country</p>

# %%
import pandas as pd
import geopandas as gpd
import folium
from folium import GeoJsonTooltip
import branca

# Load your life expectancy data
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")

# Strip trailing spaces from column names
data.columns = data.columns.str.strip()

# Load the world shapefile from your local path
world = gpd.read_file("/Users/kumarutkarsh/Desktop/minor final/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Ensure that country names match in both dataframes for merging
data['Country'] = data['Country'].str.strip()

# Merge life expectancy data with the world map data on country names
world = world.rename(columns={"NAME": "Country"})
geo_data = world.merge(data, on="Country", how="left")

# Convert the merged GeoDataFrame to GeoJSON format for folium compatibility
geo_data_json = geo_data.to_json()

# Define color bins and create a colormap
life_expectancy_bins = [40, 50, 60, 65, 70, 75, 80, 85, 90]
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
colormap = branca.colormap.StepColormap(
    colors=colors,
    vmin=min(life_expectancy_bins),
    vmax=max(life_expectancy_bins),
    caption="Life Expectancy"
)

# Create a base map
m = folium.Map(location=[20, 0], zoom_start=2)

# Add GeoJson layer with custom styling based on life expectancy
def style_function(feature):
    life_expectancy = feature["properties"].get("Life expectancy", None)
    if life_expectancy is None:
        return {"fillColor": "transparent", "color": "transparent", "weight": 0}
    return {
        "fillColor": colormap(life_expectancy),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    }

folium.GeoJson(
    geo_data_json,
    style_function=style_function,
    tooltip=GeoJsonTooltip(
        fields=["Country", "Life expectancy"],
        aliases=["Country:", "Life Expectancy:"],
        localize=True,
    ),
).add_to(m)

# Add the custom color scale to the map
colormap.add_to(m)

# Save or display the map
m.save("life_expectancy_map_custom_colors.html")
m


# %% [markdown]
# 
# # <p style="background-color:#D2D4C8;font-family:ui-rounded;color:#5E4955;font-size:120%;text-align:center;border-radius:10px 10px;">Building ANN Model</p>

# %% [markdown]
# ### **<font color = "purple">Model Structure</font>**

# %%
model = Sequential([
        Dense(64, activation='relu', input_dim=21),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
])

# %% [markdown]
# ### **<font color = "purple">Model Compiling</font>**

# %%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error','mean_squared_error'])

# %% [markdown]
# ### **<font color = "purple">Model Summary</font>**

# %%
model.summary()

# %% [markdown]
# ### **<font color = "purple">Model Visualization</font>**

# %%
# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# %% [markdown]
# ### **<font color = "purple">Model Fitting</font>**

# %%
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)

# %%
# Define needed variables
tr_loss = history.history['loss']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]

Epochs = [i+1 for i in range(len(tr_loss))]
loss_label = f'best epoch= {str(index_loss + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout
plt.show()

# %%
#metrics=['mean_absolute_error','mean_squared_error']

mae = history.history['mean_absolute_error']

acc_loss_df = pd.DataFrame({"Mean Absolute error" : mae,
                            "Loss" : tr_loss,
                            "Epoch" : Epochs})

acc_loss_df.style.bar(color = '#84A9AC',
                      subset = ['Mean Absolute error','Loss'])

# %% [markdown]
# ### **<font color = "purple">Prediction</font>**

# %%
y_pred_ann = model.predict(X_test)

# %%
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score

# %%
R2_ann = r2_score(y_test, y_pred_ann)
print("R2 Score ANN=",R2_ann )

mse_ann = mean_squared_error(y_test, y_pred_ann)
print("MSE ANN: ",mse_ann)

# %%
from sklearn.metrics import f1_score, precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming X_train, y_train, X_test, and y_test are already defined

# Fit the model
# history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)

# Get predictions
y_pred_ann = model.predict(X_test).flatten()

# Define bins and labels for classification
bins = [0, 50, 70, 90, 100]  # Example bins: low, medium, high life expectancy
labels = ['Low', 'Medium', 'High', 'Very High']

# Convert y_test and y_pred_dl to categories
y_test_categories_ann = pd.cut(y_test, bins=bins, labels=labels, right=False)
y_pred_categories_ann = pd.cut(y_pred_ann, bins=bins, labels=labels, right=False)

# Map categories to integers
category_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}

# Convert categories to integers
y_test_encoded_ann = y_test_categories_ann.map(category_map).astype(int)
y_pred_encoded_ann = y_pred_categories_ann.map(category_map).astype(int)

# Calculate F1 Score and Precision Score
f1_ann = f1_score(y_test_encoded_ann, y_pred_encoded_ann, average='weighted')
precision_ann = precision_score(y_test_encoded_ann, y_pred_encoded_ann, average='weighted')

print(f"F1 Score ANN: {f1_ann}")
print(f"Precision Score ANN: {precision_ann}")


# %%
import pandas as pd
import geopandas as gpd
import folium
from folium import GeoJsonTooltip
import branca

# Load the life expectancy data
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")

# Strip trailing spaces from column names
data.columns = data.columns.str.strip()

# Load the world shapefile from your local path
world = gpd.read_file("/Users/kumarutkarsh/Desktop/minor final/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Ensure that country names match in both dataframes for merging
data['Country'] = data['Country'].str.strip()

# Keep only the latest year data for each country
latest_data = data.sort_values('Year', ascending=False).drop_duplicates('Country')

# Merge life expectancy data with the world map data on country names
world = world.rename(columns={"NAME": "Country"})
geo_data = world.merge(latest_data, on="Country", how="left")

# Debug: Print missing countries (if any)
missing_countries = world[~world['Country'].isin(latest_data['Country'])]
if not missing_countries.empty:
    print("Missing Countries:", missing_countries['Country'].tolist())

# Convert the merged GeoDataFrame to GeoJSON format for folium compatibility
geo_data_json = geo_data.to_json()

# Define color bins and create a colormap for life expectancy
life_expectancy_bins = [40, 50, 60, 65, 70, 75, 80, 85, 90]
colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
colormap = branca.colormap.StepColormap(
    colors=colors,
    vmin=min(life_expectancy_bins),
    vmax=max(life_expectancy_bins),
    caption="Life Expectancy"
)

# Create a base map
m = folium.Map(location=[20, 0], zoom_start=2)

# Add GeoJson layer with custom styling based on life expectancy
def style_function(feature):
    life_expectancy = feature["properties"].get("Life expectancy", None)
    if life_expectancy is None:
        return {"fillColor": "grey", "color": "black", "weight": 0.5, "fillOpacity": 0.3}
    return {
        "fillColor": colormap(life_expectancy),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    }

# Define tooltips for displaying country details
tooltip_fields = [
    "Country", "Year", "Life expectancy", "Adult Mortality", "infant deaths",
    "Alcohol", "percentage expenditure", "Hepatitis B", "Measles", "BMI",
    "under-five deaths", "Polio", "Total expenditure", "Diphtheria",
    "HIV/AIDS", "GDP", "Population", "thinness  1-19 years",
    "thinness 5-9 years", "Income composition of resources", "Schooling"
]
tooltip_aliases = [f"{field}:" for field in tooltip_fields]

# Add GeoJson layer with tooltip
folium.GeoJson(
    geo_data_json,
    style_function=style_function,
    tooltip=GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        localize=True,
        sticky=True
    ),
).add_to(m)

# Add the custom color scale to the map
colormap.add_to(m)

# Save or display the map
m.save("life_expectancy_map_with_details.html")
m


# %%
import pandas as pd
import geopandas as gpd
import folium
from folium import GeoJsonTooltip
import branca

# Load the life expectancy data
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")

# Strip trailing spaces from column names
data.columns = data.columns.str.strip()

# Load the world shapefile from your local path
world = gpd.read_file("/Users/kumarutkarsh/Desktop/minor final/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Ensure that country names match in both dataframes for merging
data['Country'] = data['Country'].str.strip()

# Keep only the latest year data for each country
latest_data = data.sort_values('Year', ascending=False).drop_duplicates('Country')

# Merge HIV/AIDS data with the world map data on country names
world = world.rename(columns={"NAME": "Country"})
geo_data = world.merge(latest_data, on="Country", how="left")

# Debug: Print countries without HIV/AIDS data
missing_hiv_data = geo_data[geo_data["HIV/AIDS"].isnull()]
if not missing_hiv_data.empty:
    print("Countries without HIV/AIDS data:", missing_hiv_data["Country"].tolist())

# Fill missing values in the HIV/AIDS column with 0 for visualization
geo_data["HIV/AIDS"] = geo_data["HIV/AIDS"].fillna(0)

# Convert the merged GeoDataFrame to GeoJSON format for folium compatibility
geo_data_json = geo_data.to_json()

# Define color bins and create a colormap for HIV/AIDS prevalence
hiv_bins = [0, 0.1, 1, 5, 10, 20, 30]
colors = ["#e7e1ef", "#d4b9da", "#c994c7", "#df65b0", "#dd1c77", "#980043"]
colormap = branca.colormap.StepColormap(
    colors=colors,
    index=hiv_bins,
    vmin=min(hiv_bins),
    vmax=max(hiv_bins),
    caption="HIV/AIDS Prevalence (%)"
)

# Create a base map
m = folium.Map(location=[20, 0], zoom_start=2)

# Add GeoJson layer with custom styling based on HIV/AIDS prevalence
def style_function(feature):
    hiv_prevalence = feature["properties"].get("HIV/AIDS", None)
    if hiv_prevalence is None or hiv_prevalence == 0:
        return {"fillColor": "grey", "color": "black", "weight": 0.5, "fillOpacity": 0.3}
    return {
        "fillColor": colormap(hiv_prevalence),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    }

# Define tooltips for displaying country details (only HIV/AIDS)
tooltip_fields = ["Country", "Year", "HIV/AIDS"]
tooltip_aliases = ["Country:", "Year:", "HIV/AIDS Prevalence:"]

# Add GeoJson layer with tooltip
folium.GeoJson(
    geo_data_json,
    style_function=style_function,
    tooltip=GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        localize=True,
        sticky=True
    ),
).add_to(m)

# Add the custom color scale to the map
colormap.add_to(m)

# Save or display the map
m.save("hiv_prevalence_map_fixed.html")
m


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")

# Preprocessing: Handling missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
null_cols = data[numeric_cols].columns[data[numeric_cols].isnull().any()]

# Fill null values with column mean
data[null_cols] = data[null_cols].fillna(data[null_cols].mean())

# Feature and Target Variables
X = data.drop(columns=['Life expectancy', 'Country', 'Status'])
y = data['Life expectancy']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Training the Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=32, callbacks=[early_stop], verbose=1)

# Predictions
y_pred_dl = model.predict(X_test).flatten()

# Evaluation
mse = mean_squared_error(y_test, y_pred_dl)
r2_dl = r2_score(y_test, y_pred_dl)

print(f"Neural Network Model - Mean Squared Error: {mse}")
print(f"Neural Network Model - R2 Score: {r2_dl}")

# %%
from sklearn.metrics import f1_score, precision_score


bins = [0, 50, 70, 90, 100]  
labels = ['Low', 'Medium', 'High', 'Very High']
y_test_categories = pd.cut(y_test, bins=bins, labels=labels, right=False)
y_pred_categories = pd.cut(y_pred_dl, bins=bins, labels=labels, right=False)


category_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}


y_test_encoded = y_test_categories.map(category_map).astype(int)
y_pred_encoded = y_pred_categories.map(category_map).astype(int)


f1_dl = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
precision_dl = precision_score(y_test_encoded, y_pred_encoded, average='weighted')




# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")

# Preprocessing: Handle missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
null_cols = data[numeric_cols].columns[data[numeric_cols].isnull().any()]

# Fill null values with column mean
data[null_cols] = data[null_cols].fillna(data[null_cols].mean())

# Feature and Target Variables
X = data.drop(columns=['Life expectancy', 'Country', 'Status'])
y = data['Life expectancy']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train, axis=1)  # Add a "timesteps" dimension
X_test_lstm = np.expand_dims(X_test, axis=1)

# Build the LSTM Model
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.3),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the Model
history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_pred_lstm = model.predict(X_test_lstm).flatten()

# Evaluation
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)

print(f"LSTM Model - Mean Squared Error: {mse_lstm}")
print(f"LSTM Model - R2 Score: {r2_lstm}")

# Classification Metrics (Binned)
bins = [0, 50, 70, 90, 100]
labels = ['Low', 'Medium', 'High', 'Very High']

y_test_categories = pd.cut(y_test, bins=bins, labels=labels, right=False)
y_pred_categories = pd.cut(y_pred_lstm, bins=bins, labels=labels, right=False)

category_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
y_test_encoded = y_test_categories.map(category_map).astype(int)
y_pred_encoded = y_pred_categories.map(category_map).astype(int)

f1_lstm = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
precision_lstm = precision_score(y_test_encoded, y_pred_encoded, average='weighted')

print(f"F1 Score: {f1_lstm}")
print(f"Precision Score: {precision_lstm}")


# %%
print("ANN MODEL DIFFERENT SCORES: ")
print("MSE ANN: ",mse_ann)
print("R2 Score ANN=",R2_ann )
print(f"F1 Score ANN: {f1_ann}")
print(f"Precision Score ANN: {precision_ann}")

print("\n\n")

print("ANOTHER NEURAN NETWORK MODEL")
print(f"Neural Network Model - Mean Squared Error: {mse}")
print(f"Neural Network Model - R2 Score: {r2_dl}")
print(f"F1 Score: {f1_dl}")
print(f"Precision Score: {precision_dl}")

print("\n\n")

print("LSTM MODEL different scores")
print(f"LSTM Model - Mean Squared Error: {mse_lstm}")
print(f"LSTM Model - R2 Score: {r2_lstm}")
print(f"F1 Score: {f1_lstm}")
print(f"Precision Score: {precision_lstm}")



# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("/Users/kumarutkarsh/Desktop/minor final/Life Expectancy Data 2.csv")


print("MODIFIED CODE FOR NEURAL NETWORK USED ABOVE")
print("WITH INCREASED PRECISION AND ANOTHER SCORES")

print("\n\n")

# Preprocessing: Handling missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
null_cols = data[numeric_cols].columns[data[numeric_cols].isnull().any()]
data[null_cols] = data[null_cols].fillna(data[null_cols].mean())

# Feature and Target Variables
X = data.drop(columns=['Life expectancy', 'Country', 'Status'])
y = data['Life expectancy']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Adjusted Neural Network Architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),  # Increased neurons, added L2 regularization
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)  # Regression output
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the Model with Early Stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,  # Increased epochs for better convergence
    batch_size=16,  # Reduced batch size for more granular updates
    verbose=1
)

# Predictions
y_pred_dl = model.predict(X_test).flatten()

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred_dl)
r2_dl = r2_score(y_test, y_pred_dl)

# Binned Classification Metrics (for F1 and Precision)
bins = [0, 50, 70, 90, 100]
labels = ['Low', 'Medium', 'High', 'Very High']

y_test_categories = pd.cut(y_test, bins=bins, labels=labels, right=False)
y_pred_categories = pd.cut(y_pred_dl, bins=bins, labels=labels, right=False)

category_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
y_test_encoded = y_test_categories.map(category_map).astype(int)
y_pred_encoded = y_pred_categories.map(category_map).astype(int)

f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')

print(f"Modified Neural Network - Mean Squared Error: {mse}")
print(f"Modified Neural Network - R2 Score: {r2_dl}")
print(f"F1 Score: {f1}")
print(f"Precision Score: {precision}")


# %%



