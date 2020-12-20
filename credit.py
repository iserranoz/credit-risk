import streamlit as st 
import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
import plotly_express as px
import plotly.graph_objs as go
import seaborn as sns
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
from sklearn.externals import joblib 

warnings.filterwarnings("ignore")
def data():
	df = pd.read_csv('credit.csv', index_col = 0)
	return df

def hist(data,title, col, color, w =400):
	figure = px.histogram(data, x=col,color_discrete_sequence=[color], title = title)
	figure.update_layout( width =w)
	return figure


def bar(data,x,y, title, color,w = 300):
	fig = px.bar(data, x=data[x].unique(), y=data[y].value_counts(), title = title, color_discrete_sequence=[color])
	fig.update_layout( width =w)
	return fig


def box(data1, name1, data2, name2, x, y, title):
	trace1 = go.Box(x = data1[x], y = data1[y], name = name1)
	trace2 = go.Box(x = data2[x], y = data2[y], name = name2)
	data = [trace1, trace2]
	layout = go.Layout(yaxis = dict(title = title), boxmode = 'group')
	figure = go.Figure(data = data, layout = layout)
	return figure

def barv(data1, name1, data2, name2, x, y, title, w = 400):
	trace1 = go.Bar(x = data1[x].unique(), y = data1[y].value_counts(), name = name1)
	trace2 = go.Bar(x = data2[x].unique(), y = data2[y].value_counts(), name = name2)
	data = [trace1, trace2]
	layout = go.Layout(yaxis = dict(title = title), barmode = 'group', width = w)
	figure = go.Figure(data = data, layout = layout)
	return figure

def sex2(x):
	if x == 'Male':
		x = 1
	else:
		x = 0
	return x

def house2(x):
	if x == 'Own':
		x = 1
	elif x == 'Free':
		x = 0
	else:
		x =2
	return x

def sa2(x):
	if x == 'Little':
		x = 0
	elif x == 'Moderate':
		x = 1
	elif x == 'Quite Rich':
		x = 2
	elif x == 'Rich':
		x = 3
	return x



def job2(x):
	if x == 'Unskilled non resident':
		x = 0
	elif x == 'Unskilled and resident':
		x = 1
	elif x == 'Skilled':
		x = 2
	elif x == 'Highly skilled':
		x = 3
	return x

def pu2(x):
	
	if x == 'Business':
		x = 0
	elif x == 'Car':
		x = 1
	elif x == 'Domestic appliances':
		x = 2
	elif x == 'Education':
		x = 3
	elif x == 'Furniture/equipment':
		x = 4
	elif x == 'Repairs':
		x = 6
	elif x== 'Vacation/others':
		x = 7
	else:
		x = 5
	return x




st.set_page_config(page_title='Credit-Risk')


nav = st.sidebar.selectbox("Menu",['Dataset information','Exploring variables', 'Preprocess and Correlation Data', 'Model'])

if nav == 'Dataset information':

	df = data()
	st.info('Data visualization')
	st.write(df.head(10))
	info = pd.read_csv('info.csv')
	st.info('Data description')
	st.table(info)
	col1 , col2 = st.beta_columns(2)
	col1.info('Data Types')
	col1.code(df.dtypes)
	col2.info('Unique Values')
	col2.code(df.nunique())
	col1.info('Null values')
	col1.code(df.isnull().sum())

if nav == 'Exploring variables':

	df = data()
	good = df[df['Risk'] =='good']
	bad =  df[df['Risk'] =='bad']
	cm = sns.light_palette("green", as_cmap=True)

	with st.beta_expander("Sex"):
		
		col1 , col2 = st.beta_columns([2,1])
		col1.plotly_chart(barv(good, 'Good Credit', bad, 'Bad Credit', 'Sex', 'Sex', 'Sex',450))
		
		col2.plotly_chart(bar(df,'Sex','Sex', 'General Credit', 'blue',300))


	with st.beta_expander("Age"):
		
		col1 , col2= st.beta_columns(2)
		col1.plotly_chart(hist(good,'Good Credit','Age', 'blue'))
		col2.plotly_chart(hist(bad,'Bad Credit', 'Age', 'red'))
		st.plotly_chart(hist(df,'General Credit', 'Age', 'yellow', 800))


	with st.beta_expander("Job"):
		col1 , col2 = st.beta_columns([2,1])
		col1.plotly_chart(barv(good, 'Good Credit', bad, 'Bad Credit', 'Job', 'Job', 'Job',450))
		col2.plotly_chart(bar(df,'Job','Job', 'General Credit', 'blue',300))

	with st.beta_expander("Housing"):
		col1 , col2 = st.beta_columns([2,1])
		col1.plotly_chart(barv(good, 'Good Credit', bad, 'Bad Credit', 'Housing', 'Housing', 'Housing',450))
		col2.plotly_chart(bar(df,'Housing','Housing', 'General Credit', 'blue',300))

	with st.beta_expander("Saving accounts"):
		col1 , col2 = st.beta_columns([2,1])
		col1.plotly_chart(barv(good, 'Good Credit', bad, 'Bad Credit', 'Saving accounts', 'Saving accounts', 'Saving accounts',450))
		df.dropna(axis = 0, inplace = True)
		col2.plotly_chart(bar(df,'Saving accounts','Saving accounts', 'Saving accounts', 'blue',300))


	with st.beta_expander("Credit Amount"):
		
		col1 , col2= st.beta_columns(2)
		col1.plotly_chart(hist(good,'Good Credit','Credit amount', 'blue'))
		col2.plotly_chart(hist(bad,'Bad Credit', 'Credit amount', 'red'))
		st.plotly_chart(hist(df,'General Credit', 'Credit amount', 'yellow', 800))

	with st.beta_expander("Box Plot"):
		
		st.plotly_chart(box(good, 'Good Credit', bad, 'Bad Credit', 'Job', 'Credit amount', 'Credit amount vs Job'))
		st.plotly_chart(box(good, 'Good Credit', bad, 'Bad Credit', 'Job', 'Duration', 'Duration vs Job'))
		st.plotly_chart(box(good, 'Good Credit', bad, 'Bad Credit', 'Saving accounts', 'Age', 'Saving accounts vs Age'))
		st.plotly_chart(box(good, 'Good Credit', bad, 'Bad Credit', 'Sex', 'Credit amount', 'Credit amount vs Sex'))

	with st.beta_expander("Cross Tabs"):
		col1 , col2= st.beta_columns(2)
		col1.write(pd.crosstab(df["Purpose"],df['Sex']).style.background_gradient(cmap = cm))
		col2.write(pd.crosstab(df["Saving accounts"],df['Risk']).style.background_gradient(cmap = cm))
		col1.write(pd.crosstab(df["Purpose"],df['Risk']).style.background_gradient(cmap = cm))
		col2.subheader('')
		col2.subheader('')
		col2.subheader('')
		col2.write(pd.crosstab(df["Housing"],df['Risk']).style.background_gradient(cmap = cm))

		
if nav == 'Preprocess and Correlation Data':
	st.subheader('Preprocessing')
	st.markdown('Original Dataset')
	st.write(data().head())
	st.subheader('')
	df = data()
	st.markdown('Delete Checking acount column and drop null values')
	st.markdown('Apply label encoder for categorical variables')
	df.drop(columns = 'Checking account',axis = 1, inplace = True)
	df.dropna(axis = 0, inplace = True)
	cols = ['Sex', 'Housing', 'Saving accounts', 'Purpose', 'Risk']
	le = LabelEncoder()

	for i in cols:
	    df[i] = le.fit_transform(df[i])

	df['Risk'] = df['Risk'].apply(lambda x: 1 if x==0 else 0 )
	st.write(df.head())
	st.markdown('Sex: male = 1 y female = 0' )
	st.markdown('Housing: Own = 1, free = 0, rent = 2 ' )
	st.markdown('saving acounts: little = 0, moderate = 1 quite rich = 2, rich = 3' )
	st.markdown('Purpose: radio/tv = 5 education = 3, furniture/equipment = 4, car =1, bussines = 0, domestic appliances = 2, repairs = 6, vacation/others = 7' )
	st.markdown('Risk: good = 0 bad = 1' )
	st.subheader('')
	st.subheader('')
	st.subheader('Correlation')
	corr = df.corr()
	mask = np.triu(np.ones_like(corr, dtype = np.bool))
	cmap = sns.diverging_palette(220,10,as_cmap= True)
	f, ax = plt.subplots(figsize = (11,9))
	sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
           square = True, linewidths = 5, cbar_kws = {'shrink':.5}, annot = True)
	st.markdown('Pearson')
	st.pyplot(f)
	
	corr = df.corr(method = 'spearman')
	mask = np.triu(np.ones_like(corr, dtype = np.bool))
	cmap = sns.diverging_palette(220,10,as_cmap= True)
	f, ax = plt.subplots(figsize = (11,9))
	sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
           square = True, linewidths = 5, cbar_kws = {'shrink':.5}, annot = True)
	st.markdown('Spearman')
	st.pyplot(f)

if nav == 'Model':
	st.subheader('XGBosst Classifier')
	st.subheader('')
	st.subheader('')
	col1 , col2 = st.beta_columns(2)
	col1.markdown('XGBClassifier is for categorical target variables. They are usually called classification problems, in this case the model gives us a probability of having bad credit. Below are the evaluation metrics and then you can evaluate the probability of a specific case by filling in the fields')
	col1.info('Train Score   81.79%')
	col1.info('Accuracy Score   78.36%')
	col2.image('xg.png', width = 620, use_column_width = False)
	col2.subheader('')
	

	col1 , col2, col3, col4 = st.beta_columns([1,1,1,1.2])
	model = joblib.load('XGcredit.pkl')
	age = col1.number_input('Enter Age')
	sex = col2.selectbox('Select Sex',('Male', 'Female'))
	job = col4.selectbox('Select Job',('Unskilled non resident', 'Unskilled and resident', 'Skilled', 'Highly skilled'))
	housing = col3.selectbox('Select Housing',('Own', 'Rent', 'Free'))
	accounts = col1.selectbox('Select Saving accounts',('Little', 'Moderate', 'Quite Rich', 'Rich'))
	amount = col2.number_input('Enter Credit Amount')
	duration = col3.number_input('Enter duration in months')
	purpose = col4.selectbox('Select Purpose',('Car', 'Furniture/equipment', 'Radio/TV', 'Domestic appliances', 'Repairs', 'Education', 'Business', 'Vacation/others'))
	if col1.button('Apply'):
		
		col1, col2,col3 = st.beta_columns([1,1,.3])
		sex = sex2(sex)
		job = job2(job)
		housing = house2(housing)
		account = sa2(accounts)
		purpose = pu2(purpose)
		amount = np.log(amount)
		eva = {'Age': [age], 'Sex': [sex], 'Job': [job], 'Housing': [housing], 'Saving accounts': [account], 'Credit amount': [amount], 'Duration':[duration], 'Purpose': [purpose]}
		
		eva = pd.DataFrame(data=eva)
		
		pred = model.predict(eva)
		proba = model.predict_proba(eva)[0][1]

		col1.info('Bad Credit Probability')
		col2.info(str(np.round(proba*100,2))+' %')
		
		

		if proba < .25:
			color1 = col3.color_picker('','#83f500')
		if proba < .50	and proba >.25:
			color2 = col3.color_picker('','#fca000')
		if proba >.50 and proba <.75:
			color3 = col3.color_picker('','#ffd800')
		if proba > .75:
			color4 = col3.color_picker('','#ff0000')






















