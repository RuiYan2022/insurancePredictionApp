import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server
# Load and preprocess data
#df = pd.read_csv("C:/Users/RuiYa/MyCourses/data/medicalCostPersonalData/insurance.csv")
df = pd.read_csv("insurance.csv")
# Create single/family status
df['single_family'] = np.where(df.children==0, 'single', 'family')

# Create age groups
bins = [0, 12, 18, 35, 50, 65, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Middle-aged', 'Senior']
df['agegroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Encode categorical variables for model
encoder = LabelEncoder()
cat_cols = ['sex', 'smoker', 'region', 'single_family', 'agegroup']
df_model = df.copy()
for col in cat_cols:
    df_model[col] = encoder.fit_transform(df_model[col])

# Train model
X = df_model[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df_model['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# App layout
app.layout = html.Div([
    html.H1("Insurance Charges Analysis Dashboard", style={'textAlign': 'center'}),
    
    # Visualization Section
    html.Div([
        html.H2("Visualizations"),
        
        # Dropdown for plot selection
        dcc.Dropdown(
            id='plot-selector',
            options=[
                {'label': 'Charges by Family Status and Smoking', 'value': 'family_smoke'},
                {'label': 'Charges by Number of Children', 'value': 'children'},
                {'label': 'Average Charges by Age Group', 'value': 'age_group'},
                {'label': 'BMI vs Charges', 'value': 'bmi_charges'}
            ],
            value='family_smoke'
        ),
        
        # Graph display
        dcc.Graph(id='main-graph'),
    ]),
    
    # Prediction Section
    html.Div([
        html.H2("Insurance Charge Prediction"),
        
        # Input fields
        html.Div([
            html.Label("Age"),
            dcc.Input(id='age-input', type='number', value=30),
            
            html.Label("BMI"),
            dcc.Input(id='bmi-input', type='number', value=25),
            
            html.Label("Children"),
            dcc.Input(id='children-input', type='number', value=0),
            
            html.Label("Sex"),
            dcc.Dropdown(
                id='sex-input',
                options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}],
                value=1
            ),
            
            html.Label("Smoker"),
            dcc.Dropdown(
                id='smoker-input',
                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                value=0
            ),
            
            html.Label("Region"),
            dcc.Dropdown(
                id='region-input',
                options=[
                    {'label': 'Northeast', 'value': 0},
                    {'label': 'Northwest', 'value': 1},
                    {'label': 'Southeast', 'value': 2},
                    {'label': 'Southwest', 'value': 3}
                ],
                value=0
            ),
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),
        
        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output')
    ], style={'marginTop': '20px'})
])

# Callbacks
@app.callback(
    Output('main-graph', 'figure'),
    Input('plot-selector', 'value')
)
def update_graph(selected_plot):
    if selected_plot == 'family_smoke':
        fig = px.box(df, y='charges', x='single_family', color='smoker',
                    title='Insurance Charges by Family Status and Smoking Habit')
    
    elif selected_plot == 'children':
        fig = px.box(df, y='charges', x='children',
                    title='Insurance Charges by Number of Children')
    
    elif selected_plot == 'age_group':
        age_group_charges = df.groupby('agegroup')['charges'].mean().reset_index()
        fig = px.bar(age_group_charges, x='agegroup', y='charges',
                    title='Average Insurance Charges by Age Group')
    
    else:  # bmi_charges
        fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                        title='BMI vs Insurance Charges')
    
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('age-input', 'value'),
     State('bmi-input', 'value'),
     State('children-input', 'value'),
     State('sex-input', 'value'),
     State('smoker-input', 'value'),
     State('region-input', 'value')]
)
def predict_charges(n_clicks, age, bmi, children, sex, smoker, region):
    if n_clicks > 0:
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        return f'Predicted Insurance Charge: ${prediction:,.2f}'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
