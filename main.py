import pandas as pd
from prophet.plot import plot_components_plotly
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# Initialize the Dash app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load your time series data (replace with your own data)
file_path = r"C:\Users\manii\OneDrive\Desktop\Project Phase 1\Dataset\coin_Bitcoin.csv"
df = pd.read_csv(file_path)
df.dropna(inplace= True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df=df[["Date","Open"]]
df.columns = ['ds','y']
df['x1'] = 0;
df['x2'] = 0;
df['ds'] = pd.to_datetime(df['ds'])
df.plot(x='ds', y='y', figsize=(24, 12))
plt.title('Bitcoin')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Initialize Prophet model (you can customize this further)
train = df.iloc[:len(df)]
test = df.iloc[len(df):]
m = Prophet()
m.fit(df)
num_data_points = len(df)


# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div(

            ),
            width={"size": 6, "offset": 3}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1(
                ),
            ),
            width={"size": 6, "offset": 3}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1(
                    "BITCOIN PREDICTION USING PROPHET",
                    className="text-center mx-auto",
                    id="custom-h1"
                ),
                style={'color': '#444444', 'border': '5px solid #444444', 'padding': '5px'}
            ),
            width={"size": 6, "offset": 3}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.DatePickerRange(
            id='date-range-picker',
            start_date=df['ds'].min(),
            end_date=df['ds'].max(),
            ),
        width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='forecast-graph'), width=20),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button("Open Google", href="https://www.google.com", target="_blank", color="primary", className="mr-2"),
            width=12
        ),
    ]),
], style={'backgroundColor': '#90bdf0'},
    fluid=True
)


# Callback to update the forecast graph
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_forecast_graph(start_date, end_date):
    # Create a future dataframe for the selected date range
    future = m.make_future_dataframe(periods=num_data_points)
    future = future[(future['ds'] >= start_date) & (future['ds'] <= end_date)]
    actual_data = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

    # Return an empty figure in case of no data

    # Make a forecast for the selected date range
    prediction = m.predict(future)

    # Plot the forecast using Plotly Express, including data points
    fig = px.line(df, x='ds', y='y', title='Prediction Graph',labels={'y': 'Your Y-axis Label', 'ds': 'Your X-axis Label'},  line_shape='linear', color_discrete_sequence=['red'], markers=True)

    # Add the actual data as a line plot
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.update_traces(line=dict(color='blue'))
    # Add the forecast data as a line plot
    fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat'], mode='lines+markers', name='Predicted'))
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
