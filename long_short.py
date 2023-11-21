import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your cryptocurrency price data (replace with your dataset)
file_path = r"C:\Users\manii\OneDrive\Desktop\Project Phase 1\Dataset\coin_Aave.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Open']]
df.columns = ['ds', 'y']

# Preprocess data for LSTM
scaler = MinMaxScaler()
df['y_scaled'] = scaler.fit_transform(df[['y']])
X = df['y_scaled'].values
X = X.reshape(-1, 1, 1)


# Create LSTM model
def create_lstm_model(input_dim):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, input_dim)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Initialize LSTM model
lstm_model = create_lstm_model(input_dim=1)
lstm_model.fit(X, df['y_scaled'], epochs=50, batch_size=1, verbose=2)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the Dash app layout
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
                    "LSTM PREDICTION DASHBOARD",
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
], style={'backgroundColor': '#90bdf0'},
    fluid=True
)


# Define callback to update the forecast graph
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_forecast_graph(start_date, end_date):
    if start_date is None or end_date is None:
        # Return an empty figure or an error message as needed
        return go.Figure()

    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')

    # Create a dataframe for the selected date range
    future = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

    # Preprocess data for LSTM
    lstm_input = np.array(future['y_scaled']).reshape(-1, 1, 1)

    # Make predictions using LSTM
    lstm_predictions = lstm_model.predict(lstm_input)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Plot the forecasts
    fig = go.Figure()

    # Format date values for the x-axis
    x_values = future['ds']
    fig = px.line(df, x='ds', y='y', title='Prediction Graph',labels={'y': 'Your Y-axis Label', 'ds': 'Your X-axis Label'},  line_shape='linear', color_discrete_sequence=['red'], markers=True)
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.update_traces(line=dict(color='blue'))
    # Add LSTM forecast with formatted dates
    fig.add_trace(go.Scatter(x=x_values, y=lstm_predictions.flatten(), mode='lines + markers', name='LSTM Forecast'))
    fig.update_layout(title='Cryptocurrency Price Forecast', xaxis_title='Date', yaxis_title='Price')

    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)