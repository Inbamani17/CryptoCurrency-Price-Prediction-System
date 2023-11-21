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
from prophet import Prophet

file_path_lstm = r"C:\Users\manii\OneDrive\Desktop\Project Phase 1\Dataset\coin_Ethereum.csv"
file_path_prophet = r"C:\Users\manii\OneDrive\Desktop\Project Phase 1\Dataset\coin_Ethereum.csv"

df_lstm = pd.read_csv(file_path_lstm)
df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
df_lstm = df_lstm[['Date', 'Open']]
df_lstm.columns = ['ds', 'y']

scaler = MinMaxScaler()
df_lstm['y_scaled'] = scaler.fit_transform(df_lstm[['y']])
X = df_lstm['y_scaled'].values
X = X.reshape(-1, 1, 1)

def create_lstm_model(input_dim):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, input_dim)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

lstm_model = create_lstm_model(input_dim=1)
lstm_model.fit(X, df_lstm['y_scaled'], epochs=50, batch_size=1, verbose=2)

df_prophet = pd.read_csv(file_path_prophet)
df_prophet.dropna(inplace=True)
df_prophet.reset_index(drop=True)
df_prophet = df_prophet[["Date", "Open"]]
df_prophet.columns = ['ds', 'y']
df_prophet['x1'] = 0
df_prophet['x2'] = 0
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

m = Prophet()
m.fit(df_prophet)
num_data_points_prophet = len(df_prophet)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div(
                style={'padding': '5px'}
            ),
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1("Prophet Analysis of Ethereum", className="text-center mx-auto",),
                style={'color': '#444444', 'border': '5px solid #444444', 'padding': '5px'}
            ),
            width={"size": 6, "offset": 3}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.DatePickerRange(
                id='date-range-picker-prophet',
                start_date=df_prophet['ds'].min(),
                end_date=df_prophet['ds'].max(),
            ),
            width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='forecast-graph-prophet'), width=12),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                style={'padding': '5px'}
            ),
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1("LSTM Analysis of Ethereum", className="text-center mx-auto",),
                style={'color': '#444444', 'border': '5px solid #444444', 'padding': '5px'}
            ),
            width={"size": 6, "offset": 3}
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.DatePickerRange(
                id='date-range-picker-lstm',
                start_date=df_lstm['ds'].min(),
                end_date=df_lstm['ds'].max(),
            ),
            width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='forecast-graph-lstm'), width=12),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                style={'padding': '5px'}
            ),
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button("Proceed to Dashboard", href="https://app.powerbi.com/groups/me/reports/9aae6b55-0598-4d4d-90a0-0fb61ceea71d/ReportSectiona8668ab82475f4c7f876?experience=power-bi", target="_blank", color="primary", className="mr-2"),
            width=12
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                style={'padding': '5px'}
            ),
        ),
    ]),
], style={'backgroundColor': '#90bdf0'},
    fluid=True
)

@app.callback(
    Output('forecast-graph-lstm', 'figure'),
    [Input('date-range-picker-lstm', 'start_date'),
     Input('date-range-picker-lstm', 'end_date')]
)
def update_forecast_graph_lstm(start_date, end_date):
    if start_date is None or end_date is None:
        return go.Figure()

    start_date = datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')

    future_lstm = df_lstm[(df_lstm['ds'] >= start_date) & (df_lstm['ds'] <= end_date)]

    lstm_input = np.array(future_lstm['y_scaled']).reshape(-1, 1, 1)

    lstm_predictions = lstm_model.predict(lstm_input)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    fig = go.Figure()

    x_values_lstm = future_lstm['ds']

    fig.add_trace(go.Scatter(x=x_values_lstm, y=lstm_predictions.flatten(), mode='lines + markers', name='LSTM Forecast'))
    fig.update_layout(title='LSTM Cryptocurrency Price Forecast', xaxis_title='Date', yaxis_title='Price')

    return fig

@app.callback(
    Output('forecast-graph-prophet', 'figure'),
    [Input('date-range-picker-prophet', 'start_date'),
     Input('date-range-picker-prophet', 'end_date')]
)
def update_forecast_graph_prophet(start_date, end_date):
    if start_date is None or end_date is None:
        return go.Figure()

    future_prophet = m.make_future_dataframe(periods=num_data_points_prophet)
    future_prophet = future_prophet[(future_prophet['ds'] >= start_date) & (future_prophet['ds'] <= end_date)]

    prediction_prophet = m.predict(future_prophet)

    fig = px.line(df_prophet, x='ds', y='y', title='Prophet Prediction Graph', labels={'y': 'Your Y-axis Label', 'ds': 'Your X-axis Label'}, line_shape='linear', color_discrete_sequence=['red'], markers=True)

    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual'))
    fig.update_traces(line=dict(color='blue'))
    fig.add_trace(go.Scatter(x=prediction_prophet['ds'], y=prediction_prophet['yhat'], mode='lines+markers', name='Predicted'))
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)