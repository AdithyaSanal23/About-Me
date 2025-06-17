import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Sample Data (replace with your real CSV or Excel source)
data = {
    'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'Region': ['North', 'South', 'East', 'West'] * 25,
    'Category': ['Electronics', 'Furniture', 'Clothing', 'Toys'] * 25,
    'Revenue': [round(abs(x), 2) for x in (1000 * pd.np.random.randn(100) + 5000)]
}
df = pd.DataFrame(data)
df['Month'] = df['Date'].dt.to_period('M').astype(str)

# Calculate KPIs
total_revenue = df['Revenue'].sum()
best_category = df.groupby('Category')['Revenue'].sum().idxmax()

# Create Dashboard
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Customer Sales Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Total Revenue"),
            html.P(f"${total_revenue:,.2f}")
        ], style={'padding': '10px', 'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Best-Selling Category"),
            html.P(best_category)
        ], style={'padding': '10px', 'width': '30%', 'display': 'inline-block'}),
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        figure=px.bar(df.groupby('Category')['Revenue'].sum().reset_index(),
                      x='Category', y='Revenue',
                      title="Revenue by Category")
    ),

    dcc.Graph(
        figure=px.line(df.groupby('Month')['Revenue'].sum().reset_index(),
                       x='Month', y='Revenue',
                       title="Monthly Sales Trend")
    ),

    dcc.Graph(
        figure=px.pie(df, names='Region', values='Revenue',
                      title="Sales by Region")
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
