import plotly.graph_objs as go

def plot_portfolio_performance(data, strategy_results):
    fig = go.Figure()
    for strategy_name, result in strategy_results.items():
        fig.add_trace(go.Scatter(x=data['Date'], y=result['portfolio_values'], mode='lines', name=strategy_name))

    fig.update_layout(title='Portfolio Performance Over Time',
                      xaxis_title='Date',
                      yaxis_title='Portfolio Value')
    fig.show()
