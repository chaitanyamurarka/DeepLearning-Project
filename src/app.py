# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import numpy as np
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output

# Define the cost function
def cost(x):
    return x[0]**2 + x[1]**2 + x[2]**2

# Define the gradient of the cost function
def gradient(x):
    return np.array([2*x[0], 2*x[1], 2*x[2]])

# Define the momentum-based gradient descent algorithm
def momentum_gradient_descent(gradient, initial_position, learning_rate, momentum, num_iterations):
    # Initialize velocity to zero
    velocity = np.zeros_like(initial_position)
    # Initialize position
    position = initial_position.copy()
    # Initialize array to store position history
    positions = [position]
    
    for i in range(num_iterations):
        # Calculate gradient
        grad = gradient(position)
        # Update velocity
        velocity = momentum * velocity - learning_rate * grad
        # Update position
        position += velocity
        # Add new position to position history
        positions.append(position.copy())
    
    return np.array(positions)

# Set the initial position, learning rate, momentum, and number of iterations
initial_position_3d = np.array([1.8, 1.0, -0.5])
learning_rate = 0.01
momentum = 0.9
num_iterations = 10

# Run the momentum-based gradient descent algorithm in 3D
positions_3d = momentum_gradient_descent(gradient, initial_position_3d, learning_rate, momentum, num_iterations)

# Define the 2D plot trace for each iteration
data = []
for i in range(num_iterations+1):
    positions_2d = positions_3d[:i+1,:2]
    trace = go.Scatter(x=positions_2d[:,0], y=positions_2d[:,1], mode='lines+markers', name=f'Iteration {i}')
    data.append(trace)

# Define the layout for the 2D plot
layout = go.Layout(title='Momentum-Based Gradient Descent in 2D',
                   xaxis=dict(title='x'),
                   yaxis=dict(title='y'))
fig_2d = go.Figure(data=data, layout=layout)

# Define the 3D plot trace
trace_3d = go.Scatter3d(x=positions_3d[:,0], y=positions_3d[:,1], z=positions_3d[:,2], 
                        mode='lines+markers', line=dict(color='blue', width=5))

# Define the layout for the 3D plot
layout_3d = go.Layout(title='Momentum-Based Gradient Descent in 3D',
                   scene=dict(xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z')))
fig_3d = go.Figure(data=[trace_3d], layout=layout)


# Initialize the JupyterDash app
app = JupyterDash(__name__)
server = app.server

# Define the iterations for the dropdown menu
iteration_options = [{'label': f'Iteration {i}', 'value': i} for i in range(num_iterations+1)]


# Define the callback for updating the 2D graph based on the selected iteration
@app.callback(
    dash.dependencies.Output('2d-graph', 'figure'),
    [dash.dependencies.Input('iteration-dropdown', 'value')]
)
def update_2d_graph(iteration):
    # Define the 2D plot trace for the selected iteration
    positions_2d = positions_3d[:iteration+1,:2]
    data = [go.Scatter(x=positions_2d[:,0], y=positions_2d[:,1], mode='lines+markers', name=f'Iteration {iteration}')]
    
    # Define the layout for the 2D plot
    layout = go.Layout(title=f'Momentum-Based Gradient Descent in 2D (Iteration {iteration})',
                       xaxis=dict(title='x'),
                       yaxis=dict(title='y'))
    fig = go.Figure(data=data, layout=layout)
    
    return fig

# Define the callback for updating the 3D graph based on the selected iteration
@app.callback(
    dash.dependencies.Output('3d-graph', 'figure'),
    [dash.dependencies.Input('iteration-dropdown', 'value')]
)
def update_3d_graph(iteration):
    # Define the 3D plot trace for the selected iteration
    positions_3d_subset = positions_3d[:iteration+1,:]
    trace_3d = go.Scatter3d(x=positions_3d_subset[:,0], y=positions_3d_subset[:,1], z=positions_3d_subset[:,2], 
                            mode='lines+markers', line=dict(color='blue', width=5))
    
    # Define the layout for the 3D plot
    layout = go.Layout(title=f'Momentum-Based Gradient Descent in 3D (Iteration {iteration})',
                       scene=dict(xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z')))
    fig = go.Figure(data=[trace_3d], layout=layout)
    
    return fig

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Momentum-Based Gradient Descent'),
    html.Div(children='''
        A demonstration of momentum-based gradient descent in 2D and 3D.
    '''),
    dcc.Dropdown(
        id='iteration-dropdown',
        options=iteration_options,
        value=num_iterations,
        clearable=False
    ),
    dcc.Graph(
        id='2d-graph',
        figure=fig_2d
    ),
    dcc.Graph(
        id='3d-graph',
        figure=fig_3d
    )
])
# Run the app
app.run_server(mode='external')


# %%



