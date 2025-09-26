import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import os
from PIL import Image
import base64
import io

# Configuration
import trajectory_data, object_localization
TRAJECTORIES_DIR = f"/home/imitlearn/petr_sandbox/saw_ws/src/trajectory_data/trajectories/quantitative_study"
TRAJECTORIES_DIR = f"{trajectory_data.package_path}/trajectories"
CFG_DIR = f"/home/imitlearn/petr_sandbox/saw_ws/src/ILeSiA/franka_risk_aware_learning_from_demonstrations/object_localization/cfg"
CFG_DIR = f"{object_localization.package_path}/cfg"

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Robot Skill Visualization Dashboard"),
    html.Div([
        dcc.Dropdown(
            id='skill-dropdown',
            options=[{'label': f, 'value': f} 
                    for f in os.listdir(TRAJECTORIES_DIR) 
                    if f.endswith('.npz')],
            placeholder="Select Skill File"
        ),
        dcc.Dropdown(
            id='template-dropdown',
            options=[{'label': d, 'value': d} 
                    for d in os.listdir(CFG_DIR) 
                    if os.path.isdir(os.path.join(CFG_DIR, d))],
            placeholder="Select Template"
        )
    ], style={'margin': '20px', 'width': '500px'}),
    
    html.Div([
        dcc.Graph(
            id='3d-trajectory',
            style={'width': '49%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='gripper-plot',
            style={'width': '49%', 'display': 'inline-block'}
        )
    ]),
    
    html.Div([
        html.Div([
            html.H3("Trajectory Point View"),
            html.Img(id='trajectory-image', style={'height': '300px'})
        ], style={'width': '32%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Template Full Image"),
            html.Img(id='template-full-image', style={'height': '300px'})
        ], style={'width': '32%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Template Cropped"),
            html.Img(id='template-cropped-image', style={'height': '300px'})
        ], style={'width': '32%', 'display': 'inline-block'})
    ])
])

def load_skill_data(skill_file):
    """Load skill data from .npz file"""
    data = np.load(os.path.join(TRAJECTORIES_DIR, skill_file))
    return {
        'traj': data['traj.npy'],
        'grip': data['grip.npy'],
        'images': data['img.npy']
    }

def load_template(template_name):
    """Load template images"""
    template_dir = os.path.join(CFG_DIR, template_name)
    return {
        'full': Image.open(os.path.join(template_dir, 'full_image.png')),
        'cropped': Image.open(os.path.join(template_dir, 'template.png'))
    }

def numpy_to_base64_png(img_array):
    """Convert numpy array image to base64"""
    img = Image.fromarray(img_array.astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def numpy_to_base64(img_array):
    """Convert numpy array image to base64"""
    # Handle grayscale images (h, w) or (n, h, w)
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array.astype('uint8'), mode='L')
    elif len(img_array.shape) == 3:
        img = Image.fromarray(img_array.astype('uint8'), mode='L')
    else:
        raise ValueError("Unsupported image shape")
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

@app.callback(
    [Output('3d-trajectory', 'figure'),
     Output('gripper-plot', 'figure')],
    [Input('skill-dropdown', 'value')]
)
def update_skill_visualization(skill_file):
    if not skill_file:
        return go.Figure(), go.Figure()
    
    data = load_skill_data(skill_file)
    traj = data['traj']
    grip = data['grip'].squeeze()
    
    # Prepare point indices for customdata
    point_indices = np.arange(traj.shape[1])
    
    # 3D Trajectory Plot
    trajectory_fig = go.Figure(
        data=[go.Scatter3d(
            x=traj[0,:], 
            y=traj[1,:], 
            z=traj[2,:],
            mode='markers+lines',
            marker=dict(
                size=4,
                color=point_indices,
                colorscale='Viridis',
                colorbar=dict(title='Point Index')
            ),
            line=dict(color='royalblue', width=2),
            customdata=point_indices,
            hovertemplate='<b>Point %{customdata}</b><br>' +
                        'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        )]
    )
    trajectory_fig.update_layout(
        title='End-Effector Trajectory',
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Gripper State Plot
    gripper_fig = go.Figure(
        data=[go.Scatter(
            x=point_indices,
            y=grip,
            mode='lines+markers',
            line=dict(color='green'),
            hovertemplate='Point: %{x}<br>Value: %{y:.2f}<extra></extra>'
        )]
    )
    gripper_fig.update_layout(
        title='Gripper State',
        xaxis_title='Point Index',
        yaxis_title='Gripper opened [m], 0 is closed',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return trajectory_fig, gripper_fig

@app.callback(
    [Output('template-full-image', 'src'),
     Output('template-cropped-image', 'src')],
    [Input('template-dropdown', 'value')]
)
def update_template_images(template_name):
    if not template_name:
        return None, None
    
    template = load_template(template_name)
    full = numpy_to_base64_png(np.array(template['full']))
    cropped = numpy_to_base64_png(np.array(template['cropped']))
    return full, cropped

@app.callback(
    Output('trajectory-image', 'src'),
    [Input('3d-trajectory', 'clickData')],
    [State('skill-dropdown', 'value')]
)
def update_clicked_image(clickData, skill_file):
    if not clickData or not skill_file:
        return None
    
    try:
        point_idx = clickData['points'][0]['pointNumber']
        data = load_skill_data(skill_file)
        img_array = data['images'][point_idx]
        
        # Handle grayscale image (h, w) or (n, h, w)
        if len(img_array.shape) == 3:
            img_array = img_array[0]  # Take first frame if multiple
        
        return numpy_to_base64(img_array)
    except Exception as e:
        print(f"Error updating image: {e}")
        return None

def show_skill(skill_file, port=8090, debug=True):
    """
    Minimal viewer for a single skill (no gripper, no template).
    Uses your existing load_skill_data(...) and numpy_to_base64(...).

    Usage:
        show_skill("my_skill.npz")                # file inside TRAJECTORIES_DIR
        show_skill("/abs/path/to/my_skill.npz")   # absolute path
    """
    # --- Load data (supports filename OR absolute path) ---
    if os.path.isabs(skill_file) or os.path.sep in skill_file:
        data_npz = np.load(skill_file)
        data = {
            'traj': data_npz['traj.npy'],
            'images': data_npz['img.npy'],
        }
        skill_label = os.path.basename(skill_file)
    else:
        data = load_skill_data(skill_file)  # your existing helper (expects filename in TRAJECTORIES_DIR)
        skill_label = skill_file

    traj = data['traj']
    images = data['images']

    # --- Build the same trajectory figure you already have ---
    point_indices = np.arange(traj.shape[1])
    trajectory_fig = go.Figure(
        data=[go.Scatter3d(
            x=traj[0,:],
            y=traj[1,:],
            z=traj[2,:],
            mode='markers+lines',
            marker=dict(
                size=4,
                color=point_indices,
                colorscale='Viridis',
                colorbar=dict(title='Point Index')
            ),
            line=dict(color='royalblue', width=2),
            customdata=point_indices,
            hovertemplate='<b>Point %{customdata}</b><br>' +
                          'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        )]
    )
    trajectory_fig.update_layout(
        title=f'{skill_label} End-Effector Trajectory',
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # --- Minimal Dash app layout: left = trajectory, right = image ---
    mini = dash.Dash(__name__ + "_show_skill")
    mini.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='traj-3d',
                figure=trajectory_fig,
                style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}
            ),
            html.Div([
                html.Img(
                    id='trajectory-image',
                    style={
                        'width': '100%',
                        'maxHeight': '500px',
                        'objectFit': 'contain',
                        'border': '1px solid #ddd',
                        'borderRadius': '8px'
                    }
                ),
                html.Div(id='img-caption', style={'marginTop':'6px','fontFamily':'monospace','fontSize':'12px','color':'#666'})
            ], style={'width': '49%', 'display': 'inline-block', 'paddingLeft': '12px', 'verticalAlign': 'top'})
        ])
    ], style={'padding': '6px', 'background-color': '#FFF'})

    # --- Callback: click waypoint -> show corresponding image on the right ---
    @mini.callback(
        Output('trajectory-image', 'src'),
        Output('img-caption', 'children'),
        Input('traj-3d', 'clickData'),
        prevent_initial_call=False  # starts empty until you click
    )
    def update_clicked_image(clickData):
        if not clickData or not clickData.get('points'):
            return None, "Click a waypoint to view its image."
        try:
            # Use the same customdata index we attached to the points
            point_idx = clickData['points'][0].get('customdata', clickData['points'][0].get('pointNumber', 0))
            point_idx = int(point_idx)

            img_array = images[point_idx]
            # Handle grayscale image (H,W) or (N,H,W) by taking first frame
            if len(img_array.shape) == 3:
                img_array = img_array[0]

            return numpy_to_base64(img_array), f"Point index: {point_idx}"
        except Exception as e:
            print(f"Error updating image: {e}")
            return None, "Failed to render image for this waypoint."

    # --- Run the mini app ---
    # mini.run(debug=debug, host='0.0.0.0', port=port)
    mini.run(
        debug=debug,
        host="0.0.0.0", port=port,
        jupyter_mode="inline",     # display inside the output cell
        jupyter_height=500,        # control iframe height
        jupyter_width="40%",      # optional
        dev_tools_ui=False,         # <-- hide the bottom Dev Tools panel
    )


def run():
    app.run(debug=True, host='0.0.0.0', port=8076)

if __name__ == '__main__':
    run()