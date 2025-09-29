import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import os
from PIL import Image
import base64
import io

import socket, uuid, os, io, base64
from IPython.display import display
from IPython.display import Image as IPyImage


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

def _find_free_port(preferred=0):
    """Return an available TCP port (0 => ask OS to choose)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", preferred))
        return s.getsockname()[1]

def _fig_from_traj(traj):
    """Build the SAME 3D trajectory figure you already use."""
    point_indices = np.arange(traj.shape[1])
    fig = go.Figure(
        data=[go.Scatter3d(
            x=traj[0,:], y=traj[1,:], z=traj[2,:],
            mode='markers+lines',
            marker=dict(size=4, color=point_indices, colorscale='Viridis',
                        colorbar=dict(title='Point Index')),
            line=dict(color='royalblue', width=2),
            customdata=point_indices,
            hovertemplate=('<b>Point %{customdata}</b><br>'
                           'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>')
        )]
    )
    fig.update_layout(
        title='End-Effector Trajectory',
        scene=dict(aspectmode='data', xaxis_title='X',
                   yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def save_trajectory_png(skill_file, out_png="trajectory.png"):
    if os.path.isabs(skill_file) or os.path.sep in skill_file:
        npz = np.load(skill_file); traj = npz['traj.npy']
    else:
        traj = load_skill_data(skill_file)['traj']
    fig = _fig_from_traj(traj)
    fig.write_image(out_png, scale=2)  # needs kaleido
    return out_png

def show_skill(skill_file, port=8090, inline=True, height=520, debug=False):
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
    grip = data['grip'].squeeze()

    # Unique identity for this app instance
    app_id = str(uuid.uuid4())[:8]
    base_path = f"/{app_id}/"
    if port is None:
        port = _find_free_port(0)

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

    grip = np.asarray(grip, dtype=float)
    x = np.asarray(point_indices)

    # Define state and labels
    OPEN_THRESHOLD = 0.04  # anything >= this counts as open
    is_open = grip >= OPEN_THRESHOLD
    state_txt = np.where(is_open, "Open", "Closed")

    transition_idx = np.where(np.diff(is_open.astype(int)) != 0)[0] + 1

    # Compute "time since last change" (in points) for hover
    last_change = np.zeros_like(x, dtype=int)
    if len(x) > 0:
        lc = 0
        for i in range(len(x)):
            if i in transition_idx:
                lc = 0
            last_change[i] = lc
            lc += 1

    band_colors = np.where(is_open, "rgba(0,150,0,0.15)", "rgba(120,120,120,0.15)")
    band_trace = go.Bar(
        x=x,
        y=np.full_like(grip, 0.09, dtype=float),   # a constant tall bar spanning the y-range
        base=0.0,
        marker=dict(color=band_colors, line=dict(width=0)),
        hoverinfo="skip",
        opacity=1.0,
        showlegend=False
    )

    # Step line for the actual measurements
    main_trace = go.Scatter(
        x=x,
        y=grip,
        mode="lines",
        line=dict(width=2),
        line_shape="hv",  # step-like: hold value, then vertical jump
        name="Gripper",
        customdata=np.stack([state_txt, last_change], axis=1),
        hovertemplate=(
            "Point: %{x}<br>"
            "Value: %{y:.2f} m<br>"
            "State: %{customdata[0]}<br>"
            "Since last change: %{customdata[1]} step(s)"
            "<extra></extra>"
        )
    )

    # Markers only at transitions
    transition_trace = go.Scatter(
        x=x[transition_idx] if transition_idx.size else [],
        y=grip[transition_idx] if transition_idx.size else [],
        mode="markers",
        marker=dict(size=8, symbol="circle"),
        name="State change",
        hovertemplate="Point: %{x}<br>Value: %{y:.2f} m<br><b>State changed here</b><extra></extra>"
    )

    # (Optional) tiny rug to emphasize state on the baseline
    rug_trace = go.Scatter(
        x=x,
        y=np.full_like(grip, -0.002, dtype=float),  # just below zero line
        mode="markers",
        marker=dict(size=4, symbol="line-ns"),
        name="State rug",
        hoverinfo="skip",
        showlegend=False,
        opacity=0.6
    )

    # --- Figure ---
    gripper_fig = go.Figure(data=[band_trace, main_trace, transition_trace, rug_trace])

    gripper_fig.update_layout(
        title="",
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode="x unified",
        xaxis=dict(
            title="Point Index",
            rangeslider=dict(visible=True),  # quick zooming
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showgrid=False
        ),
        yaxis=dict(
            title="Gripper [m]",
            range=[-0.01, 0.09],
            tickmode="array",
            tickvals=[0.0, 0.08],
            ticktext=["Closed", "Open"],
            zeroline=True,
            zerolinewidth=1,
            showgrid=True,
            gridwidth=1
        ),
        legend=dict(orientation="h", x=0, y=1.1)
    )

    # Subtle horizontal reference lines (0 and 0.08)
    gripper_fig.add_hline(y=0.0, line_width=1, line_dash="dot")
    gripper_fig.add_hline(y=0.08, line_width=1, line_dash="dot")


    # --- Minimal Dash app layout: left = trajectory, right = image ---
    app = dash.Dash(__name__ + "_show_skill")
    app.layout = html.Div([
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
                html.Div(id='img-caption', style={'marginTop':'6px','fontFamily':'monospace','fontSize':'12px','color':'#666'}),
                dcc.Graph(
                    id="gripper-plot",
                    figure=gripper_fig,
                    style={"height": "240px", "width": "100%", "display": "inline-block"}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'paddingLeft': '12px', 'verticalAlign': 'top'})
        ])
    ], style={'padding': '6px', 'background-color': '#FFF'})

    # --- Callback: click waypoint -> show corresponding image on the right ---
    @app.callback(
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

    # app.run(debug=debug, host='0.0.0.0', port=port)
    run_kwargs = dict(
        debug=debug,
        host="127.0.0.1", port=port,
        jupyter_width="800px",      # optional
        dev_tools_ui=False,         # <-- hide the bottom Dev Tools panel
        dev_tools_hot_reload=False  # avoid hijacking previous iframes
    )

    # try:
    # Dash >= 2.11 supports jupyter_* args
    if inline:
        app.run(jupyter_mode="inline", jupyter_height=height, **run_kwargs)
    else:
        app.run(jupyter_mode="tab", **run_kwargs)
    
    # except TypeError:
    #     # Fallback to JupyterDash if older stack:
    #     from jupyter_dash import JupyterDash
    #     app2 = JupyterDash(name=f"show_skill_{app_id}",
    #                        routes_pathname_prefix=base_path,
    #                        requests_pathname_prefix=base_path)
    #     app2.layout = app.layout
    #     app2.callback_map = app.callback_map
    #     app2.run_server(mode="inline" if inline else "tab",
    #                     height=height if inline else None,
    #                     **run_kwargs)

    # Return the URL in case you want to open it manually too
    return f"http://127.0.0.1:{port}{base_path}"

def run():
    app.run(debug=True, host='0.0.0.0', port=8076)
    

if __name__ == '__main__':
    run()