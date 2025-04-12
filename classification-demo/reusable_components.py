from dash import dcc, html

def slider_components(default_sample=100, default_threshold=0.5):
    return html.Div([
        html.Div([
            html.Label("Sample Size (%)"),
            dcc.Slider(
                id="sample-size-slider",
                min=10,
                max=100,
                step=10,
                value=default_sample,
                marks={i: f"{i}%" for i in range(10, 101, 10)}
            )
        ], style={"margin": "20px"}),
        html.Div([
            html.Label("Threshold"),
            dcc.Slider(
                id="threshold-slider",
                min=0,
                max=1,
                step=0.05,
                value=default_threshold,
                marks={i/10: f"{i/10:.1f}" for i in range(0, 11)}
            )
        ], style={"margin": "20px"}),
        html.Button("Reset to Defaults", id="reset-button", n_clicks=0)
    ])
