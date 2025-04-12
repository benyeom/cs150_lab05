from dash import Dash, dcc, html, Input, Output
import dash
import classification
import reusable_components as drc
import figure as figs

app = Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)
app.title = "Spam Classifier Demo"
server = app.server

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "40px",
                "backgroundColor": "#d3d3d3",
            },
            children=[
                html.H2("Spam Classifier Demo", style={"margin": "0", "padding": "0", "fontSize": "40px"}),
                html.Div("CS-150 Topics. Author: Ben Yeom", style={"margin": "0", "padding": "0", "fontSize": "25px"})
            ]
        ),
        drc.slider_components(),

        html.Div([
            dcc.Graph(id="contour-plot"),
            dcc.Graph(id="roc-curve"),
            html.Div(id="confusion-matrix", style={"marginTop": "20px"})
        ]),
    ],
    style={"paddingBottom": "50px"}  # Extra bottom padding for the page.
)


@app.callback(
    [Output("contour-plot", "figure"),
     Output("roc-curve", "figure"),
     Output("confusion-matrix", "children"),
     Output("sample-size-slider", "value"),
     Output("threshold-slider", "value")],
    [Input("sample-size-slider", "value"),
     Input("threshold-slider", "value"),
     Input("reset-button", "n_clicks")]
)
def update_output(sample_size, threshold, reset_clicks):
    ctx = dash.callback_context
    #reset to defaults when reset-button is clicked.
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == "reset-button":
        sample_size = 100
        threshold = 0.5

    #run the spam classifier to update the ROC curve and confusion matrix.
    y_test, y_pred_class, y_prob, _ = classification.run_classification(
        sample_fraction=sample_size / 100.0,
        threshold=threshold
    )
    roc_fig = figs.create_roc_curve(y_test, y_prob)
    confusion_table = figs.create_confusion_matrix_table(y_test, y_pred_class)

    contour_fig = figs.create_contour_figure(sample_size=sample_size, threshold=threshold)

    return contour_fig, roc_fig, confusion_table, sample_size, threshold


if __name__ == "__main__":
    app.run(debug=True)
