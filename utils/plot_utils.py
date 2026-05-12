import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Brand colors
JEPI_COLOR = "#f97316"
SPY_COLOR = "#3b82f6"
AGG_COLOR = "#94a3b8"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#eab308"

BG_COLOR = "#0f172a"
SECONDARY_BG = "#1e293b"
TEXT_COLOR = "#f1f5f9"
GRID_COLOR = "#334155"

ASSET_COLORS = {
    "JEPI": JEPI_COLOR,
    "SPY": SPY_COLOR,
    "AGG": AGG_COLOR,
    "Cash": "#a78bfa",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=SECONDARY_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=13),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
    ),
    margin=dict(l=60, r=30, t=60, b=60),
    hovermode="x unified",
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def styled_line(df: pd.DataFrame, x: str, y: list[str], title: str,
                colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(y):
        color = (colors[i] if colors else ASSET_COLORS.get(col, SPY_COLOR))
        fig.add_trace(go.Scatter(
            x=df[x] if x in df.columns else df.index,
            y=df[col],
            name=col,
            line=dict(color=color, width=2.5),
            mode="lines",
        ))
    fig.update_layout(title=title)
    return _apply_theme(fig)


def styled_scatter(df: pd.DataFrame, x: str, y: str, title: str,
                   color: str = JEPI_COLOR, text: str | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df[x],
        y=df[y],
        mode="markers+text" if text else "markers",
        text=df[text] if text else None,
        marker=dict(color=color, size=8, opacity=0.8),
    ))
    fig.update_layout(title=title)
    return _apply_theme(fig)


def styled_bar(df: pd.DataFrame, x: str, y: str | list[str], title: str,
               colors: list[str] | None = None, barmode: str = "group") -> go.Figure:
    fig = go.Figure()
    ys = [y] if isinstance(y, str) else y
    for i, col in enumerate(ys):
        color = (colors[i] if colors else ASSET_COLORS.get(col, JEPI_COLOR))
        fig.add_trace(go.Bar(
            x=df[x] if x in df.columns else df.index,
            y=df[col],
            name=col,
            marker_color=color,
        ))
    fig.update_layout(title=title, barmode=barmode)
    return _apply_theme(fig)


def styled_area(df: pd.DataFrame, x: str, y: list[str], title: str,
                colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(y):
        color = (colors[i] if colors else ASSET_COLORS.get(col, SPY_COLOR))
        fig.add_trace(go.Scatter(
            x=df[x] if x in df.columns else df.index,
            y=df[col],
            name=col,
            fill="tozeroy",
            line=dict(color=color, width=1.5),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color
                       else color + "26",
            mode="lines",
        ))
    fig.update_layout(title=title)
    return _apply_theme(fig)


def add_annotation(fig: go.Figure, x, y, text: str, ax: int = 30, ay: int = -40) -> go.Figure:
    fig.add_annotation(
        x=x, y=y, text=text,
        showarrow=True, arrowhead=2, arrowcolor=TEXT_COLOR,
        arrowwidth=1.5, ax=ax, ay=ay,
        font=dict(color=TEXT_COLOR, size=11),
        bgcolor=SECONDARY_BG,
        bordercolor=GRID_COLOR,
        borderwidth=1,
    )
    return fig
