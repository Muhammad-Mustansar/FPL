"""
Streamlit Dashboard for FPL AI Manager (no authentication)

High-end, production-style UI for interacting with the FastAPI backend.

Features:
- KPI banner with squad metrics
- Visual pitch view with VERTICAL formation (GK at bottom)
- Interactive squad table with styling
- Sidebar controls for budget, risk, and algorithm selection
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FPL AI Manager Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------------------------

# Default to localhost for development, use environment variable for production
API_BASE_URL = os.getenv("FPL_API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def check_backend_health() -> Dict[str, Any] | None:
    """Ping the backend health endpoint to check availability."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def call_optimize_squad(
    budget: float,
    use_hill_climbing: bool,
    risk_appetite: str,
) -> Dict[str, Any] | None:
    """
    Call the FastAPI /optimize-squad endpoint and return JSON response.
    """
    # Map risk appetite to potential constraints or logic if needed
    # For now, we just pass the standard payload
    payload: Dict[str, Any] = {
        "budget": budget,
        "constraints": None,
        "max_per_club": 3,
        "use_hill_climbing": use_hill_climbing,
        "model_name": "ridge",
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{API_BASE_URL}/optimize-squad",
            json=payload,
            headers=headers,
            timeout=60,
        )
    except requests.RequestException as exc:
        st.error(f"Unable to reach backend: {exc}")
        return None

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", "Unknown error")
        except ValueError:
            detail = resp.text or "Unknown error"
        st.error(f"Backend error ({resp.status_code}): {detail}")
        return None

    return resp.json()


def select_starting_eleven(squad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose 11 players for a 4-4-2 formation (default) or best formation from the 15-player squad.
    Preference is given to highest predicted points within each position.
    """
    if squad_df.empty:
        return squad_df

    df = squad_df.copy()
    if "predicted_points" not in df.columns:
        df["predicted_points"] = 0.0

    def top_n(position: str, n: int) -> pd.DataFrame:
        pos_df = df[df["position"] == position].sort_values(
            "predicted_points", ascending=False
        )
        return pos_df.head(n)

    # Simple logic: Force a 4-4-2 or 3-5-2 based on standard FPL availability
    # Let's try to fill a valid formation: 1 GK, min 3 DEF, min 1 FWD.
    gk = top_n("GK", 1)
    
    # Get remaining best players to fill 10 spots
    outfield = df[df['position'] != 'GK'].sort_values('predicted_points', ascending=False)
    
    # For simplicity in visualization, let's stick to a fixed 4-4-2 if possible,
    # or fallback to best available.
    defs = top_n("DEF", 4)
    mids = top_n("MID", 4)
    fwds = top_n("FWD", 2)
    
    starting = pd.concat([gk, defs, mids, fwds]).drop_duplicates(subset=["id"])
    
    # If we don't have enough players for 4-4-2 (e.g. only 3 Defenders selected by optimizer), fill gaps
    if len(starting) < 11:
        needed = 11 - len(starting)
        remaining = df[~df["id"].isin(starting["id"])]
        extra = remaining.sort_values("predicted_points", ascending=False).head(needed)
        starting = pd.concat([starting, extra])

    return starting


def draw_pitch(starting_eleven: pd.DataFrame) -> plt.Figure:
    """
    Draw a VERTICAL football pitch and place the starting eleven.
    GK at Bottom, FWD at Top.
    """
    # Create figure with vertical aspect ratio
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("#0f172a")

    # Set Field Color (FPL Green)
    ax.set_facecolor('#1a472a') 

    # --- Draw Pitch Lines (Vertical Orientation) ---
    # Full Pitch Border (0 to 100 x, 0 to 100 y)
    plt.plot([0, 100], [0, 0], color='white', linewidth=2)      # Bottom Line (Goal)
    plt.plot([0, 100], [100, 100], color='white', linewidth=2)  # Top Line
    plt.plot([0, 0], [0, 100], color='white', linewidth=2)      # Left Sideline
    plt.plot([100, 100], [0, 100], color='white', linewidth=2)  # Right Sideline
    
    # Penalty Box (Bottom - GK)
    ax.add_patch(patches.Rectangle((20, 0), 60, 16, linewidth=2, edgecolor='white', facecolor='none'))
    # Goal Box (Bottom)
    ax.add_patch(patches.Rectangle((36, 0), 28, 6, linewidth=2, edgecolor='white', facecolor='none'))
    
    # Center Circle (at y=50)
    ax.add_patch(patches.Circle((50, 50), 10, linewidth=2, edgecolor='white', facecolor='none'))
    # Halfway Line
    plt.plot([0, 100], [50, 50], color='white', linewidth=2)

    # --- Plot Players ---
    # Define vertical zones for each position (0 is bottom, 100 is top)
    pos_y_map = {
        'GK': 8,    # Right in front of goal
        'DEF': 25,  # Defensive line
        'MID': 60,  # Midfield line
        'FWD': 85   # Forward line
    }
    
    # Enforce order: GK -> DEF -> MID -> FWD for correct layering/processing
    starting = starting_eleven.copy()
    starting['pos_sort'] = starting['position'].map({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    grouped = starting.sort_values('pos_sort').groupby('position', sort=False)
    
    for pos, players in grouped:
        y = pos_y_map.get(pos, 50)
        count = len(players)
        
        # Calculate x-positions to center them horizontally
        # Evenly space players across the width (0-100)
        # e.g., if 4 players: 20, 40, 60, 80
        width_step = 100 / (count + 1)
        
        for i, (_, player) in enumerate(players.iterrows()):
            x = width_step * (i + 1)
            
            # Data to display
            name = str(player.get("name", ""))
            # Use last name only for cleaner look
            if " " in name:
                name = name.split(" ")[-1]
            else:
                name = name[:10]
                
            points = float(player.get("predicted_points", 0.0))

            # 1. Draw Shirt (Circle)
            # Yellow shirt with black outline
            circle = patches.Circle((x, y), 3.5, edgecolor='black', facecolor='#FFD700', linewidth=1.5, zorder=10)
            ax.add_patch(circle)
            
            # 2. Player Name (Text Box)
            ax.text(x, y - 6, name, 
                    ha="center", va="top", 
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'),
                    zorder=11)
            
            # 3. Points (Small Text below Name)
            ax.text(x, y - 10, f"{points:.1f}", 
                    ha="center", va="top", 
                    fontsize=8, color='#ccffcc', fontweight='bold',
                    zorder=11)

    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105) # Add a little padding
    ax.axis("off")
    
    return fig


def render_kpi_banner(squad_data: Dict[str, Any], budget: float) -> None:
    """Render top KPI banner with st.metric."""
    total_points = squad_data.get("total_predicted_points", 0.0)
    total_cost = squad_data.get("total_cost", 0.0)
    budget_remaining = max(budget - total_cost, 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predicted Points", f"{total_points:.1f}")
    col2.metric("Team Cost (£m)", f"{total_cost:.1f}")
    col3.metric("Budget Remaining (£m)", f"{budget_remaining:.1f}")


def render_squad_table(squad_df: pd.DataFrame) -> None:
    """Render the squad table with styling."""
    if squad_df.empty:
        st.info("No squad data available. Click **Optimize Squad** to generate a team.")
        return

    display_df = squad_df.copy()
    preferred_cols: List[str] = [
        "name",
        "position",
        "team",
        "cost",
        "predicted_points",
        "points_per_game",
    ]
    existing_cols = [c for c in preferred_cols if c in display_df.columns]
    other_cols = [c for c in display_df.columns if c not in existing_cols]
    display_df = display_df[existing_cols + other_cols]

    # Formatting
    st.dataframe(
        display_df,
        column_config={
            "cost": st.column_config.NumberColumn("Cost", format="£%.1f"),
            "predicted_points": st.column_config.NumberColumn("Exp. Points", format="%.1f"),
            "points_per_game": st.column_config.NumberColumn("PPG", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------------------------
# MAIN DASHBOARD LAYOUT (NO AUTH)
# ---------------------------------------------------------------------------

def render_dashboard():
    """Render main dashboard."""
    # Header
    st.markdown(
        """
        ### ⚽ FPL AI Manager Dashboard
        **Intelligent squad optimization powered by machine learning and CSP search.**

        Use the controls in the sidebar to configure your budget and optimization
        strategy, then generate a data-driven FPL squad with visual pitch view and
        rich metrics.
        """.strip()
    )

    health = check_backend_health()
    if not health:
        st.warning(
            "Backend appears to be offline or unreachable. "
            "Ensure the FastAPI service is running on the configured host."
        )
    else:
        st.caption(
            f"Backend status: **{health.get('status', 'unknown')}**, "
            f"models loaded: **{health.get('models_loaded', False)}**"
        )

    # Sidebar controls
    st.sidebar.markdown("### Squad Configuration")
    budget = st.sidebar.slider("Budget (£m)", min_value=80.0, max_value=100.0, value=100.0, step=0.5)

    risk_appetite = st.sidebar.radio(
        "Risk Appetite",
        options=["Safe", "Differential"],
        index=0,
        help="Safe: focus on solid performers. Differential: open to riskier picks.",
    )

    algo_choice = st.sidebar.selectbox(
        "Optimization Algorithm",
        options=["Greedy", "Hill Climbing"],
        index=1,
        help="Greedy is fast; Hill Climbing attempts to further improve the initial team.",
    )

    optimize_button = st.sidebar.button("🔍 Optimize Squad", use_container_width=True)

    # Main content
    if optimize_button:
        with st.spinner("Optimizing squad... this may take a few seconds."):
            use_hill = algo_choice == "Hill Climbing"
            result = call_optimize_squad(budget=budget, use_hill_climbing=use_hill, risk_appetite=risk_appetite)

        if result is not None:
            # KPIs
            render_kpi_banner(result, budget=budget)

            squad_records = result.get("squad", [])
            squad_df = pd.DataFrame(squad_records)

            # Layout: pitch on the left, table on the right
            left_col, right_col = st.columns([1.1, 1.4])

            with left_col:
                st.subheader("Starting XI - Formation View")
                if not squad_df.empty:
                    starting_eleven = select_starting_eleven(squad_df)
                    fig = draw_pitch(starting_eleven)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.warning("Squad data is empty.")

            with right_col:
                st.subheader("Full Squad Details")
                render_squad_table(squad_df)

            # Constraint / validation info
            with st.expander("Validation & Optimization Details", expanded=False):
                st.write(
                    {
                        "is_valid": result.get("is_valid"),
                        "violations": result.get("violations", []),
                        "position_distribution": result.get("position_distribution", {}),
                        "club_distribution": result.get("club_distribution", {}),
                        "optimization_method": result.get("optimization_method"),
                    }
                )
    else:
        st.info(
            "Use the **sidebar** to configure your budget and optimization algorithm, "
            "then click **Optimize Squad** to generate your AI-recommended team."
        )


def main():
    """Entry point for the Streamlit app (no authentication)."""
    render_dashboard()


if __name__ == "__main__":
    main()