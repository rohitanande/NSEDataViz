import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import numpy as np
import textwrap
from datetime import datetime, timedelta
import plotly.graph_objects as go



# Set page config first
st.set_page_config(page_title="Stock Data Viewer", layout="wide")

#@st.cache_data(show_spinner=False)
def load_data():
    df_new = joblib.load('df_new.pkl')
    indices_df = joblib.load('indices_df.pkl')
    return df_new, indices_df

# Load data once and reuse (cached)
df_new, indices_df = load_data()


# Define Nifty 50 stocks for filtering
nifty_50_symbols = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "ITC", "INFY", "INDIGO",
    "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]

# Ensure date column is datetime
df_new['DATE1'] = pd.to_datetime(df_new['DATE1'], errors='coerce')


# Streamlit UI
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Security Wise Data", "Top Gainer/Loser", "Daily Analysis", "Weekly Analysis", "Monthly Analysis", "Index Analysis", "Industry Analysis", "Relative Position"])


with tab1:
    st.header("Security Wise Daily Data")
    st.title("📊 Stock Data Viewer")

    # --- Stock Data Filter Function ---
    def get_stock_data(symbol, start_date, end_date):
        try:
            filtered_stock_df = df_new[
                (df_new['SYMBOL'].str.upper() == symbol.upper()) &
                (df_new['DATE1'] >= pd.to_datetime(start_date)) &
                (df_new['DATE1'] <= pd.to_datetime(end_date))
            ]
            return filtered_stock_df[['SYMBOL', 'DATE1', 'HIGH_PRICE', 'LOW_PRICE', 'LAST_PRICE', 'TTL_TRD_QNTY', 'DELIV_QTY']]
        except Exception as e:
            return pd.DataFrame()

    # --- Index Data Filter Function ---
    def get_indices_data(index_name, start_date, end_date):
        try:
            indices_filtered_df = indices_df.copy()
            indices_filtered_df['Index Date'] = pd.to_datetime(indices_filtered_df['Index Date'], errors='coerce')
            indices_filtered_df = indices_filtered_df.dropna(subset=['Index Date'])

            # Convert inputs to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            filtered_indices_data = indices_filtered_df[
                (indices_filtered_df['Index Name'].str.upper() == index_name.upper()) &
                (indices_filtered_df['Index Date'] >= start_dt) &
                (indices_filtered_df['Index Date'] <= end_dt)
            ]
            return filtered_indices_data
        except Exception as e:
            st.error(f"⚠️ Error in get_indices_data: {e}")
            return pd.DataFrame()

    # --- Safe date conversion helper ---
    def safe_date(d):
        if d is None or (hasattr(d, 'to_pydatetime') and pd.isna(d)):
            return pd.Timestamp.today().date()
        if isinstance(d, pd.Timestamp):
            return d.date()
        if isinstance(d, str):
            try:
                return pd.to_datetime(d).date()
            except Exception:
                return pd.Timestamp.today().date()
        if isinstance(d, (pd.NaT.__class__, type(pd.NaT))):
            return pd.Timestamp.today().date()
        if isinstance(d, datetime):
            return d.date()
        return d

    # --- Date Limits for stocks ---
    min_date = safe_date(df_new['DATE1'].min())
    max_date = safe_date(df_new['DATE1'].max())

    # --- Date Limits for indices ---
    raw_index_min_date = pd.to_datetime(indices_df['Index Date'], errors='coerce').min()
    raw_index_max_date = pd.to_datetime(indices_df['Index Date'], errors='coerce').max()

    index_min_date = safe_date(raw_index_min_date)
    index_max_date = safe_date(raw_index_max_date)

    # Debug prints - can remove after confirming no issues
    print(f"index_min_date: {index_min_date} ({type(index_min_date)})")
    print(f"index_max_date: {index_max_date} ({type(index_max_date)})")


    # --- Initialize Session State ---
    for key in ["symbol", "start_date", "end_date"]:
        if key not in st.session_state:
            if "start" in key:
                st.session_state[key] = min_date
            elif "end" in key:
                st.session_state[key] = max_date
            else:
                st.session_state[key] = ""

    for key in ["index_name", "index_start", "index_end"]:
        if key not in st.session_state:
            if "start" in key:
                st.session_state[key] = index_min_date
            elif "end" in key:
                st.session_state[key] = index_max_date
            else:
                st.session_state[key] = ""

    # Clean up any session state dates to proper datetime.date objects
    for date_key in ["start_date", "end_date", "index_start", "index_end"]:
        val = st.session_state.get(date_key)
        st.session_state[date_key] = safe_date(val)

    # --- Custom CSS ---
    # Load external CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ---------- Stock Data Section ----------
    st.subheader("🔍 Stock Symbol Data")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.text_input("Enter Stock Symbol (e.g., RELIANCE)", key="symbol")
    with col2:
        st.date_input("Start Date", min_value=min_date, max_value=max_date, key="start_date")
    with col3:
        st.date_input("End Date", min_value=min_date, max_value=max_date, key="end_date")

    if st.session_state.symbol and st.button("Get Stock Data"):
        if st.session_state.start_date > st.session_state.end_date:
            st.error("❌ Start date must be before end date.")
        else:
            stock_result_df = get_stock_data(st.session_state.symbol, str(st.session_state.start_date), str(st.session_state.end_date))
            if not stock_result_df.empty:
                st.success(f"✅ Showing data for {st.session_state.symbol.upper()}")
                st.dataframe(stock_result_df)
            else:
                st.warning("⚠️ No stock data found for the given input.")

    # Ensure 'index_start' is within [index_min_date, index_max_date]
    if st.session_state["index_start"] < index_min_date:
        st.session_state["index_start"] = index_min_date
    elif st.session_state["index_start"] > index_max_date:
        st.session_state["index_start"] = index_max_date

    # Ensure 'index_end' is within [index_min_date, index_max_date]
    if st.session_state["index_end"] < index_min_date:
        st.session_state["index_end"] = index_min_date
    elif st.session_state["index_end"] > index_max_date:
        st.session_state["index_end"] = index_max_date


    # ---------- Index Data Section ----------
    st.subheader("📈 Index Data Viewer")
    col4, col5, col6 = st.columns([2, 1, 1])
    with col4:
        st.text_input("Enter Index Name (e.g., Nifty 50, Nifty Bank)", key="index_name")
    with col5:
        st.date_input("Start Date", min_value=index_min_date, max_value=index_max_date, key="index_start")
    with col6:
        st.date_input("End Date", min_value=index_min_date, max_value=index_max_date, key="index_end")

    if st.session_state.index_name and st.button("Get Index Data"):
        if st.session_state.index_start > st.session_state.index_end:
            st.error("❌ Start date must be before end date.")
        else:
            index_result_df = get_indices_data(st.session_state.index_name, str(st.session_state.index_start), str(st.session_state.index_end))

            # Specify columns to show
            columns_to_show = [
                'Index Name',
                'Index Date',
                'Open Index Value',
                'High Index Value',
                'Low Index Value',
                'Closing Index Value',
                'Change(%)',
                'Volume'
            ]

            if not index_result_df.empty:
                # Filter only the columns that exist in the dataframe to avoid errors
                cols_available = [col for col in columns_to_show if col in index_result_df.columns]

                st.success(f"✅ Showing data for {st.session_state.index_name.upper()}")
                st.dataframe(index_result_df[cols_available])
            else:
                st.warning("⚠️ No index data found for the given input.")


# Top Gainer Loser Section
with tab2:
    st.header("Top Gainer/Loser")
    st.markdown("---")
    st.header("📊 Nifty 50 - Top 5 Gainers & Losers")

    selected_date = st.date_input("Select Date to Analyze", value=datetime.today().date(), key="top_movers")

    if st.button("Show Top Gainers and Losers"):
        st.info(f"Analyzing top movers on {selected_date}...")

        df_filtered = df_new[df_new['SYMBOL'].isin(nifty_50_symbols)].copy()
        df_filtered.sort_values(by=['SYMBOL', 'DATE1'], inplace=True)

        # Get data for selected date
        current_day_df = df_filtered[df_filtered['DATE1'] == pd.to_datetime(selected_date)]

        # Get previous trading day for each symbol
        prev_day_df = (
            df_filtered[df_filtered['DATE1'] < pd.to_datetime(selected_date)]
            .groupby('SYMBOL')
            .tail(1)
        )

        # Merge and calculate % change
        merged = pd.merge(
            current_day_df[['SYMBOL', 'LAST_PRICE']],
            prev_day_df[['SYMBOL', 'LAST_PRICE']],
            on='SYMBOL',
            suffixes=('_current', '_prev')
        )

        # Calculate % change based on last price difference
        merged['% Change'] = ((merged['LAST_PRICE_current'] - merged['LAST_PRICE_prev']) / merged['LAST_PRICE_prev']) * 100

        # Sort by numeric % change
        merged = merged.sort_values(by='% Change', ascending=False)

        # Keep unformatted version for chart
        sorted_merged = merged.copy()

        # Format % Change for table display
        merged['% Change'] = merged['% Change'].map("{:.2f}%".format)

        # Get the top 5 gainers and losers
        top_gainers = merged.head(5)
        top_losers = merged.tail(5).sort_values(by='% Change')

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.success("📈 Top 5 Gainers")
            st.dataframe(top_gainers[['SYMBOL', 'LAST_PRICE_prev', 'LAST_PRICE_current', '% Change']].reset_index(drop=True))

        with col2:
            st.error("📉 Top 5 Losers")
            st.dataframe(top_losers[['SYMBOL', 'LAST_PRICE_prev', 'LAST_PRICE_current', '% Change']].reset_index(drop=True))

        # Merge and calculate % change
        merged = pd.merge(
            current_day_df[['SYMBOL', 'LAST_PRICE']],
            prev_day_df[['SYMBOL', 'LAST_PRICE']],
            on='SYMBOL',
            suffixes=('_current', '_prev')
        )

        # Calculate % change based on last price difference
        merged['% Change'] = ((merged['LAST_PRICE_current'] - merged['LAST_PRICE_prev']) / merged['LAST_PRICE_prev']) * 100

        # Drop rows with NaN values in '% Change' (if any)
        merged['% Change'] = pd.to_numeric(merged['% Change'], errors='coerce')
        merged.dropna(subset=['% Change'], inplace=True)

        # Count the number of positive (advances) and negative (declines) stocks
        positive_stocks = len(merged[merged['% Change'] > 0])
        negative_stocks = len(merged[merged['% Change'] < 0])

        # Display the counts
        st.write(f"📊 **Advances (Stocks with Positive % Change)**: {positive_stocks}")
        st.write(f"📊 **Declines (Stocks with Negative % Change)**: {negative_stocks}")


        # Heatmap
        st.markdown("### 🔥 Nifty 50 Heatmap - % Change on Selected Date")

        heatmap_df = sorted_merged[['SYMBOL', '% Change']].copy()
        heatmap_df = heatmap_df.set_index('SYMBOL')

        # Set up grid size: Fewer columns, more rows → better spacing
        grid_rows, grid_cols = 10, 5  # Can change to (5, 10) or (6, 9) etc. for more spacing
        total_cells = grid_rows * grid_cols

        symbols = heatmap_df.index.tolist()
        values = heatmap_df['% Change'].tolist()

        # Pad to fit grid
        symbols += [''] * (total_cells - len(symbols))
        values += [0] * (total_cells - len(values))

        symbols_grid = np.array(symbols).reshape(grid_rows, grid_cols)
        values_grid = np.array(values).reshape(grid_rows, grid_cols)

        fig = px.imshow(
            values_grid,
            labels=dict(color="% Change"),
            x=[f"Col {i+1}" for i in range(grid_cols)],
            y=[f"Row {i+1}" for i in range(grid_rows)],
            color_continuous_scale='RdYlGn',
            text_auto=False,
            aspect="auto"
        )

        fig.update_layout(
            title="Heatmap of % Change for Nifty 50 Stocks",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=750,
            width=1200,  # Wider!
            margin=dict(l=30, r=30, t=60, b=30)
        )

        # Add SYMBOL + % text to each tile
        for i in range(grid_rows):
            for j in range(grid_cols):
                symbol = symbols_grid[i, j]
                value = values_grid[i, j]
                if symbol:
                    fig.add_annotation(
                        text=f"{symbol}<br>{value:.2f}%",
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(size=13, color="black"),
                        xanchor="center",
                        yanchor="middle"
                    )

        st.plotly_chart(fig, use_container_width=False, key="tab2_heatmap_spaced")

    # Top Movers
    st.header("📊 Nifty 50 - Top Movers")
    st.markdown("---")

    # User selects date range
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # Filter only Nifty 50 data and sort
    nifty_df = df_new[df_new["SYMBOL"].isin(nifty_50_symbols)].copy()
    nifty_df.sort_values(by=["SYMBOL", "DATE1"], inplace=True)

    # Prepare list to collect top movers
    top_movers_data = []

    # Generate range of dates
    date_range = pd.date_range(start=start_date, end=end_date)

    for current_date in date_range:
        # Get current day's data
        current_day_df = nifty_df[nifty_df["DATE1"] == pd.to_datetime(current_date)]

        if current_day_df.empty:
            continue

        # Get previous trading day's LAST_PRICE for each symbol
        prev_day_df = (
            nifty_df[nifty_df["DATE1"] < pd.to_datetime(current_date)]
            .groupby("SYMBOL")
            .tail(1)
        )

        # Merge current and previous prices
        merged = pd.merge(
            current_day_df[["SYMBOL", "LAST_PRICE"]],
            prev_day_df[["SYMBOL", "LAST_PRICE"]],
            on="SYMBOL",
            suffixes=("_current", "_prev")
        )

        # Calculate % Change
        merged["% Change"] = ((merged["LAST_PRICE_current"] - merged["LAST_PRICE_prev"]) /
                              merged["LAST_PRICE_prev"]) * 100

        # Drop any bad data
        merged = merged.dropna(subset=["% Change"])
        if merged.empty:
            continue

        # Get top gainer & loser
        merged_sorted = merged.sort_values(by="% Change", ascending=False)
        top_gainer = merged_sorted.iloc[0]["SYMBOL"]
        top_loser = merged_sorted.iloc[-1]["SYMBOL"]

        top_movers_data.append([
            current_date.strftime('%Y-%m-%d'),
            top_gainer,
            top_loser
        ])

    # Create final DataFrame
    top_movers_df = pd.DataFrame(top_movers_data, columns=["Date", "Top Gainer", "Top Loser"])

    # Load and apply CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Plot if data exists
    if not top_movers_df.empty and "Date" in top_movers_df.columns:
        top_movers_df = top_movers_df.reset_index(drop=True)

        # Prepare data for scatter plot
        plot_data = []

        for idx, row in top_movers_df.iterrows():
            date_val = row["Date"]

            plot_data.append({
                "Date": pd.to_datetime(date_val),
                "Symbol": row["Top Gainer"],
                "Type": "Top Gainer"
            })

            plot_data.append({
                "Date": pd.to_datetime(date_val),
                "Symbol": row["Top Loser"],
                "Type": "Top Loser"
            })

        # Create plot DataFrame
        plot_df = pd.DataFrame(plot_data).sort_values(by="Date")

        # Plotly Scatter Plot
        fig = px.scatter(
            plot_df,
            x="Date",
            y="Symbol",
            color="Type",
            text="Symbol",
            title="📈 Top Gainers and Losers Over Time",
            color_discrete_map={
                "Top Gainer": "green",
                "Top Loser": "red"
            },
            height=600
        )

        fig.update_traces(
            textposition='top center',
            marker=dict(size=14, line=dict(width=2, color='DarkSlateGrey'))
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Stock Symbol",
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please select a valid date range to view the top movers chart.")



with tab3:
    st.header("📊 Nifty 50 Stocks List with Volume BO, Delivery Volume BO, % Change, Industry & Breakout Ratio")
    st.markdown("---")

    # Get min/max dates
    min_date = df_new['DATE1'].min().date()
    max_date = df_new['DATE1'].max().date()

    # Date selector
    selected_date = st.date_input(
        "Select Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="single_date_input"
    )

    # Filter only Nifty 50 stocks
    df_filtered = df_new[df_new['SYMBOL'].isin(nifty_50_symbols)]

    # Ensure proper datetime
    df_filtered['DATE1'] = pd.to_datetime(df_filtered['DATE1'])
    selected_date = pd.to_datetime(selected_date)

    # Get selected day, previous day, and day-before-previous
    selected_day_df = df_filtered[df_filtered['DATE1'] == selected_date]
    prev_day_df = df_filtered[df_filtered['DATE1'] < selected_date].groupby('SYMBOL').tail(1)
    day_before_prev_df = (
        df_filtered[df_filtered['DATE1'] < prev_day_df['DATE1'].min()]
        .groupby('SYMBOL')
        .tail(1)
    )

    # Merge selected and previous day
    merged_df = pd.merge(
        selected_day_df[['SYMBOL', 'LAST_PRICE', 'TTL_TRD_QNTY', 'DELIV_QTY', 'Industry']],
        prev_day_df[['SYMBOL', 'LAST_PRICE', 'TTL_TRD_QNTY', 'DELIV_QTY', 'HIGH_PRICE', 'LOW_PRICE']],
        on='SYMBOL',
        suffixes=('_selected', '_prev')
    )

    # Merge with day-before-previous day for Breakout Ratio
    merged_df = pd.merge(
        merged_df,
        day_before_prev_df[['SYMBOL', 'DELIV_QTY']],
        on='SYMBOL',
        how='left'
    ).rename(columns={'DELIV_QTY': 'DELIV_QTY_day_before_prev'})

    # --- Calculate Metrics ---
    merged_df['% Change'] = ((merged_df['LAST_PRICE_selected'] - merged_df['LAST_PRICE_prev']) / merged_df['LAST_PRICE_prev']) * 100
    merged_df['% Change'] = merged_df['% Change'].round(2)

    merged_df['Volume BO'] = merged_df.apply(
        lambda x: '✅' if x['TTL_TRD_QNTY_selected'] > x['TTL_TRD_QNTY_prev'] else '❌', axis=1
    )
    merged_df['Delivery Volume BO'] = merged_df.apply(
        lambda x: '✅' if x['DELIV_QTY_selected'] > x['DELIV_QTY_prev'] else '❌', axis=1
    )

    # --- Price Position Logic ---
    def classify_price_position(row):
        if row['LAST_PRICE_selected'] > row['HIGH_PRICE']:
            return 'STRONG'
        elif row['LAST_PRICE_selected'] < row['LOW_PRICE']:
            return 'WEAK'
        else:
            return 'RANGE BOUND'

    merged_df['Price vs Prev Range'] = merged_df.apply(classify_price_position, axis=1)

    # --- Breakout Ratio ---
    merged_df['Breakout Ratio'] = (
        merged_df['DELIV_QTY_day_before_prev']/merged_df['DELIV_QTY_prev']
    ).replace([np.inf, -np.inf], np.nan).round(2)

    # --- Sort and Filter Controls ---
    merged_df = merged_df.sort_values(by='% Change', ascending=False).reset_index(drop=True)

    volume_bo_filter = st.multiselect("Filter Volume BO", options=merged_df['Volume BO'].unique(), default=merged_df['Volume BO'].unique())
    delivery_bo_filter = st.multiselect("Filter Delivery Volume BO", options=merged_df['Delivery Volume BO'].unique(), default=merged_df['Delivery Volume BO'].unique())
    price_range_filter = st.multiselect("Filter Price vs Prev Range", options=merged_df['Price vs Prev Range'].unique(), default=merged_df['Price vs Prev Range'].unique())

    # --- Apply filters ---
    filtered_df = merged_df[
        (merged_df['Volume BO'].isin(volume_bo_filter)) &
        (merged_df['Delivery Volume BO'].isin(delivery_bo_filter)) &
        (merged_df['Price vs Prev Range'].isin(price_range_filter))
    ]

    # --- Final Columns ---
    final_df = filtered_df[[
        'SYMBOL', 'Industry', 'Volume BO', 'Delivery Volume BO',
        '% Change', 'Breakout Ratio', 'Price vs Prev Range'
    ]]

    # --- Display ---
    st.dataframe(final_df)



with tab4:
    st.header("📊 Nifty 50 - Weekly Comparison")
    st.markdown("---")

    # Get available date range
    min_date = df_new['DATE1'].min().date()
    max_date = df_new['DATE1'].max().date()

    # Select Week 1 Range
    st.subheader("📅 Select Week 1 Range")
    col1, col2 = st.columns(2)
    with col1:
        week1_start = st.date_input("Week 1 Start", value=min_date, min_value=min_date, max_value=max_date, key="week1_start")
    with col2:
        week1_end = st.date_input("Week 1 End", value=week1_start + timedelta(days=4), min_value=week1_start, max_value=max_date, key="week1_end")

    # Select Week 2 Range
    st.subheader("📅 Select Week 2 Range")
    col3, col4 = st.columns(2)
    with col3:
        week2_start = st.date_input("Week 2 Start", value=week1_end + timedelta(days=3), min_value=min_date, max_value=max_date, key="week2_start")
    with col4:
        week2_end = st.date_input("Week 2 End", value=week2_start + timedelta(days=4), min_value=week2_start, max_value=max_date, key="week2_end")

    # Prevent re-run refreshes with session state
    if 'compare_clicked' not in st.session_state:
        st.session_state.compare_clicked = False

    if st.button("Compare Weeks"):
        st.session_state.compare_clicked = True

    if st.session_state.compare_clicked:
        df_filtered = df_new[df_new['SYMBOL'].isin(nifty_50_symbols)].copy()

        # Add Industry info (we’ll keep it for later merge)
        symbol_industry = df_filtered[['SYMBOL', 'Industry']].drop_duplicates()

        # Create Week 1 and Week 2 DataFrames
        week1_df = (
            df_filtered[(df_filtered['DATE1'] >= pd.to_datetime(week1_start)) & (df_filtered['DATE1'] <= pd.to_datetime(week1_end))]
            .sort_values(by='DATE1')
            .groupby('SYMBOL')
            .last()
            .reset_index()[['SYMBOL', 'LAST_PRICE']]
            .rename(columns={'LAST_PRICE': 'LAST_PRICE_week1'})
        )

        week2_df = (
            df_filtered[(df_filtered['DATE1'] >= pd.to_datetime(week2_start)) & (df_filtered['DATE1'] <= pd.to_datetime(week2_end))]
            .sort_values(by='DATE1')
            .groupby('SYMBOL')
            .last()
            .reset_index()[['SYMBOL', 'LAST_PRICE']]
            .rename(columns={'LAST_PRICE': 'LAST_PRICE_week2'})
        )

        # Merge both weeks
        weekly_compare = pd.merge(week1_df, week2_df, on='SYMBOL')

        # Calculate % Change
        weekly_compare['% Change'] = ((weekly_compare['LAST_PRICE_week2'] - weekly_compare['LAST_PRICE_week1']) / weekly_compare['LAST_PRICE_week1']) * 100
        weekly_compare['% Change'] = weekly_compare['% Change'].round(2)

        # Week 1 and Week 2 full data for volume comparison
        week1_full = df_filtered[(df_filtered['DATE1'] >= pd.to_datetime(week1_start)) & (df_filtered['DATE1'] <= pd.to_datetime(week1_end))]
        week2_full = df_filtered[(df_filtered['DATE1'] >= pd.to_datetime(week2_start)) & (df_filtered['DATE1'] <= pd.to_datetime(week2_end))]

        # Get peak volumes
        week1_peak_vol = week1_full.groupby('SYMBOL')['TTL_TRD_QNTY'].max().reset_index().rename(columns={'TTL_TRD_QNTY': 'Peak_Vol_Week1'})
        week2_peak_vol = week2_full.groupby('SYMBOL')['TTL_TRD_QNTY'].max().reset_index().rename(columns={'TTL_TRD_QNTY': 'Peak_Vol_Week2'})

        # Get peak delivery volumes
        week1_peak_del = week1_full.groupby('SYMBOL')['DELIV_QTY'].max().reset_index().rename(columns={'DELIV_QTY': 'Peak_Deliv_Week1'})
        week2_peak_del = week2_full.groupby('SYMBOL')['DELIV_QTY'].max().reset_index().rename(columns={'DELIV_QTY': 'Peak_Deliv_Week2'})

        # Merge peak values
        weekly_compare = (
            weekly_compare
            .merge(week1_peak_vol, on='SYMBOL')
            .merge(week2_peak_vol, on='SYMBOL')
            .merge(week1_peak_del, on='SYMBOL')
            .merge(week2_peak_del, on='SYMBOL')
        )

        # Determine Breakouts
        weekly_compare['Volume BO'] = weekly_compare.apply(lambda x: '✅' if x['Peak_Vol_Week2'] > x['Peak_Vol_Week1'] else '❌', axis=1)
        weekly_compare['Delivery Volume BO'] = weekly_compare.apply(lambda x: '✅' if x['Peak_Deliv_Week2'] > x['Peak_Deliv_Week1'] else '❌', axis=1)

        # Merge Week 1 High/Low prices for range comparison
        week1_range = (
            df_filtered[(df_filtered['DATE1'] >= pd.to_datetime(week1_start)) & (df_filtered['DATE1'] <= pd.to_datetime(week1_end))]
            .sort_values(by='DATE1')
            .groupby('SYMBOL')
            .agg({'HIGH_PRICE': 'max', 'LOW_PRICE': 'min'})
            .reset_index()
        )

        weekly_compare = pd.merge(weekly_compare, week1_range, on='SYMBOL', how='left')

        # Price vs Range Classification
        def classify_price_position(row):
            if row['LAST_PRICE_week2'] > row['HIGH_PRICE']:
                return 'STRONG'
            elif row['LAST_PRICE_week2'] < row['LOW_PRICE']:
                return 'WEAK'
            else:
                return 'RANGEBOUND'

        weekly_compare['Closing'] = weekly_compare.apply(classify_price_position, axis=1)

        # ✅ Merge Industry info
        weekly_compare = pd.merge(weekly_compare, symbol_industry, on='SYMBOL', how='left')

        # 🔎 Filters
        col_vbo, col_dbo, col_close = st.columns(3)
        with col_vbo:
            selected_vbo = st.multiselect("Volume BO", options=['✅', '❌'], default=['✅', '❌'])
        with col_dbo:
            selected_dbo = st.multiselect("Delivery Volume BO", options=['✅', '❌'], default=['✅', '❌'])
        with col_close:
            selected_closing = st.multiselect("Closing", options=['STRONG', 'WEAK', 'RANGEBOUND'], default=['STRONG', 'WEAK', 'RANGEBOUND'])

        # Filtered Data
        filtered_df = weekly_compare[
            (weekly_compare['Volume BO'].isin(selected_vbo)) &
            (weekly_compare['Delivery Volume BO'].isin(selected_dbo)) &
            (weekly_compare['Closing'].isin(selected_closing))
        ]

        # Display
        st.markdown("### 📋 Week-to-Week % Change with Volume Breakouts")

        def highlight_closing_font(val):
            color = ''
            if val == 'STRONG':
                color = 'green'
            elif val == 'WEAK':
                color = 'red'
            elif val == 'RANGEBOUND':
                color = 'orange'
            return f'color: {color}'

        styled_df = filtered_df[['SYMBOL', 'Industry', '% Change', 'Volume BO', 'Delivery Volume BO', 'Closing']].style.applymap(
            highlight_closing_font, subset=['Closing']
        )

        st.dataframe(styled_df, use_container_width=True)

        # Heatmap
        st.markdown("### 🔥 Weekly Comparison Heatmap")

        heatmap_df = weekly_compare[['SYMBOL', '% Change']].copy()
        heatmap_df = heatmap_df.set_index('SYMBOL')

        grid_rows, grid_cols = 10, 5
        total_cells = grid_rows * grid_cols

        symbols = heatmap_df.index.tolist()
        values = heatmap_df['% Change'].tolist()

        # Sort descending so positive % changes are top left
        sorted_pairs = sorted(zip(symbols, values), key=lambda x: -x[1])
        symbols, values = zip(*sorted_pairs)
        symbols = list(symbols) + [''] * (total_cells - len(symbols))
        values = list(values) + [0] * (total_cells - len(values))

        symbols_grid = np.array(symbols).reshape(grid_rows, grid_cols)
        values_grid = np.array(values).reshape(grid_rows, grid_cols)

        fig = px.imshow(
            values_grid,
            labels=dict(color="% Change"),
            x=[f"Col {i+1}" for i in range(grid_cols)],
            y=[f"Row {i+1}" for i in range(grid_rows)],
            color_continuous_scale='RdYlGn',
            text_auto=False,
            aspect="auto"
        )

        fig.update_layout(
            title="Heatmap: % Change (Week 2 End vs Week 1 End)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=750,
            width=1200,
            margin=dict(l=30, r=30, t=60, b=30)
        )

        for i in range(grid_rows):
            for j in range(grid_cols):
                symbol = symbols_grid[i, j]
                value = values_grid[i, j]
                if symbol:
                    # Get BO values from weekly_compare
                    vol_bo = weekly_compare.loc[weekly_compare['SYMBOL'] == symbol, 'Volume BO'].values[0]
                    deliv_bo = weekly_compare.loc[weekly_compare['SYMBOL'] == symbol, 'Delivery Volume BO'].values[0]

                    fig.add_annotation(
                        text=f"{symbol}<br>{value:.2f}%",
                        hovertext=f"% Change: {value:.2f}%<br>Volume BO: {vol_bo}<br>Delivery BO: {deliv_bo}",
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(size=13, color="black"),
                        xanchor="center",
                        yanchor="middle"
                    )


        st.plotly_chart(fig, use_container_width=False)

        #Plotting Chart by Sector

        # Merge Sector info with weekly_compare
        sector_map = df_new[['SYMBOL', 'Sector']].drop_duplicates()
        weekly_compare = weekly_compare.merge(sector_map, on='SYMBOL', how='left')

        # Filter for stocks that have both Volume BO and Delivery Volume BO as '✅'
        filtered_stocks = weekly_compare[
            (weekly_compare['Volume BO'] == '✅') &
            (weekly_compare['Delivery Volume BO'] == '✅')
        ]

        # Sort the table by Sector and then % Change descending
        filtered_stocks = filtered_stocks[['Sector', 'SYMBOL', '% Change', 'Peak_Vol_Week2', 'Peak_Deliv_Week2']]

        # Sort data by Sector and % Change
        filtered_stocks = filtered_stocks.sort_values(by=['Sector', '% Change'], ascending=[True, False])

        # Reset index for clean display
        filtered_stocks = filtered_stocks.reset_index(drop=True)

        # Scatter Plot (only stocks with both Volume BO and Delivery Volume BO as '✅')
        fig_scatter = px.scatter(
            filtered_stocks,
            x="Sector",
            y="% Change",
            color="Sector",
            hover_data=["SYMBOL", "Peak_Vol_Week2", "Peak_Deliv_Week2"],
            text="SYMBOL",
            title="Scatter Plot: % Change by Sector (Only Volume & Delivery Volume Breakouts)",
            height=600
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Set custom category order for 'Closing'
        weekly_compare['Closing'] = pd.Categorical(
            weekly_compare['Closing'],
            categories=["STRONG", "RANGEBOUND", "WEAK"],
            ordered=True
        )

        '''# Group symbols by Closing status
        grouped_symbols = weekly_compare.groupby('Closing')['SYMBOL'].apply(list)

        # Convert to DataFrame with each Closing status as a column
        max_len = grouped_symbols.map(len).max()
        symbols_by_closing = pd.DataFrame({col: grouped_symbols.get(col, []) + [''] * (max_len - len(grouped_symbols.get(col, [])))
                                            for col in ["STRONG", "RANGEBOUND", "WEAK"]})

        # Display the formatted table
        st.markdown("### 📋 Symbols Grouped by Closing Status")
        st.dataframe(symbols_by_closing, use_container_width=True)'''



# Tab 5: Monthly Analysis
with tab5:

    st.header("📅 Monthly Breakout Analysis")

    # 1. User Inputs
    month1_start = st.date_input("Month 1 - Start Date")
    month1_end = st.date_input("Month 1 - End Date")

    month2_start = st.date_input("Month 2 - Start Date")
    month2_end = st.date_input("Month 2 - End Date")

    # Save to session_state
    st.session_state["month1_start"] = month1_start
    st.session_state["month1_end"] = month1_end
    st.session_state["month2_start"] = month2_start
    st.session_state["month2_end"] = month2_end

    if month1_start and month1_end and month2_start and month2_end:
        # 2. Filter Data
        month1_df = df_new[
            (df_new["DATE1"] >= pd.to_datetime(month1_start)) &
            (df_new["DATE1"] <= pd.to_datetime(month1_end)) &
            (df_new["SYMBOL"].isin(nifty_50_symbols))
        ]

        month2_df = df_new[
            (df_new["DATE1"] >= pd.to_datetime(month2_start)) &
            (df_new["DATE1"] <= pd.to_datetime(month2_end)) &
            (df_new["SYMBOL"].isin(nifty_50_symbols))
        ]

        # 3. Calculate Peak Values
        breakout_symbols = []
        volume_bo = []
        delivery_bo = []
        peak_vol_month1 = []
        peak_vol_month2 = []
        peak_deliv_month1 = []
        peak_deliv_month2 = []
        date_vol_month1 = []
        date_vol_month2 = []
        date_deliv_month1 = []
        date_deliv_month2 = []

        for symbol in nifty_50_symbols:
            m1 = month1_df[month1_df["SYMBOL"] == symbol]
            m2 = month2_df[month2_df["SYMBOL"] == symbol]

            if not m1.empty and not m2.empty:
                max_vol_m1 = m1["TTL_TRD_QNTY"].max()
                max_vol_m2 = m2["TTL_TRD_QNTY"].max()

                max_deliv_m1 = m1["DELIV_QTY"].max()
                max_deliv_m2 = m2["DELIV_QTY"].max()

                # Peak dates
                date_max_vol_m1 = m1.loc[m1["TTL_TRD_QNTY"].idxmax(), "DATE1"]
                date_max_vol_m2 = m2.loc[m2["TTL_TRD_QNTY"].idxmax(), "DATE1"]
                date_max_deliv_m1 = m1.loc[m1["DELIV_QTY"].idxmax(), "DATE1"]
                date_max_deliv_m2 = m2.loc[m2["DELIV_QTY"].idxmax(), "DATE1"]

                breakout_symbols.append(symbol)
                volume_bo.append("✅" if max_vol_m2 > max_vol_m1 else "❌")
                delivery_bo.append("✅" if max_deliv_m2 > max_deliv_m1 else "❌")

                peak_vol_month1.append(max_vol_m1)
                peak_vol_month2.append(max_vol_m2)
                peak_deliv_month1.append(max_deliv_m1)
                peak_deliv_month2.append(max_deliv_m2)

                date_vol_month1.append(date_max_vol_m1)
                date_vol_month2.append(date_max_vol_m2)
                date_deliv_month1.append(date_max_deliv_m1)
                date_deliv_month2.append(date_max_deliv_m2)

        # 5. Build Clean Table
        monthly_breakout_table = pd.DataFrame({
            "SYMBOL": breakout_symbols,
            "Volume BO": volume_bo,
            "Delivery BO": delivery_bo,
            "Date_Vol_Month1": date_vol_month1,
            "Date_Vol_Month2": date_vol_month2,
            "Date_Deliv_Month1": date_deliv_month1,
            "Date_Deliv_Month2": date_deliv_month2
        })

        # Format dates to MM-DD-YYYY
        for date_col in ["Date_Vol_Month1", "Date_Vol_Month2", "Date_Deliv_Month1", "Date_Deliv_Month2"]:
            monthly_breakout_table[date_col] = pd.to_datetime(monthly_breakout_table[date_col]).dt.strftime('%m-%d-%Y')

        # Prepare a list to store 'Price vs Range' results
        # Step 1: Add "Price vs Range" column after Monthly Breakout Table is calculated
        # Step 1: Add "Price vs Range" column after Monthly Breakout Table is calculated
        price_vs_range = []

        # Loop through each row in the monthly breakout table
        for idx, row in monthly_breakout_table.iterrows():
            symbol = row["SYMBOL"]
            # Check if the symbol has Delivery Volume Breakout (Delivery BO == "✅")
            if row["Delivery BO"] == "✅":
                breakout_date = pd.to_datetime(row["Date_Deliv_Month2"])

                # Get Month 1 High and Low
                month1_data = month1_df[month1_df["SYMBOL"] == symbol]
                month1_high = month1_data["HIGH_PRICE"].max()
                month1_low = month1_data["LOW_PRICE"].min()

                # Get Last Price on Delivery Breakout Date
                breakout_day_data = df_new[
                    (df_new["SYMBOL"] == symbol) &
                    (df_new["DATE1"] == breakout_date)
                ]
                if not breakout_day_data.empty:
                    last_price = breakout_day_data["LAST_PRICE"].values[0]

                    # Compare Last Price with Month1 High and Low
                    if last_price > month1_high:
                        price_vs_range.append("Above High")
                    elif month1_low < last_price <= month1_high:
                        price_vs_range.append("In Between")
                    else:
                        price_vs_range.append("Below Low")
                else:
                    price_vs_range.append("Data Missing")
            else:
                # If no Delivery BO, leave it as "No Breakout"
                price_vs_range.append("No Breakout")

        # Add this new column to the monthly breakout table
        monthly_breakout_table["Price vs Range"] = price_vs_range

        # Optional: Reorder columns nicely
        monthly_breakout_table = monthly_breakout_table[
            ["SYMBOL", "Volume BO", "Delivery BO",
            "Date_Vol_Month1", "Date_Vol_Month2",
            "Date_Deliv_Month1", "Date_Deliv_Month2",
            "Price vs Range"]
        ]

        # Step 2: Display Updated Table
        st.subheader("📈 Monthly Breakout Table (With Price Range Analysis)")
        st.dataframe(monthly_breakout_table, use_container_width=True)

        # Step 3: Filter for Delivery Volume Breakouts (Delivery BO == "✅")
        delivery_bo_df = monthly_breakout_table[monthly_breakout_table["Delivery BO"] == "✅"]

        # Step 4: Ensure the date is in datetime format and sort by date
        delivery_bo_df['Date_Deliv_Month2'] = pd.to_datetime(delivery_bo_df['Date_Deliv_Month2'], errors='coerce')

        # Sort by the delivery breakout date to ensure chronological order
        delivery_bo_df = delivery_bo_df.sort_values(by="Date_Deliv_Month2", ascending=True)

        # Step 5: Store Breakout Stocks in Session
        st.session_state.breakout_stocks = set(delivery_bo_df["SYMBOL"])


        # Step 5: Define color mapping based on "Price vs Range"
        color_map = {
            "Above High": "green",
            "In Between": "goldenrod",  # Dark Yellow
            "Below Low": "red",
            "No Breakout": "grey",  # For symbols that don't have a Delivery BO
            "Data Missing": "grey"
        }

        # Apply the color mapping to the 'Color' column
        delivery_bo_df["Color"] = delivery_bo_df["Price vs Range"].map(color_map)

        # Step 6: Create and display the scatter plot if data exists
        if not delivery_bo_df.empty:
            fig_scatter = px.scatter(
                delivery_bo_df,
                x="Date_Deliv_Month2",
                y="SYMBOL",
                text="SYMBOL",
                color="Price vs Range",  # Color based on Price vs Range
                color_discrete_map=color_map,
                title="📊 Delivery Volume Breakout (Symbols vs Date & Price Range)",
                height=600
            )

            # Customize marker size and appearance
            fig_scatter.update_traces(
                textposition='top center',
                marker=dict(size=14, line=dict(width=2, color='DarkSlateGrey'))
            )

            # Customize axis labels and tick angle
            fig_scatter.update_layout(
                xaxis_title="Date of Delivery Breakout",
                yaxis_title="Symbol",
                xaxis_tickangle=-45
            )

            # Display the plot
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.info("No Delivery Volume Breakouts found for the selected periods.")


        # Plot Candle Sticks with Delivery Volume BO
        # 📊 Candlestick Plot for Breakout Dates + Month 1 High/Low Lines
        st.subheader("📊 Candlestick Plot for Breakout Dates")

        selected_stock = st.selectbox("Select a breakout stock", sorted(delivery_bo_df["SYMBOL"].unique()))

        # --- REPLACE APRIL RANGE WITH USER SELECTED MONTH 1 RANGE ---
        # Make sure you have month1_start and month1_end accessible here, e.g. via st.session_state or pass as args
        month1_start = st.session_state.get("month1_start")
        month1_end = st.session_state.get("month1_end")

        if month1_start is None or month1_end is None:
            st.warning("Please select Month 1 start and end dates in the Monthly Breakout Analysis tab first.")
        else:
            # Convert to datetime if needed
            month1_start = pd.to_datetime(month1_start)
            month1_end = pd.to_datetime(month1_end)

            # Filter Month 1 data for the selected stock
            month1_data = df_new[
                (df_new["SYMBOL"] == selected_stock) &
                (df_new["DATE1"] >= month1_start) &
                (df_new["DATE1"] <= month1_end)
            ]

            # Get the two specific breakout dates for the selected stock
            selected_row = delivery_bo_df[delivery_bo_df["SYMBOL"] == selected_stock]

            if not selected_row.empty:
                date1 = pd.to_datetime(selected_row["Date_Deliv_Month1"].values[0])
                date2 = pd.to_datetime(selected_row["Date_Deliv_Month2"].values[0])

                # Filter df_new for the two breakout dates
                candle_df = df_new[
                    (df_new["SYMBOL"] == selected_stock) &
                    (df_new["DATE1"].isin([date1, date2]))
                ].sort_values("DATE1")

                # Calculate Month 1 High and Low
                month1_high = month1_data["HIGH_PRICE"].max() if not month1_data.empty else None
                month1_low = month1_data["LOW_PRICE"].min() if not month1_data.empty else None

                if len(candle_df) == 2:
                    fig = go.Figure(data=[go.Candlestick(
                        x=candle_df["DATE1"],
                        open=candle_df["OPEN_PRICE"],
                        high=candle_df["HIGH_PRICE"],
                        low=candle_df["LOW_PRICE"],
                        close=candle_df["LAST_PRICE"],
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    )])

                    # Add horizontal lines if Month 1 High/Low are valid
                    if month1_high is not None and not pd.isna(month1_high):
                        fig.add_hline(y=month1_high, line_dash="dash", line_color="blue",
                                      annotation_text="Month 1 High", annotation_position="top left")

                    if month1_low is not None and not pd.isna(month1_low):
                        fig.add_hline(y=month1_low, line_dash="dash", line_color="orange",
                                      annotation_text="Month 1 Low", annotation_position="bottom left")

                    # Adjust y-axis range to ensure visibility
                    min_y = min(candle_df["LOW_PRICE"].min(), month1_low if month1_low is not None else float('inf')) * 0.95
                    max_y = max(candle_df["HIGH_PRICE"].max(), month1_high if month1_high is not None else float('-inf')) * 1.05

                    fig.update_layout(
                        title=f"{selected_stock} - Candlestick on Delivery BO Dates",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        yaxis_range=[min_y, max_y],
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data available to plot both breakout dates.")
            else:
                st.warning("Please select a valid breakout stock.")




with tab6:

    # Assuming indices_df is already loaded

    # List of important sectors
    selected_indices = ['Nifty 50', 'Nifty Next 50', 'Nifty 100', 'Nifty Midcap 50', 'NIFTY Smallcap 100',
                        'Nifty Auto', 'Nifty Bank', 'Nifty Energy', 'Nifty Financial Services',
                        'Nifty FMCG', 'Nifty IT', 'Nifty Media', 'Nifty Metal', 'Nifty MNC', 'Nifty Pharma',
                        'Nifty PSU Bank', 'Nifty India Consumption', 'Nifty Commodities', 'Nifty Infrastructure',
                        'Nifty CPSE', 'Nifty Private Bank', 'Nifty Oil & Gas', 'Nifty Healthcare Index']

    # Streamlit UI components
    st.header("Top Gainers/Losers - Indices")
    st.markdown("---")
    st.header("📊 Top 5 Indices Gainers & Losers for a Selected Date")

    # Date input
    selected_date = st.date_input("Select Date to Analyze", value=datetime.today(), key="top_movers_date")

    # Show button for analysis
    if st.button("Show Top Gainers and Losers", key="top_movers_button"):
        st.info(f"Analyzing top movers on {selected_date}...")

        # Filter indices_df for the selected indices
        df_filtered = indices_df[indices_df['Index Name'].isin(selected_indices)].copy()

        # Convert 'Index Date' to datetime
        df_filtered['Index Date'] = pd.to_datetime(df_filtered['Index Date'])

        # Filter the data for the selected date
        current_day_df = df_filtered[df_filtered['Index Date'] == pd.to_datetime(selected_date)]

        # Get previous trading day data for each index
        prev_day_df = (
            df_filtered[df_filtered['Index Date'] < pd.to_datetime(selected_date)]
            .groupby('Index Name')
            .tail(1)
        )

        # Merge current and previous day data to calculate the % change
        merged = pd.merge(
            current_day_df[['Index Name', 'Closing Index Value']],
            prev_day_df[['Index Name', 'Closing Index Value']],
            on='Index Name',
            suffixes=('_current', '_prev')
        )

        # Calculate the percentage change
        merged['% Change'] = ((merged['Closing Index Value_current'] - merged['Closing Index Value_prev']) / merged['Closing Index Value_prev']) * 100

        # Ensure '% Change' is numeric and convert errors to NaN
        merged['% Change'] = pd.to_numeric(merged['% Change'], errors='coerce')

        # Drop rows with NaN values in '% Change'
        merged.dropna(subset=['% Change'], inplace=True)

        # Sort the data by % Change
        sorted_df = merged.sort_values(by='% Change', ascending=False)

        # Get top gainers and losers
        top_gainers = sorted_df.head(5)
        top_losers = sorted_df.tail(5).sort_values(by='% Change')


        # Display top gainers and losers
        col1, col2 = st.columns(2)
        with col1:
            st.success("📈 Top 5 Gainers")
            st.dataframe(
            top_gainers[['Index Name', 'Closing Index Value_prev', 'Closing Index Value_current', '% Change']]
            .reset_index(drop=True)
            .style.format({'% Change': '{:.2f}%'})
        )


        with col2:
            st.error("📉 Top 5 Losers")
            st.dataframe(
            top_losers[['Index Name', 'Closing Index Value_prev', 'Closing Index Value_current', '% Change']]
            .reset_index(drop=True)
            .style.format({'% Change': '{:.2f}%'})
        )


        # Count the number of positive and negative indices
        positive_indices = len(merged[merged['% Change'] > 0])
        negative_indices = len(merged[merged['% Change'] < 0])

        # Display the counts
        st.write(f"📊 **Advances (Indices with Positive % Change)**: {positive_indices}")
        st.write(f"📊 **Declines (Indices with Negative % Change)**: {negative_indices}")


        # Heatmap of % changes
        st.markdown("### 🔥 Heatmap - % Change on Selected Date")

        # Prepare data for the heatmap
        heatmap_df = merged[['Index Name', '% Change']].copy()
        heatmap_df = heatmap_df.set_index('Index Name')

        # Set up grid size for heatmap
        grid_rows, grid_cols = 5, 5  # Adjust the grid size to fit your data
        total_cells = grid_rows * grid_cols

        symbols = heatmap_df.index.tolist()
        values = heatmap_df['% Change'].tolist()

        # Pad to fit grid
        symbols += [''] * (total_cells - len(symbols))
        values += [0] * (total_cells - len(values))

        symbols_grid = np.array(symbols).reshape(grid_rows, grid_cols)
        values_grid = np.array(values).reshape(grid_rows, grid_cols)

        # Create the heatmap plot
        fig = px.imshow(
            values_grid,
            labels=dict(color="% Change"),
            x=[f"Col {i+1}" for i in range(grid_cols)],
            y=[f"Row {i+1}" for i in range(grid_rows)],
            color_continuous_scale='RdYlGn',
            text_auto=False,
            aspect="auto"
        )

        fig.update_layout(
            title="Heatmap of % Change for Indices",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=750,
            width=1200,
            margin=dict(l=30, r=30, t=60, b=30)
        )

        # Add annotations to the heatmap
        for i in range(grid_rows):
            for j in range(grid_cols):
                symbol = symbols_grid[i, j]
                value = values_grid[i, j]
                if symbol:
                    fig.add_annotation(
                        text=f"{symbol}<br>{value:.2f}%",
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(size=13, color="black"),
                        xanchor="center",
                        yanchor="middle"
                    )

        st.plotly_chart(fig, use_container_width=False, key="heatmap")

with tab7:

    # Industry Bo - Daily
    df_new["DATE1"] = pd.to_datetime(df_new["DATE1"])
    selected_date = st.date_input("Select a date to check for Volume Breakouts", key="volume_breakout_date")
    selected_date = pd.to_datetime(selected_date)

    # Get sorted unique trading dates
    trading_dates = sorted(df_new["DATE1"].unique())

    # Find previous trading date
    if selected_date not in trading_dates:
        st.error("Selected date not found in data.")
    else:
        idx = trading_dates.index(selected_date)
        if idx == 0:
            st.error("No previous trading day available.")
        else:
            prev_date = trading_dates[idx - 1]

            # Filter for selected and previous day
            df_compare = df_new[df_new["DATE1"].isin([prev_date, selected_date])]
            df_compare = df_compare[df_compare["SYMBOL"].isin(nifty_50_symbols)]


            # Group by Industry and Date
            grouped = df_compare.groupby(["Industry", "DATE1"]).agg({
                "TTL_TRD_QNTY": "sum",
                "DELIV_QTY": "sum"
            }).reset_index()

            # Pivot so each row = Industry, columns = prev/selected date
            pivot_vol = grouped.pivot(index="Industry", columns="DATE1", values="TTL_TRD_QNTY")
            pivot_deliv = grouped.pivot(index="Industry", columns="DATE1", values="DELIV_QTY")

            # Make sure both dates exist
            if selected_date in pivot_vol.columns and prev_date in pivot_vol.columns:
                # Calculate ratios
                vol_ratio = (pivot_vol[selected_date] / pivot_vol[prev_date]).fillna(0)
                deliv_ratio = (pivot_deliv[selected_date] / pivot_deliv[prev_date]).fillna(0)

                result = pd.DataFrame({
                    "Volume": vol_ratio > 1,
                    "Delivery Volume": deliv_ratio > 1,
                    "V Breakout Ratio": vol_ratio.round(2),
                    "Delivery BO Ratio": deliv_ratio.round(2)
                })

                # Format Yes/No for booleans
                result["Volume"] = result["Volume"].replace({True: "Yes", False: "No"})
                result["Delivery Volume"] = result["Delivery Volume"].replace({True: "Yes", False: "No"})

                # Add symbol count
                symbol_counts = df_compare[df_compare["DATE1"] == selected_date].groupby("Industry")["SYMBOL"].nunique().rename("Symbol Count")
                result = result.merge(symbol_counts, on="Industry", how="left")

                # Reset index to display properly
                result.reset_index(inplace=True)

                # Display
                st.dataframe(result)

            else:
                st.warning("One of the dates is missing data.")

    # ------------------ Weekly Industry BO ------------------

    st.subheader("Industry Breakout - Weekly")

    # User inputs for Week 1 and Week 2 ranges
    week1_start = st.date_input("Week 1 Start Date", key="weekly_bo_week1_start")
    week1_end = st.date_input("Week 1 End Date", key="weekly_bo_week1_end")
    week2_start = st.date_input("Week 2 Start Date", key="weekly_bo_week2_start")
    week2_end = st.date_input("Week 2 End Date", key="weekly_bo_week2_end")

    # Convert to datetime
    week1_start = pd.to_datetime(week1_start)
    week1_end = pd.to_datetime(week1_end)
    week2_start = pd.to_datetime(week2_start)
    week2_end = pd.to_datetime(week2_end)

    # Filter df_new for selected weeks
    df_week1 = df_new[(df_new["DATE1"] >= week1_start) & (df_new["DATE1"] <= week1_end)]
    df_week2 = df_new[(df_new["DATE1"] >= week2_start) & (df_new["DATE1"] <= week2_end)]

    # Keep only Nifty50 stocks
    df_week1 = df_week1[df_week1["SYMBOL"].isin(nifty_50_symbols)]
    df_week2 = df_week2[df_week2["SYMBOL"].isin(nifty_50_symbols)]

    # Group by Industry, take MAX of the week
    week1_grouped = df_week1.groupby("Industry").agg({
        "TTL_TRD_QNTY": "max",
        "DELIV_QTY": "max"
    }).rename(columns={
        "TTL_TRD_QNTY": "Week1_MaxVolume",
        "DELIV_QTY": "Week1_MaxDeliv"
    })

    week2_grouped = df_week2.groupby("Industry").agg({
        "TTL_TRD_QNTY": "max",
        "DELIV_QTY": "max"
    }).rename(columns={
        "TTL_TRD_QNTY": "Week2_MaxVolume",
        "DELIV_QTY": "Week2_MaxDeliv"
    })

    # Merge week1 and week2
    weekly_compare = week1_grouped.merge(week2_grouped, on="Industry", how="outer").fillna(0)

    # Calculate ratios
    weekly_compare["V Breakout Ratio"] = (weekly_compare["Week2_MaxVolume"] / weekly_compare["Week1_MaxVolume"]).replace([np.inf, -np.inf], 0).round(2)
    weekly_compare["Delivery BO Ratio"] = (weekly_compare["Week2_MaxDeliv"] / weekly_compare["Week1_MaxDeliv"]).replace([np.inf, -np.inf], 0).round(2)

    # Breakout flags
    weekly_compare["Volume BO"] = weekly_compare["V Breakout Ratio"].apply(lambda x: "Yes" if x > 1 else "No")
    weekly_compare["Delivery BO"] = weekly_compare["Delivery BO Ratio"].apply(lambda x: "Yes" if x > 1 else "No")

    # Add symbol count for Week 2 (active stocks in the week)
    symbol_counts = df_week2.groupby("Industry")["SYMBOL"].nunique().rename("Symbol Count")
    weekly_compare = weekly_compare.merge(symbol_counts, on="Industry", how="left")

    # Reset index for display
    weekly_compare.reset_index(inplace=True)

    # ------------------ Filters ------------------

    # Industry filter
    industry_filter = st.multiselect(
        "Select Industry",
        options=weekly_compare["Industry"].unique(),
        default=weekly_compare["Industry"].unique()
    )

    # Breakout filter
    bo_filter = st.multiselect(
        "Filter by Breakout Type",
        options=["Volume BO", "Delivery BO"],
        default=[]
    )

    filtered_df = weekly_compare.copy()

    # Apply industry filter
    if industry_filter:
        filtered_df = filtered_df[filtered_df["Industry"].isin(industry_filter)]

    # Apply breakout filter
    if "Volume BO" in bo_filter:
        filtered_df = filtered_df[filtered_df["Volume BO"] == "Yes"]
    if "Delivery BO" in bo_filter:
        filtered_df = filtered_df[filtered_df["Delivery BO"] == "Yes"]

    # Display filtered result
    st.dataframe(filtered_df)

with tab8:
    st.header("📈 Nifty 50 - Stock Overview")

    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    subtab = st.radio("Select a Subtab", ["Percentage Returns", "Relative Position"], key="subtab", index=0)

    if subtab == "Percentage Returns":
        st.subheader("Percentage Returns")

        breakout_stocks = st.session_state.get("breakout_stocks", set())

        # Filter Nifty 50 stocks
        nifty_df = df_new[df_new['SYMBOL'].isin(nifty_50_symbols)].copy()

        # ✅ Fix: Convert stock dates to datetime
        nifty_df['DATE1'] = pd.to_datetime(nifty_df['DATE1'])

        nifty_df.sort_values(['SYMBOL', 'DATE1'], inplace=True)

        # Filter Sector Index Data
        indices_df['Index Date'] = pd.to_datetime(indices_df['Index Date'])

        # Rename index columns
        sector_df = indices_df.copy()
        sector_df.rename(columns={
            'Index Name': 'Sector',
            'Index Date': 'DATE1',
            'Closing Index Value': 'LAST_PRICE',
            'High Index Value': 'HIGH_PRICE',
            'Low Index Value': 'LOW_PRICE'
        }, inplace=True)

        # Combine both
        nifty_df['Type'] = 'Stock'
        sector_df['SYMBOL'] = sector_df['Sector']
        sector_df['Type'] = 'Index'

        combined_df = pd.concat([nifty_df, sector_df], ignore_index=True)
        combined_df.sort_values(['SYMBOL', 'DATE1'], inplace=True)


        # ==== Daily Returns ====
        latest_dates = sorted(combined_df['DATE1'].unique())[-2:]
        daily_df = combined_df[combined_df['DATE1'].isin(latest_dates)].copy()
        daily_return_df = (
            daily_df.groupby('SYMBOL')['LAST_PRICE']
            .apply(lambda x: x.pct_change().iloc[-1] * 100)
            .reset_index()
            .rename(columns={'LAST_PRICE': 'Daily_Return'})
        )


        # === Weekly Returns ===
        min_date = min(combined_df['DATE1'])
        max_date = max(combined_df['DATE1'])

        col1, col2 = st.columns(2)
        with col1:
            weekly_start = st.date_input("📅 Weekly Start", value=st.session_state.get("weekly_start_tab8", min_date), min_value=min_date, max_value=max_date, key="weekly_start_tab8")
        with col2:
            weekly_end = st.date_input("📅 Weekly End", value=st.session_state.get("weekly_end_tab8", max_date), min_value=weekly_start, max_value=max_date, key="weekly_end_tab8")

        weekly_df = combined_df[(combined_df['DATE1'] >= pd.to_datetime(weekly_start)) & (combined_df['DATE1'] <= pd.to_datetime(weekly_end))].copy()
        weekly_return_df = (
            weekly_df.groupby('SYMBOL')['LAST_PRICE']
            .agg(lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100 if len(x) > 1 else None)
            .reset_index()
            .rename(columns={'LAST_PRICE': 'Weekly_Return'})
        )

        # === Monthly Returns ===
        col3, col4 = st.columns(2)
        with col3:
            monthly_start = st.date_input("📅 Monthly Start", value=st.session_state.get("monthly_start_tab8", min_date), min_value=min_date, max_value=max_date, key="monthly_start_tab8")
        with col4:
            monthly_end = st.date_input("📅 Monthly End", value=st.session_state.get("monthly_end_tab8", max_date), min_value=monthly_start, max_value=max_date, key="monthly_end_tab8")

        monthly_df = combined_df[(combined_df['DATE1'] >= pd.to_datetime(monthly_start)) & (combined_df['DATE1'] <= pd.to_datetime(monthly_end))].copy()
        monthly_return_df = (
            monthly_df.groupby('SYMBOL')['LAST_PRICE']
            .agg(lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100 if len(x) > 1 else None)
            .reset_index()
            .rename(columns={'LAST_PRICE': 'Monthly_Return'})
        )

        # === Weekly Closing Position ===
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        with wcol1:
            w1_start = st.date_input("Week 1 Start", value=st.session_state.get("w1_start", min_date), key="w1_start")
        with wcol2:
            w1_end = st.date_input("Week 1 End", value=st.session_state.get("w1_end", w1_start + timedelta(days=4)), key="w1_end")
        with wcol3:
            w2_start = st.date_input("Week 2 Start", value=st.session_state.get("w2_start", w1_end + timedelta(days=1)), key="w2_start")
        with wcol4:
            w2_end = st.date_input("Week 2 End", value=st.session_state.get("w2_end", w2_start + timedelta(days=4)), key="w2_end")

        #Convert to type datetime.
        w1_start = pd.to_datetime(w1_start)
        w1_end = pd.to_datetime(w1_end)
        w2_start = pd.to_datetime(w2_start)
        w2_end = pd.to_datetime(w2_end)



        week1_df = combined_df[(combined_df['DATE1'] >= w1_start) & (combined_df['DATE1'] <= w1_end)].copy()
        week2_df = combined_df[(combined_df['DATE1'] >= w2_start) & (combined_df['DATE1'] <= w2_end)].copy()

        weekly_closing = []
        for symbol in combined_df['SYMBOL'].unique():
            w1_data = week1_df[week1_df['SYMBOL'] == symbol]
            w2_data = week2_df[week2_df['SYMBOL'] == symbol]
            if not w1_data.empty and not w2_data.empty:
                high, low = w1_data['HIGH_PRICE'].max(), w1_data['LOW_PRICE'].min()
                last_price = w2_data.sort_values('DATE1')['LAST_PRICE'].iloc[-1]
                if last_price > high:
                    pos = "Above High"
                elif last_price < low:
                    pos = "Below Low"
                else:
                    pos = "In Between"
                weekly_closing.append({'SYMBOL': symbol, 'Weekly_Closing': pos})
        weekly_closing_df = pd.DataFrame(weekly_closing)

        # === Monthly Closing Position ===
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            m1_start = st.date_input("Month 1 Start", value=st.session_state.get("m1_start", min_date), key="m1_start")
        with mcol2:
            m1_end = st.date_input("Month 1 End", value=st.session_state.get("m1_end", m1_start + timedelta(days=20)), key="m1_end")
        with mcol3:
            m2_start = st.date_input("Month 2 Start", value=st.session_state.get("m2_start", m1_end + timedelta(days=1)), key="m2_start")
        with mcol4:
            m2_end = st.date_input("Month 2 End", value=st.session_state.get("m2_end", m2_start + timedelta(days=20)), key="m2_end")

        #Convert to type datetime.
        m1_start = pd.to_datetime(m1_start)
        m1_end = pd.to_datetime(m1_end)
        m2_start = pd.to_datetime(m2_start)
        m2_end = pd.to_datetime(m2_end)

        month1_df = combined_df[(combined_df['DATE1'] >= m1_start) & (combined_df['DATE1'] <= m1_end)].copy()
        month2_df = combined_df[(combined_df['DATE1'] >= m2_start) & (combined_df['DATE1'] <= m2_end)].copy()

        monthly_closing = []
        for symbol in combined_df['SYMBOL'].unique():
            m1_data = month1_df[month1_df['SYMBOL'] == symbol]
            m2_data = month2_df[month2_df['SYMBOL'] == symbol]
            if not m1_data.empty and not m2_data.empty:
                high, low = m1_data['HIGH_PRICE'].max(), m1_data['LOW_PRICE'].min()
                last_price = m2_data.sort_values('DATE1')['LAST_PRICE'].iloc[-1]
                if last_price > high:
                    pos = "Above High"
                elif last_price < low:
                    pos = "Below Low"
                else:
                    pos = "In Between"
                monthly_closing.append({'SYMBOL': symbol, 'Monthly_Closing': pos})
        monthly_closing_df = pd.DataFrame(monthly_closing)

        # === Final Merge ===
        final_df = daily_return_df.merge(weekly_return_df, on='SYMBOL', how='outer')
        final_df = final_df.merge(monthly_return_df, on='SYMBOL', how='outer')
        final_df = final_df.merge(weekly_closing_df, on='SYMBOL', how='left')
        final_df = final_df.merge(monthly_closing_df, on='SYMBOL', how='left')

        symbol_to_sector = nifty_df[['SYMBOL', 'Sector']].drop_duplicates()
        final_df = final_df.merge(symbol_to_sector, on='SYMBOL', how='left')

        # Add Type: Stock or Index
        type_map = combined_df[['SYMBOL', 'Type']].drop_duplicates()
        final_df = final_df.merge(type_map, on='SYMBOL', how='left')

        # Sort by Sector -> Type -> SYMBOL
        final_df.sort_values(by=['Sector', 'Type', 'SYMBOL'], inplace=True)

        # Optional: Highlight breakout stocks
        def highlight_breakouts(row):
            if row['SYMBOL'] in breakout_stocks:
                return ['background-color: gold'] * len(row)
            return [''] * len(row)

        def format_return_cell(value):
            if pd.isna(value):
                return '<td style="border:1px solid #ddd; padding:6px; text-align:center; color: #888;">—</td>'
            color = 'green' if value > 0 else 'red' if value < 0 else '#888'
            return f'<td style="border:1px solid #ddd; padding:6px; text-align:center; color: {color};">{value:.2f}</td>'

        def render_compact_table(df, index_returns, breakout_stocks):
            def format_return_cell(value):
                if value is None:
                    return '<td style="border:1px solid #ddd; padding:6px; text-align:center;">—</td>'
                color = "green" if value > 0 else "red" if value < 0 else "#888"
                return f'<td style="border:1px solid #ddd; padding:6px; text-align:center; color:{color};">{value:.2f}%</td>'

            def format_closing_cell(value):
                color = {"Above High": "green", "Below Low": "red", "In Between": "#888"}.get(value, "#888")
                return f'<td style="border:1px solid #ddd; padding:6px; text-align:center; color:{color};">{value or "—"}</td>'

            rows = ""

            # Index row first (if available)
            if index_returns:
                rows += (
                    f"<tr style='background-color:#e6f2ff;'>"
                    f"<td style='border:1px solid #ddd; padding:6px; text-align:center; font-weight:bold;'>📊 {index_returns['Index Name']}</td>"
                    f"{format_return_cell(index_returns.get('Daily_Return'))}"
                    f"{format_return_cell(index_returns.get('Weekly_Return'))}"
                    f"{format_return_cell(index_returns.get('Monthly_Return'))}"
                    f"{format_closing_cell(index_returns.get('Weekly_Closing'))}"
                    f"{format_closing_cell(index_returns.get('Monthly_Closing'))}"
                    f"</tr>"
                )

            # Then stock rows
            for i, row in df.iterrows():
                row_style = 'background-color:#f9f9f9;' if i % 2 == 0 else ''
                symbol_td_style = 'color: orange; font-weight: bold;' if row['SYMBOL'] in breakout_stocks else ''
                rows += (
                    f"<tr style='{row_style}'>"
                    f"<td style='border:1px solid #ddd; padding:6px; text-align:center; {symbol_td_style}'>{row['SYMBOL']}</td>"
                    f"{format_return_cell(row.get('Daily_Return'))}"
                    f"{format_return_cell(row.get('Weekly_Return'))}"
                    f"{format_return_cell(row.get('Monthly_Return'))}"
                    f"{format_closing_cell(row.get('Weekly_Closing'))}"
                    f"{format_closing_cell(row.get('Monthly_Closing'))}"
                    f"</tr>"
                )

            html = (
                "<table style='font-size:13px; border-collapse:collapse; width:100%;'>"
                "<thead>"
                "<tr>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>SYMBOL</th>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>Daily Return</th>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>Weekly Return</th>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>Monthly Return</th>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>Weekly Closing</th>"
                "<th style='border:1px solid #ddd; padding:6px; background-color:#eee; text-align:center;'>Monthly Closing</th>"
                "</tr>"
                "</thead>"
                "<tbody>"
                f"{rows}"
                "</tbody>"
                "</table>"
            )
            return html

        #FetchData After Button Click
        if "sector_tables" not in st.session_state:
            st.session_state.sector_tables = {}

        if st.button("🔄 Fetch Data"):

            sector_tables = {}

            for sector in sorted(final_df['Sector'].dropna().unique()):
                sector_df = final_df[final_df['Sector'] == sector][['SYMBOL', 'Daily_Return', 'Weekly_Return', 'Monthly_Return', 'Weekly_Closing', 'Monthly_Closing']].copy()
                sector_df[['Daily_Return', 'Weekly_Return', 'Monthly_Return']] = sector_df[['Daily_Return', 'Weekly_Return', 'Monthly_Return']].round(2)

                index_row = indices_df[indices_df['Index Name'] == sector].copy()
                index_returns = None

                if not index_row.empty:
                    index_row.sort_values('Index Date', inplace=True)
                    latest_dates = sorted(index_row['Index Date'].unique())[-2:]
                    daily_df = index_row[index_row['Index Date'].isin(latest_dates)]
                    daily_return = daily_df['Closing Index Value'].pct_change().iloc[-1] * 100 if len(daily_df) == 2 else None

                    weekly_df = index_row[(index_row['Index Date'] >= pd.to_datetime(weekly_start)) & (index_row['Index Date'] <= pd.to_datetime(weekly_end))]
                    weekly_return = ((weekly_df['Closing Index Value'].iloc[-1] - weekly_df['Closing Index Value'].iloc[0]) / weekly_df['Closing Index Value'].iloc[0]) * 100 if len(weekly_df) > 1 else None

                    monthly_df = index_row[(index_row['Index Date'] >= pd.to_datetime(monthly_start)) & (index_row['Index Date'] <= pd.to_datetime(monthly_end))]
                    monthly_return = ((monthly_df['Closing Index Value'].iloc[-1] - monthly_df['Closing Index Value'].iloc[0]) / monthly_df['Closing Index Value'].iloc[0]) * 100 if len(monthly_df) > 1 else None

                    # Weekly Closing Position
                    week1_df = index_row[(index_row['Index Date'] >= pd.to_datetime(week1_start)) & (index_row['Index Date'] <= pd.to_datetime(week1_end))]
                    week2_df = index_row[(index_row['Index Date'] >= pd.to_datetime(week2_start)) & (index_row['Index Date'] <= pd.to_datetime(week2_end))]

                    week1_high = week1_df['High Index Value'].max() if not week1_df.empty else None
                    week1_low = week1_df['Low Index Value'].min() if not week1_df.empty else None
                    week2_last_close = week2_df.sort_values('Index Date')['Closing Index Value'].iloc[-1] if not week2_df.empty else None

                    if week2_last_close is not None and week1_high is not None and week1_low is not None:
                        if week2_last_close > week1_high:
                            weekly_closing_pos = "Above High"
                        elif week2_last_close < week1_low:
                            weekly_closing_pos = "Below Low"
                        else:
                            weekly_closing_pos = "In Between"
                    else:
                        weekly_closing_pos = None

                    # Filter index data for selected sector
                    index_row = indices_df[indices_df["Index Name"] == sector]

                    # Monthly Closing Position (Safe Logic)
                    month1_df = index_row[(index_row['Index Date'] >= pd.to_datetime(m1_start)) & (index_row['Index Date'] <= pd.to_datetime(m1_end))]
                    month2_df = index_row[(index_row['Index Date'] >= pd.to_datetime(m2_start)) & (index_row['Index Date'] <= pd.to_datetime(m2_end))]

                    # ✅ Predefine variables to avoid NameError
                    month1_high = None
                    month1_low = None
                    month2_last_close = None

                    if not month1_df.empty and not month2_df.empty:
                        month1_high = month1_df['High Index Value'].max()
                        month1_low = month1_df['Low Index Value'].min()
                        month2_last_close = month2_df.sort_values('Index Date')['Closing Index Value'].iloc[-1]

                        if month2_last_close > month1_high:
                            monthly_closing_pos = "Above High"
                        elif month2_last_close < month1_low:
                            monthly_closing_pos = "Below Low"
                        else:
                            monthly_closing_pos = "In Between"
                    else:
                        monthly_closing_pos = "—"  # Safe fallback for missing data


                    index_returns = {
                        "Index Name": sector,
                        "Daily_Return": round(daily_return, 2) if daily_return is not None else None,
                        "Weekly_Return": round(weekly_return, 2) if weekly_return is not None else None,
                        "Monthly_Return": round(monthly_return, 2) if monthly_return is not None else None,
                        "Weekly_Closing": weekly_closing_pos,
                        "Monthly_Closing": monthly_closing_pos
                    }

                # Save rendered table HTML to dictionary
                sector_tables[sector] = (sector_df, index_returns)

            # Store result in session state
            st.session_state.sector_tables = sector_tables


        # After button block: display tables if available
        if st.session_state.get("sector_tables"):
            breakout_stocks = st.session_state.get("breakout_stocks", set())

            for sector, (sector_df, index_returns) in st.session_state.sector_tables.items():
                st.subheader(f"🏷️ Sector: {sector}")
                st.markdown(render_compact_table(sector_df, index_returns, breakout_stocks), unsafe_allow_html=True)



    elif subtab == "Relative Position":
        st.subheader("Relative Position")


