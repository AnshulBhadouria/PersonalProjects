import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.time_series import *
from io import BytesIO

# --- App Config ---
st.set_page_config(page_title="📈 Time Series Forecasting", layout="wide")
st.title("📊 Time Series Forecasting App")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload a CSV file with exactly 2 columns: [Date, Metric]", type=["csv"])

# Session storage to prevent auto execution
if "forecast_ready" not in st.session_state:
    st.session_state.forecast_ready = False
if "future_forecast" not in st.session_state:
    st.session_state.future_forecast = None
if "final_model" not in st.session_state:
    st.session_state.final_model = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None

if uploaded_file:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 2:
            st.error("❌ Please upload a CSV with exactly 2 columns: Date and a numeric metric.")
        else:
            # Identify column names
            date_col, metric_col = df.columns

            # --- Preprocessing ---
            # Step 1: Parse Date
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            # Step 2: Clean Metric Column
            import re
            df[metric_col] = (
                df[metric_col]
                .astype(str)
                .apply(lambda x: re.sub(r'[^\d.\-]', '', x))
            )
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            df = df.dropna(subset=[metric_col])

            # Step 3: Aggregate by Month-Year
            df['MonthStart'] = df[date_col].dt.to_period('M').dt.to_timestamp()
            df = df.groupby('MonthStart')[metric_col].sum().reset_index()
            df.rename(columns={'MonthStart': 'Date', metric_col: 'Metric'}, inplace=True)

            # Step 4: Set index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Preview
            st.success("✅ Data preprocessed and aggregated at Month-Year level.")
            st.subheader("📅 Preprocessed Monthly Data")
            st.dataframe(df)

            # Plot
            st.line_chart(df)

            # Store in session
            st.session_state.original_df = df.copy()

            # --- Forecast Config ---
            st.subheader("🔧 Forecast Configuration")
            fh = st.slider("📅 Forecast Horizon", 1, 6, 1)

            if st.button("🚀 Run Forecast"):
                with st.spinner("⏳ Training models and generating forecast..."):
                    setup(data=df, fold=3, fh=fh, session_id=123, verbose=False)
                    best_model = compare_models()
                    final_model = finalize_model(best_model)
                    future_forecast = predict_model(final_model, fh=fh, return_pred_int=True)

                st.session_state.forecast_ready = True
                st.session_state.future_forecast = future_forecast
                st.session_state.final_model = final_model
                st.success("✅ Forecast generated successfully!")
    except Exception as e:
        st.error(f"❌ Error: {e}")


# --- Display Forecast Results ---
if st.session_state.forecast_ready:
    df = st.session_state.original_df
    future_forecast = st.session_state.future_forecast
    final_model = st.session_state.final_model

    st.subheader("📋 Forecasted Data")
    st.dataframe(future_forecast)

    # Download CSV
    csv_buffer = BytesIO()
    future_forecast.to_csv(csv_buffer)
    st.download_button("⬇️ Download Forecast as CSV", data=csv_buffer.getvalue(),
                       file_name="forecast.csv", mime="text/csv")

    # --- Plot Selection ---
    st.subheader("📊 Select Plot Type")
    plot_type = st.radio("Choose how to visualize the forecast:", 
                         ["None", "Custom Plot (with confidence interval)", "PyCaret Forecast Plot"],
                         index=0)

    if plot_type == "Custom Plot (with confidence interval)":
        try:
            future = future_forecast.reset_index()
            future.set_index("index", inplace=True)

            # Rename known output columns
            col_map = {}
            if "Label" in future.columns:
                col_map["Label"] = "Forecast"
            if "y_pred" in future.columns:
                col_map["y_pred"] = "Forecast"    
            if "LowerBound" in future.columns:
                col_map["LowerBound"] = "lower"
            if "UpperBound" in future.columns:
                col_map["UpperBound"] = "upper"
            future.rename(columns=col_map, inplace=True)

            if future["lower"].isnull().all():
                st.warning("⚠️ Confidence intervals could not be computed for the selected forecast horizon.")
            
            if "Forecast" not in future.columns:
                st.warning("⚠️ Forecast column not found. Displaying raw forecast:")
                st.write(future.head())
    
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df["Metric"], label="Historical", color="blue")
                ax.plot(future.index, future["Forecast"], label="Forecast", color="green")

                if "lower" in future.columns and "upper" in future.columns:
                    ax.fill_between(future.index, future["lower"], future["upper"],
                                    color="lightgreen", alpha=0.3, label="Confidence Interval")

                ax.set_title("Time Series Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Metric")
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Plotting error: {e}")

    elif plot_type == "PyCaret Forecast Plot":
        try:
            plot_model(final_model, plot="forecast", display_format="streamlit")
        except Exception as e:
            st.error(f"PyCaret plot error: {e}")
