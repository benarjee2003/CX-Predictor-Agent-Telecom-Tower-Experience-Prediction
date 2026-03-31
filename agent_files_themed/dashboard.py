if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_excel = pd.read_csv(uploaded_file)
        else:
            df_excel = pd.read_excel(uploaded_file)

        df_excel.columns = [c.strip().lower().replace(" ", "_") for c in df_excel.columns]

        missing = [c for c in REQUIRED_COLS if c not in df_excel.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            result, alerts = run_predictions(df_excel)
            st.session_state.result_df = result
            st.session_state.alerts_log = alerts
            st.success("✅ Predictions updated successfully")

    except Exception as e:
        st.error(str(e))
