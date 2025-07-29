def predict_efficiency(inhibitor, source, concentration, temperature, time):
    # Encode categorical values
    inhibitor_code = le_inhibitor.transform([inhibitor])[0]
    source_code = le_source.transform([source])[0]

    # Create feature array
    input_data = pd.DataFrame([[
        inhibitor_code, source_code, concentration, temperature, time
    ]], columns=X.columns)

    # Predict
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)
