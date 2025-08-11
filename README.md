Corrosion Inhibitor Selector  
Machine Learning for Predicting Electrochemical Inhibition Efficiency in Acidic Media

Project Overview

This project uses machine learning to **predict the inhibition efficiency (%) of corrosion inhibitors used in acidic environments. It's designed to support researchers and engineers in selecting promising inhibitors based on concentration, temperature, exposure time, and inhibitor/source type.


Dataset

The dataset was manually curated from published literature and includes:

- Inhibitor type (e.g., Amino Acid, Plant Extract)
- Source (e.g., Leaves, Seeds, Bark)
- Concentration (ppm)
- Temperature (°C)
- Time (hours)
- Inhibition Efficiency (%)

Missing values are handled using a simple placeholder method (`fillna(-1)`), and categorical values are encoded using `LabelEncoder`.



Model

The model used is a Random Forest Regressor, trained using `scikit-learn`. It takes 5 input features and predicts the inhibition efficiency:

text
Features:
  - Inhibitor
  - Source
  - Concentration (ppm)
  - Temperature (°C)
  - Time (h)

 How to Use
1. Install Requirements

pip install -r requirements.txt
2. Run the Script
python corrosion_inhibitor_selector.py

This will:

Train the model

Save the model and encoders

Print evaluation metrics

Predict an example efficiency value

3. Make Predictions
After training, you can use the built-in function:


predict_efficiency("Amino Acid", "Plant Extract", 250, 60, 5)
This returns the predicted inhibition efficiency for your input conditions.

Files
dataset.csv – Input corrosion data

corrosion_inhibitor_selector.py – Main script

corrosion_model.pkl – Trained model

le_inhibitor.pkl – Label encoder for inhibitors

le_source.pkl – Label encoder for sources

requirements.txt – Python dependencies

README.md – This file

Example Output

Model trained successfully!
RMSE: 4.32
R² Score: 0.92

Predicted Inhibition Efficiency: 88.75%
Author
Prince Michael Frimpong Boateng
Machine Learning & Chemical Engineering Enthusiast
GitHub: @PrinzMike

Future Work


Expand dataset with more literature entries

Enable model export to ONNX or TensorFlow Lite for deployment

License

This project is open-source and free to use under the MIT License.
