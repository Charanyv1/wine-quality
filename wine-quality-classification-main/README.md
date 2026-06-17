# Wine Quality Prediction

This project predicts the quality of red wine using machine learning. It uses a Random Forest Classifier trained on the [Wine Quality dataset](winequality-red.csv).

## Project Structure

- `app.py` - Main Python script for data analysis, model training, and Streamlit web app.
- `model.ipynb` - Jupyter notebook for exploratory data analysis and model development.
- `winequality-red.csv` - Dataset containing physicochemical properties and quality ratings of red wine.
- `requirements.txt` - List of required Python packages.

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

3. **Usage:**
   - Enter 11 comma-separated wine feature values in the input box.
   - Click "Predict" to see if the wine is of good quality.

## Features Used

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

## Model

- **Algorithm:** Random Forest Classifier
- **Target:** Wine quality (binary: good quality if quality >= 7, else not good)

## License

This project is for educational purposes.
