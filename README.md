🚗 Car Resale Price Prediction

A machine learning project that predicts car resale price using a RandomForest Regressor model.
The web interface is built with Streamlit, allowing users to input car details and get instant predictions.

🧠 Model Input Parameters

The model predicts price based on the following features:

Car Name
Transmission Type
Fuel Type
Mileage
Power
Engine Capacity
Torque
Seats
Year of Manufacture

returns 
Predicted Sale Price 
Upper Price Bound
lower Price Bound
Confidence Score

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py

🛠 Technologies Used

Python

RandomForest Regressor (scikit-learn)

Streamlit

Pandas, NumPy
