import mlflow
import numpy as np
import pandas as pd

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001" 
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = "mlflow-artifacts:/647248505034073364/524c205bc0e04a9db788ee7491267439/artifacts/mlruns/models"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

data_df = pd.DataFrame(data)

# TODO: Predict the output for the data. You might need to use a pandas DataFrame due to a constraint from MLFlow.
prediction = loaded_model.predict(data_df)

# Print out prediction result
print(prediction)
