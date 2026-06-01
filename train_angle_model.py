import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

data = np.load("toy_detector_data.npz")
X = data["counts"]
theta = data["angles"]

# Predict sin/cos instead of raw angle to avoid 0/360 discontinuity
y = np.column_stack([np.cos(theta), np.sin(theta)])

X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
    X, y, theta, test_size=0.2, random_state=42
)

model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
theta_pred = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)

err = np.abs(np.angle(np.exp(1j * (theta_pred - theta_test))))
print("Mean angular error:", np.degrees(err).mean(), "degrees")