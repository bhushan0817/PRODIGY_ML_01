import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration
np.random.seed(42)
num_samples = 1000

# Features
square_footage = np.random.uniform(1000, 4000, num_samples)
num_bedrooms = np.random.randint(2, 6, num_samples)
num_bathrooms = np.random.uniform(1, 4, num_samples)

# True coefficients
true_intercept = 50000
coeff_square_footage = 50
coeff_num_bedrooms = 20000
coeff_num_bathrooms = 30000

# True house prices
true_prices = (
    true_intercept
    + coeff_square_footage * square_footage
    + coeff_num_bedrooms * num_bedrooms
    + coeff_num_bathrooms * num_bathrooms
    + np.random.normal(0, 10000, num_samples)
)

# Create a feature matrix X and target vector y
X = np.column_stack((square_footage, num_bedrooms, num_bathrooms))
y = true_prices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot the true prices vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("True Prices vs. Predicted Prices")
plt.show()
