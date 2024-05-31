import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%% IMPORT THE DATA SET
data = pd.read_excel('../Data/datos edgar.xlsx',skiprows=(0,2),usecols=['espAl2O3','espSiO2','R (1)','Lambda'])

data_filtered = data[(data['Lambda'] >= 580) & (data['Lambda'] <= 620)] 

# Select features and target variable
X = data_filtered[['espAl2O3', 'espSiO2', 'Lambda']]

y = data_filtered['R (1)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 

# Define the model
model = SVR(C=300, epsilon=0.0075, gamma=1.2, kernel='rbf')

# Define custom scorer
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform k-fold cross-validation
mse_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=mse_scorer)
r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

# Train the model on the entire training data
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
mse_test = mean_squared_error(y_test, model.predict(X_test_scaled))
r2_test = r2_score(y_test, model.predict(X_test_scaled))

print(f"Average MSE (Cross-Validation): {-np.mean(mse_scores)}")
print(f"Average R^2 (Cross-Validation): {np.mean(r2_scores)}")
print(f"MSE (Test): {mse_test}")
print(f"R^2 (Test): {r2_test}")



# Define the linear regression model as benchmark
lr_model = LinearRegression()

# Perform k-fold cross-validation for linear regression
lr_mse_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring=mse_scorer)
lr_r2_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2')

# Train the linear regression model on the entire training data
lr_model.fit(X_train_scaled, y_train)

# Evaluate the linear regression model on the test set
lr_mse_test = mean_squared_error(y_test, lr_model.predict(X_test_scaled))
lr_r2_test = r2_score(y_test, lr_model.predict(X_test_scaled))

print(f"Linear Regression - Average MSE (Cross-Validation): {-np.mean(lr_mse_scores)}")
print(f"Linear Regression - Average R^2 (Cross-Validation): {np.mean(lr_r2_scores)}")
print(f"Linear Regression - MSE (Test): {lr_mse_test}")
print(f"Linear Regression - R^2 (Test): {lr_r2_test}")