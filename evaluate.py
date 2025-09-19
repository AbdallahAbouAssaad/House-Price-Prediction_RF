from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Predict and Evaluate for DT
y_pred = dt_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Decision Tree MSE:", mse)
print("Decision Tree R²:", r2)
print("Decision Tree MAE:", mae)


# Predict and Evaluate for RF
y_pred_rf = rf_pipeline.predict(X_test)

print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))

print(Y.mean())
