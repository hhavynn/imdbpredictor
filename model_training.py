import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import preprocess_data

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # For Random Forest, print feature importances
        if name == 'Random Forest':
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            print("\nFeature Importances:")
            print(feature_importances.sort_values(ascending=False))
    
    # Determine the best model based on RMSE
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.4f}")
    
    return models, results, predictions, best_model_name

def visualize_results(y_test, predictions, results):
    # Create a results summary DataFrame
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[model]['RMSE'] for model in results],
        'MAE': [results[model]['MAE'] for model in results],
        'R2': [results[model]['R2'] for model in results]
    })
    
    # Plot model comparison
    plt.figure(figsize=(14, 6))
    
    # Plot RMSE
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=results_df)
    plt.title('Model Comparison - RMSE (lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot R2
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='R2', data=results_df)
    plt.title('Model Comparison - R² (higher is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Plot actual vs predicted for each model
    plt.figure(figsize=(15, 12))
    model_names = list(predictions.keys())
    
    for i, model_name in enumerate(model_names):
        plt.subplot(3, 2, i+1)
        plt.scatter(y_test, predictions[model_name], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(y_test.min(), predictions[model_name].min())
        max_val = max(y_test.max(), predictions[model_name].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        
        # Add metrics to the plot
        plt.text(min_val, max_val - 0.2, 
                f"RMSE: {results[model_name]['RMSE']:.4f}\nR²: {results[model_name]['R2']:.4f}", 
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    print("Visualizations saved as 'model_comparison.png' and 'actual_vs_predicted.png'")

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, scaler, top_genres = preprocess_data()
    
    # Train and evaluate models
    models, results, predictions, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(y_test, predictions, results)