This Python script facilitates sales prediction using machine learning, specifically linear regression. Initially, it prompts the user for the dataset's path and loads the data into a pandas DataFrame. The script performs exploratory data analysis (EDA) by displaying the first few rows, generating summary statistics, checking for missing values, and visualizing feature relationships with pair plots. It then preprocesses the data by removing any missing entries and separates the features (TV, Radio, Newspaper) from the target variable (Sales). The data is split into training and testing sets, with standardization applied to improve model performance. A Linear Regression model is trained on the standardized training data, followed by evaluation using metrics like Mean Squared Error (MSE) and R-squared (R²) on the test data. The script allows for predictions on new data points, scaling them appropriately, and finally visualizes the results through a scatter plot comparing actual versus predicted sales. Overall, this script serves as a comprehensive tool for analyzing the impact of advertising budgets on sales and developing a predictive model.

**The Output**
The output from the script provides a comprehensive analysis and prediction of sales based on advertising expenditure across TV, Radio, and Newspaper. It begins by displaying the first few rows of the dataset, showing actual values of the expenditures and corresponding sales. Basic statistics like mean, standard deviation, and quartiles are then summarized, followed by a confirmation that there are no missing values in the dataset. Visualizations, such as pair plots, reveal the relationships between variables. The dataset is then split into training and testing sets, with features standardized to ensure uniform contribution to the model. A linear regression model is trained and evaluated, yielding a Mean Squared Error (MSE) and R-squared value (R²), which indicate the model's accuracy and explanatory power, respectively. Finally, the model predicts sales for new data inputs and the results are visualized with a scatter plot comparing actual vs. predicted sales, illustrating the model's predictive performance.
