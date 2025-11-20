# Advertising Budget and Sales Analysis

This project aims to analyze the relationship between advertising budgets across different channels (TV, Radio, and Newspaper) and sales figures. The analysis includes exploratory data analysis (EDA), model building, and visualization of results.

## Project Structure

advertising-budget-sales/
├── data/
│ └── raw/
│ └── Advertising Budget and Sales.csv # Dataset
├── Advertising Budget and Sale.py # Main script (run this)
├── requirements.txt # Dependencies
└── README.md # This file

## Data

Raw Data: The raw dataset (Advertising Budget and Sales.csv) should be located in the same folder as the Python code.​ It contains the advertising expenditure for TV, radio, and newspaper, along with the corresponding sales values.​ Please note that this is a simple educational example, and real‑world scenarios are far more complex than this.

## Install dependencies:

Python 3.x
pandas — Data manipulation
numpy — Numerical computing
scikit-learn — Machine learning (Pipeline, StandardScaler, LinearRegression)
matplotlib — Visualization
seaborn — Statistical visualization

## Run the main script:
python "Advertising Budget and Sale.py"

## Actual Results from a Recent Run
See Figure 1 to Figure 8
Model paramaters:
Intercept: 14.100
Coefficients:
TV Ad Budget ($): 3.7642
Radio Ad Budget ($): 2.7923
Newspaper Ad Budget ($): 0.0560
Performance: MSE = 3.17, RMSE = 1.78, R² = 0.90
Plain-Language Explanation for Non-Experts
Baseline Sales (intercept ≈ 14.1): This is the model’s predicted sales when all ad budgets are zero. It serves as a baseline for understanding the impact of advertising.
Coefficients indicate the average increase in sales for each additional dollar spent on advertising:
TV: For every $1 increase in the TV budget, sales increase by approximately $3.76 (strongest effect).
Radio: For every $1 increase in the Radio budget, sales increase by about $2.79 (moderate effect).
Newspaper: For every $1 increase in the Newspaper budget, sales increase by only $0.056 (negligible).
Example: If you increase the TV budget by $1,000, you can expect an increase of about $3,764 in sales.

## Important consideration
This is a simple linear model, so please check for outliers, confounders, and consider testing:
    Non-linear models (polynomial regression)
    Interaction effects (e.g., TV × Radio synergy)

## License

This project is licensed under the **MIT License**.

You are FREE to:
- ✅ Use for personal or commercial projects
- ✅ Modify and adapt the code
- ✅ Share or redistribute

Just keep the original copyright notice.

---

## Acknowledgments

- Dataset: [Yasser H](https://www.kaggle.com/yasserh) on Kaggle
