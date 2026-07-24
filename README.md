# 🌤️ Predicting Temperature in London


## 📖 Overview

As the climate changes, predicting weather conditions becomes increasingly critical for businesses, urban planners, and policymakers. This project builds and compares multiple machine learning regression models to **predict the mean daily temperature in London**, using historical weather measurements tracked and compared with **MLflow**.

Three model types are experimented with across different hyperparameter configurations:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

---

## 📁 Project Structure

```
├── notebook.ipynb          # Main Jupyter notebook with full pipeline
├── london_weather.csv      # Historical London weather dataset
├── mlruns/                 # MLflow experiment tracking logs
├── tower_bridge.jpeg       # Project cover image
└── README.md               # Project documentation
```

---

## 📊 Dataset

The dataset is stored in `london_weather.csv` and contains daily weather measurements for London. It includes the following columns:

| Column | Description | Type |
|---|---|---|
| `date` | Recorded date of measurement | int |
| `cloud_cover` | Cloud cover in oktas | float |
| `sunshine` | Sunshine duration in hours (hrs) | float |
| `global_radiation` | Irradiance in Watts per square meter (W/m²) | float |
| `max_temp` | Maximum temperature in degrees Celsius (°C) | float |
| `mean_temp` | **Target** — Mean temperature in degrees Celsius (°C) | float |
| `min_temp` | Minimum temperature in degrees Celsius (°C) | float |
| `precipitation` | Precipitation in millimeters (mm) | float |
| `pressure` | Atmospheric pressure in Pascals (Pa) | float |
| `snow_depth` | Snow depth in centimeters (cm) | float |

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Parsed and extracted `year` and `month` from the raw date integer
- Aggregated daily data to monthly averages
- Visualised mean temperature trends over time
- Generated a correlation heatmap to identify the strongest predictors of mean temperature

### 2. Feature Engineering & Preprocessing
The following features were selected based on correlation analysis:

```python
feature_selection = [
    'month', 'cloud_cover', 'sunshine',
    'precipitation', 'pressure', 'global_radiation'
]
```

Preprocessing steps applied:
- Dropped rows with missing target values (`mean_temp`)
- **Train/test split**: 67% training, 33% test (`random_state=1`)
- **Missing value imputation**: `SimpleImputer` with mean strategy (fit on train, applied to test)
- **Feature scaling**: `StandardScaler` (fit on train, applied to test)

### 3. Model Training & Experiment Tracking

Three models were trained across **3 MLflow runs**, each using a different `max_depth` value `[1, 2, 10]`:

| Run | max_depth | Linear Regression RMSE | Decision Tree RMSE | Random Forest RMSE |
|---|---|---|---|---|
| run_0 | 1 | 4.695 | 4.752 | 3.867 |
| run_1 | 2 | 4.695 | 3.762 | 3.138 |
| run_2 | 10 | 4.695 | 2.226 | 1.850 |

All runs were logged using **MLflow**, tracking:
- `max_depth` parameter
- `rmse_lr` — Linear Regression RMSE
- `rmse_tr` — Decision Tree RMSE
- `rmse_fr` — Random Forest RMSE

---

## 📈 Results

The **Random Forest Regressor** at `max_depth=10` achieved the best performance with the lowest RMSE of **~1.85°C**, significantly outperforming both Linear Regression (~4.70°C) and the shallow Decision Tree configurations.

Key findings:
- Linear Regression RMSE remained constant across all runs (~4.70) since it has no `max_depth` parameter
- Both tree-based models improved substantially as depth increased
- Random Forest consistently outperformed the single Decision Tree at every depth level

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlflow
```

### Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/london-temperature-prediction.git
cd london-temperature-prediction
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

3. Run all cells from top to bottom.

4. To view the MLflow experiment dashboard:
```bash
mlflow ui
```
Then open your browser at `http://localhost:5000`

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8 | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualisation |
| scikit-learn | Machine learning models and preprocessing |
| MLflow | Experiment tracking and model logging |

---

## 📌 Key Takeaways

- Weather prediction benefits significantly from ensemble methods like Random Forest
- Increasing tree depth from 1 to 10 reduced Random Forest RMSE from ~3.87 to ~1.85
- MLflow provides an efficient way to track, compare, and reproduce ML experiments
- Features like `global_radiation`, `sunshine`, and `month` are strong predictors of mean temperature

---

## 👤 Author

**Olaniyi Olanike Gift**  
Computer Science and Engineering  

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
