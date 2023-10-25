import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def ignore_warn(*args, **kwargs):
    print(args)
    print(kwargs)
    pass


import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.graphics.tsaplots as smt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from xgboost import XGBRegressor

data = pd.read_csv("static/supermarket_sales - Sheet1.csv")
df = pd.DataFrame(data)

print("The Data Set is: ")
print(data)
print()
print(" Shape of Data is:")
print(data.shape)
print()
print("Info of DataSet")
print(data.info)
print()
print("The missing values in columns of DataSet are")
print(data.isnull().sum())
print()
print("Round Weight of Data Set is: ")
print(round(data['Cost Price / unit'].mean()))
print()
data['Cost Price / unit'] = data['Cost Price / unit'].fillna(13)
print()
print("The modes of data in a dataset are: ")
print(data['Cost Price / unit'].mode())
print()

data['Quantity'] = data['Quantity'].fillna('Medium')
print(data)

data.info()
print()
print("Description of Dataset")
print(data.describe())
print()


def monthlyORyears_sales(dataset, time=None):
    if time is None:
        time = ['monthly', 'years']
    dataData = dataset.copy()
    if time == "monthly":
        dataData.date = dataset.Date.apply(lambda x: str(x)[:-3])
    else:
        dataData.date = dataset.Date.apply(lambda x: str(x)[:4])

    dataData = dataset.groupby('Date')['Profit in %'].sum().reset_index()
    print("For each Month/year in every year Profit % Respectively")
    print(dataData)

    dataDataDemand = dataset.groupby('Date')['Demand'].sum().reset_index()
    print("For each Month/year in every year Demand Respectively")
    print(dataDataDemand)

    dataDataSupply = dataset.groupby('Date')['Supply'].sum().reset_index()
    print("For each Month/year in every year Supply Respectively")
    print(dataDataSupply)

    dataset.Date = pd.to_datetime(dataset.Date, format="%d-%m-%Y")
    return dataset


m_data = monthlyORyears_sales(data, "monthly")
m_data.to_csv("./static/monthly_data.csv")
print("The DatSet of newly created monthly_data.csv")
print(m_data.head(10))

y_data = monthlyORyears_sales(data, "yearly")


def sales_time(dataset):
    dataset.date = pd.to_datetime(dataset.Date)
    n_of_days = dataset.Date.max() - dataset.Date.min()
    n_of_years = int(n_of_days.days / 365)
    print(f"Days: {n_of_days.days}\nYears: {n_of_years}\nMonth: {12 * n_of_years}")


sales_time(data)

average_m_sales = m_data.sales.mean()
print(f"Overall Average Monthly Sales: ${average_m_sales}")
average_m_sales_1y = m_data.sales[-12:].mean()
print(f"Last 12 months average monthly sales: ${average_m_sales_1y}")
average_m_sales_1y = m_data.sales[-6:].mean()
print(f"Last 6 months average monthly sales: ${average_m_sales_1y}")

print("Sales per year")
if not pd.api.types.is_datetime64_ns_dtype(m_data['Date']):
    m_data['Date'] = pd.to_datetime(m_data['Date'], format='%Y-%m-%d')
fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(m_data['Date'], m_data['sales'], color='darkBlue', label='Total Sales')
s_mean = m_data.groupby(m_data['Date'].dt.year)['sales'].mean().reset_index()
s_mean['Date'] = pd.to_datetime(s_mean['Date'], format='%Y')
s_mean['Date'] += pd.DateOffset(months=6)
ax.scatter(s_mean['Date'], s_mean['sales'], color='red', label='Mean Sales')
ax.set(xlabel="Years",
       ylabel="Sales",
       title="Monthly Sales Before Diff Transformation")
ax.legend()
ax.grid(True)
plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/SalesPerYear_scatter_plot.jpg')
plt.show()

print("for cleaning the dataset ")
print(data['Date'].unique())


# noinspection PyShadowingNames
def GraphSection():
    data = pd.read_csv("static/supermarket_sales - Sheet1.csv")
    df = pd.DataFrame(data)
    dt_data = data.set_index('Date').drop('sales', axis=1)
    print("Data in 'dt_data' dataset")
    print(dt_data.head(10))
    law = plt.subplot(122)
    acf = plt.subplot(221)
    pacf = plt.subplot(223)
    dt_data.plot(ax=law, figsize=(10, 5))
    smt.plot_acf(dt_data['Supply'], ax=acf, lags=24, color='blue')
    smt.plot_pacf(dt_data['Supply'], lags=24, ax=pacf, color='mediumBlue')
    plt.tight_layout()
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/wholeDataSet_plot.jpg')
    plt.show()

    print("The Supply by Year and The Supply by month")
    layout = (1, 2)
    raw = plt.subplot2grid(layout, (0, 0))
    law = plt.subplot2grid(layout, (0, 1))
    years = y_data['Supply'].plot(kind="bar", color='mediumBlue', label="Supply", ax=raw, figsize=(12, 5))
    months = m_data['Supply'].plot(marker='o', color='orange', label="Supply", ax=law)
    years.set(xlabel="Years", title="Distribution of Sales Per Year")
    months.set(xlabel="Months", title="Distribution of Sales Per Month")
    sns.despine()
    plt.tight_layout()
    years.legend()
    months.legend()
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/SupplyAndDemand_ByYearAndMonth_plot.jpg')
    plt.show()

    print("Graph of demand")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Demand'], marker='o', linestyle='-', color='red')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Demand Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/Demand_plot.jpg')
    plt.show()

    print("Graph of Supply")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Supply'], marker='o', linestyle='-', color='green')
    plt.xlabel('Date')
    plt.ylabel('Supply')
    plt.title('Supply Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/Supply_plot.jpg')
    plt.show()

    print("Graph of demand and supply")
    x_values = range(1, len(df['Demand']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, df['Demand'], label='Demand', marker='o', linestyle='-', color='blue')
    plt.plot(x_values, df['Supply'], label='Supply', marker='x', linestyle='-', color='green')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Demand and Supply Over Data Points')
    plt.legend()
    plt.grid(True)
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/DemandAndSupply_plot.jpg')
    plt.show()

    print("Count of Sores in each City(Reset Index): ")
    print(df['City'].value_counts().reset_index())
    print("Count of Sores in each City(Sort Index): ")
    print(df['City'].value_counts().sort_index())
    store_counts = df['City'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(store_counts, labels=store_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Stores by City')
    plt.axis('equal')
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/NoStoreInCityVer_plot.jpg')
    plt.show()

    store_name_counts = df['StoreName'].value_counts().sort_index()
    print("In how many cities the sore is present")
    print(store_name_counts)
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(store_name_counts.index, store_name_counts.values, color='orange')
    plt.xlabel('Number of Stores')
    plt.ylabel('Store Name')
    plt.title('Number of Stores for Each Store Name')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/NoStoreInCityHor_plot.jpg')
    plt.show()  # Invert the y-axis to display the top store at the top

    print("Methods of Payment distribution")
    payment_counts = df['Payment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140,
            colors=['blue', 'coral', 'green'])
    plt.title('Payment Method Distribution')
    plt.axis('equal')
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/payment1.jpg')
    plt.show()

    print("Graph of Cost Price / unit")
    sns.displot(data['Cost Price / unit'], color="red")
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/CpPerUnit.jpg')
    plt.show()

    print("Graph of Total Cost")
    sns.displot(data['Total Cost'])
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/TotalCost.jpg')
    plt.show()

    print("Graph of Quantity")
    sns.displot(data['Quantity'])
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/quantity.jpg')
    plt.show()

    print(data['Total Cost'].value_counts())
    print("Graph of Total Cost")
    print()
    sns.histplot(data['Total Cost'])
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/TotalCost2.jpg')
    plt.show()

    print("Gross Income Value Count")
    print(data['Gross income'].value_counts())
    print("Graph of Gross income")
    print()
    sns.countplot(data['Gross income'])
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/GrossIncome.jpg')
    plt.show()

    print("Value Count of Rating in dataset ")
    print(data['Rating'].value_counts())
    print()

    def clean_col(Rating):
        if Rating <= 5:
            return "Maybe Product quality is not that good"
        else:
            return "Product quality is very good"

    print("Value Count of Rating in dataset after applying 'clean_col' function ")
    data['Rating'] = data['Rating'].apply(clean_col)
    print(data['Rating'].value_counts())
    print()

    print("Graph of Rating")
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    sns.histplot(data=df, x='Rating', bins=333)
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/Rating.jpg')
    plt.show()
    print()

    unique_varieties = df['Categories'].unique()
    print("Unique varieties:", unique_varieties)
    print("Graph of Categories")
    df['Categories'] = df['Categories'].astype(str)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Categories', palette='Set1')
    plt.title("Distribution of Categories")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/Categories.jpg')
    plt.show()
    print()

    le = LabelEncoder()
    data['Quantity'] = le.fit_transform(data.Quantity)
    data['Categories'] = le.fit_transform(data.Categories)
    data['StoreName'] = le.fit_transform(data.StoreName)
    data['Gender'] = le.fit_transform(data.Gender)
    data['City'] = le.fit_transform(data.City)
    data['Categories'] = le.fit_transform(data.Categories)
    print(data.head())
    print()


def ml_model_phase1():
    x_orig = data.drop('Supply', axis=1)
    y_orig = data['Supply']
    print("Information on the X-axis")
    print(x_orig.info())
    print()
    print("Information on the Y-axis")
    print(y_orig.info())
    print()
    x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=.2, random_state=999)
    print("X_train Shape is: ", x_train.shape)
    print("X_test Shape is: ", x_test.shape)
    print("Y_train Shape is: ", y_train.shape)
    print("Y_test Shape is: ", y_test.shape)

    print(x_train.columns)
    print(x_orig.columns)
    print("X_train", x_train.isnull().sum())
    print("y_train", y_train.isnull().sum())
    print("y_train.head\n", y_train.head())
    print("y_test.head\n", y_test.head())

    # # Ridge Regulation
    X = df[['Demand']]
    y = df[['Supply']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999 * 15)
    alpha = 1.0
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Ridge Regulation")
    print("Coefficient:", ridge_model.coef_[0])
    print("Intercept:", ridge_model.intercept_)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print()
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (Ridge Regression)')
    # plt.savefig('static/images/Ridge_scatter_plot.png')
    plt.show()

    # LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    # accuracy = accuracy_score(y_test, y_pred)
    print("\nLinear Regression")
    print("Coefficient:", linear_model.coef_[0])
    print("Intercept:", linear_model.intercept_)
    print("Mean Squared Error (Linear):", mse_linear)
    print("R-squared (Linear):", r2_linear)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred_linear, color='green', label='Predicted (Linear)')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (Linear Regression)')
    # plt.savefig('static/images/Linear_scatter_plot.png')
    plt.show()

    # # Decision Tree
    dtree_model = DecisionTreeRegressor()
    dtree_model.fit(X_train, y_train)
    y_pred = dtree_model.predict(X_test)
    print("Decision Tree")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2*100) Score:", r2 * 100)
    print("R-squared (R2) Score:", r2)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    print()
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (Decision Tree Regression)')
    # plt.savefig('static/images/decision_scatter_plot.png')
    plt.show()

    # # SVM
    svm_model = SVR(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("SVM")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2*100) Score:", r2 * 100)
    print("R-squared (R2) Score:", r2)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    print()
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (SVM Regression)')
    # plt.savefig('static/images/svm_scatter_plot.png')
    plt.show()

    # # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=999)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (Random Forest Regression)')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest")
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2*100) Score:", r2 * 100)
    print("R-squared (R2) Score:", r2)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    print()
    # plt.savefig('static/images/RM_scatter_plot.png')
    plt.show()

    # AdaBoost
    adaboost_model = AdaBoostRegressor(n_estimators=100, random_state=999)
    adaboost_model.fit(X_train, y_train)
    y_pred = adaboost_model.predict(X_test)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (AdaBoost Regression)')
    print("AdaBoost")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2*100) Score:", r2 * 100)
    print("R-squared (R2) Score:", r2)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    print()
    # plt.savefig('static/images/AB_scatter_plot.png')
    plt.show()

    # XGBRegressor
    xgb_model = XGBRegressor(n_estimators=100, random_state=999)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Demand')
    plt.ylabel('Supply')
    plt.legend()
    plt.title('Actual vs. Predicted Supply (XGBoost Regression)')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # accuracy = accuracy_score(y_test, y_pred)
    print("XGBRegressor")
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2*100) Score:", r2 * 100)
    print("R-squared (R2) Score:", r2)
    # print("Accuracy: {accuracy:.2f}", accuracy)
    # plt.savefig('static/images/XGBRegressor_scatter_plot.png')
    plt.show()

    feature1 = X
    print("Feature1: ", feature1)
    joblib.dump(ridge_model, 'static/ml_ridge_model.pkl')
    joblib.dump(linear_model, 'static/ml_linear_model.pkl')
    joblib.dump(dtree_model, 'static/ml_dtree_model.pkl')
    joblib.dump(svm_model, 'static/ml_svm_model.pkl')
    joblib.dump(rf_model, 'static/ml_rf_model.pkl')
    joblib.dump(adaboost_model, 'static/ml_adaboost_model.pkl')
    joblib.dump(xgb_model, 'static/ml_xgb_model.pkl')

    return feature1


print("ml_model_phase2")


# noinspection PyShadowingNames
def preprocess_date(df):
    # Assuming 'Date' is in datetime format
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df.drop(['Date'], axis=1)


def ml_model_phase2():
    data2 = preprocess_date(data)  # Preprocess the date column

    X = data2[['Year', 'Month', 'Day']]  # Use the extracted date features as input
    y_demand = data2[['Demand']]
    y_supply = data2[['Supply']]

    X_train, X_test, y_train_demand, y_test_demand, y_train_supply, y_test_supply = train_test_split(
        X, y_demand, y_supply, test_size=0.2, random_state=999)

    alpha = 1.0
    ridge_demand_model = Ridge(alpha=alpha)
    ridge_demand_model.fit(X_train, y_train_demand)
    y_pred_demand = ridge_demand_model.predict(X_test)

    ridge_supply_model = Ridge(alpha=alpha)
    ridge_supply_model.fit(X_train, y_train_supply)
    y_pred_supply = ridge_supply_model.predict(X_test)

    mse_demand = mean_squared_error(y_test_demand, y_pred_demand)
    r2_demand = r2_score(y_test_demand, y_pred_demand)

    mse_supply = mean_squared_error(y_test_supply, y_pred_supply)
    r2_supply = r2_score(y_test_supply, y_pred_supply)
    # Plot scatter graphs
    plt.figure(figsize=(12, 6))

    # Scatter plot for Demand
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_demand, y_pred_demand)
    # plt.scatter(y_test_demand, y_test_demand, c='blue', label='Actual', alpha=0.5)
    # plt.scatter(y_test_demand, y_pred_demand, c='red')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Demand Prediction Scatter Plot')

    # Scatter plot for Supply
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_supply, y_pred_supply)
    # plt.scatter(y_test_supply, y_test_supply, c='blue', label='Actual', alpha=0.5)
    # plt.scatter(y_test_supply, y_pred_supply, c='blue')
    plt.xlabel('Actual Supply')
    plt.ylabel('Predicted Supply')
    plt.title('Supply Prediction Scatter Plot')

    plt.tight_layout()

    # Save the scatter plot as an image (optional)
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/Ridge_scatter_plot.png')
    plt.show()
    joblib.dump(ridge_demand_model, 'static/ml_ridge_demand_model.pkl')
    joblib.dump(ridge_supply_model, 'static/ml_ridge_supply_model.pkl')

    return {
        "Demand": {
            "model": ridge_demand_model,
            "MSE": mse_demand,
            "R2": r2_demand
        },
        "Supply": {
            "model": ridge_supply_model,
            "MSE": mse_supply,
            "R2": r2_supply
        }
    }


def ml_model_phase3():
    data2 = preprocess_date(data)  # Preprocess the date column

    X = data2[['Year', 'Month', 'Day']]  # Use the extracted date features as input
    y_demand = data2[['Demand']]
    y_supply = data2[['Supply']]

    X_train, X_test, y_train_demand, y_test_demand, y_train_supply, y_test_supply = train_test_split(
        X, y_demand, y_supply, test_size=0.2, random_state=999)
    dtree_demand_model = DecisionTreeRegressor()
    dtree_demand_model.fit(X_train, y_train_demand)
    y_pred_demand = dtree_demand_model.predict(X_test)

    dtree_supply_model = DecisionTreeRegressor()
    dtree_supply_model.fit(X_train, y_train_supply)
    y_pred_supply = dtree_supply_model.predict(X_test)

    mse_demand = mean_squared_error(y_test_demand, y_pred_demand)
    r2_demand = r2_score(y_test_demand, y_pred_demand)

    mse_supply = mean_squared_error(y_test_supply, y_pred_supply)
    r2_supply = r2_score(y_test_supply, y_pred_supply)

    plt.figure(figsize=(12, 6))

    # Scatter plot for Demand
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_demand, y_pred_demand)
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Demand Prediction Scatter Plot')

    # Scatter plot for Supply
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_supply, y_pred_supply)
    plt.xlabel('Actual Supply')
    plt.ylabel('Predicted Supply')
    plt.title('Supply Prediction Scatter Plot')

    plt.tight_layout()

    # Save the scatter plot as an image (optional)
    plt.savefig('C:/Users/athar/OneDrive/Desktop/sp/static/image/dtree_scatter_plot.png')
    plt.show()
    joblib.dump(dtree_demand_model, 'static/ml_dtree_demand_model.pkl')
    joblib.dump(dtree_supply_model, 'static/ml_dtree_supply_model.pkl')

    return {
        "Demand": {
            "model": dtree_demand_model,
            "MSE": mse_demand,
            "R2": r2_demand
        },
        "Supply": {
            "model": dtree_supply_model,
            "MSE": mse_supply,
            "R2": r2_supply
        }
    }


def ml_model_phase4(data, categories_to_model):
    models = {}

    for category in categories_to_model:
        data_category = data[data['Categories'] == category]
        data2 = preprocess_date(data_category)  # Preprocess the date column

        X = data2[['Year', 'Month', 'Day']]  # Use the extracted date features as input
        y_demand = data2[['Demand']]
        y_supply = data2[['Supply']]

        X_train, X_test, y_train_demand, y_test_demand, y_train_supply, y_test_supply = train_test_split(
            X, y_demand, y_supply, test_size=0.2, random_state=999)
        dtree_demand_model = DecisionTreeRegressor()
        dtree_demand_model.fit(X_train, y_train_demand)
        y_pred_demand = dtree_demand_model.predict(X_test)

        dtree_supply_model = DecisionTreeRegressor()
        dtree_supply_model.fit(X_train, y_train_supply)
        y_pred_supply = dtree_supply_model.predict(X_test)

        mse_demand = mean_squared_error(y_test_demand, y_pred_demand)
        r2_demand = r2_score(y_test_demand, y_pred_demand)

        mse_supply = mean_squared_error(y_test_supply, y_pred_supply)
        r2_supply = r2_score(y_test_supply, y_pred_supply)

        # Save the models as .pkl artifacts
        demand_model_filename = f'static/Category_{category}_demand_model.pkl'
        supply_model_filename = f'static/Category_{category}_supply_model.pkl'

        joblib.dump(dtree_demand_model, demand_model_filename)
        joblib.dump(dtree_supply_model, supply_model_filename)

        models[category] = {
            "Demand": {
                "model": demand_model_filename,
                "MSE": mse_demand,
                "R2": r2_demand
            },
            "Supply": {
                "model": supply_model_filename,
                "MSE": mse_supply,
                "R2": r2_supply
            }
        }

    return models


def ml_model_phase5(data, categories_to_model, cities_to_model):
    models2 = {}

    for city in cities_to_model:
        for category in categories_to_model:
            data_filtered = data[(data['City'] == city) & (data['Categories'] == category)]
            if len(data_filtered) < 2:
                print(f"Skipping {city} - {category} because there is insufficient data.")
                continue
            data2 = preprocess_date(data_filtered)  # Preprocess the date column

            X = data2[['Year', 'Month', 'Day']]  # Use the extracted date features as input
            y_demand = data2[['Demand']]
            y_supply = data2[['Supply']]

            X_train, X_test, y_train_demand, y_test_demand, y_train_supply, y_test_supply = train_test_split(
                X, y_demand, y_supply, test_size=0.2, random_state=999)
            if len(X_train) < 1 or len(X_test) < 1:
                print(f"Skipping {city} - {category} due to insufficient data for the split.")
                continue
            dtree_demand_model = DecisionTreeRegressor()
            dtree_demand_model.fit(X_train, y_train_demand)
            y_pred_demand = dtree_demand_model.predict(X_test)

            dtree_supply_model = DecisionTreeRegressor()
            dtree_supply_model.fit(X_train, y_train_supply)
            y_pred_supply = dtree_supply_model.predict(X_test)

            mse_demand = mean_squared_error(y_test_demand, y_pred_demand)
            r2_demand = r2_score(y_test_demand, y_pred_demand)

            mse_supply = mean_squared_error(y_test_supply, y_pred_supply)
            r2_supply = r2_score(y_test_supply, y_pred_supply)

            # Save the models as .pkl artifacts
            demand_model_filename = f'static/City_{city}_Category_{category}_demand_model.pkl'
            supply_model_filename = f'static/City_{city}_Category_{category}_supply_model.pkl'

            joblib.dump(dtree_demand_model, demand_model_filename)
            joblib.dump(dtree_supply_model, supply_model_filename)

            if category not in models2:
                models2[category] = {}

            models2[category][city] = {
                "Demand": {
                    "model": demand_model_filename,
                    "MSE": mse_demand,
                    "R2": r2_demand
                },
                "Supply": {
                    "model": supply_model_filename,
                    "MSE": mse_supply,
                    "R2": r2_supply
                }
            }
    return models2


categories_to_model_org = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle',
                           'Fashion accessories', 'Food and beverages', 'Sports and travel']
cities_to_model_org = ['Delhi', 'Pune', 'Mumbai', 'Kanpur', 'Surat', 'Chennai', 'Bangalore',
                       'Kolkata', 'Ahmedabad', 'Hyderabad']


# print(GraphSection())
# print(ml_model_phase1())
print(preprocess_date(data))
# print(ml_model_phase2())
# print(ml_model_phase3())
print(ml_model_phase4(data, categories_to_model_org))
print()
print(ml_model_phase5(data, categories_to_model_org, cities_to_model_org))
joblib.dump(ml_model_phase1, 'static/ml_model1.pkl')
