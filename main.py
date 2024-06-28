import logging
import sys

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import pyodbc

app = Flask(__name__)

class CreditApplicationDataset:
    def __init__(self, server, database):
        self.connection = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=' + server + ';'
            'DATABASE=' + database + ';'
            'Trusted_Connection=yes;'
        )

    def fetch_data(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        dataset = pd.DataFrame(list(map(list, data)), columns=columns)

        # Преобразование числовых столбцов к нужному типу данных
        numeric_columns = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
        dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')

        print("Data types of transformed columns:")
        print(dataset.dtypes)
        return dataset

    def close_connection(self):
        if self.connection:
            self.connection.close()

    def get_train_test_data(self, query, target_column, test_size=0.2, random_state=42):
        dataset = self.fetch_data(query)
        X = dataset.drop(target_column, axis=1)
        y = dataset[target_column].replace({'approved': 1, 'rejected': 0})

        numeric_features = X.select_dtypes(include=['float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ])

        X_transformed = pipeline.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test, preprocessor

# Example usage
host = "LAPTOP-88TNBPF5"
database = "Database2"
user = "your_user"
password = "your_password"

dataset = CreditApplicationDataset(host, database)

query = "SELECT a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, result FROM form2;"

X_train, X_test, y_train, y_test, preprocessor = dataset.get_train_test_data(query, target_column='result')

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка точности модели на данных обучения
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# Оценка точности модели на тестовых данных
test_accuracy = model.score(X_test, y_test)
print(f"Testing Accuracy: {test_accuracy}")

# Вывод типов данных столбцов созданной модели
print("Data types of model columns:")
print(pd.DataFrame(model.feature_importances_, index=preprocessor.get_feature_names_out(), columns=["Importance"]).sort_values("Importance", ascending=False).head())

joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

dataset.close_connection()

# ...

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Преобразование числовых столбцов в числовой формат
        numeric_columns = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']
        for col in numeric_columns:
            if col in data and isinstance(data[col], str):
                try:
                    # Пробуем преобразовать строку в число
                    data[col] = float(data[col])
                except ValueError:
                    # Если не удается преобразовать, оставляем как есть
                    pass

        data = {key: value for key, value in data.items() if
                value is not None and value is not False and key != 'Id' and key != 'result'}
        print(f"Received prediction request with data: {data}", file=sys.stdout)

        # Применяем SimpleImputer для замены NaN значений
        imputer = SimpleImputer(strategy='mean')

        # Применяем preprocessor и fit SimpleImputer на тех же данных
        X_transformed = preprocessor.transform(pd.DataFrame([data]))
        X_transformed = imputer.fit_transform(X_transformed)

        print(f"Transformed input data: {X_transformed}", file=sys.stdout)

        model = joblib.load('model.pkl')
        print("Loaded the model", file=sys.stdout)

        prediction = model.predict(X_transformed)
        print(f"Model prediction result: {prediction}", file=sys.stdout)

        result = {'prediction': int(prediction[0])}
        print(f"Final prediction result: {result}", file=sys.stdout)

        return jsonify(result)

    except Exception as e:
        error_message = {'error': str(e)}
        print(f"Error during prediction: {error_message}", file=sys.stdout)
        return jsonify(error_message)



@app.route('/retrain_with_same_data', methods=['POST'])
def retrain_with_same_data():
    try:

        X_train, X_test, y_train, y_test, preprocessor = dataset.get_train_test_data(query, target_column='result')

        # Обучение новой модели
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)

        # Оценка точности модели на данных обучения
        train_accuracy_1 = model.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy_1}")

        # Оценка точности модели на тестовых данных
        test_accuracy_1 = model.score(X_test, y_test)
        print(f"Testing Accuracy: {test_accuracy_1}")

        # Сохранение новой модели
        joblib.dump(new_model, 'model.pkl')

        return jsonify({'message': 'Model retrained with the same data successfully'})

    except Exception as e:
        error_message = {'error': str(e)}
        print(f"Error during retraining with the same data: {error_message}", file=sys.stdout)
        return jsonify(error_message)

if __name__ == '__main__':
    app.run(debug=True)
