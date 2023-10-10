from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime

app = FastAPI()

xgb_pipe = load('../models/xgb_pipeline.joblib')

@app.get("/")
def read_root():
    """
    Get project information, objectives, and API documentation.

    Returns:
        dict: A dictionary containing project objectives, API documentation, and GitHub repo link.
    """
    project_info = {
        "Project_Objectives": "A forecasting model using a XGB time-series analysis algorithm that will forecast the total sales revenue across all stores and items for the next 7 days.",
        "Endpoints": [
            {
                "endpoint": "/",
                "description": "Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project.",
                "input_parameters": "No input parameters needed"
            },
            {
                "endpoint": "/health",
                "description": "Returning status code 200 with a string with a welcome message of your choice.",
                "input_parameters": "No input parameters needed",
                "output_message": "Sales Forecasting App is all ready to go!"
            },
            {
                "endpoint": "/sales/stores/items",
                "description": "Forecast total sales revenue for the next 7 days across all stores and items.",
                "input_parameters": [
                    {
                        "name": "date",
                        "type": "str (YYYY-MM-DD)",
                        "description": "The date for which to make the sales prediction."
                    }
                ],
                "output_format": {
                    "prediction": "float"
                }
            }
                    ],
        "github_repo": "https://github.com/anhlevn149/amla_at2"
        }

    return project_info

@app.get('/health', status_code=200)
def healthcheck():
    return 'Sales Forecasting App is all ready to go!'

@app.get("/sales/national")
def forecast( 
    date: str,
):
    """
    Forecast total sales revenue for the next 7 days across all stores and items.

    Args:
        date (string YYYY-MM-DD): The date for which to make the sales prediction.

    Returns:
        JSONResponse: A JSON response containing the sales prediction of the next 7 days.

    Example Request:
    ```
    {
        "date": "2023-10-15"
    }
    ```

    Example Response:
    ```
    {
        "prediction": 150.5
    }
    """
    try:
        input_date = datetime.strptime(date, "%Y-%m-%d")

        year = input_date.year
        quarter = (input_date.month - 1) // 3 + 1
        month = input_date.month
        day_of_week = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
        }[input_date.strftime("%A")]
        weekofyear = input_date.isocalendar().week
        dayofyear = input_date.dayofyear
        dayofmonth = input_date.day

        features = {
        "year": [year],
        "quarter": [quarter],
        "month": [month],
        "day_of_week": [day_of_week],
        "weekofyear": [weekofyear],
        "dayofyear": [dayofyear],
        "dayofmonth": [dayofmonth],
        }
        obs = pd.DataFrame(features)
        pred = xgb_pipe.predict(obs)
        return JSONResponse(pred.tolist())
    except Exception as e:
        # Handle the exception and return an appropriate error response
        return JSONResponse({"error": str(e)}, status_code=500)
