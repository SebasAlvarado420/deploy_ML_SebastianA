from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

app = FastAPI(
    title="Deploy ML Sebastian A",
    version="0.0.1"
)

# Suponiendo que tu modelo está en la carpeta 'model' y se llama 'modelo_entrenado_rf_v01.pkl'
model_rf = joblib.load('model/modelo_entrenado_rf_v01.pkl')


# Endpoint para realizar predicciones
@app.post("/api/v1/predict-stress-factors")
async def predict(
        kindly_rate_your_sleep_quality: float,
        how_many_times_a_week_do_you_suffer_headaches: float,
        how_would_you_rate_you_academic_performance: float,
        how_would_you_rate_your_study_load: float,
        how_many_times_a_week_you_practice_extracurricular_activities: float
):
    data = {
        'kindly rate your sleep quality': kindly_rate_your_sleep_quality,
        'how many times a week do you suffer headaches': how_many_times_a_week_do_you_suffer_headaches,
        'how would you rate you academic performance': how_would_you_rate_you_academic_performance,
        'how would you rate your study load': how_would_you_rate_your_study_load,
        'how many times a week you practice extracurricular activities': how_many_times_a_week_you_practice_extracurricular_activities
    }

    try:
        df_new_data = pd.DataFrame([data])
        prediction = model_rf.predict(df_new_data)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction.tolist()}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
