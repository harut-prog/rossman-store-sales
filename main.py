from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline import load_artifacts, run_pipeline, validate_columns


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
model, scaler, imputer, config = load_artifacts()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
async def predict(data: UploadFile = File()):
    content = await data.read()
    try:
        df = pd.read_excel(BytesIO(content), parse_dates=["Date"])
    except Exception as e:
        raise HTTPException(400, detail=f"Ошибка чтения Excel файла: {e}")

    missing = validate_columns(df)
    if missing:
        raise HTTPException(400, detail=f"Отсутствуют колонки: {', '.join(missing)}")

    try:
        result = run_pipeline(df, model, scaler, imputer, config)
    except Exception as e:
        raise HTTPException(500, detail=f"Ошибка при обработке: {e}")

    if result.empty:
        raise HTTPException(400, detail="Не удалось сделать предсказание, недостаточно данных")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="predictions")
    output.seek(0)

    return Response(
        content=output.read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions.xlsx"},
    )
