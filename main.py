from fastapi import FastAPI

app = FastAPI()

@app.get("/api/hola_mundo")
def hola_mundo():
    return {"message": "Hola mundo"}
