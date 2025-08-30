from fastapi import FastAPI

app = FastAPI()

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "TernakPro AI Health Check"}

def handler(request):
    return app