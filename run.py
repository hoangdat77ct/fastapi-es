import uvicorn
from test_fastapi import app
if __name__=="__main__":
    #uvicorn.run("test_fastapi:app")
    uvicorn.run(app,host='0.0.0.0',port=8000)