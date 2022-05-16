from pydantic import BaseModel

class PayloadUrl(BaseModel):
    url : str
    k : int = 0
    

class PayloadBase64(BaseModel):
    base64img : str
    k : int = 0
    
    
class PayloadCrawled(BaseModel):
    crawled : list
    
    
    
    