from pydantic import BaseModel,Field
from typing import List,Optional

class MessageRequest(BaseModel):
    message:str=Field(...,min_length=1)

class DetectionResponse(BaseModel):
    verdict:str
    risk_level:str
    confidence:float
    scam_type:Optional[str]=None
    evidence:List[str]=[]
    recommended_action:Optional[str]=None