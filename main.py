# https://platform.openai.com/docs/api-reference/chat


import os
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import Response

from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import Literal

load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
API_KEY = os.getenv("API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

print("==== STABILITY_API_KEY ", STABILITY_API_KEY)

print("==== API_KEY ", API_KEY)
client = OpenAI(api_key=API_KEY)
# Load OpenAI API key from environment variable
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4o-mini"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class ChatResponse(BaseModel):
    message: ChatMessage
    finish_reason: str
    usage: Optional[dict] = None
    
class ImageRequest(BaseModel):
    promt: str    
    output_format: str = "webp"
    

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=request.model,
            messages=[msg.dict() for msg in request.messages],
            # temperature=request.temperature,
            # max_tokens=request.max_tokens,
            # top_p=request.top_p,
            # frequency_penalty=request.frequency_penalty,
            # presence_penalty=request.presence_penalty,
        )

        # Extract the assistant's reply
        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        print("=== assistant_message ", assistant_message)
        return {
            "message": {
                "role": assistant_message.role,
                "content": assistant_message.content
            },
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if hasattr(response, 'usage') else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # TODO
    # https://platform.stability.ai/docs/api-reference#tag/Generate
    # https://platform.openai.com/docs/guides/image-generation
    
@app.post("/generate/image")
async def chat_completion(request: ImageRequest):
    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "authorization": f"Bearer {STABILITY_API_KEY}",
            "accept": "image/*"
        },
        files={"none": ''},
        data={
            "prompt": request.promt,
            "output_format": request.output_format,
        },
    )

    if response.status_code == 200:
        with open("./lighthouse.webp", 'wb') as file:
            file.write(response.content)
        # return FileResponse(
        #     "./lighthouse.webp",
        #     media_type="image/jpeg",
        #     headers={"Content-Disposition": "attachment; filename=myimage.jpg"}
        # )
        content_type = f"image/{request.output_format.lower()}"
        return Response(
            content=response.content,
            media_type=content_type,
            headers={
                "Content-Disposition": "attachment",  # or "attachment" to force download
            }
        )
    else:
        raise Exception(str(response.json()))
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)