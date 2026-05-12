from fastapi import APIRouter
from pydantic import BaseModel
import os

try:
    from gpt4all import GPT4All
    GPT_LOADED = True
except ImportError:
    GPT_LOADED = False

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

gpt_model = None

def load_gpt():
    global gpt_model
    if not GPT_LOADED:
        return False
    if gpt_model is None:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
        try:
            # Load the exact model mentioned in legacy code without trying to download
            gpt_model = GPT4All(model_name="gpt4all-falcon.Q4_0.gguf", model_path=models_dir, allow_download=False)
        except Exception as e:
            print(f"Error loading GPT4All: {e}")
            return False
    return gpt_model is not None

class ChatRequest(BaseModel):
    message: str

@router.post("/")
async def chat_with_ai(request: ChatRequest):
    if load_gpt():
        try:
            # We add a medical prompt context
            prompt = f"You are a helpful AI Medical Assistant helping a patient understand their Chest X-Ray. Keep answers short and easy to understand. User asks: {request.message}\nAssistant:"
            try:
                response = gpt_model.generate(prompt, max_tokens=150, temp=0.7)
            except TypeError:
                try:
                    response = gpt_model.generate(prompt=prompt, max_tokens=150, temperature=0.7)
                except TypeError:
                    response = gpt_model.generate(prompt)
            return {"response": response.strip()}
        except Exception as e:
            return {"response": f"I'm sorry, I encountered an error: {str(e)}"}
    else:
        # Fallback mock response if GPT4All isn't installed
        return {"response": "I am a mock AI medical assistant. To use the real AI, please ensure 'gpt4all' is installed via pip and the model file is in the models directory."}
