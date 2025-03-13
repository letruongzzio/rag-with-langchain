import os

# This line of code is used to set the environment variable `TOKENIZERS_PARALLELISM` to the value `"false"`.
# In the context of NLP libraries such as Hugging Face's `transformers`, this environment variable controls the use of multithreading (parallelism) when tokenizing text.
# Setting `TOKENIZERS_PARALLELISM` to `"false"` can help avoid multithreading-related issues, such as deadlocks or unstable performance when running code on systems with multiple CPU cores.
# This is especially useful when you encounter multithreading-related warnings or errors while tokenizing text.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

"""
- `add_routes` in `langserve` is a function used to add routes to a FastAPI application. Although `add_routes` is not part of the standard FastAPI library, it can be used to configure routes in a custom way, which can include additional logic or special configuration.
- FastAPI's `APIRouter` is a class used to create and manage route groups. It allows you to organize your routes into separate modules, making your code more manageable.

### Using `add_routes` in `langserve`

Suppose `langserve` has the following `add_routes` function:
```python
from fastapi import FastAPI
def add_routes(app: FastAPI):
    @app.get("/custom_route")
    async def custom_route():
        return {"message": "This is a custom route"}
```

In app.py, you can use `add_routes` to add routes to your FastAPI application:
```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

# Add custom routes using add_routes
add_routes(app)
```

### Using FastAPI's `APIRouter`

Here's how you can use `APIRouter` to achieve the same thing:
```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

router = APIRouter()

@router.get("/custom_route")
async def custom_route():
    return {"message": "This is a custom route"}

# Include the router in the FastAPI app
app.include_router(router)
```

### Comparison

- `add_routes` can be used to add custom routes to a FastAPI application, which can include additional logic or special configuration.
- `APIRouter` is part of FastAPI and is used to organize routes into separate modules, making the source code more manageable.

Both approaches can be used to add routes to a FastAPI application, but `APIRouter` is the standard and recommended approach when working with FastAPI.
"""
from langserve import add_routes
from fastapi_interface.src.base.llm_model import get_hf_llm
from fastapi_interface.src.rag.main import build_rag_chain, InputQA, OutputQA
from fastapi_interface.src.chat.chat import build_chat_chain, InputChat

llm = get_hf_llm(temperature=0.9)
# The `temperature` parameter in a language model like GPT-3 controls the randomness of the generated text. A higher temperature value results in more diverse and creative outputs, while a lower value produces more conservative and predictable outputs.

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
print(PROJECT_DIR)
docs = os.path.join(PROJECT_DIR, "data_src/file_storage")

# --------- Chains----------------
doc_chain = build_rag_chain(llm, data_dir=docs, data_type="pdf")
chat_chain = build_chat_chain(llm,
                              history_folder=os.path.join(PROJECT_DIR, "chat_histories"),
                              max_history_length=6)

# --------- FastAPI - App ----------------
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --------- FastAPI - Routes ----------------
@app.get("/check")
async def check():
    """
    A simple route to check if the server is running.
    """
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    """
    Route to generate answers using the document chain.
    """
    answer = doc_chain.invoke(inputs.question)
    return {"answer": answer}

@app.post("/chat")
async def chat(inputs: InputChat, request: Request):
    session_id = request.cookies.get("session_id", "default_session") # Get session from cookie (or default).
    response = chat_chain.invoke({"human_input": inputs.human_input}, config={"configurable": {"session_id": session_id}}) # Invoke the chat chain with the human input and session ID.
    return {"answer": response}

# --------- Langserve Routes - Playground ----------------
add_routes(app,
           doc_chain,
           playground_type="default", # Allow users to interact with the model in the playground
           path="/generative_ai")

add_routes(app,
           chat_chain,
           enable_feedback_endpoint=False, # Disable feedback endpoint (user can't provide feedback on model responses)
           enable_public_trace_link_endpoint=False, # Disable endpoint tracking (don't provide a public chat history link)
           playground_type="default",
           path="/chat")
