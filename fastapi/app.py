import os

# This line of code is used to set the environment variable `TOKENIZERS_PARALLELISM` to the value `"false"`.
# In the context of NLP libraries such as Hugging Face's `transformers`, this environment variable controls the use of multithreading (parallelism) when tokenizing text.
# Setting `TOKENIZERS_PARALLELISM` to `"false"` can help avoid multithreading-related issues, such as deadlocks or unstable performance when running code on systems with multiple CPU cores.
# This is especially useful when you encounter multithreading-related warnings or errors while tokenizing text.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
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
from src.base.llm_model import get_hf_llm
from src.main import build_rag_chain, InputQA, OutputQA
from src.chat.main import build_chat_chain