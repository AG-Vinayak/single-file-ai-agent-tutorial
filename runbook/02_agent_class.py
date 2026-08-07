# ./// . script
# . requires-python.=">=3.12"
# . depndencies.=[
# ....."google", .#. type:. ignore
# ....."pydantic",
# .]


import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class AIAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.message: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        print("Agent initialized")


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)
    agent = AIAgent(api_key)


# ```bash
#  export GEMINI_API_KEY="your-api-key"
# uv.run.runbook/02_agent_class.py
# ```
# Should print: Agent initialized
