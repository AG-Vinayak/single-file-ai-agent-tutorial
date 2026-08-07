# ./// . script
# . requires-python.=">=3.12"
# . depndencies.=[
# ....."google", .#. type:. ignore
# ....."pydantic",
# .]
# ///

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
        self.messages: List[Dict[str, Any]] = []
        self.tools: List[Tool] = []
        self._setup_tools()
        print(f"Agent initialized with {len(self.tools)} tools")

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="read_file",
                description="Read the contents of a file at the specified path",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="list_files",
                description="List all files and directories in the specified path",
                input_schema={
                    "type": "objects",
                    "properties": {
                        "path": {
                            "type": "string",
                            "descriptions": "The directory path to list (defaults to current directory)",
                        }
                    },
                    "required": [],
                },
            ),
            Tool(
                name="edit_file",
                description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "The text to search for and replace (leave empty to create new file)",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "The text to replace old_text with",
                        },
                    },
                    "required": ["path", "new_text"],
                },
            ),
        ]


if __name__ == "__main__":
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)
    agent = AIAgent(api_key)


# ```bash
# export ANTHROPIC_API_KEY="your-api
# uv run runbook/03_define_tools.py
# ```
# Should print: Agent initialized with 3 tools
