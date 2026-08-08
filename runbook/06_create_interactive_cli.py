# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-genai",
#     "python-dotenv",
#     "pydantic",
# ]
# ///

import os
import sys
import argparse  # NEW
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.genai import types
from google import genai
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)

# Suppress verbose HTTP logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AIAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.messages: List[Dict[str, Any]] = []

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _list_files(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"

            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")

            if not items:
                return f"Empty directory: {path}"

            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_text not in content:
                    return f"Text not found in file: {old_text}"

                content = content.replace(old_text, new_text)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"Successfully edited {path}"
            else:
                # Only create directory if path contains subdirectories
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)

                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute one of the available tools.
        """

        try:
            if tool_name == "_read_file":
                return self._read_file(tool_input["path"])

            elif tool_name == "_list_files":
                return self._list_files(tool_input.get("path", "."))

            elif tool_name == "_edit_file":
                return self._edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )

            return f"Unknown tool: {tool_name}"

        except Exception as e:
            return str(e)

    def chat(self, user_input: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        """
        Chat with Gemini using manual function calling.
        """

        contents = []

        for msg in self.messages:
            contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[
                        types.Part.from_text(text=msg["content"]),
                    ],
                )
            )

        while True:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=self._build_tools(),
                ),
            )

            # No function call -> final answer
            if not response.function_calls:
                assistant_reply = response.text

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_reply,
                    }
                )

                return assistant_reply

            # Execute every requested function
            function_response_parts = []

            for function_call in response.function_calls:
                tool_result = self._execute_tool(
                    function_call.name,
                    dict(function_call.args),
                )

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={
                            "result": tool_result,
                        },
                    )
                )

            # Add model function call
            contents.append(response.candidates[0].content)

            # Add tool response
            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

    def _build_tools(self):

        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="_read_file",
                        description="Read a file.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                }
                            },
                            "required": ["path"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="_list_files",
                        description="List files in a directory.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                }
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="_edit_file",
                        description="Edit a file.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"},
                            },
                            "required": ["path", "new_text"],
                        },
                    ),
                ]
            )
        ]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Code Assistant - A conversational AI agent with file editing capabilities"
    )
    parser.add_argument(
        "--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: Please provide an API key via --api-key or GEMINI_API_KEY environment variable"
        )
        sys.exit(1)

    agent = AIAgent(api_key)

    print("AI Code Assistant")
    print("=================")
    print("A conversational AI agent that can read, list, and edit files.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print()


if __name__ == "__main__":
    main()

# ```bash
# export GEMINI_API_KEY="your-api-key-here"
# ```
# Should print:
# AI Code Assistant
# ================
# A conversational AI agent that can read, list, and edit files.
# Type 'exit' or 'quit' to end the conversation.

# You: <<Your input here>>
# ===============
# Type `exit` or `quit` to end the conversation.
