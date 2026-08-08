# /// script
# requires-python = ">=3.12"
# # dependencies = [
#     "google-genai",
#     "python-dotenv",
#     "pydantic",
# ]
# ///

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.read_file import read_file
from tools.list_files import list_files
from tools.edit_file import edit_file

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
        self.system_instruction = (
            "You are a helpful coding assistant operating in a terminal "
            "environment. Output only plain text without markdown formatting. "
            "Be concise but thorough, providing clear and practical advice "
            "with a friendly tone."
        )

    def _build_tools(self) -> list[types.Tool]:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="read_file",
                        description="Read the contents of a file at the specified path.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path of the file to read.",
                                }
                            },
                            "required": ["path"],
                        },
                    ),
                    types.FunctionDeclaration(
                        name="list_files",
                        description="List all files and directories in the specified path.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Directory to list.",
                                }
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="edit_file",
                        description="Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the file.",
                                },
                                "old_text": {
                                    "type": "string",
                                    "description": "Text to replace.",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement text.",
                                },
                            },
                            "required": [
                                "path",
                                "new_text",
                            ],
                        },
                    ),
                ]
            )
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        logging.info(f"Executing tool: {tool_name} with input: {tool_input}")

        try:
            if tool_name == "read_file":
                return read_file(tool_input["path"])

            elif tool_name == "list_files":
                return list_files(tool_input.get("path", "."))

            elif tool_name == "edit_file":
                return edit_file(
                    tool_input["path"],
                    tool_input.get("old_text", ""),
                    tool_input["new_text"],
                )
            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logging.exception(f"Error executing {tool_name}")
            return f"Error executing {tool_name}: {str(e)}"

    def chat(self, user_input: str) -> str:

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        ]

        while True:
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        tools=self._build_tools(),
                    ),
                )

                if response.function_calls:
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

                    contents.append(response.candidates[0].content)

                    contents.append(
                        types.Content(
                            role="user",
                            parts=function_response_parts,
                        )
                    )

                    continue

                assistant_reply = response.text

                return assistant_reply

            except Exception as e:
                logging.exception(e)

                return f"Error: {e}"


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
    print("================")
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
