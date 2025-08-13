from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
# llm
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json
from typing import Any, Dict

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    env=None,
)


def convert_to_llm_tool(tool) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"]
            }
        }
    }


def call_llm(prompt, functions):
    try:
        # Get GitHub token from environment
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is not set")

        # GitHub Copilot endpoint
        endpoint = "https://api.githubcopilot.com/chat/completions"
        model_name = "gpt-4"

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

        print("CALLING LLM")
        response = client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model_name,
            tools=functions,
            temperature=1.,
            max_tokens=1000,
            top_p=1.
        )
        # ...existing code...
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        return []

    response_message = response.choices[0].message
    functions_to_call = []

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            print("TOOL: ", tool_call)
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            functions_to_call.append({"name": name, "args": args})

    return functions_to_call


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available resources
            resources = await session.list_resources()
            print("LISTING RESOURCES")
            for resource in resources:
                print("Resource: ", resource)

            # List available tools and convert them to LLM format
            tools = await session.list_tools()
            functions = []
            for tool in tools.tools:
                print("Tool: ", tool.name)
                print("Tool", tool.inputSchema["properties"])

            # Process LLM request
            prompt = "Add 2 to 20"
            functions_to_call = call_llm(prompt, functions)

            # Call suggested functions
            for f in functions_to_call:
                result = await session.call_tool(f["name"], arguments=f["args"])
                print("TOOLS result: ", result.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
