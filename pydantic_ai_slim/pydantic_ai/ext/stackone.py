from __future__ import annotations

from collections.abc import Sequence

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

try:
    import stackone_ai
except ImportError as _import_error:
    raise ImportError('Please install `stackone-ai` to use StackOne tools.') from _import_error


def tool_from_stackone(
    tool_name: str,
    *,
    account_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Creates a Pydantic AI tool proxy from a StackOne tool.

    Args:
        tool_name: The name of the StackOne tool to wrap (e.g., "hris_list_employees").
        account_id: The StackOne account ID. If not provided, uses STACKONE_ACCOUNT_ID env var.
        api_key: The StackOne API key. If not provided, uses STACKONE_API_KEY env var.
        base_url: Custom base URL for StackOne API. Optional.

    Returns:
        A Pydantic AI tool that corresponds to the StackOne tool.
    """
    # Initialize StackOneToolSet
    stackone_toolset = stackone_ai.StackOneToolSet(
        api_key=api_key,
        account_id=account_id,
        **({'base_url': base_url} if base_url else {}),
    )

    # Get tools that match the specific tool name
    tools = stackone_toolset.get_tools([tool_name])

    # Get the specific tool
    stackone_tool = tools.get_tool(tool_name)

    if stackone_tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in StackOne")

    # Extract JSON schema from the OpenAI function representation
    openai_function = stackone_tool.to_openai_function()
    json_schema: JsonSchemaValue = openai_function['function']['parameters']

    return Tool.from_schema(
        function=lambda *args, **kwargs: stackone_tool.call(*args, **kwargs),
        name=stackone_tool.name,
        description=stackone_tool.description,
        json_schema=json_schema,
    )


class StackOneToolset(FunctionToolset):
    """A toolset that wraps StackOne tools."""

    def __init__(
        self,
        tools: Sequence[str] | None = None,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        id: str | None = None,
    ):
        # when tools is specified, use it; otherwise, use filtering features
        if tools is not None:
            tool_names = list(tools)
        else:
            # Fetch all available tools and apply filtering manually
            temp_toolset = stackone_ai.StackOneToolSet(
                api_key=api_key,
                account_id=account_id,
                **({'base_url': base_url} if base_url else {}),
            )
            all_tools = temp_toolset.get_tools()
            tool_names = [tool.name for tool in all_tools]

            # Apply include_tools filter if specified
            if include_tools is not None:
                tool_names = [name for name in tool_names if name in include_tools]

            # Apply exclude_tools filter if specified
            if exclude_tools is not None:
                tool_names = [name for name in tool_names if name not in exclude_tools]

        super().__init__(
            [
                tool_from_stackone(
                    tool_name,
                    account_id=account_id,
                    api_key=api_key,
                    base_url=base_url,
                )
                for tool_name in tool_names
            ],
            id=id,
        )
