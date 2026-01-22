from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

try:
    from stackone_ai import StackOneToolSet
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
    stackone_toolset = StackOneToolSet(
        api_key=api_key,
        account_id=account_id,
        **({'base_url': base_url} if base_url else {}),
    )

    tools = stackone_toolset.fetch_tools(actions=[tool_name])
    stackone_tool = tools.get_tool(tool_name)
    if stackone_tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in StackOne")

    openai_function = stackone_tool.to_openai_function()
    json_schema: JsonSchemaValue = openai_function['function']['parameters']

    def implementation(**kwargs: Any) -> Any:
        return stackone_tool.execute(kwargs)

    return Tool.from_schema(
        function=implementation,
        name=stackone_tool.name,
        description=stackone_tool.description or '',
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
        filter_pattern: str | list[str] | None = None,
        id: str | None = None,
    ):
        if tools is not None:
            tool_names = list(tools)
        else:
            temp_toolset = StackOneToolSet(
                api_key=api_key,
                account_id=account_id,
                **({'base_url': base_url} if base_url else {}),
            )
            actions = [filter_pattern] if isinstance(filter_pattern, str) else filter_pattern
            filtered_tools = temp_toolset.fetch_tools(actions=actions)
            tool_names = [tool.name for tool in filtered_tools]

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
