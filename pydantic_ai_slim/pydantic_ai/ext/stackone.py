from __future__ import annotations

import os
import json
from typing import Annotated, Any, TypeAlias, cast
from collections.abc import Sequence
from typing import Any

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

try:
    from stackone_ai import StackOneToolSet
except ImportError as _import_error:
    # Friendly error when StackOne SDK is not installed
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
    stackone_toolset = StackOneToolSet(
        api_key=api_key,
        account_id=account_id,
        **({'base_url': base_url} if base_url else {}),
    )

    # Get tools that match the specific tool name
    tools = stackone_toolset.get_tools([tool_name])

    # Get the specific tool
    stackone_tool = tools.get_tool(tool_name)

    # define schema
    json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}, 'additionalProperties': True, 'required': []}

    return Tool.from_schema(
        # return json_schema
        function=lambda *args, **kwargs: stackone_tool.call(*args, **kwargs),
        name=tool_name,
        description=f'StackOne tool: {tool_name}',
        json_schema=json_schema,
    )
