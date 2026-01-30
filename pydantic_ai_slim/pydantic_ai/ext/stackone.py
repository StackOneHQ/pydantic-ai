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

# StackOneTools is dynamically typed as stackone_ai doesn't provide complete type stubs
StackOneTools = Any


def _tool_from_stackone_tool(stackone_tool: Any) -> Tool:
    """Creates a Pydantic AI tool from a StackOneTool instance.

    Args:
        stackone_tool: A StackOneTool instance.

    Returns:
        A Pydantic AI tool that wraps the StackOne tool.
    """
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


def tool_from_stackone(
    tool_name: str,
    *,
    account_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Creates a Pydantic AI tool proxy from a StackOne tool.

    Args:
        tool_name: The name of the StackOne tool to wrap (e.g., "stackone_list_employees").
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

    return _tool_from_stackone_tool(stackone_tool)


def search_tool(
    tools: StackOneTools,
    *,
    hybrid_alpha: float | None = None,
) -> Tool:
    """Creates a search tool for discovering StackOne tools using natural language.

    This tool uses hybrid BM25 + TF-IDF search to find relevant tools based on
    a natural language query.

    Args:
        tools: A StackOne Tools collection (returned from `fetch_tools()`).
        hybrid_alpha: Weight for BM25 in hybrid search (0-1). Default is 0.2,
            which has been shown to provide better tool discovery accuracy.

    Returns:
        A Pydantic AI tool for searching StackOne tools.
    """
    utility_tools = tools.utility_tools(hybrid_alpha=hybrid_alpha)
    search = utility_tools.get_tool('tool_search')
    if search is None:
        raise ValueError('tool_search not found in StackOne utility tools')

    return _tool_from_stackone_tool(search)


def execute_tool(
    tools: StackOneTools,
) -> Tool:
    """Creates an execute tool for running discovered StackOne tools by name.

    This tool allows executing any tool from the provided collection by name,
    typically used after discovering tools with `search_tool`.

    Args:
        tools: A StackOne Tools collection (returned from `fetch_tools()`).

    Returns:
        A Pydantic AI tool for executing StackOne tools.
    """
    utility_tools = tools.utility_tools()
    execute = utility_tools.get_tool('tool_execute')
    if execute is None:
        raise ValueError('tool_execute not found in StackOne utility tools')

    return _tool_from_stackone_tool(execute)


def feedback_tool(
    *,
    api_key: str | None = None,
    account_id: str | None = None,
    base_url: str | None = None,
) -> Tool:
    """Creates a feedback tool for collecting user feedback on StackOne tools.

    This tool allows users to provide feedback on their experience with StackOne tools,
    which helps improve the tool ecosystem.

    Args:
        api_key: The StackOne API key. If not provided, uses STACKONE_API_KEY env var.
        account_id: The StackOne account ID. If not provided, uses STACKONE_ACCOUNT_ID env var.
        base_url: Custom base URL for StackOne API. Optional.

    Returns:
        A Pydantic AI tool for collecting feedback.
    """
    try:
        from stackone_ai.feedback.tool import create_feedback_tool
    except ImportError as e:
        raise ImportError('Please install `stackone-ai` with feedback support to use the feedback tool.') from e

    # Get API key from environment if not provided
    if api_key is None:
        import os

        api_key = os.environ.get('STACKONE_API_KEY')
        if api_key is None:
            raise ValueError(
                'API key is required. Provide it as an argument or set STACKONE_API_KEY environment variable.'
            )

    fb_tool = create_feedback_tool(
        api_key=api_key,
        account_id=account_id,
        **({'base_url': base_url} if base_url else {}),
    )

    return _tool_from_stackone_tool(fb_tool)


class StackOneToolset(FunctionToolset):
    """A toolset that wraps StackOne tools.

    This toolset provides access to StackOne's integration infrastructure for AI agents,
    offering 200+ connectors across HR, ATS, CRM, and other business applications.
    It can operate in two modes:

    1. **Direct mode** (default): Each StackOne tool is exposed directly to the agent.
       This is best when you have a small, known set of tools.

    2. **Utility tools mode** (`include_utility_tools=True`): Instead of exposing all tools
       directly, provides `tool_search` and `tool_execute` that allow the agent to
       dynamically discover and execute tools. This is better for large tool sets
       where the agent needs to search for the right tool.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.ext.stackone import StackOneToolset

    toolset = StackOneToolset(api_key='your-api-key')
    agent = Agent('openai:gpt-4o', toolsets=[toolset])
    ```
    """

    def __init__(
        self,
        tools: Sequence[str] | None = None,
        *,
        account_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        filter_pattern: str | list[str] | None = None,
        include_utility_tools: bool = False,
        include_feedback_tool: bool = False,
        hybrid_alpha: float | None = None,
        id: str | None = None,
    ):
        """Creates a StackOne toolset.

        Args:
            tools: Specific tool names to include (e.g., ["stackone_list_employees"]).
            account_id: The StackOne account ID. Uses STACKONE_ACCOUNT_ID env var if not provided.
            api_key: The StackOne API key. Uses STACKONE_API_KEY env var if not provided.
            base_url: Custom base URL for StackOne API.
            filter_pattern: Glob pattern(s) to filter tools (e.g., "stackone_*").
            include_utility_tools: If True, includes search and execute utility tools instead of
                individual tools. Default is False.
            include_feedback_tool: If True, includes the feedback collection tool.
                Default is False.
            hybrid_alpha: Weight for BM25 in hybrid search (0-1) when using utility tools.
                Default is 0.2.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        stackone_toolset = StackOneToolSet(
            api_key=api_key,
            account_id=account_id,
            **({'base_url': base_url} if base_url else {}),
        )

        if tools is not None:
            actions = list(tools)
        else:
            actions = [filter_pattern] if isinstance(filter_pattern, str) else filter_pattern

        fetched_tools = stackone_toolset.fetch_tools(actions=actions)

        pydantic_tools: list[Tool] = []

        if include_utility_tools:
            # Utility tools mode: provide search and execute tools
            pydantic_tools.append(search_tool(fetched_tools, hybrid_alpha=hybrid_alpha))
            pydantic_tools.append(execute_tool(fetched_tools))
        else:
            # Direct mode: expose each tool individually
            for stackone_tool in fetched_tools:
                pydantic_tools.append(_tool_from_stackone_tool(stackone_tool))

        if include_feedback_tool:
            pydantic_tools.append(
                feedback_tool(
                    api_key=api_key,
                    account_id=account_id,
                    base_url=base_url,
                )
            )

        super().__init__(pydantic_tools, id=id)
