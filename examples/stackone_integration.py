"""Example of integrating StackOne tools with Pydantic AI."""

import os

from pydantic_ai import Agent
from pydantic_ai.ext.stackone import (
    StackOneToolset,
    execute_tool,
    feedback_tool,
    search_tool,
    tool_from_stackone,
)


def single_tool_example():
    """Example using a single StackOne tool."""
    employee_tool = tool_from_stackone(
        'stackone_list_employees',
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', tools=[employee_tool])
    result = agent.run_sync('List all employees')
    print(result.output)


def toolset_with_filter_example():
    """Example using StackOne toolset with filter pattern."""
    toolset = StackOneToolset(
        filter_pattern='stackone_*',  # Get all StackOne tools
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('Get employee information')
    print(result.output)


def toolset_with_specific_tools_example():
    """Example using StackOne toolset with specific tools."""
    toolset = StackOneToolset(
        tools=[
            'stackone_list_employees',
            'stackone_get_employee',
        ],  # Specific tools only
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('Get information about all employees')
    print(result.output)


def utility_tools_example():
    """Example using StackOne with utility tools for dynamic discovery.

    Utility tools mode provides two special tools:
    - tool_search: Search for relevant tools using natural language
    - tool_execute: Execute a discovered tool by name

    This is useful when you have a large number of tools and want the agent
    to dynamically discover and use the right ones.
    """
    toolset = StackOneToolset(
        filter_pattern='stackone_*',  # Load all StackOne tools
        include_utility_tools=True,  # Enable dynamic discovery
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('Find a tool to list employees and use it')
    print(result.output)


def standalone_utility_tools_example():
    """Example using standalone search_tool and execute_tool functions.

    This gives you more control over how the tools are created and used.
    """
    from stackone_ai import StackOneToolSet

    # Fetch tools from StackOne
    stackone = StackOneToolSet()
    tools = stackone.fetch_tools(actions=['stackone_*'])

    # Create search and execute tools
    agent = Agent(
        'openai:gpt-5',
        tools=[search_tool(tools), execute_tool(tools)],
    )

    result = agent.run_sync('Search for employee-related tools')
    print(result.output)


def feedback_example():
    """Example using the feedback collection tool via StackOneToolset.

    The feedback tool allows users to provide feedback on their experience
    with StackOne tools, which helps improve the tool ecosystem.
    """
    toolset = StackOneToolset(
        filter_pattern='stackone_*',
        include_feedback_tool=True,
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('List employees and then ask for feedback')
    print(result.output)


def standalone_feedback_example():
    """Example using the standalone feedback_tool function.

    This gives you more control over when and how to include the feedback tool.
    """
    fb_tool = feedback_tool(
        api_key=os.getenv('STACKONE_API_KEY'),
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
    )

    agent = Agent('openai:gpt-5', tools=[fb_tool])
    result = agent.run_sync('Collect feedback about a recent tool usage')
    print(result.output)


def full_featured_example():
    """Example using all StackOne features together.

    This example demonstrates:
    - Utility tools for dynamic discovery
    - Feedback collection
    - Custom hybrid_alpha for search tuning
    """
    toolset = StackOneToolset(
        filter_pattern=['stackone_*'],  # Filter pattern
        include_utility_tools=True,
        include_feedback_tool=True,
        hybrid_alpha=0.3,  # Custom search tuning
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync(
        'Find the best tool to get employee information, use it, '
        'and then collect feedback about the experience'
    )
    print(result.output)


if __name__ == '__main__':
    single_tool_example()
    toolset_with_filter_example()
    toolset_with_specific_tools_example()
    utility_tools_example()
    standalone_utility_tools_example()
    feedback_example()
    standalone_feedback_example()
    full_featured_example()
