"""Example of integrating StackOne tools with Pydantic AI."""

import os

from pydantic_ai import Agent
from pydantic_ai.ext.stackone import StackOneToolset, tool_from_stackone


def single_tool_example():
    """Example using a single StackOne tool."""
    employee_tool = tool_from_stackone(
        'hris_list_employees',
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', tools=[employee_tool])
    result = agent.run_sync('List all employees')
    print(result.output)


def toolset_with_filter_example():
    """Example using StackOne toolset with filter pattern."""
    toolset = StackOneToolset(
        filter_pattern='hris_*',  # Get all HRIS tools
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('Get employee information')
    print(result.output)


def toolset_with_specific_tools_example():
    """Example using StackOne toolset with specific tools."""
    toolset = StackOneToolset(
        tools=['hris_list_employees', 'hris_get_employee'],  # Specific tools only
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    result = agent.run_sync('Get information about all employees')
    print(result.output)


if __name__ == '__main__':
    single_tool_example()
    toolset_with_filter_example()
    toolset_with_specific_tools_example()
