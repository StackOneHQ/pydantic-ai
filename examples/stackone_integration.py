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

    agent = Agent('openai:gpt-4o-mini', tools=[employee_tool])
    result = agent.run_sync('List all employees')
    print(result.output)


def toolset_example():
    """Example using StackOne toolset with filters."""
    toolset = StackOneToolset(
        include_tools=['hris_*'],
        exclude_tools=['hris_delete_*'],
        account_id=os.getenv('STACKONE_ACCOUNT_ID'),
        api_key=os.getenv('STACKONE_API_KEY'),
    )

    agent = Agent('openai:gpt-4o-mini', toolsets=[toolset])
    result = agent.run_sync('Get employee information')
    print(result.output)


if __name__ == '__main__':
    single_tool_example()
    toolset_example()
