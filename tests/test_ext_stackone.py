"""Tests for StackOne integration with Pydantic AI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset

if TYPE_CHECKING:
    from unittest.mock import MagicMock

try:
    import stackone_ai  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: lax no cover
    stackone_installed = False
else:
    stackone_installed = True


class TestStackOneImportError:
    """Test import error handling."""

    def test_import_error_without_stackone(self):
        """Test that ImportError is raised when stackone-ai is not available."""
        # Test that importing the module raises the expected error when stackone_ai is not available
        with patch.dict('sys.modules', {'stackone_ai': None}):
            with pytest.raises(ImportError, match='Please install `stackone-ai`'):
                # Force reimport by using importlib
                import importlib

                import pydantic_ai.ext.stackone

                importlib.reload(pydantic_ai.ext.stackone)


@pytest.mark.skipif(not stackone_installed, reason='stackone-ai not installed')
class TestToolFromStackOne:
    """Test the tool_from_stackone function."""

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_creation(self, mock_stackone_toolset_class: MagicMock) -> None:
        """Test creating a single tool from StackOne."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne tool
        mock_tool = Mock()
        mock_tool.name = 'hris_list_employees'
        mock_tool.description = 'List all employees'
        mock_tool.execute.return_value = {'employees': []}
        mock_tool.to_openai_function.return_value = {
            'type': 'function',
            'function': {
                'name': 'hris_list_employees',
                'description': 'List all employees',
                'parameters': {
                    'type': 'object',
                    'properties': {'limit': {'type': 'integer', 'description': 'Limit the number of results'}},
                },
            },
        }

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create the tool
        tool = tool_from_stackone('hris_list_employees', account_id='test-account', api_key='test-key')

        # Verify tool creation
        assert isinstance(tool, Tool)
        assert tool.name == 'hris_list_employees'
        assert tool.description == 'List all employees'

        # Verify StackOneToolSet was called with correct parameters
        mock_stackone_toolset_class.assert_called_once_with(api_key='test-key', account_id='test-account')

        # Verify fetch_tools was called with actions parameter
        mock_stackone_toolset.fetch_tools.assert_called_once_with(actions=['hris_list_employees'])
        mock_tools.get_tool.assert_called_once_with('hris_list_employees')
        # Verify returned Tool has correct JSON schema based on StackOne definition
        expected = mock_tool.to_openai_function()['function']['parameters']
        assert tool.function_schema.json_schema == expected

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_not_found(self, mock_stackone_toolset_class: MagicMock) -> None:
        """Test error when tool is not found."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the tools to return None for the requested tool
        mock_tools = Mock()
        mock_tools.get_tool.return_value = None

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Should raise ValueError when tool not found
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not found in StackOne"):
            tool_from_stackone('unknown_tool', api_key='test-key')

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_with_base_url(self, mock_stackone_toolset_class: MagicMock) -> None:
        """Test creating a tool with custom base URL.

        Note: base_url is not commonly used by end users, but this test exists for coverage.
        """
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne tool
        mock_tool = Mock()
        mock_tool.name = 'hris_list_employees'
        mock_tool.description = 'List all employees'
        mock_tool.to_openai_function.return_value = {
            'type': 'function',
            'function': {
                'name': 'hris_list_employees',
                'description': 'List all employees',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create tool with base URL and verify json_schema conversion
        tool = tool_from_stackone('hris_list_employees', api_key='test-key', base_url='https://custom.api.stackone.com')
        # Verify base URL was passed to StackOneToolSet
        mock_stackone_toolset_class.assert_called_once_with(
            api_key='test-key', account_id=None, base_url='https://custom.api.stackone.com'
        )
        # Verify returned Tool has correct schema
        expected = mock_tool.to_openai_function()['function']['parameters']
        assert tool.function_schema.json_schema == expected

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_default_parameters(self, mock_stackone_toolset_class: MagicMock) -> None:
        """Test default account_id and base_url are None when not provided."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        mock_tool = Mock()
        mock_tool.name = 'foo'
        mock_tool.description = 'bar'
        mock_tool.to_openai_function.return_value = {
            'type': 'function',
            'function': {'name': 'foo', 'description': 'bar', 'parameters': {}},
        }
        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool
        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        tool = tool_from_stackone('foo', api_key='key-only')
        mock_stackone_toolset_class.assert_called_once_with(api_key='key-only', account_id=None)
        expected = mock_tool.to_openai_function()['function']['parameters']
        assert tool.function_schema.json_schema == expected

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_with_none_description(self, mock_stackone_toolset_class: MagicMock) -> None:
        """Test creating a tool when description is None."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        mock_tool = Mock()
        mock_tool.name = 'test_tool'
        mock_tool.description = None  # None description should become empty string
        mock_tool.to_openai_function.return_value = {
            'function': {'name': 'test_tool', 'parameters': {'type': 'object'}},
        }

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        tool = tool_from_stackone('test_tool', api_key='test-key')
        assert tool.description == ''


@pytest.mark.skipif(not stackone_installed, reason='stackone-ai not installed')
class TestStackOneToolset:
    """Test the StackOneToolset class."""

    @patch('pydantic_ai.ext.stackone.tool_from_stackone')
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_specific_tools(
        self, mock_stackone_toolset_class: MagicMock, mock_tool_from_stackone: MagicMock
    ) -> None:
        """Test creating a StackOneToolset with specific tools."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock tool_from_stackone to return different mock tools
        def create_mock_tool(tool_name: str) -> Mock:
            mock_tool = Mock(spec=Tool)
            mock_tool.name = tool_name
            mock_tool.max_retries = None
            return mock_tool

        def side_effect_func(name: str, **kwargs: object) -> Mock:
            return create_mock_tool(name)

        mock_tool_from_stackone.side_effect = side_effect_func

        # Create the toolset with specific tools
        toolset = StackOneToolset(
            tools=['hris_list_employees', 'hris_get_employee'], account_id='test-account', api_key='test-key'
        )

        # Verify it's a FunctionToolset
        assert isinstance(toolset, FunctionToolset)

        # Verify tool_from_stackone was called for each tool
        assert mock_tool_from_stackone.call_count == 2
        mock_tool_from_stackone.assert_any_call(
            'hris_list_employees', account_id='test-account', api_key='test-key', base_url=None
        )
        mock_tool_from_stackone.assert_any_call(
            'hris_get_employee', account_id='test-account', api_key='test-key', base_url=None
        )

    @patch('pydantic_ai.ext.stackone.tool_from_stackone')
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_filter_pattern(
        self, mock_stackone_toolset_class: MagicMock, mock_tool_from_stackone: MagicMock
    ) -> None:
        """Test creating a StackOneToolset with filter_pattern."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock the StackOneToolSet to return tool names
        mock_tool_obj = Mock()
        mock_tool_obj.name = 'hris_list_employees'

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = [mock_tool_obj]
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tool_from_stackone
        mock_tool = Mock(spec=Tool)
        mock_tool.name = 'hris_list_employees'
        mock_tool.max_retries = None
        mock_tool_from_stackone.return_value = mock_tool

        # Create toolset with filter_pattern
        toolset = StackOneToolset(filter_pattern='hris_*', account_id='test-account', api_key='test-key')

        # Verify StackOneToolSet was created correctly
        mock_stackone_toolset_class.assert_called_once_with(api_key='test-key', account_id='test-account')

        # Verify fetch_tools was called with actions parameter (list)
        mock_stackone_toolset.fetch_tools.assert_called_once_with(actions=['hris_*'])

        # Verify tools were created
        assert isinstance(toolset, FunctionToolset)

    @patch('pydantic_ai.ext.stackone.tool_from_stackone')
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_list_filter_pattern(
        self, mock_stackone_toolset_class: MagicMock, mock_tool_from_stackone: MagicMock
    ) -> None:
        """Test creating a StackOneToolset with list filter_pattern."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock the StackOneToolSet to return tool names
        mock_tool1 = Mock()
        mock_tool1.name = 'hris_list_employees'
        mock_tool2 = Mock()
        mock_tool2.name = 'ats_get_job'

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = [mock_tool1, mock_tool2]
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tool_from_stackone to return different tools for each call
        def mock_tool_side_effect(tool_name: str, **kwargs: object) -> Mock:
            mock_tool = Mock(spec=Tool)
            mock_tool.name = tool_name
            mock_tool.max_retries = None
            return mock_tool

        mock_tool_from_stackone.side_effect = mock_tool_side_effect

        # Create toolset with list filter_pattern
        toolset = StackOneToolset(filter_pattern=['hris_*', 'ats_*'], account_id='test-account', api_key='test-key')

        # Verify fetch_tools was called with list filter_pattern as actions
        mock_stackone_toolset.fetch_tools.assert_called_once_with(actions=['hris_*', 'ats_*'])

        # Verify tools were created for both returned tools
        assert mock_tool_from_stackone.call_count == 2

        # Verify it's a FunctionToolset
        assert isinstance(toolset, FunctionToolset)

    @patch('pydantic_ai.ext.stackone.tool_from_stackone')
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_without_filter_pattern(
        self, mock_stackone_toolset_class: MagicMock, mock_tool_from_stackone: MagicMock
    ) -> None:
        """Test creating a StackOneToolset without filter_pattern (gets all tools)."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock the StackOneToolSet to return all tools
        mock_tool = Mock()
        mock_tool.name = 'all_tools'

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.fetch_tools.return_value = [mock_tool]
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tool_from_stackone
        mock_pydantic_tool = Mock(spec=Tool)
        mock_pydantic_tool.name = 'all_tools'
        mock_pydantic_tool.max_retries = None
        mock_tool_from_stackone.return_value = mock_pydantic_tool

        # Create toolset without filter_pattern
        toolset = StackOneToolset(account_id='test-account', api_key='test-key')

        # Verify fetch_tools was called with None actions (no filter)
        mock_stackone_toolset.fetch_tools.assert_called_once_with(actions=None)

        # Verify tools were created
        assert isinstance(toolset, FunctionToolset)

    @patch('pydantic_ai.ext.stackone.tool_from_stackone')
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_base_url(
        self, mock_stackone_toolset_class: MagicMock, mock_tool_from_stackone: MagicMock
    ) -> None:
        """Test creating a StackOneToolset with custom base URL.

        Note: base_url is not commonly used by end users, but this test exists for coverage.
        """
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock tool_from_stackone
        mock_tool = Mock(spec=Tool)
        mock_tool.name = 'hris_list_employees'
        mock_tool.max_retries = None
        mock_tool_from_stackone.return_value = mock_tool

        # Create toolset with base URL
        toolset = StackOneToolset(
            tools=['hris_list_employees'],
            account_id='test-account',
            api_key='test-key',
            base_url='https://custom.api.stackone.com',
        )

        # Verify tool_from_stackone was called with base URL
        mock_tool_from_stackone.assert_called_once_with(
            'hris_list_employees',
            account_id='test-account',
            api_key='test-key',
            base_url='https://custom.api.stackone.com',
        )

        # Verify it's a FunctionToolset
        assert isinstance(toolset, FunctionToolset)
