"""
Centralized tool initialization for AERIS.

Creates a ToolRegistry and registers all available tool modules.
Keeps registration logic out of main.py.
"""

from __future__ import annotations

import logging

from tools.registry import ToolRegistry
from tools.applications import register_application_tools
from tools.websites import register_website_tools
from tools.system import register_system_tools
from tools.system_audio import register_system_audio_tools
from tools.system_media import register_system_media_tools
from tools.files import register_file_tools

logger = logging.getLogger("aeris.core.tool_setup")


def create_tool_registry() -> ToolRegistry:
    """
    Build and return a fully populated ToolRegistry with all AERIS tools.
    """
    registry = ToolRegistry()

    register_application_tools(registry)
    logger.info("Registered application tools.")

    register_website_tools(registry)
    logger.info("Registered website tools.")

    register_system_tools(registry)
    logger.info("Registered system tools.")

    register_system_audio_tools(registry)
    logger.info("Registered system audio tools.")

    register_system_media_tools(registry)
    logger.info("Registered system media tools.")

    register_file_tools(registry)
    logger.info("Registered file tools.")

    logger.info("Tool registry ready with %d tools: %s", len(registry.list_names()), registry.list_names())
    return registry
