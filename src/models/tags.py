from enum import Enum

class Tag(str, Enum):
    PLAN = "plan"
    REASONING = "reasoning"
    TEXT = "text"
    FULL_TEXT = "full_text"
    TOOL = "tool"
    SUMMARY = "summary"
    STATUS = "status"
    DEBUG = "debug"
    ERROR = "error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    UPDATED_CONTEXT = "updated_context"