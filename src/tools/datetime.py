import re
from datetime import datetime, timedelta
from typing import ClassVar, Optional

from dateutil.relativedelta import relativedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DateTimeArgs(BaseModel):
    format: Optional[str] = Field(
        default="%Y-%m-%d %H:%M:%S %A",
        description="Format string for the datetime output. Default includes day of week. Use Python's strftime directives."
    )
    timezone: Optional[str] = Field(
        default="UTC",
        description="Timezone for the datetime. Default is UTC. Use IANA timezone names (e.g., 'America/New_York', 'Europe/London') or 'UTC'."
    )
    offset: Optional[str] = Field(
        default=None,
        description="Offset from current time. Format: (+/-)(\d+)(minutes|hours|days|weeks|months|years). E.g., '+1day', '-2weeks'."
    )

class DateTimeTool(BaseTool):
    name: ClassVar[str] = "get_datetime"
    description: ClassVar[str] = (
        "Returns the current date and time, or a date/time with a specified offset. "
        "You can specify the format, timezone, and offset. "
        "Use this to get accurate, up-to-date time information or calculate future/past dates."
    )
    args_schema: ClassVar[type[DateTimeArgs]] = DateTimeArgs

    def _run(self, format: Optional[str] = "%Y-%m-%d %H:%M:%S %A", timezone: Optional[str] = "UTC", offset: Optional[str] = None) -> str:
        try:
            if timezone.upper() == "UTC":
                tz = datetime.timezone.utc
            else:
                tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            return f"Invalid timezone: {timezone}. Please use a valid IANA timezone name or 'UTC'."

        try:
            current_datetime = datetime.now(tz)
            
            if offset:
                current_datetime = self._apply_offset(current_datetime, offset)
            
            formatted_datetime = current_datetime.strftime(format)
            offset_str = f" with offset {offset}" if offset else ""
            return f"Date and time{offset_str} ({timezone}): {formatted_datetime}"
        except ValueError as e:
            return f"Error: {str(e)}. Please check your format string and offset."

    def _apply_offset(self, dt: datetime, offset: str) -> datetime:
        match = re.match(r'([+-])(\d+)(minutes|hours|days|weeks|months|years)$', offset)
        if not match:
            raise ValueError(f"Invalid offset format: {offset}")
        
        sign, amount, unit = match.groups()
        amount = int(amount) * (-1 if sign == '-' else 1)
        
        if unit == 'minutes':
            return dt + timedelta(minutes=amount)
        elif unit == 'hours':
            return dt + timedelta(hours=amount)
        elif unit == 'days':
            return dt + timedelta(days=amount)
        elif unit == 'weeks':
            return dt + timedelta(weeks=amount)
        elif unit == 'months':
            return dt + relativedelta(months=amount)
        elif unit == 'years':
            return dt + relativedelta(years=amount)

    async def _arun(self, format: Optional[str] = "%Y-%m-%d %H:%M:%S %A", timezone: Optional[str] = "UTC", offset: Optional[str] = None) -> str:
        return self._run(format, timezone, offset)