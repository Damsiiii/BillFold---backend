"""Financial data context integration interface.

Extracts real financial metrics (income, expenses, category spending, budget recommendations)
from Member 2 (Database) and Member 3 (Analytics) to ground chatbot responses.
"""

from typing import Optional
from app.schemas.chatbot import FinancialContext


class FinancialContextProvider:
    """Interface to fetch user financial metrics from backend services.

    When Member 2 (Database) and Member 3 (Analytics) deliver their modules,
    hook their database queries and calculation routines into `get_user_context`.
    """

    def __init__(self, db_service: Optional[object] = None, analytics_service: Optional[object] = None):
        self.db_service = db_service
        self.analytics_service = analytics_service

    def get_user_context(self, user_id: str) -> FinancialContext:
        """Retrieves structured financial facts for the given user.

        If actual data is unavailable, returns has_data=False so the chatbot
        does not hallucinate or guess numbers.
        """
        # If external modules are connected in the future:
        if self.db_service and self.analytics_service:
            try:
                # Example future hook:
                # summary = self.analytics_service.get_summary(user_id)
                # categories = self.analytics_service.get_categories(user_id)
                pass
            except Exception:
                return FinancialContext(user_id=user_id, has_data=False)

        # Default placeholder: no data loaded yet
        # shortcut: no persistent DB connected yet, upgrade when Member 2 and 3 modules arrive
        return FinancialContext(user_id=user_id, has_data=False)


# Global default context provider instance
_default_context_provider = FinancialContextProvider()


def get_context_provider() -> FinancialContextProvider:
    return _default_context_provider


def set_context_provider(provider: FinancialContextProvider) -> None:
    global _default_context_provider
    _default_context_provider = provider

