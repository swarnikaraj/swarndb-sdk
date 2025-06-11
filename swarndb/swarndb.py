"""
Base connection interface for SwarndbDB
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from .common import URI


class SwarndbConnection(ABC):
    """Abstract base class for SwarndbDB connections."""
    __slots__ = ('uri',)  # Performance optimization using slots

    def __init__(self, uri: URI):
        self.uri = uri

    @abstractmethod
    def create_database(
        self,
        db_name: str,
        options: Optional[Dict[str, Any]] = None,
        comment: Optional[str] = None
    ) -> None:
        """Create a new database."""
        pass

    @abstractmethod
    def list_databases(self) -> List[str]:
        """List all databases."""
        pass

    @abstractmethod
    def show_database(self, db_name: str) -> Dict[str, Any]:
        """Show database details."""
        pass

    @abstractmethod
    def drop_database(
        self,
        db_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Drop a database."""
        pass

    @abstractmethod
    def get_database(self, db_name: str) -> 'SwarndbDatabase':
        """Get a database instance."""
        pass

    @abstractmethod
    def show_current_node(self) -> Dict[str, Any]:
        """Show current node information."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the database."""
        pass