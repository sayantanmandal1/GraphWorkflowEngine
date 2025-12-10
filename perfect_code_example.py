"""
Perfect Code Example - Demonstrates best practices for the workflow engine to review.
This file contains well-structured, documented, and properly formatted Python code.
"""

import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    """Represents a user in the system."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

    def __post_init__(self):
        """Validate user data after initialization."""
        if not self.username or len(self.username) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if "@" not in self.email:
            raise ValueError("Invalid email format")


class UserManager:
    """Manages user operations with proper error handling and logging."""

    def __init__(self):
        self.users: Dict[int, User] = {}
        self.logger = logging.getLogger(__name__)

    def create_user(self, username: str, email: str) -> User:
        """
        Create a new user with validation.

        Args:
            username: The username for the new user
            email: The email address for the new user

        Returns:
            User: The created user object

        Raises:
            ValueError: If username or email is invalid
        """
        try:
            user_id = len(self.users) + 1
            user = User(
                id=user_id,
                username=username,
                email=email,
                created_at=datetime.now()
            )
            self.users[user_id] = user
            self.logger.info(f"Created user: {username} with ID: {user_id}")
            return user
        except ValueError as e:
            self.logger.error(f"Failed to create user {username}: {e}")
            raise

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Retrieve a user by ID.

        Args:
            user_id: The ID of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        return self.users.get(user_id)

    def get_active_users(self) -> List[User]:
        """
        Get all active users.

        Returns:
            List[User]: List of active users
        """
        return [user for user in self.users.values() if user.is_active]

    def deactivate_user(self, user_id: int) -> bool:
        """
        Deactivate a user account.

        Args:
            user_id: The ID of the user to deactivate

        Returns:
            bool: True if user was deactivated, False if not found
        """
        user = self.get_user(user_id)
        if user:
            user.is_active = False
            self.logger.info(f"Deactivated user: {user.username}")
            return True
        return False


def calculate_user_statistics(users: List[User]) -> Dict[str, Union[int, float]]:
    """
    Calculate statistics for a list of users.

    Args:
        users: List of users to analyze

    Returns:
        Dict containing user statistics
    """
    if not users:
        return {"total": 0, "active": 0, "inactive": 0, "active_percentage": 0.0}

    total_users = len(users)
    active_users = sum(1 for user in users if user.is_active)
    inactive_users = total_users - active_users
    active_percentage = (active_users / total_users) * 100 if total_users > 0 else 0.0

    return {
        "total": total_users,
        "active": active_users,
        "inactive": inactive_users,
        "active_percentage": round(active_percentage, 2)
    }


if __name__ == "__main__":
    # Example usage with proper error handling
    logging.basicConfig(level=logging.INFO)

    manager = UserManager()

    try:
        # Create some users
        user1 = manager.create_user("alice_smith", "alice@example.com")
        user2 = manager.create_user("bob_jones", "bob@example.com")
        user3 = manager.create_user("charlie_brown", "charlie@example.com")

        # Deactivate one user
        manager.deactivate_user(user2.id)

        # Get statistics
        all_users = list(manager.users.values())
        stats = calculate_user_statistics(all_users)

        print(f"User Statistics: {stats}")
        print(f"Active users: {len(manager.get_active_users())}")

    except ValueError as e:
        logging.error(f"Error in main execution: {e}")