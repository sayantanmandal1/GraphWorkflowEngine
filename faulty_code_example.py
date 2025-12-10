"""
Faulty Code Example - Contains various issues for the workflow engine to detect.
This file demonstrates common coding problems and anti-patterns.
"""

import logging
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class user:  # Class name should be capitalized
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

    def __post_init__(self):
        # Missing docstring and weak validation
        if len(self.username) < 3:
            raise ValueError("Username too short")
        # No email validation


class UserManager:
    # Missing docstring
    
    def __init__(self):
        self.users = {}  # Missing type hints
        # No logger initialization
        
    def create_user(self, username, email):  # Missing type hints and docstring
        user_id = len(self.users) + 1
        # No error handling
        new_user = user(
            id=user_id,
            username=username,
            email=email,
            created_at=datetime.now()
        )
        self.users[user_id] = new_user
        print(f"Created user: {username}")  # Using print instead of logging
        return new_user
    
    def get_user(self, user_id):  # Missing type hints and return type
        # No docstring
        return self.users.get(user_id)
    
    def get_active_users(self):  # Missing type hints
        active = []
        for user in self.users.values():
            if user.is_active == True:  # Redundant comparison
                active.append(user)
        return active
    
    def deactivate_user(self, user_id):  # Missing type hints and return type
        user = self.get_user(user_id)
        if user != None:  # Should use 'is not None'
            user.is_active = False
            print(f"Deactivated user: {user.username}")  # Using print instead of logging
            return True
        else:
            return False


def calculate_user_statistics(users):  # Missing type hints and docstring
    total_users = len(users)
    active_users = 0
    
    # Inefficient loop
    for i in range(len(users)):
        if users[i].is_active:
            active_users += 1
    
    inactive_users = total_users - active_users
    
    # Potential division by zero
    active_percentage = (active_users / total_users) * 100
    
    # Inconsistent return structure
    return {
        "total": total_users,
        "active": active_users,
        "inactive": inactive_users,
        "percentage": active_percentage  # Different key name than expected
    }


# Poor main execution without proper error handling
if __name__ == "__main__":
    manager = UserManager()
    
    # No try-catch blocks
    user1 = manager.create_user("al", "alice@example.com")  # Username too short
    user2 = manager.create_user("bob_jones", "invalid-email")  # Invalid email
    user3 = manager.create_user("charlie_brown", "charlie@example.com")
    
    manager.deactivate_user(user2.id)
    
    all_users = list(manager.users.values())
    stats = calculate_user_statistics(all_users)
    
    print(f"User Statistics: {stats}")
    print(f"Active users: {len(manager.get_active_users())}")
    
    # Accessing non-existent user without checking
    missing_user = manager.get_user(999)
    print(f"Missing user active status: {missing_user.is_active}")  # Will cause AttributeError