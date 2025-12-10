#!/usr/bin/env python3
"""Debug database schema and content."""

import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.storage.database import create_tables

def debug_database():
    """Debug the database schema and content."""
    print("🐛 Debugging Database")
    print("=" * 40)
    
    # Create tables
    create_tables()
    print("✅ Tables created")
    
    # Connect to database directly
    conn = sqlite3.connect("workflow_engine.db")
    cursor = conn.cursor()
    
    try:
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\n📋 Tables: {[table[0] for table in tables]}")
        
        # Check workflow_runs table schema
        cursor.execute("PRAGMA table_info(workflow_runs);")
        columns = cursor.fetchall()
        print(f"\n📊 workflow_runs columns:")
        for col in columns:
            print(f"   {col[1]} ({col[2]}) - PK: {col[5]}, NotNull: {col[3]}")
        
        # Check if there are any records
        cursor.execute("SELECT COUNT(*) FROM workflow_runs;")
        count = cursor.fetchone()[0]
        print(f"\n📈 workflow_runs record count: {count}")
        
        if count > 0:
            cursor.execute("SELECT * FROM workflow_runs LIMIT 5;")
            records = cursor.fetchall()
            print(f"\n📝 Sample records:")
            for record in records:
                print(f"   {record}")
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    debug_database()