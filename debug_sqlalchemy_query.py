#!/usr/bin/env python3
"""Debug SQLAlchemy query issue."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.storage.database import get_db, create_tables
from app.storage.models import WorkflowRunModel

def debug_sqlalchemy_query():
    """Debug the SQLAlchemy query that's causing the tuple index error."""
    print("🐛 Debugging SQLAlchemy Query")
    print("=" * 40)
    
    # Create tables
    create_tables()
    print("✅ Tables created")
    
    try:
        # Get database session
        db = next(get_db())
        
        # Try to query all workflow runs
        print("Querying all workflow runs...")
        try:
            runs = db.query(WorkflowRunModel).all()
            print(f"✅ Found {len(runs)} workflow runs")
            
            for run in runs:
                print(f"   Run ID: {run.id}")
                print(f"   Status: {run.status}")
                print(f"   Graph ID: {run.graph_id}")
                
        except Exception as e:
            print(f"❌ Error querying all runs: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Try to query a specific run
        print("\nQuerying specific run...")
        try:
            run_id = "d0b22523-5f6a-458a-9dc3-53d6fa66f6d2"  # From the database debug
            run = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
            
            if run:
                print(f"✅ Found run: {run.id}")
                print(f"   Status: {run.status}")
                print(f"   Current state type: {type(run.current_state)}")
            else:
                print("❌ Run not found")
                
        except Exception as e:
            print(f"❌ Error querying specific run: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Database connection error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            db.close()
        except:
            pass

if __name__ == "__main__":
    debug_sqlalchemy_query()