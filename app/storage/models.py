"""SQLAlchemy database models for the workflow engine."""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class GraphModel(Base):
    """Database model for workflow graphs."""
    __tablename__ = "graphs"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    definition = Column(JSON, nullable=False)  # Stores the complete graph definition
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to workflow runs
    runs = relationship("WorkflowRunModel", back_populates="graph")


class WorkflowRunModel(Base):
    """Database model for workflow execution runs."""
    __tablename__ = "workflow_runs"
    
    id = Column(String, primary_key=True)
    graph_id = Column(String, ForeignKey("graphs.id"), nullable=False)
    status = Column(String, nullable=False)  # pending, running, completed, failed, cancelled
    initial_state = Column(JSON)
    current_state = Column(JSON)
    final_state = Column(JSON)
    current_node = Column(String)
    execution_path = Column(JSON)  # List of executed node IDs
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    graph = relationship("GraphModel", back_populates="runs")
    logs = relationship("LogEntryModel", back_populates="run")


class LogEntryModel(Base):
    """Database model for execution log entries."""
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("workflow_runs.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    node_id = Column(String)
    event_type = Column(String, nullable=False)  # node_start, node_complete, node_error, etc.
    message = Column(Text, nullable=False)
    state_snapshot = Column(JSON)
    
    # Relationship
    run = relationship("WorkflowRunModel", back_populates="logs")


class ToolRegistryModel(Base):
    """Database model for registered tools."""
    __tablename__ = "tool_registry"
    
    name = Column(String, primary_key=True)
    description = Column(Text)
    function_module = Column(String, nullable=False)  # Module path where function is defined
    function_name = Column(String, nullable=False)    # Function name within the module
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)