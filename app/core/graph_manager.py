"""Graph Manager for workflow definition handling."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..models.core import (
    GraphDefinition, 
    GraphSummary, 
    ValidationResult
)
from ..storage.database import get_db
from ..storage.models import GraphModel
from .exceptions import GraphValidationError, StorageError
from .logging import get_logger

logger = get_logger(__name__)


class GraphManager:
    """Manages workflow graph definitions, validation, and storage."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize GraphManager with optional database session."""
        self._db_session = db_session
    
    def _get_db_session(self) -> Session:
        """Get database session, creating one if not provided."""
        if self._db_session:
            return self._db_session
        return next(get_db())
    
    def create_graph(self, graph_definition: GraphDefinition) -> str:
        """
        Create a new workflow graph and return its unique identifier.
        
        Args:
            graph_definition: The graph definition to create
            
        Returns:
            str: Unique graph identifier
            
        Raises:
            GraphValidationError: If graph validation fails
            StorageError: If storage operation fails
        """
        logger.info(f"Creating new graph: {graph_definition.name}")
        
        # Validate the graph structure
        validation_result = self.validate_graph(graph_definition)
        if not validation_result.is_valid:
            error_msg = f"Graph validation failed: {'; '.join(validation_result.errors)}"
            logger.error(error_msg)
            raise GraphValidationError(error_msg)
        
        # Log warnings if any
        if validation_result.warnings:
            logger.warning(f"Graph validation warnings: {'; '.join(validation_result.warnings)}")
        
        # Generate unique ID
        graph_id = self._generate_unique_id()
        
        # Store in database
        try:
            db = self._get_db_session()
            
            # Check if graph with same name already exists
            existing_graph = db.query(GraphModel).filter(GraphModel.name == graph_definition.name).first()
            if existing_graph:
                raise GraphValidationError(f"Graph with name '{graph_definition.name}' already exists")
            
            # Create database model
            graph_model = GraphModel(
                id=graph_id,
                name=graph_definition.name,
                description=graph_definition.description,
                definition=graph_definition.model_dump(),
                created_at=datetime.utcnow()
            )
            
            db.add(graph_model)
            db.commit()
            
            logger.info(f"Successfully created graph '{graph_definition.name}' with ID: {graph_id}")
            return graph_id
            
        except GraphValidationError:
            # Re-raise validation errors as-is
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error while creating graph: {str(e)}")
            raise StorageError(f"Failed to store graph: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while creating graph: {str(e)}")
            raise StorageError(f"Failed to create graph: {str(e)}")
    
    def get_graph(self, graph_id: str) -> GraphDefinition:
        """
        Retrieve a graph definition by its ID.
        
        Args:
            graph_id: The unique identifier of the graph
            
        Returns:
            GraphDefinition: The graph definition
            
        Raises:
            StorageError: If graph not found or storage operation fails
        """
        logger.debug(f"Retrieving graph with ID: {graph_id}")
        
        try:
            db = self._get_db_session()
            graph_model = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
            
            if not graph_model:
                raise StorageError(f"Graph with ID '{graph_id}' not found")
            
            # Convert stored definition back to GraphDefinition
            graph_definition = GraphDefinition(**graph_model.definition)
            logger.debug(f"Successfully retrieved graph: {graph_definition.name}")
            return graph_definition
            
        except SQLAlchemyError as e:
            logger.error(f"Database error while retrieving graph: {str(e)}")
            raise StorageError(f"Failed to retrieve graph: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while retrieving graph: {str(e)}")
            raise StorageError(f"Failed to retrieve graph: {str(e)}")
    
    def validate_graph(self, graph_definition: GraphDefinition) -> ValidationResult:
        """
        Validate a graph definition for structural correctness.
        
        Args:
            graph_definition: The graph definition to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        logger.debug(f"Validating graph: {graph_definition.name}")
        
        try:
            # Use the built-in validation from the GraphDefinition model
            validation_result = graph_definition.validate_structure()
            
            # Collect additional errors and warnings
            additional_errors = []
            additional_warnings = []
            
            # Additional custom validations
            self._validate_cycles(graph_definition, additional_errors, additional_warnings)
            self._validate_unreachable_nodes(graph_definition, additional_errors, additional_warnings)
            self._validate_invalid_references(graph_definition, additional_errors, additional_warnings)
            
            # Combine all errors and warnings
            all_errors = validation_result.errors + additional_errors
            all_warnings = validation_result.warnings + additional_warnings
            
            # Create final validation result
            final_result = ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings
            )
            
            logger.debug(f"Graph validation completed. Valid: {final_result.is_valid}, "
                        f"Errors: {len(final_result.errors)}, Warnings: {len(final_result.warnings)}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error during graph validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[]
            )
    
    def list_graphs(self) -> List[GraphSummary]:
        """
        List all available graphs with summary information.
        
        Returns:
            List[GraphSummary]: List of graph summaries
            
        Raises:
            StorageError: If storage operation fails
        """
        logger.debug("Listing all graphs")
        
        try:
            db = self._get_db_session()
            graph_models = db.query(GraphModel).order_by(GraphModel.created_at.desc()).all()
            
            summaries = []
            for model in graph_models:
                # Count nodes from the stored definition
                definition = model.definition
                node_count = len(definition.get('nodes', []))
                
                summary = GraphSummary(
                    id=model.id,
                    name=model.name,
                    description=model.description or "",
                    created_at=model.created_at,
                    node_count=node_count
                )
                summaries.append(summary)
            
            logger.debug(f"Retrieved {len(summaries)} graph summaries")
            return summaries
            
        except SQLAlchemyError as e:
            logger.error(f"Database error while listing graphs: {str(e)}")
            raise StorageError(f"Failed to list graphs: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while listing graphs: {str(e)}")
            raise StorageError(f"Failed to list graphs: {str(e)}")
    
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph by its ID.
        
        Args:
            graph_id: The unique identifier of the graph to delete
            
        Returns:
            bool: True if graph was deleted, False if not found
            
        Raises:
            StorageError: If storage operation fails
        """
        logger.info(f"Deleting graph with ID: {graph_id}")
        
        try:
            db = self._get_db_session()
            graph_model = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
            
            if not graph_model:
                logger.warning(f"Graph with ID '{graph_id}' not found for deletion")
                return False
            
            db.delete(graph_model)
            db.commit()
            
            logger.info(f"Successfully deleted graph with ID: {graph_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error while deleting graph: {str(e)}")
            raise StorageError(f"Failed to delete graph: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while deleting graph: {str(e)}")
            raise StorageError(f"Failed to delete graph: {str(e)}")
    
    def _generate_unique_id(self) -> str:
        """Generate a unique identifier for a graph."""
        return str(uuid.uuid4())
    
    def _validate_cycles(self, graph_definition: GraphDefinition, errors: List[str], warnings: List[str]):
        """
        Validate graph for cycles and update error/warning lists.
        
        Args:
            graph_definition: The graph to validate
            errors: List to append errors to
            warnings: List to append warnings to
        """
        if graph_definition._has_cycles():
            if not graph_definition.exit_conditions:
                errors.append(
                    "Graph contains cycles but no exit conditions are specified. "
                    "This may result in infinite loops."
                )
            else:
                warnings.append(
                    "Graph contains cycles. Ensure exit conditions are properly configured "
                    "to prevent infinite loops."
                )
    
    def _validate_unreachable_nodes(self, graph_definition: GraphDefinition, errors: List[str], warnings: List[str]):
        """
        Validate graph for unreachable nodes and update error/warning lists.
        
        Args:
            graph_definition: The graph to validate
            errors: List to append errors to
            warnings: List to append warnings to
        """
        if not graph_definition.entry_point or not graph_definition.edges:
            return
        
        try:
            node_ids = {node.id for node in graph_definition.nodes}
            reachable = graph_definition._find_reachable_nodes(graph_definition.entry_point, graph_definition.edges)
            unreachable = node_ids - reachable
            
            if unreachable:
                errors.append(
                    f"Unreachable nodes detected: {', '.join(sorted(unreachable))}. "
                    "All nodes must be reachable from the entry point."
                )
        except Exception as e:
            warnings.append(f"Could not validate node reachability: {str(e)}")
    
    def _validate_invalid_references(self, graph_definition: GraphDefinition, errors: List[str], warnings: List[str]):
        """
        Validate graph for invalid node references and update error/warning lists.
        
        Args:
            graph_definition: The graph to validate
            errors: List to append errors to
            warnings: List to append warnings to
        """
        node_ids = {node.id for node in graph_definition.nodes}
        
        # Check entry point reference
        if graph_definition.entry_point and graph_definition.entry_point not in node_ids:
            errors.append(
                f"Entry point '{graph_definition.entry_point}' references non-existent node"
            )
        
        # Check edge references
        for edge in graph_definition.edges:
            if edge.from_node not in node_ids:
                errors.append(
                    f"Edge references non-existent source node: '{edge.from_node}'"
                )
            if edge.to_node not in node_ids:
                errors.append(
                    f"Edge references non-existent target node: '{edge.to_node}'"
                )