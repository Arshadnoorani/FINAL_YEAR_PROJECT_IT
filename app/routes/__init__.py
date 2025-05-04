# __init__.py - Initializes the routes package

from app.routes import predict, health

# Define a list of available routes
__all__ = ["predict", "health"]