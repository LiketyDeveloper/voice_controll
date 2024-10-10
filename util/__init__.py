import os

def get_path(*args):
    """Return the path to a file in the main directory."""
    # Check that the arguments are not empty
    if not args:
        raise ValueError("At least one argument is required")

    # Check that the arguments are strings
    if not all(isinstance(arg, str) for arg in args):
        raise ValueError("All arguments must be strings")

    path = os.path.join(os.path.dirname(__file__), "..", *args)

    return path
