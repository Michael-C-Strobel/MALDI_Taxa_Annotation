import logging
from decimal import Decimal

def init_logging(debug:bool=True):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def convert_to_serializable(obj):
    """Convert Decimal types in the object to float."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float
    return obj
