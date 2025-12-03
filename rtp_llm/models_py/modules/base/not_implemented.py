"""
Base module utilities and special operation classes.
"""


class NotImplementedOp:
    """
    Placeholder class for operations not implemented on certain architectures.
    Raises an error when instantiated to provide clear feedback to users.
    """

    def __init__(self, op_name: str = None, device_type: str = "current device"):
        """
        Initialize NotImplementedOp.

        Args:
            op_name: Name of the operation that is not implemented
            device_type: The device/architecture where this operation is not available

        Raises:
            NotImplementedError: Always raised to indicate the operation is not supported
        """
        if op_name is None:
            op_name = self.__class__.__name__

        raise NotImplementedError(
            f"{op_name} is not implemented for {device_type}. "
            f"This operation may only be available on specific hardware architectures."
        )
