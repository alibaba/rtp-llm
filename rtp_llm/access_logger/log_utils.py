import logging
import os
import shutil
from typing import Optional

from rtp_llm.access_logger.async_log_handler import AsyncRotatingFileHandler


def copy_aggregate_logs_script(log_path: str) -> None:
    """
    Copy aggregate_logs.py script to the same directory as log files.
    This allows users to easily access the log aggregation tool.

    Args:
        log_path: The directory where log files are stored
    """
    if not log_path or log_path == "":
        return

    try:
        # Find the path to aggregate_logs.py in the same directory (access_logger)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_script = os.path.join(current_dir, "aggregate_logs.py")

        # Target location in log directory
        target_script = os.path.join(log_path, "aggregate_logs.py")

        # Copy if source exists and target doesn't exist or is older
        if os.path.exists(source_script):
            should_copy = True
            if os.path.exists(target_script):
                # Only copy if source is newer than target
                source_mtime = os.path.getmtime(source_script)
                target_mtime = os.path.getmtime(target_script)
                should_copy = source_mtime > target_mtime

            if should_copy:
                os.makedirs(log_path, exist_ok=True)
                shutil.copy2(source_script, target_script)
                # Make it executable
                os.chmod(target_script, 0o755)
                logging.info(f"Copied aggregate_logs.py to {target_script}")
    except Exception as e:
        # Log the error but don't fail the application
        logging.warning(f"Failed to copy aggregate_logs.py to log directory: {e}")


def get_process_log_filename(
    base_filename: str, rank_id: Optional[int] = None, server_id: Optional[int] = None
) -> str:
    """
    Generate process-specific log filename.

    Args:
        base_filename: Base filename like 'access.log' or 'query_access.log'
        rank_id: Process rank ID
        server_id: Server ID

    Returns:
        Process-specific filename like 'access_r0_s1.log'
    """
    if rank_id is None:
        rank_id = 0
    if server_id is None:
        server_id = 0

    # Split base filename and extension
    name_parts = base_filename.rsplit(".", 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        return f"{name}_r{rank_id}_s{server_id}.{ext}"
    else:
        return f"{base_filename}_r{rank_id}_s{server_id}"


def get_handler(
    file_name: str,
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> Optional[logging.Handler]:
    """
    Create log handler with process-specific filename.

    Args:
        file_name: Base log filename
        rank_id: Process rank ID
        server_id: Server ID
        async_mode: Whether to use async log handler (default: True)

    Returns:
        RotatingFileHandler or AsyncRotatingFileHandler for process-specific log file
    """
    if log_path == "":
        return None

    # Ensure log directory exists
    os.makedirs(log_path, exist_ok=True)

    # Copy aggregate_logs.py to log directory during first handler creation
    copy_aggregate_logs_script(log_path)

    # Generate process-specific filename
    process_filename = get_process_log_filename(file_name, rank_id, server_id)

    if async_mode:
        # Use async handler
        return AsyncRotatingFileHandler(
            filename=f"{log_path}/{process_filename}",
            mode="a",
            max_bytes=100 * 1024 * 1024,
            backup_count=backup_count,
            encoding="utf-8",
            max_queue_size=100000,  # Large queue to handle bursts
            flush_interval=1.0,  # Flush every second
        )
    else:
        # Use synchronous handler
        from logging.handlers import RotatingFileHandler

        return RotatingFileHandler(
            filename=f"{log_path}/{process_filename}",
            mode="a",
            maxBytes=100 * 1024 * 1024,
            backupCount=backup_count,
            use_gzip=True,
        )
