import resource
import logging
import os
import sys

# Tee class for dual output
class Tee:
    """Write to multiple outputs simultaneously with terminal compatibility"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        if obj is None: return # Safety check
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        # Returns True if the first file handle (usually stdout) is a terminal
        return bool(self.files) and hasattr(self.files[0], 'isatty') and self.files[0].isatty()

def rss_mib() -> float:
    """
    Return the current process RSS (Resident Set Size) in MiB.
    Linux: resource.getrusage returns ru_maxrss in KiB.
    macOS: ru_maxrss is in bytes — not used on this cluster but handled gracefully.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

class SafeFormatter(logging.Formatter):
    """
    Custom formatter that gracefully handles log records that are missing
    the 'rank' and 'batch_id' extra fields.
    Used by setup_file_logger and setup_stream_logger.
    """
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'rank'):
            record.rank = 'N/A'       # type: ignore[attr-defined]
        if not hasattr(record, 'batch_id'):
            record.batch_id = 'N/A'   # type: ignore[attr-defined]
        return super().format(record)

_LOG_FMT = "%(asctime)s - Rank:%(rank)s - Batch:%(batch_id)s - PID:%(process)d - %(message)s"

def setup_file_logger(
        rank:     int,
        batch_id: int,
        log_dir:  str,
        run_name: str = "",
) -> logging.Logger:
    """
    Creates (or retrieves) a logger that writes to a dedicated .log file
    for the given (rank, batch_id) pair. Overwrites on each run (mode='w').

    Parameters
    ----------
    rank     : MPI rank of the parent process.
    batch_id : 0-based micro-batch index within this rank.
    log_dir  : Directory where the .log file will be written.
    run_name : Optional tag inserted into the filename (default: "").
    """
    log_file    = os.path.join(log_dir, f"worker_log{run_name}-rank_{rank}_batch_{batch_id}.log")
    logger_name = f"file_logger_rank_{rank}_batch_{batch_id}"
    logger      = logging.getLogger(logger_name)

    if not logger.handlers:    # guard against duplicate handlers on worker reuse
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(SafeFormatter(_LOG_FMT))
        logger.addHandler(handler)

    return logger

def setup_stream_logger(
        rank:     int,
        batch_id: int,
) -> logging.Logger:
    """
    Creates (or retrieves) a logger that writes to sys.stdout.
    Workers are forked after Tee is installed, so this output lands in
    both the per-rank Tee .txt file and the SLURM .out file.

    Parameters
    ----------
    rank     : MPI rank of the parent process.
    batch_id : 0-based micro-batch index within this rank.
    """
    logger_name = f"stream_logger_rank_{rank}_batch_{batch_id}"
    logger      = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SafeFormatter(_LOG_FMT))
        logger.addHandler(handler)

    return logger