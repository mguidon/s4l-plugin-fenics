import logging
import shutil
import sys
from io import StringIO
from pathlib import Path

from mpi4py import MPI

RANK = MPI.COMM_WORLD.rank


def init_results_dir(results_dir: Path) -> None:

    if RANK == 0:

        if results_dir.is_dir():
            shutil.rmtree(results_dir)

        results_dir.mkdir(parents=True, exist_ok=True)

    MPI.COMM_WORLD.barrier()


def init_logging(results_dir: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        filename=results_dir / f"py_log_rank{RANK}.log",
    )
    logger = logging.getLogger("main")
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
            )

        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = handle_exception

    return logger


class TqdmLogger(StringIO):
    def __init__(self, logger):
        super().__init__()

        self._logger = logger

    def write(self, buffer):
        self._logger.info(buffer)

    def flush(self):
        pass
