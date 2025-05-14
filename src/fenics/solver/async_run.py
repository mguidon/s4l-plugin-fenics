import asyncio
import logging
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import aiodocker
import click
from s4l_core.simulator_plugins.base.solver.project_runner import check_src_dir
from s4l_core.simulator_plugins.base.solver.project_runner_dind import (
    ProjectRunnerDind,
    create_default_config,
    create_tar_buffer_from_dir,
)

logger = logging.getLogger(__name__)


async def run_project_in_container(work_dir: str, src_dir: str) -> bool:
    runner = ProjectRunnerDind(work_dir, src_dir)
    await runner.start()
    return await runner.wait_for_result()


async def run_pyright_in_container(src_dir) -> bool:
    docker = aiodocker.Docker()
    ok = True

    try:
        src_tar_buffer = create_tar_buffer_from_dir(Path(src_dir))

        container = await docker.containers.create(
            name=str(uuid4()),
            config=create_default_config(
                ["source /root/.nvm/nvm.sh && echo checking: *.py && pyright *.py"]
            ),
        )

        logger.info("Uploading src code to container")
        await container.put_archive("/", src_tar_buffer.getvalue())

        logger.info("Starting container")
        await container.start()

        logger.info("Waiting for container to finish check with pyright")
        result = await container.wait()

        stdout = await container.log(stdout=True)
        stderr = await container.log(stderr=True)
        logger.info(
            f"finished pyright check:\n StatusCode: {result['StatusCode']} \n\n----stdout---:\n\n{''.join(stdout)}\n----stderr---:\n{''.join(stderr)}"
        )

        ok = result["StatusCode"] == 0

    except Exception:
        logger.info(f"Encountered exception: {traceback.format_exc()}")
        ok = False
    finally:
        await docker.close()

    return ok


async def run(work_dir: str, src_dir: str, run_pyright: bool) -> None:
    check_src_dir(src_dir)

    if run_pyright:
        ok = await run_pyright_in_container(src_dir)
        if not ok:
            logger.error("source code invalid, exiting")
            sys.exit(1)

        if work_dir == "":
            logger.info("No work_dir specified, finishing after pyright check")
            sys.exit(0)

    await run_project_in_container(work_dir, src_dir)


@click.command()
@click.option(
    "--work_dir",
    default="",
    help="Work directory, should have a sub-folder called 'input_files/' with the problem specification.",
)
@click.option(
    "--src_dir",
    default=(Path(__file__).parent / "driver").resolve(),
    help="Source code directory should contain the main.py (and *utils) which processes the 'input_files/'.",
)
@click.option(
    "--run_pyright",
    default=False,
    help="Check the src code using pyright",
)
def main(work_dir: str, src_dir: str, run_pyright: bool):
    """
    Run the fenics project in the work_dir using the code in src_dir.
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(work_dir, src_dir, run_pyright))
    loop.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
