import argparse
import asyncio
import logging
import os.path

from aiohttp import web

from whisperx.worker import setup_model, work_on_file


def _parse_args():
    parser = argparse.ArgumentParser(description="Process some flags.")

    parser.add_argument(
        "--device",
        default=os.environ.get("WHISPER_DEVICE", "cpu"),
        help="Specify the device (default: cpu)",
    )

    parser.add_argument(
        "--torch_threads",
        type=int,
        default=int(os.getenv("TORCH_THREADS", 1)),
        help="Specify the number of Torch threads (default: 1)",
    )

    parser.add_argument(
        "--model_size",
        default=os.environ.get("WHISPER_MODEL", "large-v2"),
        choices=["small", "medium", "large", "large-v2"],
        help="Specify the model size (default: large-v2)",
    )
    args = parser.parse_args()

    if args.model_size not in ["small", "medium", "large", "large-v2"]:
        logging.error(
            "invalid WHISPER_MODEL value. Must be one of ['small', 'medium', 'large', 'large-v2']"
        )
    return args.model_size, args.device, args.torch_threads


def _with_worker(model):
    async def handler(request):
        params = await request.json()
        audio_path = params.get("audioPath")
        output_dir = params.get("outputDir")
        task = params.get("task")
        sync = params.get("sync")
        if audio_path is None:
            return web.Response(
                body={"message": "'audioPath' key is missing"}, status=400
            )
        if not audio_path:
            return web.Response(
                body={"message": "'pathToFile' value is not a string"}, status=400
            )
        if not os.path.isfile(audio_path):
            return web.Response(
                body={"message": f"the file at path '{audio_path}' was not found"},
                status=400,
            )
        if task not in [None, "translate", "transcribe"]:
            return web.Response(
                body={
                    "message": f"Invalid task value [{task}]. Must be one of [translate, transcribe]"
                },
                status=400,
            )
        logging.info(f"request received: {task} [{audio_path} -> {output_dir}]")
        future = asyncio.get_event_loop().run_in_executor(
            None,
            work_on_file,
            model,
            audio_path,
            output_dir or os.path.dirname(os.path.abspath(audio_path)),
            task or "transcribe",
        )
        if sync:
            await future
        return web.Response(status=200)

    return handler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app.add_routes([web.post("/transcribe", _with_worker(setup_model(*_parse_args())))])
    web.run_app(app, port=8080)
