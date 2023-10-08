import argparse
import logging
import os.path
import threading

from aiohttp import web

from whisperx.worker import Worker


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


def _with_worker(worker):
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
        if not str(audio_path):
            return web.Response(
                body={"message": "'pathToFile' value is not a string"}, status=400
            )
        if not os.path.isfile(audio_path):
            return web.Response(
                body={"message": f"the file at path '{audio_path}' was not found"},
                status=404,
            )
        if task is not None and str(task) not in ["translate", "transcribe"]:
            return web.Response(
                body={
                    "message": f"Invalid task value [{task}]. Must be one of [translate, transcribe]"
                },
                status=404,
            )
        if task is None:
            task = "transcribe"
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(audio_path))
        logging.info(f"request received: {task} [{audio_path} -> {output_dir}]")
        transcribe_thread = threading.Thread(
            target=worker.work,
            name="Transcriber Function",
            args=[audio_path, output_dir, task],
        )
        transcribe_thread.start()
        if sync:
            transcribe_thread.join()
        return web.Response(status=200)

    return handler


if __name__ == "__main__":
    app = web.Application()
    app.add_routes([web.post("/transcribe", _with_worker(Worker(*_parse_args())))])
    web.run_app(app, port=8080)
