import argparse
import logging
import os.path
import threading

from flask import Flask, request
from flask_restful import Api, Resource
from waitress import serve

from whisperx.worker import Worker

app = Flask(__name__)
api = Api(app)


@app.route("/transcribe/")
class TranscribeHandler(Resource):
    def post(self):
        global model
        transcription_request = request.json

        audio_path = transcription_request.get("audioPath")
        output_dir = transcription_request.get("outputDir")
        task = transcription_request.get("task")
        sync = transcription_request.get("sync")
        if audio_path is None:
            return {"message": "'audioPath' key is missing"}, 400

        if not str(audio_path):
            return {"message": "'pathToFile' value is not a string"}, 400

        if not os.path.isfile(audio_path):
            return {"message": f"the file at path '{audio_path}' was not found"}, 404

        if task is not None and str(task) not in ["translate", "transcribe"]:
            return {
                "message": f"the value of task is not valid: '{audio_path}'. Must be one of [translate, transcribe]"
            }, 404

        if task is None:
            logging.info(
                "\"task\" query parameter not provided - using the default 'transcribe'"
            )
            task = "transcribe"

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(audio_path))

        logging.info(f"request received: {task} [{audio_path} -> {output_dir}]")

        if worker.is_busy():
            return {
                "message": "a video is currently being processed, try again later"
            }, 529

        transcribe_thread = threading.Thread(
            target=worker.work,
            name="Transcriber Function",
            args=[audio_path, output_dir, task],
        )
        transcribe_thread.start()
        if sync:
            transcribe_thread.join()
        return {}, 200


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


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

device = args.device
torch_threads = args.torch_threads
model_size = args.model_size

if model_size not in ["small", "medium", "large", "large-v2"]:
    logging.error(
        "invalid WHISPER_MODEL value. Must be one of ['small', 'medium', 'large', 'large-v2']"
    )

worker = Worker(model_size, device, torch_threads)


api.add_resource(TranscribeHandler, "/transcribe")


if __name__ == "__main__":
    serve(app, host="localhost", port=8080)
