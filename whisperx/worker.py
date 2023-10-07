import json
import logging
import os
import time
from datetime import datetime

import torch

from . import load_model
from .alignment import align, load_align_model
from .transcribe import transcribe
from .utils import write_srt, write_txt


class Worker:
    _model = None
    _working_status = False
    _device = "cpu"

    def __init__(self, model_size, device, torch_threads):
        self._model_size = model_size
        self._device = device
        torch.set_num_threads(torch_threads)
        logging.info(
            f"initialized worker with device={device}, torch_threads={torch_threads}, model_size={model_size}"
        )

    def is_busy(self):
        return self._working_status

    def work(self, audio_path, output_dir, task):
        if self.working_status:
            logging.info("Already working! sorry..")
            return

        self.working_status = True
        model = self.model
        self.work_on_file(model, audio_path, output_dir, task)
        self.working_status = False

    def work_on_file(self, model, audio_path, output_dir, task):
        logging.info(f"Will {task} {audio_path} with {self._device}...")
        start = time.time()

        args = {
            "language": None,
            "task": str(task),
            "verbose": False,
        }

        if self._device == "cpu":
            args["fp16"] = False

        result = transcribe(model, audio_path, temperature=[0], **args)

        lan = result["language"]
        logging.info(f"Language of {audio_path} is {lan}")

        logging.info(f"Alligning {audio_path}...")
        align_model, align_metadata = load_align_model(lan, self._device)
        result_aligned = align(
            result["segments"],
            align_model,
            align_metadata,
            audio_path,
            self._device,
            extend_duration=2,
            start_from_previous=True,
            interpolate_method="nearest",
        )

        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.splitext(audio_path)[0]

        # save TXT
        with open(
            os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8"
        ) as txt:
            write_txt(result_aligned["segments"], file=txt)

        # save SRT
        with open(
            os.path.join(output_dir, f"{filename}.srt"), "w", encoding="utf-8"
        ) as srt:
            write_srt(result_aligned["segments"], file=srt)

        # save metadata JSON
        end = time.time()
        logging.info(
            f"Finished processing {audio_path} within {round(end - start)} seconds (model: {self._model_size})"
        )

        now = datetime.now()  # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        metadata = {
            "language": lan,
            "task": "transcribe",
            "source": audio_path,
            "whisper_model": self._model_size,
            "duration": f"{round(end - start)} seconds",
            "finished_at": date_time,
        }
        with open(
            os.path.join(output_dir, f"{filename}.json"),
            "w",
            encoding="utf-8",
        ) as fd:
            json.dump(metadata, fd)

    def _get_working_status(self):
        return self._working_status

    def _set_working_status(self, value):
        self._working_status = value

    working_status = property(
        fget=_get_working_status,
        fset=_set_working_status,
        doc="Whether the worker is currently working or not.",
    )

    @property
    def model(self):
        if not self._model:
            logging.info(f"lazy loading Whisper model... (size: {self._model_size})")
            self._model = load_model(self._model_size, device=self._device)
            logging.info("Whisper model loaded successfully!")
        return self._model

        # Diarization Code:
        # logging.info("Performing diarization...")
        # hf_token: str = os.environ.get('HF_TOKEN')
        # if hf_token is None:
        #     logging.info("Warning, no huggingface token used, needs to be saved in environment variable HF_TOKEN, otherwise will throw error loading VAD model...")
        # diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
        #                             use_auth_token=hf_token)
        # diarize_segments = diarize_pipeline(audio_path)
        # diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True))
        # diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        # diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        # # assumes each utterance is single speaker (needs fix)
        # result_segments, word_segments = assign_word_speakers(diarize_df, result_aligned["segments"], fill_nearest=True)
        # result_aligned["segments"] = result_segments
        # result_aligned["word_segments"] = word_segments


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
