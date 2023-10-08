import logging
import os
import time

import torch

from . import load_model
from .alignment import align, load_align_model
from .transcribe import transcribe
from .utils import write_srt, write_txt


def _work_on_file(device, model, audio_path, output_dir, task):
    logging.info(f"Will {task} {audio_path} with {device}...")
    start = time.time()
    args = {
        "language": None,
        "task": str(task),
        "verbose": False,
    }

    if device == "cpu":
        args["fp16"] = False

    result = transcribe(model, audio_path, temperature=[0], **args)

    lan = result["language"]
    logging.info(f"Language of {audio_path} is {lan}")

    logging.info(f"Alligning {audio_path}...")
    align_model, align_metadata = load_align_model(lan, device)
    result_aligned = align(
        result["segments"],
        align_model,
        align_metadata,
        audio_path,
        device,
        extend_duration=2,
        start_from_previous=True,
        interpolate_method="nearest",
    )
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(audio_path)[0]
    with open(
        os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8"
    ) as txt:
        write_txt(result_aligned["segments"], file=txt)
    with open(
        os.path.join(output_dir, f"{filename}.srt"), "w", encoding="utf-8"
    ) as srt:
        write_srt(result_aligned["segments"], file=srt)
    end = time.time()
    logging.info(f"Processing {audio_path} took {round(end - start)} seconds")


def _load_model(model_size, device):
    logging.info(f"loading Whisper model... (size: {model_size})")
    model = load_model(model_size, device=device)
    logging.info("Whisper model loaded successfully!")
    return model


class Worker:
    def __init__(self, model_size, device, torch_threads):
        self._model = _load_model(model_size, device)
        self._device = device
        torch.set_num_threads(torch_threads)
        logging.info(
            f"initialized worker with device={device}, torch_threads={torch_threads}, model_size={model_size}"
        )

    def is_busy(self):
        return self._working_status

    def work(self, audio_path, output_dir, task):
        try:
            _work_on_file(self._device, self._model, audio_path, output_dir, task)
        except Exception as e:
            logging.error(f"error occurred: {e}")

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


logging.basicConfig(
    format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)
