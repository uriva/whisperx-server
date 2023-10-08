import logging
import os
import time

import torch

from . import load_model
from .alignment import align, load_align_model
from .transcribe import transcribe
from .utils import write_srt, write_txt


def work_on_file(model, audio_path, output_dir, task):
    logging.info(f"Will {task} {audio_path} with {model.device}...")
    start = time.time()
    args = {
        "language": None,
        "task": str(task),
        "verbose": False,
    }

    if model.device == "cpu":
        args["fp16"] = False

    result = transcribe(model, audio_path, temperature=[0], **args)

    lan = result["language"]
    logging.info(f"Language of {audio_path} is {lan}")

    logging.info(f"Alligning {audio_path}...")
    align_model, align_metadata = load_align_model(lan, model.device)
    result_aligned = align(
        result["segments"],
        align_model,
        align_metadata,
        audio_path,
        model.device,
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


def setup_model(model_size, device, num_threads):
    torch.set_num_threads(num_threads)
    logging.info("loading Whisper model...")
    model = load_model(model_size, device=device)
    logging.info("Whisper model loaded successfully!")
    return model
