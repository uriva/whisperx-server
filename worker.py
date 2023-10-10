import logging
import os
import time
from typing import Iterator

import gamla
import torch
from whisperx import load_model
from whisperx.alignment import align, load_align_model
from whisperx.utils import format_timestamp


def write_srt(transcript: Iterator[dict]) -> str:
    result = ""
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        result += (
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
        )
    return result


@gamla.throttle(1)
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
    result = model.transcribe(model, audio_path, temperature=[0], **args)
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
    result = write_srt(result_aligned["segments"])
    end = time.time()
    logging.info(f"Processing {audio_path} took {round(end - start)} seconds")
    return result


def setup_model(model_size, device, num_threads):
    torch.set_num_threads(num_threads)
    logging.info("loading Whisper model...")
    model = load_model(model_size, device=device)
    logging.info("Whisper model loaded successfully!")
    return model
