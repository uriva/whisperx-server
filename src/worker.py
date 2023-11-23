import logging
from typing import Iterator, Optional

import gamla
from whisperx.alignment import align, load_align_model
from whisperx.asr import FasterWhisperPipeline
from whisperx.utils import format_timestamp


def write_srt(transcript: Iterator[dict]) -> str:
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += "\n".join(
            [
                str(i),
                " --> ".join(
                    [
                        format_timestamp(
                            segment["start"],
                            always_include_hours=True,
                            decimal_marker=",",
                        ),
                        format_timestamp(
                            segment["end"],
                            always_include_hours=True,
                            decimal_marker=",",
                        ),
                    ]
                ),
                segment["text"].strip().replace("-->", "->"),
                "",
            ]
        )
    return result


@gamla.throttle(1)
@gamla.timeit
def work_on_file(
    model: FasterWhisperPipeline, audio_path: str, task: str, language: Optional[str]
):
    logging.info(f"{task} {audio_path} {model.device} {language}")
    try:
        result = model.transcribe(audio_path, language=language, task=task)
        align_model, align_metadata = load_align_model(result["language"], model.device)
        return write_srt(
            align(
                result["segments"],
                align_model,
                align_metadata,
                audio_path,
                model.device,
            )["segments"]
        )
    except Exception as e:
        logging.error(e)
        return None
