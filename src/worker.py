import logging
from typing import Iterator, Optional

import gamla


def write_srt(transcript: Iterator[dict]) -> str:
    from whisperx import utils

    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += "\n".join(
            [
                str(i),
                " --> ".join(
                    [
                        utils.format_timestamp(
                            segment["start"],
                            always_include_hours=True,
                            decimal_marker=",",
                        ),
                        utils.format_timestamp(
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
    logging.info(result)
    return result


@gamla.timeit
def work_on_file(
    model,
    audio_path: str,
    task: str,
    language: Optional[str],
):
    from whisperx import alignment

    logging.info(f"{task} {audio_path} {model.device} {language}")
    try:
        result = model.transcribe(audio_path, language=language, task=task)
        align_model, align_metadata = alignment.load_align_model(
            result["language"], model.device
        )
        return write_srt(
            alignment.align(
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
