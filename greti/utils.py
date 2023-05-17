from datetime import datetime


def get_dt(filename: str) -> datetime:
    """
    Get datetime from filename.
    Args:
        filename (str): Filename of the spectrogram.
    Returns:
        datetime: Datetime of the spectrogram.
    """
    date_time_str = "".join(filename.split("_")[1:3])
    return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
