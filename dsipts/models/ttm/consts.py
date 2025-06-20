DEFAULT_FREQUENCY_MAPPING = {
    "oov": 0,
    "min": 1,  # minutely
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,  # hourly
    "H": 7,  # hourly, for compatibility
    "d": 8,  # daily, for compatibility
    "D": 8,  # daily
    "W": 9,  # weekly
}

TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT = 512