CACHE_DIR = "src/project/apis/model_serving/cache"

APP_CONFIG = {
    "DEBUG": False,
    "CACHE_TYPE": "FileSystemCache",  # for multithread server with gunicorn, use 'FileSystemCache'
    "CACHE_DEFAULT_TIMEOUT": 300,  # 1 hour,
    "CACHE_THRESHOLD": 10000,
    "CACHE_DIR": f"{CACHE_DIR}",
}


ENV = "base"
