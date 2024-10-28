import os

def load_env_vars(fn=".env"):
    with open(fn, "r") as f:
        for line in f:
            if line.count("=") == 1 and not line.strip().startswith("#"):
                key, value = line.strip().split("=")
                os.environ[key] = value.strip('"')
