import os

local_file = os.path.join(os.path.dirname(__file__), "..", "local.env")

if os.path.exists(local_file):
    with open(local_file, "r") as contents:
        for f in contents.readlines():
            k, v  = f.split("=")
            os.environ[k] = v