try:
    import subprocess
    version = subprocess.check_output(["git", "describe"]).decode().strip()
except:
    version = "unknown"

