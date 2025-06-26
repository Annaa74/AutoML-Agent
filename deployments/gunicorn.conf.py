# deployments/gunicorn.conf.py

# Number of worker processes.
# A common pattern is `2 * NUM_CORES + 1`.
workers = 2 # Adjust based on your server's CPU cores

# The socket to bind to (IP address and port)
bind = "0.0.0.0:5000" # For the Flask backend

# Worker class (sync, gevent, eventlet, etc.)
worker_class = "sync"

# Logging
accesslog = "-" # Log to stdout
errorlog = "-"  # Log to stderr

# Debugging (set to False in production)
# reload = True # Reload workers on code changes (useful for development)
loglevel = "info"

# Timeout for workers (in seconds)
timeout = 120 # Adjust if your ML tasks take a long time to start
