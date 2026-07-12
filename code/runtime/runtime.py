class Runtime:
    def __init__(self):
        self.state = {}
        self.is_running = False
    
    def start(self):
        """Start the runtime."""
        self.is_running = True
    
    def stop(self):
        """Stop the runtime."""
        self.is_running = False
    
    def get_state(self):
        """Get the current runtime state."""
        return self.state


# module level singleton ----------------------------------------------------
# For convenience the package exposes a single shared runtime instance and a
# helper method to retrieve it.  This mirrors patterns often used by
# application frameworks and makes it easy for other modules (like the API)
# to grab the runtime without needing to manage their own instance.

_runtime_instance: Runtime | None = None


def get_runtime() -> Runtime:
    """Return the shared :class:`Runtime` instance.

    The first time this is called the runtime object is created and started.
    Subsequent calls return the same object.
    """

    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = Runtime()
        _runtime_instance.start()
    return _runtime_instance


if __name__ == "__main__":
    runtime = Runtime()
    runtime.start()
