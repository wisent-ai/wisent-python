"""
The defaults every client and configuration in this package starts from.
"""

#: One request to the Wisent API waits at most a minute.
DEFAULT_TIMEOUT_SECONDS = 60

#: A listing answers this many records unless the caller asks for a page size.
DEFAULT_PAGE_SIZE = 100

#: Generation sampling: the completion budget, temperature, nucleus and top-k cut.
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
