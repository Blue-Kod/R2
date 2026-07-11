class EveryLLMError(Exception):
    pass


class ProviderError(EveryLLMError):
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    pass


class AuthenticationError(ProviderError):
    pass


class TimeoutError(ProviderError):
    pass
