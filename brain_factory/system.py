from contextlib import contextmanager
from typing import Iterator

from brain_factory.lib.omegaconf_resolvers import register_resolvers


@contextmanager
def configure_hydra() -> Iterator[None]:
    register_resolvers()
    yield
