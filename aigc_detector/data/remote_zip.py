"""Read selected members out of a remote ZIP without downloading the whole archive.

WildFake ships as monolithic ZIPs (6-54 GB each). We only want a few hundred
images for a hackathon-scale subset, so downloading a whole archive is wasteful
and will fill a laptop disk. ZIP stores its central directory at the *end* of
the file, so if the server supports HTTP range requests we can:

  1. read the last few KB to parse the central directory (the file listing),
  2. seek directly to the handful of members we actually want,

downloading only a few MB total. `zipfile.ZipFile` accepts any seekable
file-like object, so we just implement one backed by ranged GETs.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request


def _with_retries(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry a flaky network call with exponential backoff.

    Range requests against large WildFake archives occasionally stall or drop
    mid-transfer (observed on multi-GB archives like DDPM.zip) even though the
    same request succeeds a moment later -- transient host/CDN hiccups rather
    than anything wrong with the request itself, so a short retry is the right
    fix rather than raising immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file-like object backed by HTTP range requests."""

    def __init__(self, url: str, timeout: int = 60, user_agent: str = "curl/8"):
        self.url = url
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._pos = 0
        self._size = _with_retries(self._resolve_and_size)

    def _resolve_and_size(self) -> int:
        """Fetch the total size, and pin the post-redirect URL for later requests.

        Hosts like ModelScope redirect the API path to a signed CDN object. That
        redirect costs several seconds *per request*, which dominates runtime
        when reading many small members. Resolving it once and reusing the final
        URL turns ~6s requests into ~0.1s ones.
        """
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise RuntimeError(
                    f"Server does not support HTTP range requests (status {r.status}): {self.url}"
                )
            content_range = r.headers.get("Content-Range", "")
            self.url = r.url  # pin the resolved (possibly signed CDN) URL
            r.read()
        try:
            return int(content_range.split("/")[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"Could not parse Content-Range {content_range!r}") from exc

    # -- io plumbing -------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""
        start, end = self._pos, self._pos + size - 1

        def do_request() -> bytes:
            req = urllib.request.Request(
                self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()

        data = _with_retries(do_request)
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    @property
    def size(self) -> int:
        return self._size


def open_remote_zip(url: str, timeout: int = 60):
    """Return a `zipfile.ZipFile` over a remote archive, read lazily via range requests."""
    import zipfile

    # Buffer so zipfile's many small reads don't each become an HTTP request.
    raw = HttpRangeFile(url, timeout=timeout)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
