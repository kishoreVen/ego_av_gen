
import base64
import logging
import os
import time
from typing import Any, Dict, List, cast

import requests as http_requests
from PIL import Image
from together import Together

from model_router.model_interface import (
    ModelInterface,
    Capability,
    Query,
    ImageGenQuery,
    VideoGenQuery,
)
from model_router.scaffold import ImageGenerationModelInterface
from model_router.lib.media import compress_for_reference, image_to_base64

logger = logging.getLogger(__name__)


class TogetherInterface(ModelInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        return {}


class TogetherFlux2DevStyleTransfer(ImageGenerationModelInterface):
    """FLUX.2 [dev] style transfer via Together AI REST API.

    Together's images.generate() accepts reference_images for style transfer,
    making it a simpler synchronous alternative to the Runware WebSocket API.
    """

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self.client = None

    def initialize_client(self) -> None:
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def _fetch_image_response(
        self, query: ImageGenQuery, capability: Capability | None = None
    ) -> Dict[str, Any]:
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        text_prompt = query.make_query() or "apply style"

        height = query.image_resolution[1] if query.image_resolution else 512
        width = query.image_resolution[0] if query.image_resolution else 512

        if height <= 0 or width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        # Convert reference images to base64 data URIs
        reference_images = None
        if query.images:
            raw = query.images if isinstance(query.images, list) else [query.images]
            reference_images = compress_for_reference(raw)

        response = self.client.images.generate(
            model="black-forest-labs/FLUX.2-dev",
            prompt=text_prompt,
            height=height,
            width=width,
            seed=self.seed,
            steps=query.generation_steps if query.generation_steps else 25,
            guidance_scale=3.5,
            n=query.number_of_results if query.number_of_results else 1,
            response_format="base64",
            reference_images=reference_images,
        )

        images = [img.b64_json for img in response.data]
        return {"images": images}


class TogetherKling21(ModelInterface):
    """Kling 2.1 video generation via Together AI REST API.

    Uses direct HTTP calls to POST /v2/videos since the installed Together SDK
    doesn't include video support yet.  Submits a job and polls until complete.

    Supports text-to-video and image-to-video (first frame via frame_images).
    """

    _BASE_URL = "https://api.together.xyz"
    _POLL_INTERVAL = 5.0
    _MAX_POLL_TIME = 600.0

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name = model_name
        self._session: http_requests.Session | None = None

    def initialize_client(self) -> None:
        self._session = http_requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
            "Content-Type": "application/json",
        })

    def requires_initialization(self) -> bool:
        return self._session is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.VIDEO_GEN]

    def _resolve_resolution(
        self, query: VideoGenQuery
    ) -> tuple[int, int]:
        """Pick the closest supported 1080p resolution for Kling 2.1.

        Supported: 1920x1080 (16:9), 1080x1920 (9:16), 1080x1080 (1:1).
        """
        if query.aspect_ratio == "9:16" or (
            query.video_resolution
            and query.video_resolution[0] < query.video_resolution[1]
        ):
            return 1080, 1920
        if query.aspect_ratio == "1:1" or (
            query.video_resolution
            and query.video_resolution[0] == query.video_resolution[1]
        ):
            return 1080, 1080
        return 1920, 1080  # default 16:9

    def _image_to_data_uri(self, image: str | Image.Image) -> str:
        """Convert image to data URI for Together API frame_images."""
        if isinstance(image, Image.Image):
            return image_to_base64(image, include_data_uri_prefix=True)
        if isinstance(image, str) and not image.startswith("data:"):
            return f"data:image/jpeg;base64,{image}"
        return image

    def _submit_job(self, query: VideoGenQuery) -> str:
        """Submit a video generation job, return job ID."""
        if self._session is None:
            raise ValueError("Call initialize_client before querying")

        text_prompt = query.make_query()
        if not text_prompt:
            raise ValueError("Video generation requires a text prompt")

        width, height = self._resolve_resolution(query)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": text_prompt,
            "width": width,
            "height": height,
            "output_format": "MP4",
        }

        if query.duration is not None:
            payload["seconds"] = str(int(query.duration))
        if query.fps is not None:
            payload["fps"] = query.fps
        if query.generation_steps is not None:
            payload["steps"] = query.generation_steps
        if query.cfg_scale is not None:
            payload["guidance_scale"] = query.cfg_scale
        if query.negative_prompt:
            payload["negative_prompt"] = query.negative_prompt
        if self.seed is not None:
            payload["seed"] = self.seed

        # First-frame image guidance
        if query.seed_image:
            seed_uri = self._image_to_data_uri(query.seed_image)
            frame_images = [{"url": seed_uri, "position": "first"}]
            if query.tail_image:
                tail_uri = self._image_to_data_uri(query.tail_image)
                frame_images.append({"url": tail_uri, "position": "last"})
            payload["frame_images"] = frame_images

        logger.info(
            f"Submitting Together video job: model={self.model_name}, "
            f"{width}x{height}"
        )

        resp = self._session.post(
            f"{self._BASE_URL}/v2/videos",
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            try:
                err = resp.json()
            except Exception:
                resp.raise_for_status()
            raise ValueError(
                f"Together video API error {resp.status_code}: "
                f"{err.get('message', resp.text)}"
            )
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise ValueError(f"No job ID in response: {data}")
        logger.info(f"Together video job submitted: {job_id}")
        return job_id

    def _poll_job(self, job_id: str) -> Dict[str, Any]:
        """Poll until the job completes or fails."""
        if self._session is None:
            raise ValueError("Session not initialized")

        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > self._MAX_POLL_TIME:
                raise TimeoutError(
                    f"Together video job {job_id} did not complete "
                    f"within {self._MAX_POLL_TIME}s"
                )

            resp = self._session.get(
                f"{self._BASE_URL}/v2/videos/{job_id}",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "completed":
                outputs = data.get("outputs", {})
                video_url = outputs.get("video_url", "")
                if not video_url:
                    raise ValueError(
                        f"Job completed but no video_url: {data}"
                    )
                logger.info(
                    f"Together video job {job_id} completed in {elapsed:.1f}s"
                )

                # Download video and base64-encode (matching Veo 3.1 pattern)
                dl = self._session.get(video_url, timeout=120)
                dl.raise_for_status()
                video_b64 = base64.b64encode(dl.content).decode("utf-8")

                result: Dict[str, Any] = {"base64": video_b64, "id": job_id}
                return result

            elif status == "failed":
                error = data.get("error", {})
                raise ValueError(
                    f"Together video job {job_id} failed: "
                    f"{error.get('message', 'Unknown error')}"
                )

            else:
                logger.debug(
                    f"Together video job {job_id} status={status}, "
                    f"elapsed={elapsed:.1f}s"
                )
                time.sleep(self._POLL_INTERVAL)

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        query = cast(VideoGenQuery, query)
        job_id = self._submit_job(query)
        video_data = self._poll_job(job_id)
        return {"videos": [video_data]}


class TogetherKling21Master(TogetherKling21):
    """Kling 2.1 Master — highest quality video generation."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("kwaivgI/kling-2.1-master", seed)


class TogetherKling21Standard(TogetherKling21):
    """Kling 2.1 Standard — good quality-to-price ratio."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("kwaivgI/kling-2.1-standard", seed)


class TogetherKling21Pro(TogetherKling21):
    """Kling 2.1 Pro — professional-grade HD video generation."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("kwaivgI/kling-2.1-pro", seed)
