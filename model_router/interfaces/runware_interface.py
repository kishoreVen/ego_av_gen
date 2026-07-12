
import asyncio
import base64
import os
import threading
from typing import Any, Dict, List, cast

import requests as http_requests

from runware import Runware
from runware.types import (
    IImageInference as RunwareImageInference,
    IOutpaint as RunwareOutpaint,
    IVideoInference,
    IFrameImage,
    IInputFrame,
    IVideoInputs,
    IMinimaxProviderSettings,
    IKlingAIProviderSettings,
)
from model_router.model_interface import (
    ModelInterface,
    Capability,
    Query,
    ImageGenQuery,
    VideoGenQuery,
)
from model_router.scaffold import ImageGenerationModelInterface
from model_router.lib.media import (
    convert_query_images_to_base64_list,
    image_to_base64,
    base64_to_image,
    OutpaintingExtent,
    compress_for_reference,
)
from model_router.lib.constants import DEFAULT_NEGATIVE_PROMPTS_FOR_IMAGE_GEN

from PIL import Image

import logging

logger = logging.getLogger(__name__)


class RunwareInterface(ImageGenerationModelInterface):

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)

        self.model_name: str = model_name

        self.client = None
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()

    def initialize_client(self) -> None:
        self.client = Runware(api_key=os.environ["RUNWARE_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_GEN]

    async def _await_image_result(self, request_image: RunwareImageInference):
        """Call imageInference and handle async delivery polling."""
        result = await self.client.imageInference(requestImage=request_image)
        if not isinstance(result, list):
            result = await self.client.getResponse(
                taskUUID=result.taskUUID,
                numberResults=request_image.numberResults or 1,
            )
        return result

    async def _async_generate(self, query: ImageGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Only connect if not already connected
        if not self.client.connected:
            await self.client.connect()

        if query.system_prompt is None and query.query_text is None:
            raise ValueError("Query must have either system_prompt or query_text")

        text_prompt = query.make_query()

        height = query.image_resolution[1] if query.image_resolution else 512
        width = query.image_resolution[0] if query.image_resolution else 512

        if height <= 0 or width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        reference_images = None
        if query.images:
            reference_images = convert_query_images_to_base64_list(
                query.images, include_data_uri_prefix=True
            )

        mask_image = None
        if query.mask_image:
            if isinstance(query.mask_image, str):
                mask_image = query.mask_image
            elif isinstance(query.mask_image, Image.Image):
                mask_image = image_to_base64(
                    query.mask_image, include_data_uri_prefix=True
                )

        request_image = RunwareImageInference(
            positivePrompt=text_prompt,
            model=self.model_name,
            numberResults=query.number_of_results if query.number_of_results else 1,
            negativePrompt=(
                query.negative_prompt
                if query.negative_prompt
                else DEFAULT_NEGATIVE_PROMPTS_FOR_IMAGE_GEN
            ),
            height=height,
            width=width,
            seed=self.seed,
            steps=query.generation_steps if query.generation_steps else 25,
            outputType="base64Data",
            outputFormat=query.image_format if query.image_format else "JPG",  # type: ignore
            referenceImages=reference_images,
            maskImage=mask_image,
            deliveryMethod="async",
        )

        return await self._await_image_result(request_image)

    def _ensure_event_loop(self):
        """Ensure we have a running event loop in a background thread."""
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                # Create new event loop in background thread
                self._loop = asyncio.new_event_loop()

                def run_loop():
                    asyncio.set_event_loop(self._loop)
                    self._loop.run_forever()

                self._thread = threading.Thread(target=run_loop, daemon=True)
                self._thread.start()

    def _fetch_image_response(
        self, query: ImageGenQuery, capability: Capability | None = None
    ) -> Dict[str, Any]:
        # Ensure we have an event loop running
        self._ensure_event_loop()

        # Submit the coroutine to the event loop and wait for result
        future = asyncio.run_coroutine_threadsafe(
            self._async_generate(query),
            self._loop
        )

        try:
            # 360s > SDK's IMAGE_INFERENCE_TIMEOUT (300s) so SDK errors propagate cleanly
            images = future.result(timeout=360)
        except Exception:
            future.cancel()
            raise

        if images is None or len(images) == 0:
            raise ValueError("Content generation failed")

        image_response_as_text: List[str] = []

        for image_response in images:
            if image_response.imageBase64Data is not None:
                image_response_as_text.append(image_response.imageBase64Data)

        return {"images": image_response_as_text}


class RunwareKontextPro(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("bfl:3@1", seed)

    async def _async_generate(self, query: ImageGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Only connect if not already connected
        if not self.client.connected:
            await self.client.connect()

        if query.system_prompt is None and query.query_text is None:
            raise ValueError("Query must have either system_prompt or query_text")

        text_prompt = query.make_query()

        height = query.image_resolution[1] if query.image_resolution else 512
        width = query.image_resolution[0] if query.image_resolution else 512

        if height <= 0 or width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        if query.generation_steps is not None:
            logger.warning(
                "Generation steps is specified but not supported by RunwareKontextPro."
            )

        if query.negative_prompt is not None:
            logger.warning(
                "Negative prompt is specified but not supported by RunwareKontextPro."
            )

        reference_images = None
        if query.images:
            reference_images = convert_query_images_to_base64_list(
                query.images, include_data_uri_prefix=True
            )
            # Debug logging for reference images
            if reference_images:
                logger.info(f"RunwareKontextPro: Processing {len(reference_images)} reference images")
                for i, ref_img in enumerate(reference_images):
                    if ref_img:
                        # Log the first 100 chars to see format
                        preview = ref_img[:100] if len(ref_img) > 100 else ref_img
                        logger.info(f"Reference image {i} preview: {preview}")
                        # Check if it has proper data URI prefix
                        if ref_img.startswith('data:'):
                            mime_part = ref_img.split(',')[0] if ',' in ref_img else 'unknown'
                            logger.info(f"Reference image {i} data URI header: {mime_part}")
                        else:
                            logger.warning(f"Reference image {i} missing data URI prefix")

        mask_image = None
        if query.mask_image:
            if isinstance(query.mask_image, str):
                mask_image = query.mask_image
            elif isinstance(query.mask_image, Image.Image):
                mask_image = image_to_base64(
                    query.mask_image, include_data_uri_prefix=True
                )

        request_image = RunwareImageInference(
            positivePrompt=text_prompt,
            model=self.model_name,
            numberResults=query.number_of_results if query.number_of_results else 1,
            height=height,
            width=width,
            seed=self.seed,
            outputType="base64Data",
            outputFormat=query.image_format if query.image_format else "JPG",  # type: ignore
            referenceImages=reference_images,
            maskImage=mask_image,
            deliveryMethod="async",
        )

        return await self._await_image_result(request_image)


class RunwareKontextDev(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:106@1", seed)


class RunwareKontextMax(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("bfl:4@1", seed)


class RunwareFlux1Schnell(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:100@1", seed)


class RunwareFlux1Dev(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:101@1", seed)


class RunwareFlux2DevStyleTransfer(RunwareInterface):
    """FLUX.2 [dev] style transfer interface.

    Reference images are passed via inputs.referenceImages (not top-level),
    matching the working Runware API pattern for FLUX.2 models.
    Reference images are downscaled to max 1024px to avoid WebSocket payload limits.
    """

    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:400@1", seed)

    async def _async_generate(self, query: ImageGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        if not self.client.connected:
            await self.client.connect()

        text_prompt = query.make_query() or "apply style"

        height = query.image_resolution[1] if query.image_resolution else 512
        width = query.image_resolution[0] if query.image_resolution else 512

        if height <= 0 or width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        # Downscale + JPEG-compress references to reduce WebSocket payload
        reference_images = None
        if query.images:
            raw = query.images if isinstance(query.images, list) else [query.images]
            reference_images = compress_for_reference(raw)

        request_image = RunwareImageInference(
            positivePrompt=text_prompt,
            model=self.model_name,
            numberResults=query.number_of_results if query.number_of_results else 1,
            height=height,
            width=width,
            seed=self.seed,
            steps=query.generation_steps if query.generation_steps else 25,
            outputType="base64Data",
            outputFormat="JPG",  # type: ignore
            CFGScale=3.5,
            scheduler="FlowMatchEulerDiscreteScheduler",
            acceleration="high",
            referenceImages=reference_images or [],
        )

        return await self.client.imageInference(requestImage=request_image)


class RunwareFluxDevFillInpaint(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:102@1", seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_INPAINT]

    async def _async_generate(self, query: ImageGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Only connect if not already connected
        if not self.client.connected:
            await self.client.connect()

        if query.system_prompt is None and query.query_text is None:
            raise ValueError("Query must have either system_prompt or query_text")

        text_prompt = query.make_query()

        height = query.image_resolution[1] if query.image_resolution else 512
        width = query.image_resolution[0] if query.image_resolution else 512

        if height <= 0 or width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        if query.negative_prompt is not None:
            logger.warning(
                "Negative prompt is specified but not supported by RunwareKontextPro."
            )

        reference_images = None
        if query.images:
            reference_images = convert_query_images_to_base64_list(
                query.images, include_data_uri_prefix=True
            )

        if reference_images is None or len(reference_images) > 1:
            raise ValueError(
                "RunwareFluxDevFillInpaint only supports a single reference image which is the base image "
                "for inpainting."
            )

        seed_image = reference_images[0]

        mask_image = None
        if query.mask_image:
            if isinstance(query.mask_image, str):
                mask_image = query.mask_image
            elif isinstance(query.mask_image, Image.Image):
                mask_image = image_to_base64(
                    query.mask_image, include_data_uri_prefix=True
                )

        request_image = RunwareImageInference(
            positivePrompt=text_prompt,
            model=self.model_name,
            numberResults=query.number_of_results if query.number_of_results else 1,
            height=height,
            width=width,
            seed=self.seed,
            outputType="base64Data",
            outputFormat=query.image_format if query.image_format else "JPG",  # type: ignore
            steps=query.generation_steps if query.generation_steps else 40,
            seedImage=seed_image,
            maskImage=mask_image,
            deliveryMethod="async",
        )

        return await self._await_image_result(request_image)


class RunwareFluxDevFillOutpaint(RunwareInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("runware:102@1", seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_OUTPAINT]

    def _preprocess_input(self, image: Image.Image, extent: OutpaintingExtent) -> str:
        scaled_image = image
        if extent.needs_preprocessing:
            input_width, input_height = image.size
            new_width, new_height = extent.working_input_size
            logger.debug(
                f"Downsampling outpainting input from {input_width}x{input_height} to {new_width}x{new_height}"
            )
            scaled_image = image.resize(
                (new_width, new_height),
                (
                    Image.LANCZOS
                    if hasattr(Image, "LANCZOS")
                    else Image.Resampling.LANCZOS
                ),
            )
        return image_to_base64(scaled_image, include_data_uri_prefix=True)

    def _postprocess_outputs(
        self,
        outpainted_images: List[str],
        extent: OutpaintingExtent,
        crop_to_exact: bool = True,
    ) -> List[str]:
        result_images = []

        for i, image_b64 in enumerate(outpainted_images):
            image = base64_to_image(image_b64)

            # Step 1: Upscale if needed (when working at reduced resolution)
            if extent.needs_postprocessing:
                outpainted_width, outpainted_height = extent.scaled_output_size
                # First upscale to the padded size (divisible by 64)
                # This is larger than or equal to target_size
                actual_width = extent.working_input_size[0] + extent.left + extent.right
                actual_height = (
                    extent.working_input_size[1] + extent.top + extent.bottom
                )
                scaled_width = int(actual_width / extent.pre_scale)
                scaled_height = int(actual_height / extent.pre_scale)

                logger.debug(
                    f"Upsampling outpainting result {i+1} from {outpainted_width}x{outpainted_height} to {scaled_width}x{scaled_height}"
                )
                image = image.resize(
                    (scaled_width, scaled_height),
                    (
                        Image.LANCZOS
                        if hasattr(Image, "LANCZOS")
                        else Image.Resampling.LANCZOS
                    ),
                )

            # Step 2: Crop to exact target size if requested
            if crop_to_exact:
                target_width, target_height = extent.target_size
                current_width, current_height = image.size

                if current_width > target_width or current_height > target_height:
                    # Calculate center crop coordinates
                    left_crop = (current_width - target_width) // 2
                    top_crop = (current_height - target_height) // 2
                    right_crop = left_crop + target_width
                    bottom_crop = top_crop + target_height

                    logger.debug(
                        f"Cropping from {current_width}x{current_height} to exact target {target_width}x{target_height}"
                    )
                    image = image.crop((left_crop, top_crop, right_crop, bottom_crop))

            result_image_b64 = image_to_base64(image)
            result_images.append(result_image_b64)

        return result_images

    def _fetch_image_response(
        self, query: ImageGenQuery, capability: Capability | None = None
    ) -> Dict[str, Any]:
        # Grab image to be outpainted
        if not query.images:
            raise ValueError("No input image provided")
        image_b64 = convert_query_images_to_base64_list(
            query.images, include_data_uri_prefix=True
        )[0]
        image = base64_to_image(image_b64)
        input_width, input_height = image.size

        # Grab target resolution
        target_height = query.image_resolution[1] if query.image_resolution else 512
        target_width = query.image_resolution[0] if query.image_resolution else 512

        # Calculate outpainting parameters with position if provided
        extent = OutpaintingExtent.from_image_sizes(
            (input_width, input_height),
            (target_width, target_height),
            image_position=query.image_position,  # Use the position from the query
        )

        # Preprocess input if needed
        scaled_image_b64 = self._preprocess_input(image, extent)

        # Do API call for outpainting
        logger.debug(
            f"Outpainting with position={query.image_position}, extent: top={extent.top}, bottom={extent.bottom}, left={extent.left}, right={extent.right}"
        )
        images = asyncio.run(self._async_generate(query, scaled_image_b64, extent))
        if images is None or len(images) == 0:
            raise ValueError("Content generation failed")
        image_response_as_text: List[str] = []
        for image_response in images:
            if image_response.imageBase64Data is not None:
                image_response_as_text.append(image_response.imageBase64Data)

        # Postprocess outputs if needed
        processed_images_b64 = self._postprocess_outputs(image_response_as_text, extent)

        return {"images": processed_images_b64}

    async def _async_generate(
        self, query: ImageGenQuery, scaled_image_b64: str, extent: OutpaintingExtent
    ):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Only connect if not already connected
        if not self.client.connected:
            await self.client.connect()

        text_prompt = query.make_query() or "__BLANK__"  # __BLANK__ is a special value specified by Runware

        target_height = query.image_resolution[1] if query.image_resolution else 512
        target_width = query.image_resolution[0] if query.image_resolution else 512

        if target_height <= 0 or target_width <= 0:
            raise ValueError("Image resolution must be greater than 0")

        if query.negative_prompt is not None:
            logger.warning(
                "Negative prompt is specified but not supported by RunwareKontextPro."
            )

        request_image = RunwareImageInference(
            positivePrompt=text_prompt,
            model=self.model_name,
            numberResults=query.number_of_results if query.number_of_results else 1,
            outpaint=RunwareOutpaint(
                top=extent.top,
                right=extent.right,
                bottom=extent.bottom,
                left=extent.left,
            ),
            height=extent.scaled_output_size[1],
            width=extent.scaled_output_size[0],
            seed=self.seed,
            outputType="base64Data",
            outputFormat=query.image_format if query.image_format else "JPG",  # type: ignore
            steps=query.generation_steps if query.generation_steps else 40,
            seedImage=scaled_image_b64,
            deliveryMethod="async",
        )
        return await self._await_image_result(request_image)

class RunwareMinimax(ModelInterface):
    """Runware Minimax video generation model (minimax:3@1)."""

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name = "minimax:3@1"
        self.client = None
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()

    def initialize_client(self) -> None:
        self.client = Runware(api_key=os.environ["RUNWARE_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.VIDEO_GEN]

    async def _async_generate_video(self, query: VideoGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Only connect if not already connected
        if not self.client.connected:
            await self.client.connect()

        if query.system_prompt is None and query.query_text is None:
            raise ValueError("Query must have either system_prompt or query_text")

        text_prompt = query.make_query()

        # Process resolution - support both 512x512 and 1366x768
        if query.video_resolution:
            requested_width, requested_height = query.video_resolution
            # Check if it's one of the supported resolutions
            if (requested_width, requested_height) == (512, 512):
                width, height = 512, 512
            elif (requested_width, requested_height) == (1366, 768):
                width, height = 1366, 768
            else:
                # Default to 512x512 for unsupported resolutions
                width, height = 512, 512
                logger.info(f"Requested resolution {requested_width}x{requested_height} not supported, using {width}x{height}")
        else:
            width, height = 512, 512

        # Process seed image if provided
        seed_image = None
        if query.seed_image:
            if isinstance(query.seed_image, str):
                seed_image = query.seed_image
                # Add data URI prefix if not present
                if not seed_image.startswith('data:'):
                    seed_image = f"data:image/jpeg;base64,{seed_image}"
            elif isinstance(query.seed_image, Image.Image):
                seed_image = image_to_base64(
                    query.seed_image, include_data_uri_prefix=True
                )

        # Prepare video request using the actual IVideoInference dataclass fields
        # Convert seed image to IFrameImage format if provided
        frame_images = None
        if seed_image:
            # For 512x512, we can only use first frame (API limitation)
            # For 1366x768, we can use both first and last frames for looping
            if width == 512 and height == 512:
                first_frame = IFrameImage(
                    inputImage=seed_image,
                    frame="first"
                )
                frame_images = [first_frame]
            else:  # 1366x768
                # Use both first and last frames for smooth looping
                first_frame = IFrameImage(
                    inputImage=seed_image,
                    frame="first"
                )
                last_frame = IFrameImage(
                    inputImage=seed_image,
                    frame="last"
                )
                frame_images = [first_frame, last_frame]
                logger.info(f"Using first and last frames for smooth looping at {width}x{height}")

        # Duration must be 6 or 10 for minimax model
        duration = 6.0  # Default to 6 seconds
        if query.duration:
            if query.duration <= 6:
                duration = 6.0
            else:
                duration = 10.0
            if query.duration not in [6.0, 10.0]:
                logger.info(f"Requested duration {query.duration} not supported, using {duration}")

        # Enable prompt optimization for better results
        minimax_settings = IMinimaxProviderSettings(
            promptOptimizer=True  # Enhance the prompt for better video generation
        )

        request_video = IVideoInference(
            model=self.model_name,
            positivePrompt=text_prompt,
            negativePrompt=query.negative_prompt,
            width=width,
            height=height,
            duration=duration,
            outputType="URL",  # Video API only supports URL output
            outputFormat="MP4",  # MP4 for iOS compatibility
            frameImages=frame_images,
            providerSettings=minimax_settings,
        )

        result = await self.client.videoInference(requestVideo=request_video)
        return result

    def _ensure_event_loop(self):
        """Ensure we have a running event loop in a background thread."""
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                # Create new event loop in background thread
                self._loop = asyncio.new_event_loop()

                def run_loop():
                    asyncio.set_event_loop(self._loop)
                    self._loop.run_forever()

                self._thread = threading.Thread(target=run_loop, daemon=True)
                self._thread.start()

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        query = cast(VideoGenQuery, query)

        # Ensure we have an event loop running
        self._ensure_event_loop()

        # Submit the coroutine to the event loop and wait for result
        future = asyncio.run_coroutine_threadsafe(
            self._async_generate_video(query),
            self._loop
        )

        try:
            # 360s > SDK's internal timeout so SDK errors propagate cleanly
            videos = future.result(timeout=360)
        except Exception:
            future.cancel()
            raise

        if videos is None or len(videos) == 0:
            raise ValueError("Video generation failed")

        video_response = []

        for video in videos:
            video_data = {}
            # Since we're using outputType="URL", we'll get videoURL
            if hasattr(video, 'videoURL') and video.videoURL:
                video_data['url'] = video.videoURL
            # We won't get base64 data with URL output type
            # if hasattr(video, 'videoBase64Data') and video.videoBase64Data:
            #     video_data['base64'] = video.videoBase64Data
            if hasattr(video, 'videoUUID') and video.videoUUID:
                video_data['id'] = video.videoUUID

            if video_data:
                video_response.append(video_data)

        return {"videos": video_response}


def _resolve_image_data_uri(image: str | Image.Image) -> str:
    """Convert a seed/tail image to a data-URI string (Runware expects this)."""
    if isinstance(image, Image.Image):
        return image_to_base64(image, include_data_uri_prefix=True)
    if isinstance(image, str) and not image.startswith("data:"):
        return f"data:image/jpeg;base64,{image}"
    return image


class RunwareKlingV3(ModelInterface):
    """Kling Video 3.0 via Runware (klingai:kling-video@3-standard).

    Supports text-to-video and image-to-video with:
    - Duration: 5 or 10 seconds
    - Resolution: 1920x1080 (16:9), 1080x1920 (9:16), 1080x1080 (1:1)
    - First-frame and last-frame image guidance via IFrameImage
    - Native audio generation via IKlingAIProviderSettings
    - Negative prompts
    """

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name = "klingai:kling-video@3-standard"
        self.client = None
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()

    def initialize_client(self) -> None:
        self.client = Runware(api_key=os.environ["RUNWARE_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.VIDEO_GEN]

    def _resolve_resolution(
        self, query: VideoGenQuery
    ) -> str:
        """Kling 3.0 via Runware only supports '720p'."""
        return "720p"

    async def _async_generate_video(self, query: VideoGenQuery):
        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        if not self.client.connected:
            await self.client.connect()

        text_prompt = query.make_query()
        if not text_prompt:
            raise ValueError("Video generation requires a text prompt")

        resolution = self._resolve_resolution(query)

        # Build frame images for first/last frame guidance
        # Kling uses `inputs.frameImages` with IInputFrame (not top-level frameImages)
        input_frames = []
        if query.seed_image:
            seed_uri = _resolve_image_data_uri(query.seed_image)
            input_frames.append(IInputFrame(image=seed_uri, frame="first"))

            if query.tail_image:
                tail_uri = _resolve_image_data_uri(query.tail_image)
                input_frames.append(IInputFrame(image=tail_uri, frame="last"))
                logger.info("Using first and last frame guidance for Kling 3.0")

        # Duration: Kling 3.0 supports 5 or 10 seconds via Runware
        duration = 5.0
        if query.duration:
            duration = 10.0 if query.duration > 5 else 5.0
            if query.duration not in [5.0, 10.0]:
                logger.info(
                    f"Kling 3.0 supports 5 or 10s; "
                    f"requested {query.duration}s, using {duration}s"
                )

        # Kling-specific provider settings
        kling_settings = IKlingAIProviderSettings(
            sound=query.generate_audio or False,
        )

        # Kling requires frame images inside `inputs`, not at top level
        video_inputs = None
        if input_frames:
            video_inputs = IVideoInputs(frameImages=input_frames)

        request_video = IVideoInference(
            model=self.model_name,
            positivePrompt=text_prompt,
            negativePrompt=query.negative_prompt,
            resolution=resolution,
            duration=duration,
            outputType="URL",
            outputFormat="MP4",
            inputs=video_inputs,
            providerSettings=kling_settings,
        )

        result = await self.client.videoInference(requestVideo=request_video)
        # videoInference may return an IAsyncTaskResponse; poll for the
        # actual video result via getResponse (same pattern as image inference).
        if not isinstance(result, list):
            result = await self.client.getResponse(
                taskUUID=result.taskUUID,
                numberResults=1,
            )
        return result

    def _ensure_event_loop(self):
        """Ensure we have a running event loop in a background thread."""
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                self._loop = asyncio.new_event_loop()

                def run_loop():
                    asyncio.set_event_loop(self._loop)
                    self._loop.run_forever()

                self._thread = threading.Thread(target=run_loop, daemon=True)
                self._thread.start()

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        query = cast(VideoGenQuery, query)

        self._ensure_event_loop()

        future = asyncio.run_coroutine_threadsafe(
            self._async_generate_video(query),
            self._loop,
        )

        try:
            videos = future.result(timeout=360)
        except Exception:
            future.cancel()
            raise

        if videos is None or len(videos) == 0:
            raise ValueError("Video generation failed")

        video_response = []
        for video in videos:
            video_url = getattr(video, "videoURL", None)
            if not video_url:
                continue

            # Download video bytes and base64-encode (matching Veo 3.1 pattern)
            dl = http_requests.get(video_url, timeout=120)
            dl.raise_for_status()
            video_b64 = base64.b64encode(dl.content).decode("utf-8")

            video_data: Dict[str, Any] = {"base64": video_b64}
            if hasattr(video, "videoUUID") and video.videoUUID:
                video_data["id"] = video.videoUUID
            video_response.append(video_data)

        return {"videos": video_response}
