#!/usr/bin/env python3

"""
Interactive demo script showcasing the ModelRouter functionality.

At the interface selection prompt, you can enter:
    - A number to select a specific interface
    - "all" to test all interfaces
    - An exact interface name (e.g., "gemini_pro3")
    - A prefix ending with _ (e.g., "gemini_" for all Gemini models)
    - Space-separated names/prefixes (e.g., "gemini_ anthropic_sonnet4")
"""

import logging
import sys
import termios
import tty
from typing import List
from PIL import Image
import numpy as np
from model_router.router import ModelRouter
from model_router.model_interface import (
    Query,
    ImageGenQuery,
    AudioGenQuery,
    SoundEffectsGenQuery,
    VideoGenQuery,
    Capability,
)
from model_router.lib.media import (
    base64_to_image,
    save_image_to_temp,
    base64_to_audio,
    save_audio_to_temp,
    save_video_to_temp,
)

import os

# Set up logging (suppress router logging)
logging.basicConfig(level=logging.WARNING)


def filter_interfaces(
    all_interfaces: List[str], filters: List[str]
) -> List[str]:
    """Filter interfaces based on exact names or prefixes.

    Args:
        all_interfaces: List of all available interface names
        filters: List of exact names or prefixes (ending with _)

    Returns:
        Filtered list of interface names
    """
    if not filters:
        return all_interfaces

    result = []
    for f in filters:
        if f.endswith("_"):
            # Prefix match
            prefix = f
            for iface in all_interfaces:
                if iface.startswith(prefix) and iface not in result:
                    result.append(iface)
        else:
            # Exact match
            if f in all_interfaces and f not in result:
                result.append(f)
            elif f not in all_interfaces:
                print(f"⚠️  Warning: Interface '{f}' not found")

    return result


def getch():
    """Get a single character from stdin without pressing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def create_sample_image(color="red") -> Image.Image:
    """Create a simple test image for multimodal demos."""
    color_map = {
        "red": [255, 0, 0],
        "blue": [0, 0, 255],
        "green": [0, 255, 0],
        "white": [255, 255, 255],
        "black": [0, 0, 0],
    }
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:, :] = color_map.get(color, [255, 0, 0])
    return Image.fromarray(img_array)


def create_mask_image() -> Image.Image:
    """Create a simple mask image for inpainting/outpainting demos."""
    # Create a mask with a white circle in the center on black background
    mask_array = np.zeros((100, 100, 3), dtype=np.uint8)
    center = (50, 50)
    radius = 25

    # Create circular mask
    for i in range(100):
        for j in range(100):
            if ((i - center[0]) ** 2 + (j - center[1]) ** 2) <= radius**2:
                mask_array[i, j] = [255, 255, 255]  # White for mask area

    return Image.fromarray(mask_array)


def create_checkerboard_with_hole(
    size: int = 1024, square_size: int = 64, hole_radius: int = 150
) -> tuple[Image.Image, Image.Image]:
    """Create a checkerboard pattern with a hole in the middle and corresponding mask."""
    # Create checkerboard pattern
    checkerboard = np.zeros((size, size, 3), dtype=np.uint8)

    # Fill with checkerboard pattern
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            # Determine if this square should be white or black
            square_i = i // square_size
            square_j = j // square_size
            if (square_i + square_j) % 2 == 0:
                color = [255, 255, 255]  # White
            else:
                color = [128, 128, 128]  # Gray

            # Fill the square
            end_i = min(i + square_size, size)
            end_j = min(j + square_size, size)
            checkerboard[i:end_i, j:end_j] = color

    # Create mask for the hole (white circle in center)
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)

    # Create circular hole and mask
    for i in range(size):
        for j in range(size):
            distance = ((i - center[0]) ** 2 + (j - center[1]) ** 2) ** 0.5
            if distance <= hole_radius:
                # Make hole black in checkerboard
                checkerboard[i, j] = [0, 0, 0]
                # Make mask white where hole is
                mask[i, j] = [255, 255, 255]

    return Image.fromarray(checkerboard), Image.fromarray(mask)


def create_video_frames() -> List[Image.Image | str]:
    """Create a sequence of frames for video demo."""
    frames = []
    for i in range(3):
        frame_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Vary the red intensity to simulate motion
        frame_array[:, :] = [255 - i * 50, i * 50, 0]
        frames.append(Image.fromarray(frame_array))
    return frames


def _get_interfaces_to_test(
    router: ModelRouter, interface_name: str | List[str] | None
) -> List[str]:
    """Get list of interfaces to test based on input."""
    if interface_name is None:
        return list(router.loaded_registry.keys())
    elif isinstance(interface_name, list):
        return interface_name
    else:
        return [interface_name]


def demo_text_capability(
    router: ModelRouter,
    interface_name: str | List[str] | None = None,
):
    """Demonstrate text capability."""
    print("\n💬 Enter your text query:")
    user_query = input("Your question: ").strip()

    if not user_query:
        user_query = "What is the capital of France?"  # Fallback to default
        print(f"Using default query: {user_query}")

    query = Query(
        system_prompt="You are a helpful assistant. Respond concisely.",
        query_text=user_query,
    )

    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with TEXT ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.TEXT not in interface.supported_capabilities():
                print(f"❌ {iface} does not support TEXT capability")
                continue

            response = router.get_response(query, Capability.TEXT, iface)
            print(f"✅ Response: {response['text'][:200]}...")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_text_thinking_capability(
    router: ModelRouter,
    interface_name: str | List[str] | None = None,
):
    """Demonstrate text thinking capability (extended thinking mode)."""
    print("\n🧠 Enter your text query for extended thinking:")
    user_query = input("Your question: ").strip()

    if not user_query:
        user_query = "What are the key considerations when designing a distributed system?"
        print(f"Using default query: {user_query}")

    query = Query(
        system_prompt="You are a helpful assistant. Think through this carefully and respond thoroughly.",
        query_text=user_query,
    )

    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with TEXT_THINKING ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.TEXT_THINKING not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support TEXT_THINKING capability")
                continue

            response = router.get_response(query, Capability.TEXT_THINKING, iface)
            print(f"✅ Response: {response['text'][:500]}...")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_image_capability(
    router: ModelRouter,
    interface_name: str | List[str] | None = None,
    multiple_images: bool = False,
):
    """Demonstrate image encoding capability."""
    if multiple_images:
        # Use test image and create a second image
        test_image_path = os.path.join(
            os.path.dirname(__file__), "resources", "test_image_encoding.jpg"
        )
        test_image = Image.open(test_image_path).convert("RGB")
        blue_image = create_sample_image("blue")
        query = Query(
            system_prompt="You are a helpful assistant. Compare the images.",
            query_text="What do you see in the first image and what color is the second image?",
            images=[test_image, blue_image],
        )
        print("\n📸 Testing with MULTIPLE IMAGES (test image + synthetic)")
    else:
        # Load the actual test image
        test_image_path = os.path.join(
            os.path.dirname(__file__), "resources", "test_image_encoding.jpg"
        )
        test_image = Image.open(test_image_path).convert("RGB")
        query = Query(
            system_prompt="You are a helpful assistant. Describe what you see in detail.",
            query_text="What do you see in this image? Describe the content, objects, people, and any text if present.",
            images=test_image,
        )
        print("\n📸 Testing with TEST IMAGE from resources")

    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with IMAGE_ENC ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.IMAGE_ENC not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support IMAGE_ENC capability")
                continue

            # Save input images
            input_paths = []
            if query.images:
                if isinstance(query.images, Image.Image):
                    input_path = save_image_to_temp(query.images, f"input_{iface}")
                    input_paths.append(input_path)
                elif isinstance(query.images, list):
                    for i, img in enumerate(query.images):
                        if isinstance(img, Image.Image):
                            input_path = save_image_to_temp(img, f"input_{iface}_{i}")
                            input_paths.append(input_path)

            response = router.get_response(query, Capability.IMAGE_ENC, iface)
            print(f"✅ Response: {response['text'][:200]}...")

            # Show input image paths
            if input_paths:
                print(f"📁 Input images saved to: {', '.join(input_paths)}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_video_capability(
    router: ModelRouter,
    interface_name: str | List[str] | None = None,
):
    """Demonstrate video encoding capability."""
    frames = create_video_frames()
    query = Query(
        system_prompt="You are a helpful assistant. Analyze the sequence.",
        query_text="Describe the changes in this sequence of frames.",
        video=frames,
    )

    print("\n🎥 Testing with VIDEO FRAMES")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with VIDEO_ENC ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.VIDEO_ENC not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support VIDEO_ENC capability")
                continue

            # Save input video frames
            input_paths = []
            if query.video:
                for i, frame in enumerate(query.video):
                    if isinstance(frame, Image.Image):
                        input_path = save_image_to_temp(
                            frame, f"video_frame_{iface}_{i}"
                        )
                        input_paths.append(input_path)

            response = router.get_response(query, Capability.VIDEO_ENC, iface)
            print(f"✅ Response: {response['text'][:200]}...")

            # Show input frame paths
            if input_paths:
                print(f"📁 Video frames saved to: {', '.join(input_paths)}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_video_generation_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate video generation capability (text-to-video and image-to-video)."""
    print("\n🎬 Video Generation Demo")
    print("Choose mode:")
    print("  1. Text-to-video (prompt only)")
    print("  2. Image-to-video (prompt + first frame image)")
    print("  3. Image-to-video with tail (prompt + first frame + last frame)")
    print("Enter choice (1/2/3, default 1): ", end="", flush=True)

    try:
        mode_choice = input().strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    mode_choice = mode_choice or "1"

    # Get prompt
    print("\nEnter video prompt (or Enter for default): ", end="", flush=True)
    user_prompt = input().strip()
    if not user_prompt:
        user_prompt = "A golden retriever running through a sunlit meadow, slow motion, cinematic"
        print(f"Using default: {user_prompt}")

    # Get duration
    print("Duration in seconds (3-15, default 5): ", end="", flush=True)
    dur_input = input().strip()
    duration = float(dur_input) if dur_input else 5.0

    # Get aspect ratio
    print("Aspect ratio (16:9, 9:16, 1:1, default 16:9): ", end="", flush=True)
    ar_input = input().strip()
    aspect_ratio = ar_input if ar_input else "16:9"

    # Build query
    seed_image = None
    tail_image = None

    if mode_choice in ("2", "3"):
        # Use a test image as seed (first frame)
        test_image_path = os.path.join(
            os.path.dirname(__file__), "resources", "test_image_encoding.jpg"
        )
        if os.path.exists(test_image_path):
            seed_image = Image.open(test_image_path).convert("RGB")
            print(f"Using test image as first frame: {test_image_path}")
        else:
            seed_image = create_sample_image("blue")
            print("Using synthetic blue image as first frame")

        if mode_choice == "3":
            # Use a different color image as tail (last frame)
            tail_image = create_sample_image("red")
            print("Using synthetic red image as last frame")

    query = VideoGenQuery(
        query_text=user_prompt,
        duration=duration,
        aspect_ratio=aspect_ratio,
        mode="pro" if tail_image else "std",
        seed_image=seed_image,
        tail_image=tail_image,
        negative_prompt="blurry, low quality, distorted",
    )

    print("\n🎬 Testing with VIDEO GENERATION")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with VIDEO_GEN ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.VIDEO_GEN not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support VIDEO_GEN capability")
                continue

            # Save input images if any
            if seed_image and isinstance(seed_image, Image.Image):
                input_path = save_image_to_temp(seed_image, f"videogen_seed_{iface}")
                print(f"📁 First frame: {input_path}")
            if tail_image and isinstance(tail_image, Image.Image):
                tail_path = save_image_to_temp(tail_image, f"videogen_tail_{iface}")
                print(f"📁 Last frame: {tail_path}")

            response = router.get_response(query, Capability.VIDEO_GEN, iface)

            if "videos" in response:
                print(f"✅ Generated {len(response['videos'])} video(s)")
                for i, video in enumerate(response["videos"]):
                    if "base64" in video:
                        import base64 as b64
                        video_bytes = b64.b64decode(video["base64"])
                        video_path = save_video_to_temp(
                            video_bytes, "mp4", f"videogen_{iface}_{i+1}"
                        )
                        print(f"   📁 Video {i+1} saved to: {video_path}")
                        print(f"   Size: {len(video_bytes):,} bytes")
                    elif "url" in video:
                        print(f"   Video {i+1} URL: {video['url'][:120]}...")
                    if "id" in video:
                        print(f"   Video {i+1} task ID: {video['id']}")
                    if "duration" in video:
                        print(f"   Video {i+1} duration: {video['duration']}s")
            else:
                print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_audio_capability(router: ModelRouter, interface_name: str | List[str] | None = None):
    """Demonstrate audio generation capability."""
    print("\n🎤 Enter text to convert to speech:")
    user_text = input("Your text: ").strip()

    if not user_text:
        user_text = "Hello! This is a test of the ElevenLabs text-to-speech system. It can generate natural sounding speech from text."
        print(f"Using default text: {user_text[:50]}...")

    # Let user choose voice settings
    print("\nVoice settings (press Enter for defaults):")
    print("Stability (0.0-1.0, default 0.5): ", end="")
    stability_input = input().strip()
    stability = float(stability_input) if stability_input else None

    print("Similarity boost (0.0-1.0, default 0.75): ", end="")
    similarity_input = input().strip()
    similarity_boost = float(similarity_input) if similarity_input else None

    # Create audio generation query
    query = AudioGenQuery(
        query_text=user_text,
        output_format="mp3_44100_128",
        stability=stability,
        similarity_boost=similarity_boost,
        stream=False,  # Non-streaming for demo
    )

    print("\n🎵 Testing with AUDIO GENERATION")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with AUDIO_GEN ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.AUDIO_GEN not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support AUDIO_GEN capability")
                continue

            response = router.get_response(query, Capability.AUDIO_GEN, iface)

            if "audio" in response:
                # Convert base64 audio to bytes and save
                audio_bytes = base64_to_audio(response["audio"])
                audio_format = response.get("format", "mp3")
                audio_path = save_audio_to_temp(
                    audio_bytes, audio_format, f"output_{iface}"
                )

                print(f"✅ Generated audio successfully")
                print(f"📁 Audio saved to: {audio_path}")
                print(f"   Format: {audio_format}")
                print(f"   Voice: {response.get('voice_id', 'default')}")
                print(f"   Model: {response.get('model', 'unknown')}")
                print(f"   Size: {len(audio_bytes):,} bytes")
            else:
                print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_sound_effects_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate sound effects generation capability."""
    print("\n🔊 Enter description for sound effect:")
    user_text = input("Sound description: ").strip()

    if not user_text:
        user_text = "Thunder rumbling in the distance followed by heavy rain"
        print(f"Using default: {user_text}")

    # Let user choose duration
    print("\nSound effect settings:")
    print("Duration in seconds (0.5-22, or Enter for auto): ", end="")
    duration_input = input().strip()
    duration = float(duration_input) if duration_input else None

    print("Prompt influence (0.0-1.0, default 0.3): ", end="")
    influence_input = input().strip()
    prompt_influence = float(influence_input) if influence_input else None

    # Create sound effects generation query
    query = SoundEffectsGenQuery(
        query_text=user_text,
        duration_seconds=duration,
        prompt_influence=prompt_influence,
        output_format="mp3_44100_128",
    )

    print("\n🎵 Testing with SOUND EFFECTS GENERATION")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with SOUND_EFFECTS_GEN ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.SOUND_EFFECTS_GEN not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support SOUND_EFFECTS_GEN capability")
                continue

            response = router.get_response(query, Capability.SOUND_EFFECTS_GEN, iface)

            if "audio" in response:
                # Convert base64 audio to bytes and save
                audio_bytes = base64_to_audio(response["audio"])
                audio_format = response.get("format", "mp3")
                audio_path = save_audio_to_temp(
                    audio_bytes, audio_format, f"sound_effect_{iface}"
                )

                print(f"✅ Generated sound effect successfully")
                print(f"📁 Audio saved to: {audio_path}")
                print(f"   Format: {audio_format}")
                print(f"   Duration: {response.get('duration', 'auto-determined')}")
                print(f"   Type: {response.get('type', 'sound_effect')}")
                print(f"   Size: {len(audio_bytes):,} bytes")
            else:
                print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_image_generation_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate image generation capability."""
    print("Enter number of images to generate (1-8, default 3): ", end="")
    try:
        num_images_input = input().strip()
        num_images = int(num_images_input) if num_images_input else 3
        num_images = max(1, min(8, num_images))  # Clamp between 1-8
    except (ValueError, KeyboardInterrupt):
        num_images = 3
        print(f"Using default: {num_images} images")

    query = ImageGenQuery(
        system_prompt="Create a high-quality digital art image.",
        query_text="A serene mountain landscape at sunset with a lake in the foreground",
        image_resolution=(1024, 1024),
        number_of_results=num_images,
        generation_steps=20,
        negative_prompt="blurry, low quality, distorted",
    )

    print("\n🎨 Testing with IMAGE GENERATION")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with IMAGE_GEN ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.IMAGE_GEN not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support IMAGE_GEN capability")
                continue

            response = router.get_response(query, Capability.IMAGE_GEN, iface)
            if "images" in response:
                print(f"✅ Generated {len(response['images'])} image(s)")

                # Convert base64 back to images and save
                output_paths = []
                for i, img_base64 in enumerate(response["images"]):
                    try:
                        output_image = base64_to_image(img_base64)
                        output_path = save_image_to_temp(
                            output_image, f"output_{iface}_generated_{i+1}"
                        )
                        output_paths.append(output_path)
                    except Exception as img_err:
                        print(f"⚠️  Could not save generated image {i+1}: {img_err}")

                if output_paths:
                    print(f"📁 Generated images saved to: {', '.join(output_paths)}")
            elif "image" in response:
                print(
                    f"✅ Generated image (base64 length: {len(response['image'])} chars)"
                )

                # Convert base64 back to image and save
                try:
                    output_image = base64_to_image(response["image"])
                    output_path = save_image_to_temp(
                        output_image, f"output_{iface}_generated"
                    )
                    print(f"📁 Generated image saved to: {output_path}")
                except Exception as img_err:
                    print(f"⚠️  Could not save generated image: {img_err}")
            else:
                print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_image_inpaint_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate image inpainting capability."""
    # Load the test inpainting image from resources
    base_image_path = os.path.join(
        os.path.dirname(__file__), "resources", "test_inpaint_image.png"
    )
    base_image = Image.open(base_image_path).convert("RGB")

    # Create a mask for inpainting (white circle in center)
    mask_image = create_mask_image()

    query = ImageGenQuery(
        system_prompt="Fill in the missing area of this checkerboard pattern.",
        query_text="Complete the checkerboard pattern by filling the black hole with appropriate squares",
        images=base_image,
        mask_image=mask_image,
        image_resolution=(1024, 1024),
        number_of_results=1,
        generation_steps=20,
        negative_prompt="blurry, low quality, distorted, artifacts, incorrect pattern",
    )

    print("\n🖌️ Testing with IMAGE INPAINTING")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with IMAGE_INPAINT ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.IMAGE_INPAINT not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support IMAGE_INPAINT capability")
                continue

            # Save input images
            input_path = save_image_to_temp(base_image, f"inpaint_input_{iface}")
            mask_path = save_image_to_temp(mask_image, f"inpaint_mask_{iface}")

            response = router.get_response(query, Capability.IMAGE_INPAINT, iface)
            if "images" in response:
                print(f"✅ Generated {len(response['images'])} inpainted image(s)")

                # Convert base64 back to images and save
                output_paths = []
                for i, img_base64 in enumerate(response["images"]):
                    try:
                        output_image = base64_to_image(img_base64)
                        output_path = save_image_to_temp(
                            output_image, f"output_{iface}_inpainted_{i+1}"
                        )
                        output_paths.append(output_path)
                    except Exception as img_err:
                        print(f"⚠️  Could not save inpainted image {i+1}: {img_err}")

                print(f"📁 Input image: {input_path}")
                print(f"📁 Mask image: {mask_path}")
                if output_paths:
                    print(f"📁 Inpainted results: {', '.join(output_paths)}")
            elif "image" in response:
                print(
                    f"✅ Inpainted image (base64 length: {len(response['image'])} chars)"
                )

                # Convert base64 back to image and save
                try:
                    output_image = base64_to_image(response["image"])
                    output_path = save_image_to_temp(
                        output_image, f"output_{iface}_inpainted"
                    )
                    print(f"📁 Input image: {input_path}")
                    print(f"📁 Mask image: {mask_path}")
                    print(f"📁 Inpainted result: {output_path}")
                except Exception as img_err:
                    print(f"⚠️  Could not save inpainted image: {img_err}")
                    print(f"📁 Input image: {input_path}")
                    print(f"📁 Mask image: {mask_path}")
            else:
                print(f"✅ Response: {response}")
                print(f"📁 Input image: {input_path}")
                print(f"📁 Mask image: {mask_path}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_image_outpaint_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate image outpainting capability."""
    print("Enter number of images to generate (1-8, default 3): ", end="")
    try:
        num_images_input = input().strip()
        num_images = int(num_images_input) if num_images_input else 3
        num_images = max(1, min(8, num_images))  # Clamp between 1-8
    except (ValueError, KeyboardInterrupt):
        num_images = 3
        print(f"Using default: {num_images} images")

    # Load the test inpainting image from resources for outpainting
    base_image_path = os.path.join(
        os.path.dirname(__file__), "resources", "test_inpaint_image.png"
    )
    base_image = Image.open(base_image_path).convert("RGB")

    # For outpainting, mask typically indicates areas to extend (inverse of inpainting)
    mask_image = create_sample_image("white")  # White mask for outpainting areas

    query = ImageGenQuery(
        system_prompt="Extend this image beyond its current boundaries.",
        query_text="Add more scenery around the existing content, maintaining consistent style",
        images=base_image,
        mask_image=mask_image,
        image_resolution=(1024, 1024),  # Larger resolution for outpainting
        number_of_results=num_images,
        generation_steps=25,
        negative_prompt="cropped, cut off, incomplete, blurry, low quality",
    )

    print("\n🔍 Testing with IMAGE OUTPAINTING")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with IMAGE_OUTPAINT ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.IMAGE_OUTPAINT not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support IMAGE_OUTPAINT capability")
                continue

            # Save input images
            input_path = save_image_to_temp(base_image, f"outpaint_input_{iface}")

            response = router.get_response(query, Capability.IMAGE_OUTPAINT, iface)
            if "images" in response:
                print(f"✅ Generated {len(response['images'])} outpainted image(s)")

                # Convert base64 back to images and save
                output_paths = []
                for i, img_base64 in enumerate(response["images"]):
                    try:
                        output_image = base64_to_image(img_base64)
                        output_path = save_image_to_temp(
                            output_image, f"output_{iface}_outpainted_{i+1}"
                        )
                        output_paths.append(output_path)
                    except Exception as img_err:
                        print(f"⚠️  Could not save outpainted image {i+1}: {img_err}")

                print(f"📁 Input image: {input_path}")
                if output_paths:
                    print(f"📁 Outpainted results: {', '.join(output_paths)}")
            else:
                print(f"✅ Response: {response}")
                print(f"📁 Input image: {input_path}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def demo_style_transfer_capability(
    router: ModelRouter, interface_name: str | List[str] | None = None
):
    """Demonstrate style transfer capability using FLUX.2 [dev]."""
    # Load base image and style reference from resources
    base_image_path = os.path.join(
        os.path.dirname(__file__), "resources", "test_style_transfer_base.png"
    )
    style_ref_path = os.path.join(
        os.path.dirname(__file__), "resources", "test_style_reference.png"
    )
    base_image = Image.open(base_image_path).convert("RGB")
    style_reference = Image.open(style_ref_path).convert("RGB")

    style_prompt = """Apply the artistic style of image 2 to the content of image 1.

CRITICAL PRESERVATION RULES:
- PRESERVE the exact color palette from image 1. Do NOT transfer colors from image 2.
- PRESERVE all character expressions, poses, and facial features from image 1 exactly.
- ONLY transfer the artistic TECHNIQUE from image 2: brushwork, texture, line quality and rendering style.
- Do NOT change character species, clothing colors, accessory colors, or prop colors.
- PRESERVE original background. Do not paint-in empty background.
- Do NOT add new elements or change character expressions/poses.
"""

    query = ImageGenQuery(
        query_text=style_prompt,
        image_resolution=(base_image.width, base_image.height),
        images=[base_image, style_reference],
        number_of_results=1,
        generation_steps=25,
    )

    print("\n🎨 Testing with STYLE TRANSFER")
    interfaces_to_test = _get_interfaces_to_test(router, interface_name)

    for iface in interfaces_to_test:
        print(f"\n--- Testing {iface} with STYLE TRANSFER ---")
        try:
            interface = router.loaded_registry[iface]
            if Capability.IMAGE_GEN not in interface.supported_capabilities():
                print(f"⏭️  {iface} does not support IMAGE_GEN capability")
                continue

            # Save input images
            input_path = save_image_to_temp(base_image, f"style_base_{iface}")
            ref_path = save_image_to_temp(style_reference, f"style_ref_{iface}")

            response = router.get_response(query, Capability.IMAGE_GEN, iface)
            if "images" in response:
                print(f"✅ Generated {len(response['images'])} styled image(s)")

                output_paths = []
                for i, img_base64 in enumerate(response["images"]):
                    try:
                        output_image = base64_to_image(img_base64)
                        output_path = save_image_to_temp(
                            output_image, f"output_{iface}_styled_{i+1}"
                        )
                        output_paths.append(output_path)
                    except Exception as img_err:
                        print(f"⚠️  Could not save styled image {i+1}: {img_err}")

                print(f"📁 Base image: {input_path}")
                print(f"📁 Style reference: {ref_path}")
                if output_paths:
                    print(f"📁 Styled results: {', '.join(output_paths)}")
            else:
                print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error with {iface}: {e}")


def show_interface_menu(router: ModelRouter) -> str | List[str]:
    """Show interface selection menu and return choice.

    Returns:
        - "quit" to exit
        - "all" to test all interfaces
        - A single interface name (str)
        - A list of interface names matching filters
    """
    print("\n" + "=" * 50)
    print("🤖 SELECT INTERFACE")
    print("=" * 50)

    interfaces = list(router.loaded_registry.keys())
    print("Available interfaces:")
    for i, interface in enumerate(interfaces, 1):
        capabilities = router.loaded_registry[interface].supported_capabilities()
        cap_str = ", ".join([cap.value for cap in capabilities])
        print(f"  {i}. {interface} ({cap_str})")

    print(f"  {len(interfaces) + 1}. all (test all listed interfaces)")
    print("  0. quit")
    print("\nEnter choice (number, name, prefix with _, or space-separated list): ", end="", flush=True)

    try:
        choice = input().strip()

        if choice == "0":
            return "quit"
        elif choice == str(len(interfaces) + 1) or choice.lower() == "all":
            return "all"
        elif choice.isdigit() and 1 <= int(choice) <= len(interfaces):
            selected = interfaces[int(choice) - 1]
            print(f"Selected: {selected}")
            return selected
        else:
            # Try to parse as model name(s) or prefix(es)
            filters = choice.split()
            matched = filter_interfaces(interfaces, filters)

            if not matched:
                print(f"❌ No interfaces matched '{choice}'. Please try again.")
                return show_interface_menu(router)
            elif len(matched) == 1:
                print(f"Selected: {matched[0]}")
                return matched[0]
            else:
                print(f"Selected {len(matched)} interfaces: {', '.join(matched)}")
                return matched
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return "quit"


def show_capability_menu(router: ModelRouter, interface_choice: str | List[str]) -> str:
    """Show capability selection menu and return choice."""
    print("\n" + "=" * 50)
    print("🎯 SELECT CAPABILITY")
    print("=" * 50)

    # Get supported capabilities for the selected interface(s)
    if isinstance(interface_choice, list):
        # Multiple interfaces - get union of all capabilities
        all_caps = set()
        for iface in interface_choice:
            all_caps.update(router.loaded_registry[iface].supported_capabilities())
        supported_caps = list(all_caps)
        print(f"Interfaces: {', '.join(interface_choice)}")
        print(
            f"Combined capabilities: {', '.join([cap.value for cap in supported_caps])}"
        )
        print()
    elif interface_choice != "all":
        interface = router.loaded_registry[interface_choice]
        supported_caps = interface.supported_capabilities()
        print(f"Interface: {interface_choice}")
        print(
            f"Supported capabilities: {', '.join([cap.value for cap in supported_caps])}"
        )
        print()

    # Map capabilities to their corresponding demo functions and requirements
    all_capabilities = {
        "1": ("text", "Text processing", [Capability.TEXT]),
        "2": ("text_thinking", "Text with extended thinking", [Capability.TEXT_THINKING]),
        "3": ("image", "Single image processing", [Capability.IMAGE_ENC]),
        "4": ("images", "Multiple image processing", [Capability.IMAGE_ENC]),
        "5": ("video", "Video frame processing", [Capability.VIDEO_ENC]),
        "6": ("imagegen", "Image generation", [Capability.IMAGE_GEN]),
        "7": ("inpaint", "Image inpainting", [Capability.IMAGE_INPAINT]),
        "8": ("outpaint", "Image outpainting", [Capability.IMAGE_OUTPAINT]),
        "9": ("styletransfer", "Style transfer (FLUX.2)", [Capability.IMAGE_GEN]),
        "10": ("videogen", "Video generation (text/image-to-video)", [Capability.VIDEO_GEN]),
        "11": ("audio", "Audio generation (text-to-speech)", [Capability.AUDIO_GEN]),
        "12": (
            "soundeffects",
            "Sound effects generation",
            [Capability.SOUND_EFFECTS_GEN],
        ),
        "13": ("all", "Test all capabilities", []),  # Special case
    }

    # Filter capabilities based on interface selection
    if interface_choice == "all":
        available_capabilities = all_capabilities
    else:
        # Get supported capabilities (already computed above for display)
        if isinstance(interface_choice, list):
            all_caps = set()
            for iface in interface_choice:
                all_caps.update(router.loaded_registry[iface].supported_capabilities())
            supported_caps = list(all_caps)
        else:
            supported_caps = router.loaded_registry[interface_choice].supported_capabilities()

        available_capabilities = {}
        counter = 1

        for key, (choice, desc, required_caps) in all_capabilities.items():
            if choice == "all":
                # Always include "all" option
                available_capabilities[str(counter)] = (choice, desc, required_caps)
                counter += 1
            elif any(cap in supported_caps for cap in required_caps):
                available_capabilities[str(counter)] = (choice, desc, required_caps)
                counter += 1

    for key, (_, desc, _) in available_capabilities.items():
        print(f"  {key}. {desc}")
    print("  0. back to interface selection")
    print("\nEnter your choice: ", end="", flush=True)

    try:
        choice = input().strip()

        if choice == "0":
            return "back"
        elif choice in available_capabilities:
            selected_capability, desc, _ = available_capabilities[choice]
            print(f"Selected: {desc}")
            return selected_capability
        else:
            print(f"❌ Invalid choice '{choice}'. Please try again.")
            return show_capability_menu(router, interface_choice)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return "back"


def run_demo(router: ModelRouter, interface_choice: str | List[str], capability_choice: str):
    """Run the selected demo."""
    # Determine interface_name for demo functions
    if interface_choice == "all":
        interface_name = None  # None means test all
    else:
        interface_name = interface_choice  # str or List[str]

    # Format display string
    if isinstance(interface_choice, list):
        display_name = f"{len(interface_choice)} interfaces"
    else:
        display_name = interface_choice.upper()

    print(f"\n" + "=" * 60)
    print(f"🚀 RUNNING DEMO: {capability_choice.upper()} on {display_name}")
    print("=" * 60)

    if capability_choice == "text":
        demo_text_capability(router, interface_name)
    elif capability_choice == "text_thinking":
        demo_text_thinking_capability(router, interface_name)
    elif capability_choice == "image":
        demo_image_capability(router, interface_name, multiple_images=False)
    elif capability_choice == "images":
        demo_image_capability(router, interface_name, multiple_images=True)
    elif capability_choice == "video":
        demo_video_capability(router, interface_name)
    elif capability_choice == "imagegen":
        demo_image_generation_capability(router, interface_name)
    elif capability_choice == "inpaint":
        demo_image_inpaint_capability(router, interface_name)
    elif capability_choice == "outpaint":
        demo_image_outpaint_capability(router, interface_name)
    elif capability_choice == "styletransfer":
        demo_style_transfer_capability(router, interface_name)
    elif capability_choice == "videogen":
        demo_video_generation_capability(router, interface_name)
    elif capability_choice == "audio":
        demo_audio_capability(router, interface_name)
    elif capability_choice == "soundeffects":
        demo_sound_effects_capability(router, interface_name)
    elif capability_choice == "all":
        # Get supported capabilities for this interface
        if interface_name is None:
            # For "all" interfaces, run all demos
            supported_caps = list(Capability)
        elif isinstance(interface_name, list):
            # Multiple interfaces - get union of capabilities
            all_caps = set()
            for iface in interface_name:
                all_caps.update(router.loaded_registry[iface].supported_capabilities())
            supported_caps = list(all_caps)
        else:
            supported_caps = router.loaded_registry[interface_name].supported_capabilities()

        if Capability.TEXT in supported_caps:
            demo_text_capability(router, interface_name)
        if Capability.TEXT_THINKING in supported_caps:
            demo_text_thinking_capability(router, interface_name)
        if Capability.IMAGE_ENC in supported_caps:
            demo_image_capability(router, interface_name, multiple_images=False)
            demo_image_capability(router, interface_name, multiple_images=True)
        if Capability.VIDEO_ENC in supported_caps:
            demo_video_capability(router, interface_name)
        if Capability.IMAGE_GEN in supported_caps:
            demo_image_generation_capability(router, interface_name)
        if Capability.IMAGE_INPAINT in supported_caps:
            demo_image_inpaint_capability(router, interface_name)
        if Capability.IMAGE_OUTPAINT in supported_caps:
            demo_image_outpaint_capability(router, interface_name)
        if Capability.VIDEO_GEN in supported_caps:
            demo_video_generation_capability(router, interface_name)
        if Capability.AUDIO_GEN in supported_caps:
            demo_audio_capability(router, interface_name)
        if Capability.SOUND_EFFECTS_GEN in supported_caps:
            demo_sound_effects_capability(router, interface_name)

    print(f"\n✅ Demo completed for {capability_choice} on {display_name}")


if __name__ == "__main__":
    print("🚀 Interactive Model Router Demo")
    print("This script lets you test specific interfaces with specific capabilities.")
    print("Use Ctrl+C at any time to quit.")

    router = ModelRouter()

    try:
        while True:
            # Interface selection
            interface_choice = show_interface_menu(router)
            if interface_choice == "quit":
                break

            # Capability selection
            capability_choice = show_capability_menu(router, interface_choice)
            if capability_choice == "back":
                continue

            # Run demo
            run_demo(router, interface_choice, capability_choice)

            # Ask if user wants to continue
            print("\n" + "-" * 50)
            print("Press Enter to continue or 'q' to quit: ", end="", flush=True)
            ch = input().strip()
            if ch.lower() == "q":
                break

    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")
