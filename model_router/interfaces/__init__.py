
from typing import Dict, Type

from model_router.model_interface import ModelInterface
from model_router.interfaces import anthropic_interface
from model_router.interfaces import openai_interface
from model_router.interfaces import runware_interface
from model_router.interfaces import elevenlabs_tts_interface
from model_router.interfaces import elevenlabs_sfx_interface
from model_router.interfaces import gemini_interface
from model_router.interfaces import together_interface
from model_router.interfaces import dummy_interface
from model_router.interfaces import qwen_vl_interface
from model_router.interfaces import sam_interface


INTERFACE_REGISTRY: Dict[str, Type[ModelInterface]] = {
    "dummy": dummy_interface.DummyInterface,
    "anthropic_haiku35": anthropic_interface.AnthropicHaiku35,
    "anthropic_sonnet37": anthropic_interface.AnthropicSonnet37,
    "anthropic_sonnet4": anthropic_interface.AnthropicSonnet4,
    "anthropic_haiku45": anthropic_interface.AnthropicHaiku45,
    "anthropic_sonnet45": anthropic_interface.AnthropicSonnet45,
    "anthropic_opus45": anthropic_interface.AnthropicOpus45,
    "anthropic_opus41": anthropic_interface.AnthropicOpus41,
    "openai_gpt4o": openai_interface.OpenAiGPT4o,
    "openai_gpt4o_mini": openai_interface.OpenAiGPT4oMini,
    "openai_gpt4_turbo": openai_interface.OpenAiGPT4Turbo,
    "openai_o1_mini": openai_interface.OpenAiO1Mini,
    "openai_gpt5": openai_interface.OpenAiGPT5,
    "openai_gpt5_mini": openai_interface.OpenAiGPT5Mini,
    "openai_gpt5_nano": openai_interface.OpenAiGPT5Nano,
    "openai_gpt52": openai_interface.OpenAiGPT52,
    "openai_image_generation": openai_interface.OpenAiImageGeneration,
    "runware_kontext_pro": runware_interface.RunwareKontextPro,
    "runware_kontext_dev": runware_interface.RunwareKontextDev,
    "runware_kontext_max": runware_interface.RunwareKontextMax,
    "runware_flux1_schnell": runware_interface.RunwareFlux1Schnell,
    "runware_flux1_dev": runware_interface.RunwareFlux1Dev,
    "runware_flux2_dev_style_transfer": runware_interface.RunwareFlux2DevStyleTransfer,
    "together_flux2_dev_style_transfer": together_interface.TogetherFlux2DevStyleTransfer,
    "runware_flux_dev_fill_inpaint": runware_interface.RunwareFluxDevFillInpaint,
    "runware_flux_dev_fill_outpaint": runware_interface.RunwareFluxDevFillOutpaint,
    "runware_minimax": runware_interface.RunwareMinimax,
    "elevenlabs_multilingual_v2": elevenlabs_tts_interface.ElevenLabsMultilingualV2,
    "elevenlabs_flash_v2": elevenlabs_tts_interface.ElevenLabsFlashV2,
    "elevenlabs_turbo_v2": elevenlabs_tts_interface.ElevenLabsTurboV2,
    "elevenlabs_monolingual_v1": elevenlabs_tts_interface.ElevenLabsMonolingualV1,
    "elevenlabs_sound_effects": elevenlabs_sfx_interface.ElevenLabsSoundEffects,
    "gemini_pro25": gemini_interface.GeminiPro25,
    "gemini_pro3": gemini_interface.GeminiPro3,
    "gemini_flash25": gemini_interface.GeminiFlash25,
    "gemini_flash_lite25": gemini_interface.GeminiFlashLite25,
    "gemini_flash3": gemini_interface.GeminiFlash3,
    "gemini_nano_banana": gemini_interface.GeminiNanoBanana,
    "gemini_nano_banana2": gemini_interface.GeminiNanoBanana2,
    "gemini_pro3_image": gemini_interface.GeminiPro3Image,
    "gemini_veo31": gemini_interface.GeminiVeo31,
    "runware_kling_v3": runware_interface.RunwareKlingV3,
    "together_kling21_master": together_interface.TogetherKling21Master,
    "together_kling21_standard": together_interface.TogetherKling21Standard,
    "together_kling21_pro": together_interface.TogetherKling21Pro,
    # Local models
    "local_qwen_vl": qwen_vl_interface.QwenVLInterface,
    "local_sam": sam_interface.SAMInterface,
}
