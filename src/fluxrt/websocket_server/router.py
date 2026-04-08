from fastapi import APIRouter, Depends, WebSocket, Response
from pydantic import BaseModel
from fluxrt.websocket_server.stream_processor_service import (
    StreamProcessorService,
)

router = APIRouter(tags=["Stream Processor"])


def get_stream_processor_service() -> StreamProcessorService:
    raise RuntimeError("Service not initialized")


class PromptRequest(BaseModel):
    prompt: str


class StepsRequest(BaseModel):
    steps: int


class InversionStrengthRequest(BaseModel):
    inversion_strength: float


class ConditioningScaleRequest(BaseModel):
    controlnet_conditioning_scale: float


class AttentionIntervalRequest(BaseModel):
    attention_chain_update_interval: int


class CannyLowRequest(BaseModel):
    canny_low_threshold: int


class CannyHighRequest(BaseModel):
    canny_high_threshold: int


class SeedRequest(BaseModel):
    seed: int


@router.get("/input-shared-tensor-name")
def get_input_shared_tensor_name(
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    return {"input_shared_tensor_name": service.get_input_shared_tensor_name()}


@router.get("/output-shared-tensor-name")
def get_output_shared_tensor_name(
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    return {"output_shared_tensor_name": service.get_output_shared_tensor_name()}


@router.post("/prompt")
def set_prompt(
    req: PromptRequest,
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    service.set_prompt(req.prompt)
    return {"status": "ok"}


@router.post("/steps")
def set_steps(
    req: StepsRequest,
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    service.set_steps(req.steps)
    return {"status": "ok"}


@router.post("/seed")
def set_seed(
    req: SeedRequest,
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    service.set_seed(req.seed)
    return {"status": "ok"}


@router.get("/resolution")
def get_resolution(
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    return service.get_resolution()


@router.websocket("/ws/stream")
async def run_websocket_stream(
    websocket: WebSocket,
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    await service.handle_websocket_stream(websocket)


@router.get("/latest-frame")
def get_latest_frame(
    service: StreamProcessorService = Depends(get_stream_processor_service),
):
    jpeg_data = service.get_latest_frame_jpeg()
    return Response(content=jpeg_data, media_type="image/jpeg")
