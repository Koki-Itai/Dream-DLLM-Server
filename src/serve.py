import argparse
import queue
import threading
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import uvicorn
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class GenerationRequest(BaseModel):
    """Request model for text generation"""

    messages: List[Dict[str, str]]
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    steps: Optional[int] = 512
    alg: Optional[str] = "entropy"
    alg_temp: Optional[float] = 0.0


class GenerationResponse(BaseModel):
    """Response model for text generation"""

    generated_text: str
    model_name: str
    gpu_id: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Response model for model information"""

    model_name: str
    available_gpus: List[int]
    model_info: Dict[str, Any]


def get_serializable_config(config: Any) -> Dict[str, Any]:
    config_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, torch.dtype):
            config_dict[key] = str(value)
        elif hasattr(value, "__dict__"):
            config_dict[key] = str(value)
        else:
            try:
                import json

                json.dumps(value)
                config_dict[key] = value
            except (TypeError, OverflowError):
                config_dict[key] = str(value)
    return config_dict


class GPUWorker:
    """Worker class for handling inference on a specific GPU"""

    def __init__(self, model_name: str, gpu_id: int) -> None:
        """Initialize worker with model on specific GPU"""
        print(f"Initializing model on GPU {gpu_id}")
        self.gpu_id: int = gpu_id
        self.model_name: str = model_name

        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = self.model.to(f"cuda:{gpu_id}").eval()

        self.request_queue: Queue = queue.Queue()
        self.response_queue: Queue = queue.Queue()

        self.worker_thread: threading.Thread = threading.Thread(
            target=self._process_requests
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _process_requests(self) -> None:
        """Process requests from the queue"""
        while True:
            request_id, request = self.request_queue.get()
            try:
                inputs = self.tokenizer.apply_chat_template(
                    request.messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )

                input_ids = inputs.input_ids.to(device=f"cuda:{self.gpu_id}")
                attention_mask = inputs.attention_mask.to(device=f"cuda:{self.gpu_id}")

                with torch.no_grad():
                    output = self.model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_new_tokens,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=request.steps,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        alg=request.alg,
                        alg_temp=request.alg_temp,
                    )

                generated_text = self.tokenizer.decode(
                    output.sequences[0][len(input_ids[0]) :].tolist()
                ).split(self.tokenizer.eos_token)[0]

                self.response_queue.put(
                    (
                        request_id,
                        GenerationResponse(
                            generated_text=generated_text,
                            model_name=self.model_name,
                            gpu_id=self.gpu_id,
                        ),
                    )
                )
            except Exception as e:
                self.response_queue.put((request_id, str(e)))
            finally:
                self.request_queue.task_done()

    def generate(self, request_id: int, request: GenerationRequest) -> None:
        """Queue a generation request"""
        self.request_queue.put((request_id, request))

    def get_result(self) -> Tuple[int, Union[GenerationResponse, str]]:
        """Get result from response queue"""
        return self.response_queue.get()


def create_single_gpu_app(model_name: str, device: int) -> FastAPI:
    """Create the FastAPI application with a single GPU"""
    app = FastAPI(title="Dream Model Server (Single GPU)")

    print(f"Loading model {model_name} on device {device}...")
    model: PreTrainedModel = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    if device >= 0:
        model = model.to(f"cuda:{device}").eval()
    else:
        model = model.to("cpu").eval()

    @app.get("/", response_model=ModelInfoResponse)
    def get_model_info() -> ModelInfoResponse:
        """Get information about the loaded model"""
        return ModelInfoResponse(
            model_name=model_name,
            available_gpus=[device] if device >= 0 else [],
            model_info=get_serializable_config(model.config),
        )

    @app.post("/generate", response_model=GenerationResponse)
    def generate(request: GenerationRequest = Body(...)) -> GenerationResponse:
        """Generate text based on input messages"""
        try:
            inputs = tokenizer.apply_chat_template(
                request.messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )

            if device >= 0:
                input_ids = inputs.input_ids.to(device=f"cuda:{device}")
                attention_mask = inputs.attention_mask.to(device=f"cuda:{device}")
            else:
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask

            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=request.steps,
                temperature=request.temperature,
                top_p=request.top_p,
                alg=request.alg,
                alg_temp=request.alg_temp,
            )

            generated_text = tokenizer.decode(
                output.sequences[0][len(input_ids[0]) :].tolist()
            ).split(tokenizer.eos_token)[0]

            return GenerationResponse(
                generated_text=generated_text,
                model_name=model_name,
                gpu_id=device if device >= 0 else None,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def create_multi_gpu_app(model_name: str, gpu_ids: List[int]) -> FastAPI:
    """Create the FastAPI application with models loaded on multiple GPUs"""
    app = FastAPI(title="Dream Model Multi-GPU Server")

    workers: List[GPUWorker] = [GPUWorker(model_name, gpu_id) for gpu_id in gpu_ids]

    request_counter: int = 0
    request_lock = threading.Lock()

    model_info: Dict[str, Any] = (
        get_serializable_config(workers[0].model.config) if workers else {}
    )

    @app.get("/", response_model=ModelInfoResponse)
    def get_model_info() -> ModelInfoResponse:
        """Get information about the loaded model and available GPUs"""
        return ModelInfoResponse(
            model_name=model_name, available_gpus=gpu_ids, model_info=model_info
        )

    @app.post("/generate", response_model=GenerationResponse)
    async def generate(
        request: GenerationRequest = Body(...), background_tasks: BackgroundTasks = None
    ) -> GenerationResponse:
        """Generate text based on input messages, distributing load across GPUs"""
        nonlocal request_counter

        worker_idx = np.argmin([worker.request_queue.qsize() for worker in workers])
        worker = workers[worker_idx]

        with request_lock:
            request_id = request_counter
            request_counter += 1

        worker.generate(request_id, request)

        while True:
            result_id, result = worker.get_result()
            if result_id == request_id:
                if isinstance(result, str):
                    raise HTTPException(status_code=500, detail=result)
                return result
            else:
                worker.response_queue.put((result_id, result))

    return app


def create_app(model_name: str, devices: List[int]) -> FastAPI:
    """Create the appropriate app based on number of devices"""
    if len(devices) == 1:
        return create_single_gpu_app(model_name, devices[0])
    else:
        return create_multi_gpu_app(model_name, devices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dream Model Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="GPU device to use (-1 for CPU)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (overrides --device when provided)",
    )

    args = parser.parse_args()

    if args.gpus:
        devices = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
    else:
        devices = [args.device]

    app = create_app(args.model, devices)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
