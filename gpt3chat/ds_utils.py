import os
import json
import torch.distributed as dist
import torch

from functools import partial
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode
from pydantic import BaseModel
from typing import Dict, Any

try:
    import mii
except:
    raise ImportError("Please install Deepspeed-MII for deepspeed deployment.")

DEPLOYMENT_NAME = "ds_inference_grpc_server"


def run_rank_n(func: partial, barrier: bool = False, rank: int = 0, other_rank_output: Any = None) -> Any:
    # runs function on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            output = func()
            if barrier:
                dist.barrier()
            return output
        else:
            if barrier:
                dist.barrier()
            return other_rank_output
    else:
        return func()


def get_downloaded_model_path(model_name: str):
    f = partial(
        snapshot_download,
        repo_id=model_name,
        allow_patterns=["*"],
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
    )
    # download only on 1 process
    run_rank_n(f, barrier=True)
    # now since the snapshot is downloaded, pass the model_path to all processes
    return f()


def init_deepspeed_grpc(model_name: str, dtype: str = "fp16", hf_cache: str = None):
    mii_config = {
        "dtype": dtype,
        "tensor_parallel": torch.cuda.device_count(),
        "port_number": 50010,
    }
    if "microsoft/bloom" in model_name:
        downloaded_model_path = get_downloaded_model_path(model_name)
        checkpoints_json = os.path.join(downloaded_model_path, "ds_inference_config.json")
        mii_config["checkpoint_dict"] = json.load(open(checkpoints_json, "r"))
        additional_kwargs = {"model_path": downloaded_model_path}
    else:
        downloaded_model_path = hf_cache if hf_cache is not None else os.getenv("TRANSFORMERS_CACHE", None)
        downloaded_model_path = os.path.join(downloaded_model_path, model_name)
        ds_config = {
            "fp16": {
                "enabled": dtype == "fp16"
            },
            "bf16": {
                "enabled": dtype == "bf16"
            },
            "train_micro_batch_size_per_gpu": 1,
        }
        additional_kwargs = {
            "local_model_path": downloaded_model_path,
            "ds_config": ds_config,
        }
    mii.deploy(
        task="text-generation",
        # for bloom we should pass args.model_name but can't since the new
        # weights are not supported yet. So, this is a hack
        model=model_name,
        deployment_name=DEPLOYMENT_NAME,
        mii_config=mii_config,
        **additional_kwargs
    )

    return mii.mii_query_handle(DEPLOYMENT_NAME)


def inference_deepspeed_grpc(model, text: str, generate_kwargs: Dict):
    response = model.query({"query": text}, **generate_kwargs)
    print(f"got response")
    output_text = response.response
    print(f"ds output: {output_text}")
    output_text = [_ for _ in output_text]

    # TODO: simulate hf output

    return output_text


def terminate_grpc():
    mii.terminate(DEPLOYMENT_NAME)