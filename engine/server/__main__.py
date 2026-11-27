

import argparse
import uvicorn
import torch
from engine.server.api import create_app

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--device", type=str,default="cuda")
parser.add_argument("--dtype", type=str,default="float16")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)


args = parser.parse_args()
dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}

dtype = dtype_map[args.dtype]

app = create_app(args.model,args.device,dtype)


uvicorn.run(
    app,
    host=args.host,
    port=args.port
)

