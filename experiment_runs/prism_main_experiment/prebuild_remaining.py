"""Populate the unchanged frozen builder's cache without training a model."""
import argparse
import json
import time
import torch
import suite

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--methods',nargs='+',required=True)
    args=ap.parse_args()
    suite.protocol()  # Refuse changed model, preprocessing or configuration code.
    suite.seed_all(suite.CONSTRUCTION_SEED)
    inputs=suite.Inputs(args.dataset)
    for method in args.methods:
        start=time.perf_counter()
        suite.make_context(inputs,method,suite.candidates(args.dataset,method)[0])
        print(json.dumps({'stage':'prebuilt','dataset':args.dataset,'method':method,
                          'seconds':time.perf_counter()-start}),flush=True)
        inputs.contexts.clear()
        torch.cuda.empty_cache()
