"""Single-A800 validation-first PRISM tuning runner.

The runner uses only CUDA device 0, selects candidates by validation score,
and records progress atomically so a remote monitor can read it while the job
is running. Test scores are written only after a candidate is selected.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "server_runs/a800_tuning_20261112"
PROGRESS = OUT / "progress.json"
LOG = OUT / "tuning.log"
sys.path.insert(0, str(HERE))
import a800_suite as suite
from prepare_data import DATASETS, prepare

def tuned_make_context(inputs, method, spec):
    """Build views from each trial config without changing the frozen suite."""
    cfg = spec.get("config", {})
    key = "symmetric_structural" if method in ("hgnn", "hcha") else method
    if key in inputs.contexts:
        return inputs.contexts[key]
    model = cfg.get("model", {})
    builder = suite.FastBuilder(
        hop_k=int(model.get("hop_k", 2)),
        structural_cap=int(model.get("structural_cap", 32)),
        knn_k=int(model.get("knn_k", 12)),
        n_clusters=int(model.get("n_clusters", 64)),
        random_state=suite.CONSTRUCTION_SEED,
        soft_membership=method not in ("hgnn", "hcha", "legacy_ce"),
        elastic_margin=cfg.get("loss", {}).get("elastic_margin", .18),
        max_elastic_pairs=cfg.get("loss", {}).get("max_elastic_pairs", 10000),
        structure_filter=method == "prism",
        structural_candidate_cap=int(model.get("structural_candidate_cap", 32)),
        densify_k=int(model.get("densify_k", 1)),
        densify_min_similarity=float(model.get("densify_min_similarity", .3)),
        hyperedge_keep_ratio=float(model.get("hyperedge_keep_ratio", .95)),
        hyperedge_score=model.get("hyperedge_score", "coherence_coverage"),
        preserve_node_coverage=bool(model.get("preserve_node_coverage", True)),
        prototype_soft_topk=int(model.get("prototype_soft_topk", 3)),
    )
    enabled = ["structural"] if method in ("hgnn", "hcha") else model.get(
        "enabled_views", ["structural", "feature_knn", "prototype"])
    views = builder.build(inputs.x.numpy(), inputs.adj, enabled_views=enabled,
                          include_elastic=method == "legacy_full",
                          include_knn_pairs=method == "legacy_full")
    if method in ("hgnn", "hcha"):
        h = views.incidences[0].copy()
        h.data[:] = 1.
        ni, ei = builder._degree_stats(h)
        context = {"h": suite.tensor_sparse(h), "ht": suite.tensor_sparse(h.T.tocsr()),
                   "dv_inv_sqrt": suite.torch.from_numpy(suite.np.sqrt(ni)).to(suite.DEVICE),
                   "de_inv": suite.torch.from_numpy(suite.np.asarray(ei)).to(suite.DEVICE)}
    else:
        ops, names, prior = suite.run_node.build_view_ops(
            views, {"enabled_views": enabled,
                    "view_gate_prior": model.get("view_gate_prior", {
                        "structural": .34, "feature_knn": .33, "prototype": .33})})
        context = {"ops": [{k: v.to(suite.DEVICE) for k, v in op.items()} for op in ops],
                   "prior": prior, "filter_stats": views.filter_stats,
                   "pairs": suite.torch.from_numpy(views.elastic_pairs).long().to(suite.DEVICE),
                   "lower": suite.torch.from_numpy(views.elastic_lower).to(suite.DEVICE),
                   "upper": suite.torch.from_numpy(views.elastic_upper).to(suite.DEVICE)}
    inputs.contexts[key] = context
    return context

suite.make_context = tuned_make_context

PAPER_TARGETS = {
    "actor": 0.3919, "amazon_ratings": 0.5652, "coauthor_cs": 0.9660,
    "coauthor_physics": 0.9768, "dblp": 0.8535, "ogbn_arxiv": 0.7431,
    "wikics": 0.8107, "questions": 0.8038, "tolokers": 0.8712,
}
PAPER_BEST_METHODS = {
    "actor": "MVHGCL", "amazon_ratings": "MVHGCL", "coauthor_cs": "MVHGCL",
    "coauthor_physics": "MVHGCL", "dblp": "MVHGCL", "ogbn_arxiv": "MVHGCL",
    "wikics": "GAT", "questions": "MVHGCL", "tolokers": "MVHGCL",
}
COMPARABILITY = {
    "dblp": "not_direct: local conversion uses all 3 feature blocks and four type indicators; paper table lists 334 features",
    "ogbn_arxiv": "not_direct: paper caption specifies ROC-AUC, but local benchmark evaluates 40-class accuracy",
}

def now():
    return datetime.now(timezone.utc).isoformat()

def save_progress(state):
    state["updated_at"] = now()
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROGRESS)

def monitored_fit(inputs, spec, split, init, epochs, state, stage, label,
                  candidate_index=None, candidate_total=None, final=False):
    run_path = OUT / inputs.name / stage / f"{spec['id']}_split_{split}_init_{init}.json"
    if run_path.exists():
        saved = suite.read(run_path)
        assert saved['spec'] == spec and saved['epochs_limit'] == epochs
        assert saved['input_sha256'] == inputs.meta['sha256']
        assert saved['test_evaluations'] == int(final)
        return saved
    state.update(stage=stage, dataset=inputs.name, candidate=label,
                 split=split, init=init, current_epoch=0,
                 current_loss=None, current_stopping_score=None,
                 current_best_stopping_score=None, current_best_epoch=0)
    save_progress(state)
    def epoch_callback(info):
        state.update(stage=stage, dataset=inputs.name, candidate=label,
                     split=split, init=init, current_epoch=info["epoch"],
                     current_loss=info["loss"],
                     current_stopping_score=info["stopping_score"],
                     current_best_stopping_score=info["best_stopping_score"],
                     current_best_epoch=info["best_epoch"],
                     patience_used=info["patience_used"], completed=False)
        if candidate_index is not None:
            state.update(candidate_index=candidate_index,
                         candidate_total=candidate_total)
        save_progress(state)
        print(json.dumps({"event":"epoch", "dataset":inputs.name,
                          "stage":stage,"candidate":label,"split":split,
                          "init":init,**info}, ensure_ascii=False), flush=True)
    suite.EPOCH_CALLBACK = epoch_callback
    try:
        result = suite.fit(inputs, "prism", spec, split=split, init=init,
                           final=final, epochs=epochs)
        result.update(spec=copy.deepcopy(spec), epochs_limit=epochs,
                      input_sha256=inputs.meta['sha256'], split=split, init=init,
                      execution_device=str(suite.DEVICE), finished_at=now())
        suite.write(result, run_path)
        return result
    finally:
        suite.EPOCH_CALLBACK = None

def candidate_grid(name):
    base = suite.candidates(name, "prism")[0]
    candidates = []
    for label, config in [("base", base), ("compact", suite.candidates(name, "prism")[1])]:
        config = copy.deepcopy(config)
        config["label"] = label
        candidates.append(config)
    cfg = copy.deepcopy(base)
    cfg["id"], cfg["label"] = "wide", "wide"
    m, loss, optim = cfg["config"]["model"], cfg["config"]["loss"], cfg["config"]["optim"]
    m.update(hidden_dim=256, proj_dim=128, K=2, dropout=.35,
             gate_temperature=1.3, gate_residual=.05, private_gate_strength=.7,
             structural_cap=48, structural_candidate_cap=48, knn_k=16,
             n_clusters=96, densify_k=2, densify_min_similarity=.20,
             hyperedge_keep_ratio=.85)
    loss.update(lambda_align=.02, lambda_var=.003, lambda_decorrelation=.01,
                distill_warmup_epochs=10, distill_temperature=1.5)
    optim.update(lr=.0015, weight_decay=.0005)
    candidates.append(cfg)
    cfg = copy.deepcopy(base)
    cfg["id"], cfg["label"] = "stable", "stable"
    m, loss, optim = cfg["config"]["model"], cfg["config"]["loss"], cfg["config"]["optim"]
    m.update(hidden_dim=128, proj_dim=64, K=3, dropout=.40,
             gate_temperature=1.8, gate_residual=.10, private_gate_strength=.5,
             structural_cap=32, structural_candidate_cap=40, knn_k=12,
             n_clusters=64, densify_k=1, densify_min_similarity=.30,
             hyperedge_keep_ratio=.90)
    loss.update(lambda_align=.05, lambda_var=.005, lambda_decorrelation=.01,
                distill_warmup_epochs=10, distill_temperature=1.5)
    optim.update(lr=.002, weight_decay=.001)
    candidates.append(cfg)
    return candidates

def run_dataset(name, state, epochs):
    state.update(stage="preparing", dataset=name, completed=False)
    save_progress(state)
    prepare(name)
    inputs = suite.Inputs(name)
    records = []
    specs = candidate_grid(name)
    for index, spec in enumerate(specs, 1):
        label = spec["label"]
        state.update(stage="tuning", dataset=name, candidate=label,
                     candidate_index=index, candidate_total=len(specs),
                     epochs=epochs, completed=False)
        save_progress(state)
        started = time.time()
        result = monitored_fit(inputs, spec, split=0, init=0, epochs=epochs,
                               state=state, stage="tuning", label=label,
                               candidate_index=index, candidate_total=len(specs))
        # The test set has not been touched during candidate selection.
        record = {
            "dataset": name, "candidate": label, "config_id": spec["id"],
            "best_val_score": result["best_val_score"],
            "best_stopping_score": result["best_stopping_score"],
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "seconds": result["seconds"], "target_from_paper": PAPER_TARGETS[name],
            "paper_best_method": PAPER_BEST_METHODS[name],
            "paper_comparability": COMPARABILITY.get(name, "nominal_metric_match; confirm split alignment"),
            "config": spec["config"],
        }
        records.append(record)
        inputs.contexts.clear()
        state.setdefault("trials", []).append(record)
        save_progress(state)
        print(json.dumps({"event":"trial_complete", **{k: record[k] for k in
              ("dataset","candidate","best_val_score","best_epoch","seconds")}},
              ensure_ascii=False), flush=True)
    finalists = sorted(records,
                       key=lambda row: (-row["best_val_score"], row["config_id"]))[:2]
    confirm_split = min(1, inputs.num_splits - 1)
    confirmations = []
    for finalist in finalists:
        spec = next(s for s in specs if s["id"] == finalist["config_id"])
        result = monitored_fit(inputs, spec, split=confirm_split, init=42,
                              epochs=epochs, state=state, stage="confirmation",
                              label=finalist["candidate"])
        confirmation = {"dataset":name,"candidate":finalist["candidate"],
                        "split":confirm_split,"init":42,
                        "best_val_score":result["best_val_score"],
                        "best_stopping_score":result["best_stopping_score"],
                        "best_epoch":result["best_epoch"]}
        finalist["confirmation_val_score"] = result["best_val_score"]
        finalist["selection_val_mean"] = (finalist["best_val_score"]+
                                          result["best_val_score"])/2
        confirmations.append(confirmation)
        inputs.contexts.clear()
        state.setdefault("confirmations", []).append(confirmation)
        save_progress(state)
    selected = max(finalists,
                   key=lambda row: (row["selection_val_mean"], row["config_id"]))
    selected_spec = next(s for s in specs if s["id"] == selected["config_id"])
    state.update(stage="selected", dataset=name, candidate=selected["candidate"],
                 best_val_score=selected["selection_val_mean"], completed=False)
    save_progress(state)

    # Final test scores follow the repository benchmark's all-splits/10-seed protocol.
    final_indices = list(range(inputs.num_splits)) if inputs.num_splits > 1 else list(range(10))
    final_records = []
    for number in final_indices:
        split = number if inputs.num_splits > 1 else 0
        init = 201 + number
        final = monitored_fit(inputs, selected_spec, split=split, init=init,
                             epochs=epochs, state=state, stage="final", 
                             label=selected["candidate"], final=True)
        record = {"dataset":name,"candidate":selected["candidate"],
                  "split":split,"init":init,
                  "best_val_score":final["best_val_score"],
                  "best_test_score":final["best_test_score"],
                  "best_epoch":final["best_epoch"],
                  "test_evaluations":final["test_evaluations"],
                  "metric":final["metric"],"seconds":final["seconds"]}
        assert final["test_evaluations"] == 1
        final_records.append(record)
        state.setdefault("final_runs", []).append(record)
        values = [r["best_test_score"] for r in final_records]
        state["current_final_test_mean"] = suite.mean(values)
        state["current_final_test_std"] = suite.stdev(values) if len(values)>1 else 0.0
        state["current_final_runs"] = len(final_records)
        inputs.contexts.clear()
        save_progress(state)
        print(json.dumps({"event":"final_run_complete", **record,
                          "running_test_mean":state["current_final_test_mean"],
                          "running_test_std":state["current_final_test_std"]},
                          ensure_ascii=False),flush=True)
    values = [r["best_test_score"] for r in final_records]
    selected.update({"best_test_score_mean":suite.mean(values),
                     "best_test_score_std":suite.stdev(values) if len(values)>1 else 0.0,
                     "final_run_count":len(final_records),
                     "test_evaluations":sum(r["test_evaluations"] for r in final_records),
                     "test_gap_to_paper_target":suite.mean(values)-PAPER_TARGETS[name],
                     "confirmations":confirmations})
    state.setdefault("selected", []).append(selected)
    state["dataset_complete"] = name
    inputs.contexts.clear()
    save_progress(state)
    print(json.dumps({"event":"dataset_complete", **{k:selected[k] for k in
          ("dataset","candidate","best_val_score","best_test_score_mean",
           "best_test_score_std","final_run_count")}},
          ensure_ascii=False), flush=True)

def main():
    requested = sys.argv[1:] or DATASETS
    epochs = int(os.environ.get("PRISM_TUNING_EPOCHS", "200"))
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status":"running", "stage":"starting", "device":"cuda:0",
             "gpu":suite.torch.cuda.get_device_name(0), "datasets":requested,
             "epochs_per_trial":epochs, "started_at":now(), "trials":[], "selected":[],
             "paper_targets":PAPER_TARGETS, "paper_best_methods":PAPER_BEST_METHODS,
             "comparability_notes":COMPARABILITY,
             "selection_protocol":"four candidates selected on validation split 0; top two confirmed on split 1 where available; final all-splits/10-seed evaluation after config selection; test never used to choose configs"}
    save_progress(state)
    assert suite.DEVICE.type == "cuda" and suite.torch.cuda.device_count() >= 1
    print(json.dumps({"event":"job_start", "device":"cuda:0",
                      "gpu":suite.torch.cuda.get_device_name(0),
                      "datasets":requested, "epochs":epochs}), flush=True)
    failures = []
    for name in requested:
        try:
            run_dataset(name, state, epochs)
        except Exception as error:
            failures.append({'dataset':name,'error':repr(error),'time':now()})
            state.update(stage="dataset_error", errors=failures)
            save_progress(state)
            traceback.print_exc()
            suite.torch.cuda.empty_cache()
    state.update(status="completed_with_errors" if failures else "passed",
                 stage="complete", completed=True, finished_at=now())
    save_progress(state)
    print(json.dumps({"event":"job_complete", "datasets":requested}), flush=True)
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
