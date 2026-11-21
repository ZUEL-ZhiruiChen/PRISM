# Single-A800 adaptive PRISM search

The first pass writes `server_runs/a800_tuning_20261112`.
The adaptive continuation writes `server_runs/a800_adaptive_20261112`.
Both use the same advisory GPU lock at the first folder's `campaign.lock`.
The adaptive worker waits for the first pass to exit and then uses GPU 0 only.
The worker has its own session, redirected input/output, and requires no client
connection. Turning off the client computer does not terminate server training.

Each dataset is searched in rounds of eight new configurations. Initial
parents come exclusively from the first pass's validation tuning records;
test/final result directories are never read to select parents. Later rounds
combine mutations of the best confirmed configurations with global proposals.
Three views, structure filtering, shared/private branches and distillation stay
enabled. The model class is `models.model.PRISM`; core model files are unchanged.

Every fit allows 400 epochs and uses stopping patience 60. The top two
configurations in a round are checked on three fixed validation runs in total:
split 0 / seed 41, split 1 / seed 42, and split 2 / seed 43. Single-split datasets
reuse their official validation split with the three initializations. WikiCS
uses its stopping mask to select checkpoints and its validation mask to rank
configurations. Training masks and full input features match the first pass.

A dataset stops after at least four rounds and three consecutive rounds with
less than 0.0005 improvement in its best mean validation score (0.05 percentage
points). This defines an empirical plateau within the declared search space,
not a certified global optimum or a guarantee of beating paper scores. Datasets
are visited in round-robin order. Once every dataset converges or a repeated
error is recorded, all chosen configurations are frozen before final test
evaluation: ten official splits or ten initializations, and all twenty WikiCS
splits. DBLP conversion and OGBN-Arxiv metric differences must be retained when
comparing with the manuscript. Every final fit evaluates test labels once.

`search_state.json` stores population/round state; per-fit files store validation
history and config/input/source fingerprints. Completed fits are reused after
a restart. `best_validation_config.json` stores each incumbent. Final fits save
both JSON results and CPU model-state checkpoints (`.pt`). Frozen code hashes
in `protocol.json` protect result reuse. Changing those sources requires a new
campaign directory. Three consecutive dataset-level failures stop only that
dataset and are reported in the progress file.

Monitoring:

```bash
source /home/czr/PRISM/activate_PRISM.sh
cd /data1/Chenzr/PRISM
python experiment_runs/prism_full_suite_20261111_v2/read_a800_status.py
tail -f server_runs/a800_adaptive_20261112/adaptive.log
```

The client panel at `http://127.0.0.1:8766/` reads this status through SSH every
ten seconds. It displays queue status, then switches to the adaptive campaign
automatically. If the client is turned off, its panel cannot refresh; server
logs/results continue to update. The panel can be reconnected later.

To resume after a server process interruption, first confirm the PID in the
adaptive folder is no longer running, then launch `launch_adaptive.sh` with
nohup and detached standard streams. Do not start multiple workers in the same
campaign directory. Do not change data, metric or split definitions to chase a
paper target. The adaptive worker does not use paper targets as stopping rules.
