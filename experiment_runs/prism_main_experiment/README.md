# PRISM full-size node classification benchmark

The final table has one PRISM row and eleven comparison rows: MLP, GCN, GAT,
GraphSAGE, APPNP, GPR-GNN, GCNII, HGNN, HCHA, the original CE-only framework,
and the original full framework. These are local implementations; published
leaderboard numbers are not mixed into the table.

## Data and evaluation

The intended main matrix comprises Actor, Amazon-Ratings, Coauthor-CS,
Coauthor-Physics, DBLP, OGBN-Arxiv, WikiCS, Questions, and Tolokers. There is
no node subsampling or random feature projection. Every method uses the same
row-L2-normalized node features, undirected adjacency, labels, and masks.
Questions/Tolokers use binary AUROC; the other datasets use accuracy.

Actor/Amazon/Questions/Tolokers use all ten official splits. Coauthor uses
ten class-stratified splits with 20 training nodes per class, 1000 validation
nodes and 1000 test nodes. WikiCS uses all twenty official splits: stopping
masks choose checkpoints and validation masks select configurations. DBLP
and Arxiv use their fixed official split with ten model initializations.
Final model seeds are 201-210 (201-220 for WikiCS); construction seed is 0.
Reported SD is across the ten/twenty completed fits. For multi-split datasets
it includes both split and initialization variability; it does not measure
initialization variability alone. WikiCS test nodes are shared across splits.

DBLP uses the MAGNN/PyG dataset, with all 26128 nodes and only author nodes
labeled. Its common homogeneous conversion preserves disjoint native feature
blocks (334 author, 4233 paper, 50 term dimensions) and adds four node-type
indicators. Its resulting 4621 feature dimensions differ from the 334 author
dimensions in the manuscript. This conversion must be disclosed in any paper;
its scores cannot be merged with unrelated heterogeneous DBLP protocols.
The Dropbox source was inaccessible. Numeric data were recovered from the
public MSGNN research-code mirror. That mirror has 4233 paper vocabulary
dimensions (PyG documents 4231); author features and all node-type counts match.
The original MAGNN 400/400/3257 author split was reconstructed using the exact
two train_test_split calls and seed 1566911444 from its public notebook.
Raw hashes, source URLs and this version distinction are recorded in
`data/raw/full_suite/dblp_provenance.json` under the project root.

## Tuning and models

Every method gets four preset candidates on split 0 with seed 41. The best
two are confirmed on split 1 with seed 42; fixed-split datasets keep split 0.
Mean validation score selects the configuration. Candidate IDs break ties.
All fits have at most 200 epochs and patience 40. Test is evaluated exactly
once after restoring the selected checkpoint in each final fit, and never
during tuning. Final test scores do not select candidates.

GPR-GNN uses a two-layer MLP followed by ten trainable propagation steps with
PPR initialization (alpha=.1). GCNII uses PyG GCN2Conv with 16 layers,
alpha=.1 and theta=.5. HGNN/HCHA share a binary, unfiltered two-hop structural
hypergraph capped at 32 nodes per hyperedge. HGNN uses symmetric normalized
propagation; HCHA uses two attention-enabled PyG HypergraphConv layers and
mean node attributes for hyperedge attributes. Hypergraph input construction
is therefore explicitly part of this local benchmark.

PRISM uses the repository's shared/private branches, gated leave-one-view-out
distillation, structural densification and hyperedge selection. Original
framework controls use MVHGCL without the PRISM branches/filter. Original full
uses its scheduled alignment/distillation, variance and discriminative elastic
losses. CUDA adaptation reuses repository model/loss code unchanged. Only
feature nearest-neighbor search is replaced by exact blockwise float32 CUDA
cosine search; self is excluded. Tie order is deterministic within this suite.
Prototype fitting retains the repository's sklearn implementation.

This is a common, modest tuning budget, not an exhaustive search for every
baseline's published best result. GraphGPS, NodeFormer, SGFormer, Polynormer
and other unimplemented literature methods are not claimed as reproduced.

## Reproduction and audit

Use the isolated PyTorch runtime at `D:\tmp\prism_full_suite_cuda130` via
PYTHONPATH with `D:\anoconda\python.exe`. It supports the laptop GPU's sm_120.
The earlier global/runtime installation is not modified.

`check_protocol.py` checks CUDA nearest-neighbor agreement with sklearn,
all twelve methods' forward/backward and AUROC behavior, checkpoint boundaries,
and invariance to held-out test-label changes. Its log is `protocol_tests.log`.

`suite.py` freezes the sources/configurations, records every tuning/final fit,
and writes the parent engine's results. Native PyG GraphSAGE materializes a
large edge-message tensor for high-dimensional inputs. `sparse_sage_linear.py` is a
separately frozen adaptation that computes exactly the same mean aggregation
as sparse matrix multiplication followed by degree division. Linearity allows
the neighbor projection before aggregation; its bias is added afterwards,
including for isolated nodes. This avoids aggregation over the full input
feature dimension. It retains the
native SAGEConv weights, input/hidden dropout, biases and root contributions.
Output and parameter-gradient agreement, with and without dropout, is checked
in `sparse_sage_linear_tests.log`. All nine final GraphSAGE rows use this sparse
execution engine, with the same four candidates and seeds as every other row.
The preliminary native GraphSAGE rows are excluded from the final comparison.

`serial_remaining.py` schedules the remaining frozen jobs one GPU training
process at a time after the Tolokers worker exits. This avoids memory contention
between large graphs and resumes atomic completed records without changing
configurations, seeds or selection. `prebuild_remaining.py` calls the unchanged
builder ahead of training to populate its existing source/input-keyed cache;
it trains no models and evaluates no labels or test metrics. Job state and
individual logs are retained beside the results.

OGBN-Arxiv's HCHA also uses a separately frozen equivalent computation in
`sparse_hcha.py`: scalar attention projections avoid concatenating every
node/hyperedge pair, and sparse incidence multiplication uses bounded chunks
for its attention-weight gradients. It preserves PyG's attention softmax,
dropout, normalization, biases and parameterization; forward and parameter
gradient agreement is checked in `sparse_hcha_tests.log`. Other HCHA datasets
use the original PyG implementation. This changes execution memory needs,
not the node count or the architecture.

OGBN-Arxiv's GCNII uses `sparse_gcnii.py`, also frozen under its own source
signature. It retains all sixteen PyG GCN2Conv parameter sets, dropout,
self-loop normalization, alpha/beta and identity mapping operations, using
sparse matrix multiplication to evaluate the weighted message sum. Its
logits, input and parameter gradients, including training dropout, agree in
`sparse_gcnii_tests.log`. Other datasets use native PyG GCNII. Preliminary
native Arxiv GCNII tuning records are excluded from the final matrix.

`report.py` composes the parent-engine methods and the three documented sparse
execution adaptations in
`comparison/main_results.md`, with numeric results in `comparison/summary.json`
and explicit source-record hashes. Workers can resume completed records.
`audit_suite.py` independently verifies counts,
provenance, masks, validation-only selection and summary values.

The neighboring directory without `_v2` preserves preliminary runs before
correcting two builder-parameter forwarding details. Those results are not
part of the final v2 matrix. No core repository source is edited by the suite.

Model references: [GPR-GNN](https://arxiv.org/abs/2006.07988),
[GCNII](https://arxiv.org/abs/2007.02133),
[HGNN](https://arxiv.org/abs/1809.09401),
[HCHA](https://arxiv.org/abs/1901.08150).

## Completed results

The matrix completed on 2026-11-11: 108 method/dataset cells, 1200 final fits
and 648 validation-only tuning fits. The strict independent audit passed;
see `comparison/audit.json`, `comparison/summary.json` and
`comparison/main_results.md`. No training jobs remain queued or active.

PRISM has the highest mean in this matrix on Amazon-Ratings (47.31%),
Coauthor-CS (92.43%) and OGBN-Arxiv (68.39%). These small mean differences do
not establish statistical superiority. It does not lead the remaining six
datasets. HCHA has unusually low results on several datasets under the common
presets and constructed structural hypergraph. The table records these actual
runs; it does not establish HCHA's best attainable or published performance.
Method-specific reproduction and convergence checks would be needed before
using those results as evidence of superiority in a manuscript.
