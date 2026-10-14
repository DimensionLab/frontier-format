<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pipeline run events v1

_Status: draft (2026-10-12). Implemented: the vocabulary (`frontier_canvas::run_events`);
the fold on the general engine (`crates/foldstate` through `frontier_canvas::run_machine`),
used by the server, the live board and the scrubber; the runner's emitter
(`frontier_data.board_run`, training progress and run id from `frontier_train`); the
server's log, stream, step logs and "run from here" (`routes/pipeline_runs.rs`); the app's
live board, run history and scrubber, replay of a run's data to the scrubbed event,
explanations in words, the timeline of a run, fork, re-run from an event and restoring
the board of a run, archive, the archived filter and deleting an archived run, with, when
asked, the outputs nothing else uses; the snapshot of a run and the comparison of two runs
built on it. Not built: retries of a step inside a run._

A pipeline run is a state machine. Its history is an append-only, ordered log of events;
the state of the run at any point is the fold of the events up to that sequence number.
The same log drives the live board, the run history, time travel, forks and the audit of
how a model was produced. Producers: the server (submit, cancel, fork, archive), the worker
(claim, lease), the Python runner (`frontier_data pipeline`, and the train / evaluate steps
it launches). Consumers: the app, the CLI and TUI, the agent harness, ACP.

## 1. Step state machine

```
                 ┌─────────────────────────────── cancel ───────────────────────────────┐
                 │                                                                       ▼
 pending ──► waiting ──► queued ──► running ──► succeeded                            cancelled
    │           │           │          │
    │           │           │          └──► failed ──► (retry) ──► queued
    │           │           └──► cached        ▲
    │           └──► skipped                    │
    └──► skipped   (an upstream step failed, was cancelled or skipped)
```

| State       | Meaning                                                                                                                   | At rest        |
| ----------- | ------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `pending`   | In the plan; nothing has looked at it yet.                                                                                | no             |
| `waiting`   | Its upstream steps are not all done.                                                                                      | no             |
| `queued`    | Ready; waiting for a worker or a slot.                                                                                    | no             |
| `running`   | Executing. Carries `progress` (0..1, optional) and a message.                                                             | no             |
| `cached`    | Not executed: an earlier run produced the same fingerprint; its outputs are reused.                                       | yes (success)  |
| `succeeded` | Finished; outputs are bound.                                                                                              | yes (success)  |
| `failed`    | Finished with an error (one line + a log reference). An ordinary state tagged terminal: it has one transition, the retry. | yes, retryable |
| `skipped`   | Not executed because something upstream did not succeed, or the step is outside the requested range.                      | yes            |
| `cancelled` | Stopped by a person or by the run being cancelled.                                                                        | yes            |

Allowed transitions (anything else is rejected by the server and is a bug in the producer):

| From      | To                                                                                          |
| --------- | ------------------------------------------------------------------------------------------- |
| `pending` | `waiting`, `queued`, `skipped`, `cached`, `cancelled`                                       |
| `waiting` | `queued`, `skipped`, `cached`, `cancelled`                                                  |
| `queued`  | `running`, `cached`, `skipped`, `cancelled`, `failed` (queue timeout, cause `timeout`)      |
| `running` | `succeeded`, `failed`, `cancelled`, `queued` (the worker's lease was lost, cause `timeout`) |
| `failed`  | `queued` (a retry, cause `retry`)                                                           |

`running → running` is not a transition: progress is a `step.progress` event that changes
no state. `step.progress`, `step.metric` and `step.log` for a step that is at rest are
ignored, not errors (reports race with the end of a step). An illegal `step.state` is
rejected.

`waiting → queued` is derived: the engine moves a step when its `needs` are `succeeded` or
`cached`. Producers need not send it; the server appends it to the log as a `step.state`
event with cause `upstream`, so consumers see every transition in one place.

Time never enters the fold: a timeout is an event. A lease that expires is `running →
queued` with cause `timeout` (the step is tried again by another worker); a step that waited
too long for a worker is `queued → failed` with cause `timeout`.

## 2. Run state

Derived from the step states, never stored on its own:

| Run state   | When                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| `submitted` | no step has left `pending`                                                 |
| `running`   | some step is `waiting`, `queued` or `running` and none decided the outcome |
| `succeeded` | every step is `succeeded`, `cached` or `skipped` by range                  |
| `failed`    | some step is `failed` and nothing can still run                            |
| `cancelled` | the run was cancelled and nothing is running                               |

## 3. Event

One JSON object per line. `seq` starts at 1 and has no gaps within a run; `prev` is the
SHA-256 of the previous event serialised as canonical JSON per RFC 8785 (JCS); the first
event's `prev` is 64 zeros. The chaining is that of the audit log, so a log can be verified
offline.

```jsonc
{
  "v": 1,
  "run": "prun_01a13…", // pipeline run id (not a training run id)
  "seq": 17,
  "prev": "9f2c…", // hash of event 16
  "at": "2026-10-12T14:03:11.482Z",
  "type": "step.state", // see section 4
  "step": "train", // node id on the board; absent on run.* events
  "from": "queued",
  "to": "running", // step.state only
  "cause": { "kind": "worker", "by": "node-3", "detail": "claimed" },
  "data": {}, // per type, section 4
}
```

`cause.kind`: `user` (a person pressed something; `by` is the login name), `upstream`
(a dependency finished; `detail` is its step id), `worker` (`by` is the worker id), `runner`
(the pipeline runner itself), `cache` (fingerprint hit; `detail` is the run that produced
it), `error`, `policy` (refused; `detail` is the policy key), `retry`, `timeout` (a lease or a
queue limit expired; `detail` says which), `rerun` (seeded from an earlier run), `trigger`
(an automatic rule of a step; `detail` is the rule, or the inference that made an output).

## 4. Event types

| `type`                          | `data`                                                                                                                                                                                                 | Changes state                                                                                                                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.submitted`                 | `{machine: {schema: "foldstate/1", hash}, board, board_version, project, by, plan: {steps: [{id, kind, params, fingerprint, needs: [step ids]}], edges}, parent?: {run, seq}, range?: {from?, only?}}` | creates the run; every step `pending`. `machine.hash` is the SHA-256 of the JCS-canonical machine definition generated from `plan` (step ids and `needs`), so an old log replays under the rules it was written with |
| `step.state`                    | on `running`: `{worker?, job?, attempt}`; on `succeeded`/`cached`: `{outputs: [Output], seconds, cache_of?}`; on `failed`: `{error: {line, code?, log_ref}}`; on `skipped`: `{because}`                | yes                                                                                                                                                                                                                  |
| `step.progress`                 | `{fraction?: 0..1, message?, done?, total?}`                                                                                                                                                           | no                                                                                                                                                                                                                   |
| `step.output`                   | `{output: Output}` — an output that exists before the step ends (the training run id as soon as the run is created)                                                                                    | binds an output                                                                                                                                                                                                      |
| `step.metric`                   | `{key, value, step?}` — a headline number worth showing on the node; full metrics stay on the run                                                                                                      | no                                                                                                                                                                                                                   |
| `step.log`                      | `{ref, offset, bytes}` — where the step's log grew; never the log text                                                                                                                                 | no                                                                                                                                                                                                                   |
| `step.checkpoint`               | `{name: "last.pt" \| "best.pt", path, epoch}` — a training step saved a checkpoint                                                                                                                     | no                                                                                                                                                                                                                   |
| `trigger.fired`                 | `{rule, inference?, job?, predictions?, checkpoint?, detail?}` — an automatic rule of the step started work; `cause.kind` is `trigger`                                                                 | no                                                                                                                                                                                                                   |
| `run.cancel_requested`          | `{by, reason?}`                                                                                                                                                                                        | no (steps follow with `cancelled`)                                                                                                                                                                                   |
| `run.finished`                  | `{state, seconds, summary: {ok, failed, cached, skipped}}`                                                                                                                                             | closes the log                                                                                                                                                                                                       |
| `run.archived` / `run.restored` | `{by}`                                                                                                                                                                                                 | visibility only                                                                                                                                                                                                      |

```jsonc
// Output: what a step made, by reference.
{"port": "model",                         // output port of the node
 "kind": "dataset" | "dataset_version" | "run" | "artifact" | "model" | "report" | "predictions",
 "id": "…", "version": "v_5830…",         // version for dataset_version
 "run": "…", "name": "eval/report.json",  // artifact: the run it is attached to and its name
 "fingerprint": "…"}
```

An output of kind `predictions` names the run that holds them (`id`, also `run`) and the
directory they were written to (`name`); a node wired after it shows that run's
predictions.

`fingerprint` of a step = hash(kind, op version, canonical params, fingerprints of the
outputs it consumes). Two steps with the same fingerprint produce the same outputs; that is
what `cached` and forks rely on.

## 5. Fold

```
state(0)   = {steps: {}, outputs: {}, run: none}
state(n)   = apply(state(n-1), event(n))
```

A log is well formed only if: its first event, at `seq` 1, is `run.submitted`; there is
no second `run.submitted`; no two steps of the plan share an id. The engine refuses a log
that breaks one of these, and the server never writes one (it refuses a plan with a
repeated step id with 400).

`apply` is total and deterministic: `run.submitted` creates the steps; `step.state` checks
the transition table and sets the step's state, attempt, timing, error and outputs;
`step.output` binds an output; `step.progress` / `step.metric` set the last progress and
metric; everything else leaves the state alone. The board at "seq N" is `state(N)` rendered
onto the graph in `run.submitted.plan`: each node shows its step state, and each node wired
downstream of a bound output shows that output.

## 6. Time travel, fork, re-run, restore

- **Scrub**: the app renders `state(N)` for any N. What the nodes show is read as of the
  time of event N (`until_time` on the run's metrics, artifacts and predictions,
  `docs/spec/api.md`): a plot draws the curve up to that moment, a viewer has the
  predictions that existed by then. While a past state is shown the board is read-only:
  no command changes the document, no gesture starts, and nothing is saved.
- **Why**: for event N the engine's explanation of it in `state(N-1)` (the transitions it
  could take, the guard of each, part by part) is shown as sentences
  (`frontier_canvas::run_words`): what happened and its cause, which rule did not apply
  and the condition that failed, and for every step waiting in `state(N)` the upstream
  steps that had not succeeded with the state each was in. The explanation as data stays
  available next to them.
- **Timeline**: a past run is also drawn over time (`frontier_canvas::run_timeline`): one
  lane per step of the plan, a segment for every stretch a step spent in one state, a mark
  at every `step.state` event (a failure larger, a reused step as a diamond), a line at
  event N. Choosing a mark sets N to its event. Progress, metric and log events are
  counted, not drawn, so the picture has a few elements per step for a log of any length
  (100 000 events lay out in a few milliseconds). A run with `parent` names the run and
  event it continues.
- **Run tree**: the runs of a board are also drawn as a tree
  (`frontier_canvas::run_diff::tree`): a row per run, a run that continues another under
  its parent, its row starting where it leaves the parent (`parent.seq` along the parent's
  events) and joined to it there. A row opens its run; the joint opens the parent at that
  event. A run whose parent is not listed (deleted, archived and hidden, or older than the
  newest 200) is a root marked as continuing a run that is not listed. The layout is
  linear in the number of runs and draws a few elements per run.
- **Fork** (engine primitive): continue from the exact state at sequence number N, sharing
  the log prefix 1..N. The fork is a new log whose first N events are the parent's.
  `POST …/pipeline-runs/{run}/fork {seq?, backend?, compute?}` -> 201 `{run, parent: {run,
seq}, job?, state}`. The new run's events 1..N are the parent's with the new run id and
  their own hash chain; its `run.submitted` carries `parent` and who forked. It runs under
  the parent's plan (steps, parameters, fingerprints): a step that had succeeded keeps its
  outputs, a step that was queued, waiting or running runs, a step that had failed is
  queued again (`step.state failed -> queued`, cause `retry`), a step that was skipped or
  cancelled by then stays so. 409 when the run had ended by event N (nothing is left to
  continue: re-run instead); 400 when the run has no event N. `seq` absent: the last
  event.
- **Re-run from here** (Frontier operation): `POST …/pipeline-runs/{run}/rerun {seq?,
pipeline?, document?, backend?, compute?}` creates a new run of the board (`pipeline`: its
  steps as they are now; absent: the stored board's) whose `run.submitted` carries `parent:
{run, seq}`. Steps whose fingerprint was `succeeded` or `cached` in `state(seq)` of the
  parent start as `cached` (cause `rerun`, `detail` the parent run); the rest run, with the
  edited parameters or graph. The history of a board is therefore a tree of runs:
  `RunSummary.parent` is `{run, seq}` or null.
- **Restore**: never a deletion. At the engine level it is an appended restore entry; the
  Frontier-level effect is a new board version holding the graph and parameters of that
  state, and, when asked, a re-run from it. A run keeps the board document it was started
  from (`document` of the submit, else the stored board at that moment; a fork has its
  parent's). `POST …/pipeline-runs/{run}/restore-board` -> `{board, run}` saves that
  document as the newest version of the board with the note `restored from run …`; the
  version it replaces stays in the board's history. A run from before documents were kept
  falls back to the board version it names, and is 404 when that version is gone.
- **Archive / delete**: `POST …/pipeline-runs/{run}/archive` appends `run.archived` to a
  run that has ended (409 while it is going) and the lists leave it out unless asked
  (`?archived=true` adds archived runs, `?archived=only` shows them alone); `…/unarchive`
  appends `run.restored`. The person who started a run archives it; a run of someone else
  needs `runs.delete` (maintainer). `GET …/pipeline-runs/{run}/deletion` says what a delete
  would do: `{run, archived, removes: {events, work_files}, outputs_kept: [{kind, id,
version, referenced_by: [run]}], outputs_unused, continued_by: [run], blocked}`. `DELETE
…/pipeline-runs/{run}` (`runs.delete`; 409 unless archived) removes the event log and
  the run's work files and leaves a tombstone `{run, state, created_by, created_at,
deleted_by, at, parent, removed: {events, outputs, continued_by}}`
  (`GET …/boards/{board}/runs/deleted`), so a run forked from it still names its parent.
  Both changes are audit records (`pipeline_run`). What the run produced (training runs,
  datasets, reports) is kept by default, including what no other run references.
  `DELETE …?outputs=unused` also removes `outputs_unused` of the preview, `[{kind: run |
predictions | dataset, id, name, artifacts?}]`: a training or predictions run of the
  project (with its metrics, artifacts and files), or a dataset the server wrote under the
  project, that no other pipeline run binds in any version, no board of the project names,
  and, for a dataset, that is not shared with another project and that no run which stays
  was made from. The tombstone lists them as `removed.outputs_deleted`. 400 for any other
  value of `outputs`.
- **Compare**: `GET …/pipeline-runs/{run}/snapshot` is a run as a whole in one answer:
  `{summary, document, plan: {steps: [{id, kind, params, fingerprint, needs}], edges},
range, state, datasets: [{step, port, id, version}], runs: [{step, port, id}],
predictions: [{step, port, id}], compute: {backend, request, image, worker}}`.
  `document` is the board the run was started from (null when it is no longer kept),
  `state` the fold at the head, `datasets` / `runs` / `predictions` what its steps bound,
  `compute` what its job was asked for and the worker that took it (null when every step
  was reused). The app compares two snapshots, or one with the board as it is now: the
  pipeline and every step's parameters as text, the two graphs with what was added,
  removed and changed, and the results (step states and durations, metrics, loss curves,
  fields of a sample both predicted).

## 7. Transport

Every route is under the project, so the permission table scopes it to the project
(`{id}` is a project id or its unique name; `{board}` a board id or `default`).

```
POST /api/v1/projects/{id}/boards/{board}/runs {pipeline?, range?: {from?, to?}, backend?, compute?}
                                                              -> 201 {run, job?, state}
GET  /api/v1/projects/{id}/boards/{board}/runs[?limit=]       -> [RunSummary], newest first
GET  /api/v1/projects/{id}/pipeline-runs[?active=true]        -> [RunSummary] of the project
GET  /api/v1/projects/{id}/pipeline-runs/{run}                -> {summary, state}   (state = fold at head)
GET  /api/v1/projects/{id}/pipeline-runs/{run}/events[?since=<seq>]
                                                              -> [Event] with seq > since
POST /api/v1/projects/{id}/pipeline-runs/{run}/cancel         -> RunSummary
POST /api/v1/projects/{id}/pipeline-runs/{run}/fork {seq?}    -> 201 {run, parent, job?, state}
POST /api/v1/projects/{id}/pipeline-runs/{run}/rerun {seq?, pipeline?, document?}
                                                              -> 201 {run, job?, state}
POST /api/v1/projects/{id}/pipeline-runs/{run}/restore-board  -> {board, run}
POST /api/v1/projects/{id}/pipeline-runs/{run}/archive | unarchive -> RunSummary
GET  /api/v1/projects/{id}/pipeline-runs/{run}/snapshot       -> the run as a whole (section 6, Compare)
GET  /api/v1/projects/{id}/pipeline-runs/{run}/deletion       -> what a delete removes and keeps
DELETE /api/v1/projects/{id}/pipeline-runs/{run}[?outputs=unused]
                                                              -> {deleted, deleted_by, at, removed}
GET  /api/v1/projects/{id}/boards/{board}/runs/deleted        -> [tombstone], newest first
```

The two list routes take `?archived=true|only`.

`RunSummary` = `{id, project_id, board_id, board_version, job_id?, state, head_seq,
created_by, created_at, finished_at?, parent?: {run, seq}, archived, stats?}`. `stats` is
written when the log closes: `{ok, cached, failed, skipped, slowest?: {step, seconds},
metric?: {key, value, run}}`, `metric` being `val/loss` (else `train/loss`) at the last step
of the training run the pipeline run bound. The newest run of a board is its active run: the
app attaches to it when the board opens, so the pointer lives on the server and not in a
browser.

**Stream.** The same `…/events` route with `Accept: text/event-stream` (what `EventSource`
sends) answers with server-sent events: every stored event with `seq > since` (or greater
than the `Last-Event-ID` header), then live ones as they are appended. Each is one SSE
message with `id: <seq>` and the event JSON as `data` and no `event:` name, so
`EventSource.onmessage` receives all of them and a reconnect resumes where it stopped.
After `run.finished` the server sends `event: end` and closes the stream.

**Submit.** `pipeline` is the board's steps as the canvas exports them (`frontier-pipeline`
JSON: `nodes[] {id, kind, params, inputs}` and `edges[]`). The server computes the
fingerprints, writes `run.submitted` and the first `step.state` of every step (`queued`,
`waiting`, `cached` or `skipped`), and queues one job: `python -m frontier_data pipeline
--json <document>`, the document being the pipeline plus a `run` section `{id, project,
board, work_dir, datasets_dir, steps: {<id>: {action: run | cached | skip, outputs}}}`.
The job has no image on the `local` backend; elsewhere it is `compute.image`, or the
training image of this version in `compute.registry`.

**Steps.** A node of the board is a step when it has what it needs wired in:

| Node kind                    | Step when                                                                                                                                                                                                                                                                                                                                                           | Runs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Binds                                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `import`                     | always                                                                                                                                                                                                                                                                                                                                                              | by `source`: `example` `frontier_examples install`; `hub` `frontier_hub pull --convert`; `attach` `frontier_hub attach <ref> <files>` (files the user already has for a hub entry; `subset`, `mode`); `connector` `frontier_connectors convert <ref>` with `connector`, `cases` (a glob or `auto`: one sample per case), `case_params` and `name`, or `frontier_hub attach --name` when `ref` is an object-store or `https` location; all into the datasets directory (skipped when the dataset is already there); `dataset`: a registered dataset is used as it is. A local path in `ref` (`connector`) or `files` (`attach`) must lie inside the directories requests may use (`storage.import_roots`) when requests are confined, else the submit is 403; a location that carries a credential is 400. The run's job gets what the import steps that run need, as the hub's own jobs do: the stored credential of an object-store location's provider, the Hugging Face token and the organisation's mirror for a hub entry, and the caller as `FRONTIER_USER` for files they supply (`docs/spec/hub-catalog.md`, "Contract for the app") | `dataset` (`dataset_version`)                                                                                                                   |
| `hf_source` … `dataset_sink` | always                                                                                                                                                                                                                                                                                                                                                              | the data kinds of `frontier_data.pipeline`, in process                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | the sink: `dataset`                                                                                                                             |
| `recipe`                     | a dataset reaches `dataset`                                                                                                                                                                                                                                                                                                                                         | `frontier_prep versions init` if the dataset has no versions, then the recipe in the node's `text`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `dataset` with the new `version`                                                                                                                |
| `training_run`               | a dataset reaches `dataset`                                                                                                                                                                                                                                                                                                                                         | `frontier_train train` (`config`, `epochs`, `overrides`; the train-ready gate applies unless `allow_ungated`). With a base model (`base_source` `hub`, `package` or `run` and `base_ref`: a hub model id, `name@version`, a run id) it runs `frontier_train finetune --base` instead: a hub model goes by its id and is fetched by the job into the raw files of the hub (`storage.raw_dir`), a package or a run is resolved by the server to the file the job opens (`base_dir`). A base that is asked for and not chosen fails the step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `model` (`run`, as soon as the run exists; `name` is its output directory); when it fine-tunes, `base` (below)                                  |
| `inference`                  | a `training_run` step feeds `model`, or the node names a model (`model_source` `run` or `package` with `model_ref`; the server resolves it to `model_dir`)                                                                                                                                                                                                          | `frontier_train infer` on the samples (`split`, `ids`, `first`) of the dataset wired in, named, or trained on; the predictions are attached to a run made for them                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `predictions` (`predictions`: that run)                                                                                                         |
| `evaluate`                   | an `inference` step feeds `predictions`, or a `training_run` step feeds `model`, or the node names a model                                                                                                                                                                                                                                                          | fed by predictions: `frontier_eval run` per benchmark on them, no second inference, attached to the run that holds them; otherwise `frontier_eval evaluate`; with `include_uq` and a checkpoint, `frontier_train predict` on the validation split and `frontier_eval uq`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `report` (`artifact`: the evaluation directory), `model` (the run, when there is one)                                                           |
| `input`                      | always                                                                                                                                                                                                                                                                                                                                                              | `frontier_data inputs`: typed parameters (`fields`, or a `table`) and a geometry (`geometry_source`: a parametric `geometry` such as `ahmed` at `resolution`, `geometry_files`, or the samples `ids` of the dataset wired in) as an input dataset, every case in the `test` split; checked against the contract of the model wired in or named (`model_source`, `model_ref`), when there is one                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `dataset` (`dataset_version`)                                                                                                                   |
| `adapt`                      | a dataset reaches `dataset` and a model reaches `model` or is named                                                                                                                                                                                                                                                                                                 | `frontier_train compat` for the verdict (unless `fixes` is given), then `frontier_prep adapt` on a copy of the dataset: the fixes as a recipe (`values` for parameters the model needs); a dataset that fits as it is passes through                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `dataset` (`dataset_version`)                                                                                                                   |
| `optimise`                   | a model reaches `model` or is named                                                                                                                                                                                                                                                                                                                                 | `frontier_train optimise`: the parameters `free` of the parametric `geometry` moved (`method`, `budget`, `starts`, `bounds`, `fixed`, `conditions`, `checks`, `minimise`, `trust_z`, `resolution`) to minimise what the model predicts; each evaluation is a `step.metric` (`optimise/best`) and a metric of the run that holds the results; `optimum.json` is attached to it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `metrics` (`run`: the one that holds the search), `dataset` (`dataset_version`: the optimum's geometry), `optimum` (`artifact`: `optimum.json`) |
| `report`                     | an `evaluate` step feeds `report`, or a `training_run` step feeds `model` (then it evaluates first, without uncertainty)                                                                                                                                                                                                                                            | `frontier_eval report`, attached to the run                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `report`, `model` (the run)                                                                                                                     |
| `service.in`, `service.out`  | never: the Request and Response nodes of a served board are boundaries, not steps (`docs/spec/api.md`, "Services"). The canvas leaves them out of the export; a runner that meets one passes through it (`succeeded` at once, a Response passing on the outputs wired into it). The service runtime, not a board run, executes the steps between them once per call | nothing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `service.out`: what reaches it                                                                                                                  |

An evaluation is complete without a report after it.

**What a fine-tuned step started from.** Once the base weights are loaded the training
tool reports two events for the step: `step.output` with `{port: "base", kind: "model",
id, name, fingerprint}` (`id`: the hub id, `name@version` of a package, the run id, else
the file name; `name`: `hub`, `package`, `run` or `file`; `fingerprint`: the SHA-256 of
the weights file that was read), and `step.metric` with key `finetune/fraction_loaded`
(0..1: the share of the new model's parameters that came from the base). `base` is not a
port of the node and nothing is wired to it; it is bound like an output so that the fold,
a scrubbed state and a re-run all carry it. The same facts are in the run's summary
(`finetune.identity`, `finetune.fraction_loaded`) and in `finetune_report.json`, which
lists every tensor. A run validated on the cases it trained on (or not validated) has
`summary.smoke`, one line that says so ("2 training cases, no held-out split"); an
evaluation of the training split is a smoke evaluation without a verdict.

**Automatic rules.** A step's `trigger` parameter says when it runs without being asked;
the server decides, from events it receives anyway:

| `trigger`         | Fires when                                                                                    | What runs                                                                                                                                                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `checkpoint`      | a `step.checkpoint` of a step it needs arrives while the run is open                          | an inference job next to the running pipeline (`POST …/inference` semantics, cause `trigger`) on that checkpoint. `trigger.fired` is appended for the step; when the job succeeds, `step.output` binds its predictions on the step, which is still `waiting` |
| `upstream`        | a run ended in which the step was left out by the range while what it needs ran and succeeded | a run of the board restricted to the step (`range {from: X, to: X}`), with `trigger.fired` in it                                                                                                                                                             |
| `dataset_version` | a data-preparation job wrote a new version of the dataset the step names                      | the same                                                                                                                                                                                                                                                     |
| `schedule`        | `every_min` minutes passed since it last ran (the period starts when the rule is first seen)  | the same                                                                                                                                                                                                                                                     |

Per node: one job in flight, the newest request replaces one that waits, and starts are
`min_interval_s` apart. A rule of a board run waits while the board has a run that is not
finished; waiting requests and schedules are looked at every 15 seconds. Events are not
appended to a closed log, which is why a rule that fires after a run ended starts a run of
its own.

A dataset reaches a step from the step that writes it, through dataset cards, or from a
dataset card that names a registered dataset.

`pipeline` may be left out: the server then exports the steps of the stored board. What a
run imports or writes goes under the project (`<data dir>/projects/<project id>/datasets`),
and its work files under `<data dir>/pipeline-runs/<run id>`.

**Range.** `range.to` names the last step that may run: it and what is upstream of it; the
rest is `skipped`. `{to: X}` runs X and whatever upstream of it has no reusable result;
`{from: X, to: X}` runs exactly X. 400 when `to` is not `from` or downstream of it.

**Run from here.** `range.from` names a step. That step and everything downstream of it
run. A step upstream of it whose fingerprint `succeeded` or was `cached` in one of the
board's last 30 runs starts as `cached` (cause `cache`, `detail` and `data.cache_of` = the
run that made the outputs, which are bound again); one without a hit runs. Steps on other
branches are `skipped` (`because: "outside the requested range"`). When nothing has to
run, no job is queued and the log closes at once. This is the Frontier operation of section
6 with the board's history as the parent; `POST …/rerun {seq}` from a chosen state of one
run is not built.

**Producers.** The runner reports on stdout, one line per event:

```
frontier-event: {"type": "step.state", "step": "train", "to": "running", "at": "…", "data": {…}}
```

without `v`, `run`, `seq`, `prev` and `from`. The worker forwards a job's output to
`POST /api/v1/jobs/{id}/status` as it does for every job, with the job's own credentials;
the server takes the event lines of a job that belongs to a run (they also stay in the
job's log), assigns `seq`, `prev` and `from`, and appends them in order. Chosen over a
second channel because it needs nothing new on the worker, in a container or on a cluster:
if a job can log, it can report. A step that runs another tool (`frontier_train`,
`frontier_eval`) sets `FRONTIER_PIPELINE_STEP=<step id>` for it; the tool prints its own
event lines (`step.output` with the training run id as soon as the run exists,
`step.progress` per epoch, `step.checkpoint` after it saves `last.pt` or `best.pt`) and the
runner passes them through.

The server, on append:

- drops `run.submitted`, `run.cancel_requested`, `run.archived` and `run.restored` from a
  runner: only the server says those;
- fills in what a producer may leave out: `queued` before `running` for a step that was
  `pending` or `waiting`, and `waiting -> queued` (cause `upstream`) for every step whose
  `needs` have all succeeded, written before the producer's next event;
- drops, with a warning in its own log, an event the transition table does not allow, and
  a repeated report of the state a step is already in;
- registers a dataset output that names a directory (`name`) and has no `id`, and writes
  the id into the output;
- writes `step.log {ref: "jobs/<job>", offset, bytes}` when a step ends: the part of the
  job's log between the step's `running` line and the line that ended it (a runner's
  own `step.log` is dropped). `GET …/pipeline-runs/{run}/steps/{step}/log[?tail=200]`
  returns those lines without the event lines, and, while the step runs, what it has
  written so far;
- closes the log when the job ends without `run.finished`: on failure the reason (the
  job's error, else the last line of its log, or that line's `message` when it is a JSON
  log record) goes on the step that was running; when
  none was running and a step had already failed, nothing else fails; when nothing had
  failed either, the first step due does. The rest are `skipped` because of that step. On
  cancel the open steps are `cancelled`. A runner that prints nothing therefore still
  yields a complete log;
- closes the log in the same way when the job is failed because its worker stopped
  renewing its lease: the step that was running is `failed` with cause `timeout`.

`prev` is the SHA-256 of the previous event as canonical JSON by the engine's exporter
(`foldstate::canon`): RFC 8785, with one documented deviation: integers are written in
full beyond 2^53, so a sequence number or a millisecond timestamp is never rounded.
`run.submitted.data.machine` is `{schema, hash}` as the engine names the machine generated
from the plan; a `run.submitted` whose hash does not match its plan is refused.

## 8. What the engine must provide

So that this schema can sit on top of it unchanged:

1. A pure transition function `(state, event) -> (state, effects)` with no clock and no
   I/O; time and identity arrive inside events.
2. A machine definition that is data (serialisable, versioned) so the server, the worker,
   the CLI and the app (wasm) run the same one, and a snapshot of the state that is data
   too.
3. Parallel regions or an equivalent: a run is the product of its steps' machines, and the
   number of steps is only known at `run.submitted`.
4. Guards on transitions that can see other regions (a step becomes `queued` when its
   `needs` are `succeeded` or `cached`).
5. Replay from the log to any sequence number at a cost linear in the events, and a fork
   that starts a new instance from a snapshot.
6. Rejection of an event the definition does not allow, as a value, not a panic.

## 9. Open questions

1. Granularity for training: one step, with epochs as `step.progress` and metrics on the
   training run (proposed), or sub-steps.
2. Retention of event logs and of outputs referenced only by archived runs.
3. Whether `step.metric` belongs in the log at all, or only a pointer to the run's metrics.
