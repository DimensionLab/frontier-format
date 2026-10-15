<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontier control-plane API and job specification

The wire contract spoken by the SDK, the worker and the app.

## Server API (frontier-server, JSON unless noted)

The control plane's own README has a curl example of every route and the exact bodies;
the configuration guide has the settings, credentials and policy contracts. Errors are `{"error": "..."}` with 400 / 403 / 404 / 409 / 422 /
500; a policy refusal is 403 `{error, status: "blocked_by_policy", policy, message, locked,
reason}`.

```
GET    /healthz
GET    /api/v1/projects                       -> [Project]
POST   /api/v1/projects {name}                -> Project
GET    /api/v1/runs?project=<id>[&kind=inference|all] -> [Run]   (runs that only hold an inference's predictions are left out unless asked for)
POST   /api/v1/runs {project,name,config,tags}-> Run
GET    /api/v1/runs/{id}                      -> Run
PATCH  /api/v1/runs/{id} {state,...}          -> Run
POST   /api/v1/runs/{id}/metrics  [Metric]    -> 204   (batch; Metric = {step,timestamp,values:{key:number}})
GET    /api/v1/runs/{id}/metrics?keys=a,b&since_step=&until_step=&until_time=&downsample=  -> {key:[[step,value],...]}
GET    /api/v1/runs/{id}/events (SSE)         -> metric batches and run updates as they arrive
POST   /api/v1/runs/{id}/artifacts (multipart)-> {name,size,url}; GET .../artifacts, GET .../artifacts/{name}
GET    /api/v1/runs/{id}/predictions          -> [{sample_id, kind, n_steps, url}]; .../predictions/{sid}/{kind,times,surface,mesh,grid,field,field-stats}
POST   /api/v1/datasets {name,uri}            -> Dataset (reads manifest.json)
GET    /api/v1/datasets, /api/v1/datasets/{id}, .../quality, .../samples, .../samples/{sid}/{meta,kind,times,surface,mesh,grid,field,field-stats}
GET    .../samples/{sid}/topology?lod=        -> TOP1 block (and /runs/{id}/predictions/{sid}/topology): see "Typed binary frames"
GET    .../samples/{sid}/frames?t=a-b&fields=a,b&lod=&component=  -> FRM1 block (and .../predictions/{sid}/frames)
DELETE /api/v1/datasets/{id}                  -> {id,name,unregistered:true,files_deleted:false,path}   (files never touched)
GET    /api/v1/hub/datasets?domain=&tag=&q=…  -> {total,count,datasets,facets}; GET .../{id}
POST   /api/v1/hub/datasets/{id}/pull {subset,max_bytes,target}
                                              -> 201 {job,hub_id,subset,target}; 409 {status:"needs_auth",dataset,credential,url,message,next_steps,policy}
POST   /api/v1/hub/datasets/{id}/access-check {subset?}
                                              -> {ok:true,files,bytes,source} | {ok:false,status,message,next_steps}   (asks the publisher's host with the caller's stored credential, else the organisation's)
POST   /api/v1/hub/datasets/{id}/attach {source,subset?,mode?,replace?,target?,connector?,cases?,case_params?,backend?}
POST   /api/v1/hub/attach {name,source,connector?,cases?,case_params?,mode?,target?,backend?}
                                              -> 201 {job,hub_id|name,subset,source,target}; 403 a path outside storage.import_roots; a refused job has {status:"mismatch",message} as its error
PUT    /api/v1/hub/uploads/{name}  (the file as the body)  -> 201 {source,name,bytes,sha256}   (`source` is what attach takes; removed when that job ends, or after a day)
GET    /api/v1/hub/mirrors                    -> {mirrors:[{pattern,location}]}   (setting hub.mirrors; changed with PATCH /api/v1/settings)
GET    /api/v1/hub/mirror/{id}?subset=        -> {configured,held,location,files,bytes,pushed:{by,at}} | {configured:false,held:false} | {configured:true,held:false,readable:false,message}
POST   /api/v1/hub/mirror/{id}/push {subset?,acknowledged?,backend?}
                                              -> 201 {job,hub_id,subset,location}; 409 {status:"needs_acknowledgement",reason,license,location} | {status:"no_mirror"}
                                                 details, roles and audit events: docs/spec/hub-catalog.md "Server routes"
POST   /api/v1/jobs {spec: JobSpec}           -> Job + pending_reason (+ allocation once claimed; + result: the JSON object a hub pull, attach or mirror job printed last); 403 blocked_by_policy ; GET /api/v1/jobs, /api/v1/jobs/{id}, .../log ; POST .../cancel
POST   /api/v1/jobs/{id}/status {state, log}  -> 204   (a refused hub pull puts the gated-dataset object, as a JSON string, into the job's error)
POST   /api/v1/workers/heartbeat {NodeInfo}    -> {jobs:[JobSpec + allocation]}   (worker pulls work; renews allocation leases; credentials only here)
GET    /api/v1/workers[?all=true]              -> [{worker_id,hostname,node_id,backend,last_seen,online,seen_secs_ago,info}]   (silent > 1 h left out)
GET    /api/v1/compute/inventory              -> {generated_at,nodes:[{node_id,hostname,workers,online,seen_secs_ago,cpus,memory_mb,labels,driver_version,cuda_version,container,address,gpus:[{id,index,name,vendor,memory_mb,memory_used_mb,utilization_pct,nvlink_peers,pools,roles,held_by,rendering}]}],pools,summary}
GET    /api/v1/compute/pools                  -> {pools,resolved,locked,reason,rendering_pool,allocation_policy,overlaps}
PUT    /api/v1/compute/pools {pools,force?}   -> the same; 400 invalid, 403 locked, 409 {status:"overlap",overlaps,suggestion}
GET    /api/v1/compute/allocations[?all=true] -> {allocations:[Allocation],pending:[{job_id,reason}],lease_secs}
POST   /api/v1/compute/allocate {pool?,role?,gpus?,nodes?,gpu_memory_gb?,exclusive?,node?,holder?,ttl_secs?}
                                              -> 201 Allocation; 409 {error,status:"no_capacity",needed,available,suggestion}
POST   /api/v1/compute/allocations/{id}/release -> Allocation
GET    /api/v1/compute/rendering              -> {pool,reserved,devices,adapter_hint}
GET    /api/v1/compute/workers                -> {online,local,local_running,local_executor_setting,worker_binary,start_command}
POST   /api/v1/compute/workers/local/{start|stop} -> 201 LocalWorker (503 no_worker_binary) | 204
GET    /api/v1/plugins[?available=&refresh=]  -> {plugins,connectors,exporters,importers,python,state_file,...}
GET    /api/v1/plugins/{id}, .../health, .../files/{path}; GET /api/v1/plugins/doctor, /exporters, /ops, /ops/{op}
POST   /api/v1/plugins/{id}/{install|uninstall|enable|disable} -> 202 {op}   (output grows line by line in /ops/{op})
POST   /api/v1/plugins/exporters/{vendor}/install {dest,force}; POST /api/v1/plugins/try/{detect,describe,convert}
GET    /api/v1/benchmarks, /api/v1/benchmarks/{name}; POST /api/v1/benchmarks/{name}/run; GET /api/v1/benchmark-candidates
GET    /api/v1/reports, /api/v1/evidence, /api/v1/evidence/{id}; PUT /api/v1/evidence/{id}
GET    /api/v1/settings                       -> {schema,sections,values:{key:{value,source,locked,reason,origin,default,editable,...}},files,problems,startup,server}
                                                 source: default | system | server | user | project | env | cli
PATCH  /api/v1/settings[?confirm=key,...] {key: value|null}
                                              -> the same body + written:[{key,file}]; 400 unknown key; 403 locked or read-only start-up setting
                                                 ({read_only}); 409 {status:"needs_confirmation",confirm:[key]}; 422 wrong JSON type or out of range,
                                                 or {status:"would_not_start"}
POST   /api/v1/settings/startup/discard       -> the same body + discarded:[key]   (start-up settings back to what the server runs with)
GET    /api/v1/settings/schema, /api/v1/settings/audit
GET    /api/v1/server                         -> {boot_id,started_at,pid,version,restart:{how,command,unit,allowed,reason},pending_restart,jobs,scheduled}
POST   /api/v1/server/restart {when?: now|idle} -> 202 {restarting|scheduled,boot_id,jobs}; 409 {status:"manual",command}; 403 blocked_by_policy
DELETE /api/v1/server/restart                 -> {cancelled}   (a restart scheduled for when the built-in worker is idle)
GET    /api/v1/credentials[?scope=]           -> {credentials:[{id,title,kind,fields,configured,source,last_test,...}],store}
                                                 source: null | {store: keychain|file|server|env|job, detail}
                                                 last_test: null | {at, ok, identity, detail}
GET    /api/v1/credentials/{id}; PUT (write-only fields); POST .../test -> {ok,identity,error,detail,status,tested_at}
DELETE /api/v1/credentials/{id}               -> the row; 404 when nothing is configured; 409 when the value comes from the environment
POST   /api/v1/credentials/{id}/oauth/start, .../oauth/poll; GET .../oauth/callback
```

**Dataset samples by version.** `GET /api/v1/datasets/{id}/version` -> `{id, version}`: the
content version of what the dataset holds now (`c_<hash of its manifest>`); it changes
whenever the dataset is written again under the same id. Every read under
`/api/v1/datasets/{id}/samples/…` takes `v=<version>`: with it the answer is that version's
and carries `Cache-Control: public, max-age=31536000, immutable`; when the dataset has been
written again since, the answer is 409 `{status: "version_changed", version: <now>}`, never
data of another version. Without `v` the answer is the current data with `no-cache`. Both
carry `ETag` (version, path and query) and `X-Frontier-Dataset-Version`, and a request with
a matching `If-None-Match` gets 304 without the sample being read, on the Arrow routes too.
The app asks for sample blocks by version and asks for the version again on 409.

**A run as of a moment.** `until_time=<RFC 3339>` on a read under `/api/v1/runs/{id}/` asks
for the run as it was then: `…/metrics` ends at the last step of each key reported by that
time (inclusive; it combines with the step bounds and with `downsample`), `…/artifacts`
lists what was attached by then, and `…/artifacts/{name}` and `…/predictions…` answer 404
for something attached later. A time that is not RFC 3339 is 400. The board uses it while a
past state of a pipeline run is shown (`docs/spec/pipeline-run-events.md`, section 6).

### Data preparation (`docs/spec/data-prep.md` section 8)

Reads come from the dataset's `.frontier/` store directly (JSON files, no interpreter);
writes and array statistics go through `python -m frontier_prep versions ...` with the
server's `compute.python`. `{id}` is a dataset id or its (unique) name; `{v}` is a version
reference: `HEAD`, `HEAD~n`, a tag, a full id or a unique prefix. A dataset without a store
answers 409 `{status: "no_version_store", next_steps}`.

```
GET    /api/v1/datasets/{id}/versions[?limit=]      -> {head, tags:{name:version}, versions:[VersionRecord]}
GET    /api/v1/datasets/{id}/versions/{v}           -> {version: VersionRecord, op: OpRecord|null, assertions:[Assertion],
                                                       effective_assertions:[Assertion], approvals:[ProvenanceRecord]}
GET    /api/v1/datasets/{id}/versions/{a}/diff/{b}[?arrays=false] -> the statistical diff (data-prep.md section 5)
POST   /api/v1/datasets/{id}/versions/{v}/revert {message?} -> 201 {version, message}   (new version; the manifest is refreshed)
POST   /api/v1/datasets/{id}/versions/{v}/tags/{name}        -> {tag, version, tags}
GET    /api/v1/datasets/{id}/provenance[?kind=&since=&format=graph|records|prov]
                                                    -> {head, tags, graph:{nodes:[version|op nodes], edges:[{from,to,kind:parent|input|output}]}, records:[...]}
                                                       format=prov: W3C PROV-JSON (entity/activity/worker/used/wasAssociatedWith/...)
GET    /api/v1/datasets/{id}/provenance/approvals   -> {provenance:[approval records], audit:[ApprovalRecord]}
POST   /api/v1/datasets/{id}/provenance/approvals {version?, decision:"approve"|"reject", note?, approved_by?}
                                                    -> 201 ApprovalRecord   (recorded in provenance by Python and in the audit table)
GET    /api/v1/datasets/{id}/assertions[?version=HEAD&effective=true] -> {version, effective, counts:{pass,warn,fail,skip}, assertions:[Assertion]}
GET    /api/v1/datasets/{id}/lineage?target=samples/run_1 -> {target, lineage:[{version,created_at,message,op,op_record,params,n_files,added,removed,modified}]}
GET    /api/v1/prep/ops[?refresh=true]              -> {frontier_prep_version, cached_at, from_cache, contract, tools:[{name,description,input_schema,x-frontier}], types}
                                                       (export-tools, cached in memory and <data_dir>/prep/ops.json keyed by the installed frontier-prep version;
                                                        one interpreter call on the first request or on refresh; 503 no_frontier_prep when the interpreter lacks it)
GET    /api/v1/prep/ops/{op}                        -> one tool (by op id `prep.fields.map` or tool name `prep_fields_map`)
POST   /api/v1/prep/{op}/plan|preview|run  PrepRequest -> 201 {op, mode, dataset, dataset_root, job: Job, approval?: ApprovalRecord, result_url}
                                                       PrepRequest = {dataset, params:{}, selection?:{max_samples,samples,renders}, seed?, workers?, message?, pool?,
                                                                      approval?:{plan_id?, approved_by?, decision?, note?}}
                                                       a job `python -m frontier_prep <mode> <op> --dataset <root> --params ... --json` with compute role `prep`;
                                                       run without an approval: 403 blocked_by_policy (policy.ai.autonomous_op_classes) unless the op's class is
                                                       autonomous; a `reject` approval: 409 {status:"rejected"}; the reviewer becomes `--approve <name>` of the op
GET    /api/v1/prep/jobs/{id}                       -> {job, op, mode, dataset, result: the op's JSON (Plan | Preview | Result) once the job is terminal}
GET    /api/v1/prep/approvals[?dataset=|session=]   -> [ApprovalRecord]   (the audit; tokens are never returned here)
```

```jsonc
// VersionRecord
{"id": "v_5830a015a9e79275", "parent": "v_9cd1...", "created_at": "2026-10-11T14:04:56+00:00",
 "message": "prep.quality.check (ahmedml-40)", "actor": {"kind": "human", "name": "cleanup-pass"},
 "op_record": "c389ac55...", "op": "prep.quality.check", "stats": {"n_files": 5350, "n_bytes": 2412831046, "n_new_bytes": 223626, "n_samples": 40},
 "tags": ["train-ready"], "n_samples": 40}
// Assertion
{"id": "...", "version": "v_...", "check": "force_closure", "status": "pass" | "warn" | "fail" | "skip", "target": "samples/run_1" | "dataset",
 "detail": {...}, "op_record": "...", "at": "..."}
// ApprovalRecord
{"id": "...", "session_id": null, "dataset": "<dataset id>", "op": "prep.fields.rename", "plan_id": "...",
 "decision": "approve", "reviewer": "alice", "note": "...", "at": "...", "token": "<only in the response that minted it>"}
```

### Agent sessions

A session is a job running `python -m frontier_agent session run ...`; its record is the
harness's `<data_dir>/agent-sessions/<id>.jsonl`, which the server reads and streams. Policy
is checked before the job exists: `Policy::check_ai_provider` (`policy.ai.data_leaves_site`,
`policy.ai.allowed_providers`) answers 403 `blocked_by_policy` with the key.

```
POST   /api/v1/agent/sessions {task, dataset?, provider?, model?, hosted?, harness?: auto|builtin|pi, approve?: [op ids], max_tool_calls?, backend?, pool?}
                                                    -> 201 {session: AgentSession, job: Job, approvals: [ApprovalRecord]}
                                                       provider/model default to `local` + policy.ai.local_model, or policy.ai.hosted_provider/hosted_model with
                                                       `hosted` (the job then carries FRONTIER_CREDENTIALS=<provider> for the credential hand-over at claim time);
                                                       `approve` are ops a reviewer approves in advance (recorded); every other plan is refused by the harness
GET    /api/v1/agent/sessions[?dataset=&limit=]     -> [{session: AgentSession, summary: SessionSummary}]
GET    /api/v1/agent/sessions/{id}[?events=false]   -> {session, job, summary, approvals:[ApprovalRecord], events:[SessionEvent]}
GET    /api/v1/agent/sessions/{id}/events[?since=<seq>&kind=] -> [SessionEvent]
GET    /api/v1/agent/sessions/{id}/stream (SSE)     -> one `event: <kind>` per record line as it is written; `event: end {state, events}` once the job is terminal
GET    /api/v1/agent/sessions/{id}/approvals        -> [ApprovalRecord]
POST   /api/v1/agent/sessions/{id}/approvals/{plan_id} {decision?: approve|reject, note?, approved_by?}   (reviewer identity required: body or X-Frontier-Actor)
                                                    -> 201 ApprovalRecord with `token` = HMAC-SHA256(server secret, "<session>:<plan_id>")[:32] on approve;
                                                       also appended to <record>.approvals.jsonl for the harness reviewer; 409 once the session ended
POST   /api/v1/agent/sessions/{id}/replay {dataset?} -> 201 {session (harness "replay"), job}   (`python -m frontier_agent session replay <record> [--dataset <root>]`)
GET    /api/v1/agent/judgments[?dataset=&session=&limit=] -> {dataset, sessions:[ids], judgments:[judgment events + session, dataset],
                                                               judge_store: null | {path, rows:[log rows of ~/.frontier/agent/judge.sqlite for those sessions]}}
```

```jsonc
// AgentSession
{"id": "20261011-185150-c63ec1", "dataset": "<dataset id>", "dataset_root": "/.../datasets/ahmedml-40", "task": "...",
 "provider": "local", "model": "qwen3:8b", "harness": "builtin", "state": "queued" | "running" | "succeeded" | "failed" | "cancelled",
 "job_id": "...", "record_path": "/.../agent-sessions/<id>.jsonl", "created_by": "alice", "created_at": "...", "updated_at": "..."}
// SessionSummary
{"id", "task", "model", "harness", "tool_calls", "tools_used": [...], "approvals_requested", "approvals_granted", "ops_run": [...],
 "judgments", "errors", "outcome", "final_text", "usage", "duration_s", "n_events"}
// SessionEvent: {"seq", "t", "kind": session_start|message|tool_call|tool_result|preview|judgment|approval_request|approval|op_run|note|error|session_end, ...}
```

Persistence: SQLite file (`server.db_path`, default `./frontier.db`) via sqlx; artifacts
under `server.data_dir`. Must run with zero config.

### Typed binary frames (playback engine)

Surface and mesh samples (datasets and prediction stores, the same sub-routes under
`/runs/{id}/predictions/{sid}/`) are streamed to viewers as little-endian typed blocks a
client uploads to the GPU as they are: no Arrow, no per-frame decode. Both blocks are
`magic (4 bytes) | u32 header length | UTF-8 JSON header padded with spaces to a multiple
of 4 | payload`, so every typed array in the payload is 4-byte aligned.

`GET .../topology?lod=<L>` -> `application/vnd.frontier.topology`, magic `TOP1`:

```jsonc
{"version": 1, "sample_id": "...", "dataset_id" | "run_id": "...", "lod": 2, "lods": [0, 1, 2, 3],
 "n_points": 23011, "n_faces": 20000, "n_full_points": 376593, "dim": 3, "moving": true,
 "bounds": [xmin, xmax, ymin, ymax, zmin, zmax], "positions_bytes": 276132, "indices_bytes": 240000}
```

payload: `positions f32[n_points * 3]` (step 0) then `indices u32[n_faces * 3]`.

`GET .../frames?t=<a>[-<b>]&fields=<f1,f2>&lod=<L>&component=<mag|index>` ->
`application/vnd.frontier.frames`, magic `FRM1`:

```jsonc
{
  "version": 1,
  "sample_id": "...",
  "lod": 2,
  "n_points": 23011,
  "n_faces": 20000,
  "moving": true,
  "fields": ["stress_vm"],
  "t": [3, 4, 5],
  "times": [0.03, 0.04, 0.05],
  "n_steps": 13,
  "frame_bytes": 368176,
  "positions_bytes": 276132,
  "scalar_bytes": 92044,
}
```

payload, per step of `t` in order: `positions f32[n_points * 3]` when `moving`, then one
`f32[n_points]` per entry of `fields` (one scalar per vertex: vectors reduced by
`component`, default magnitude; cell values averaged onto the vertices). A request is
refused with 400 when it would exceed 96 MB (ask for fewer steps). `t` past the end is
400; a missing `fields` is 400; a grid sample is 400.

Levels of detail: surface stores serve the levels they carry (`lod/<L>/`). Mesh stores
serve level 0 (the drawable surface of the cells) and, when the surface has more
triangles than a level's target (1: 100k, 2: 20k, 3: 5k), levels 1-3 as _vertex maps_:
`lod/<L>/{points,faces,vertex_map}` written at ingest by `frontier_data` (also added to
existing datasets by `python -m frontier_data mesh-lods <dataset>`), else built on the
fly by vertex clustering on first use and cached in memory. Per-step positions and
fields at a mapped level are gathered through the map, so a moving mesh plays at any
level. `GET .../kind` lists the levels in `lods`; `GET .../surface?lod=` (Arrow) serves
mapped levels of mesh stores too.

Caching: responses carry `ETag` (from the store's modification time and the query) and
`Cache-Control: no-cache` (dataset samples are rewritten under the same id by a recipe or a
re-import, prediction stores while a run trains: a client always revalidates);
`If-None-Match` answers 304. The server
keeps the last 512 MB of encoded blocks in memory (LRU), so a second viewer of the same
sample costs a memcpy. Content encoding is not applied: f32 payloads compress poorly
and on a local link the CPU time is a net loss; put a compressing proxy in front for
remote users.

### Project lifecycle, example projects, per-user state

| Method   | Path                                             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET      | `/api/v1/projects/{id}/deletion`                 | What a delete would remove: `{project, confirm: <name>, removes: {boards: [{id, name}], runs: [{id, name}], artifacts, jobs, datasets_deleted: [{id, name, path}], datasets_unregistered: [...], dataset_references, bytes}, active_jobs, blocked: <reason> \| null}`. `datasets_deleted` are owned datasets stored under the project's own directory (`<data dir>/projects/<id>`); an owned dataset stored elsewhere is only unregistered and its files stay                                                                                                                                                                                                                                                                           |
| POST     | `/api/v1/projects/{id}/delete`                   | `{confirm: <the project's name>}` -> the same body with `deleted: true`, `deleted_by`, `files_not_removed`. Removes the project's boards and their history, runs with metrics and artifacts, jobs with logs, pipeline runs with their work directories, agent sessions, project-scoped credentials, members, the datasets above and the project's directory. 400 when `confirm` is not the name; 409 while a job of the project is queued or running, and for the default project                                                                                                                                                                                                                                                       |
| POST     | `/api/v1/projects/{id}/archive`, `.../unarchive` | `{id, name, archived, archived_at}`. An archived project keeps everything; `summary.meta.archived_at` is set and the app lists it under _Archived_. 409 for the default project                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| GET      | `/api/v1/examples`                               | `{examples: [{id, title, summary, minutes, dataset: {samples, bytes, licence, licence_url, attribution}, stages: [{id, title, step}], pipeline: {source, ref, recipe, config, epochs, overrides, loss_keys, error_keys}}]}`: the examples of the `frontier-examples` Python package in the job interpreter (`compute.python`). The server asks the interpreter once where the package data is (or reads `FRONTIER_EXAMPLES_DIR`) and reads `catalog.json` there on each request. `pipeline` is what a board needs to run on the example's sample; `recipe` is the recipe's JSON text. Empty when the package is not installed. A server built with `--no-default-features` (no `examples` feature) has none of the three example routes |
| POST     | `/api/v1/projects/from-example`                  | `{example, name?}` -> 201 `{id, project, board_id, example: Marker}`. An ordinary project (`meta.template = "example:<id>"`) whose main board is an ordinary pipeline: Import (`source: "example"`), Recipe (the example's recipe in the node's `text`), Training run, Evaluate, Report, wired in that order, with a dataset card, viewers and plots. A taken name gets a number. Nothing runs and no dataset exists yet                                                                                                                                                                                                                                                                                                                |
| GET      | `/api/v1/projects/{id}/example`                  | `Marker`: `{example, title, dataset, available, project_id, stages: [{id, title, step}]}`: which example the project was made from and which step of its main board each stage of the app's guide runs. 404 for a project that was not made from an example. The state of the stages is the state of the board's pipeline run; a stage is run with `POST /projects/{id}/boards/{board}/runs {range: {from: <step>, to: <step>}}`                                                                                                                                                                                                                                                                                                        |
| GET, PUT | `/api/v1/me/state/{key}`                         | Small per-user interface state (which onboarding steps were finished): `{key, value}`; `value` is any JSON up to 16 KiB, `null` when unset. Keyed by the caller's user id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

Permissions: the deletion preview, delete, archive and unarchive need `members.manage` in
the project (maintainer); `from-example` needs `projects.write`; `/me/state` any signed-in
caller. All changes are audited
(the delete's audit record carries the digest of the response that lists what was removed).

### Pipeline runs of a board (`docs/spec/pipeline-run-events.md`)

| Method | Path                                                               | Notes                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| POST   | `/api/v1/projects/{id}/boards/{board}/runs`                        | `{pipeline?, range?: {from?, to?}, backend?, compute?}` -> 201 `{run, job?, state}`. Runs the board's steps as one job (`pipeline` absent: the stored board's); `range.to` is the last step that may run; with `range.from`, that step and what is downstream of it run and steps upstream are reused by fingerprint (`cached`). 400 when the board has no steps, the steps form a cycle or `from` is not a step |
| GET    | `/api/v1/projects/{id}/boards/{board}/runs`                        | `?limit=` -> `[RunSummary]`, newest first; the first is the board's active run                                                                                                                                                                                                                                                                                                                                   |
| GET    | `/api/v1/projects/{id}/pipeline-runs`                              | `?active=true&limit=` -> `[RunSummary]` of the project (`active`: not finished)                                                                                                                                                                                                                                                                                                                                  |
| GET    | `/api/v1/projects/{id}/pipeline-runs/{run}`                        | `{summary, state}`; `state` is the fold of the log: `{run, seq, steps: {<id>: {state, attempt, started_at, changed_at, progress, message, error, outputs, metrics, since_seq, ...}}, order, finished, ...}`                                                                                                                                                                                                      |
| GET    | `/api/v1/projects/{id}/pipeline-runs/{run}/events`                 | `?since=<seq>` -> `[Event]`. With `Accept: text/event-stream`: server-sent events, one message per event (`id: <seq>`, the event as `data`), from `since` or `Last-Event-ID`, live until `run.finished`, then `event: end`                                                                                                                                                                                       |
| POST   | `/api/v1/projects/{id}/pipeline-runs/{run}/cancel`                 | cancels the run's job; open steps become `cancelled`. 409 when the run has ended                                                                                                                                                                                                                                                                                                                                 |
| POST   | `/api/v1/projects/{id}/pipeline-runs/{run}/fork`                   | `{seq?, backend?, compute?}` -> 201 `{run, parent: {run, seq}, job?, state}`: a new run whose log starts with events 1..`seq` of this one and continues from that state under this run's plan. 409 when the run had ended by that event, 400 when it has no such event                                                                                                                                           |
| POST   | `/api/v1/projects/{id}/pipeline-runs/{run}/rerun`                  | `{seq?, pipeline?, document?, backend?, compute?}` -> 201 `{run, job?, state}`: a new run of the board as it is now; steps whose fingerprint had succeeded by event `seq` of this run start `cached` (cause `rerun`)                                                                                                                                                                                             |
| POST   | `/api/v1/projects/{id}/pipeline-runs/{run}/archive`, `…/unarchive` | -> `RunSummary`. Archive hides an ended run from the lists (`?archived=true\|only` on the two list routes shows them); 409 while the run is going; 403 for a run of someone else without `runs.delete`                                                                                                                                                                                                           |
| GET    | `/api/v1/projects/{id}/pipeline-runs/{run}/snapshot`               | `{summary, document, plan: {steps, edges}, range, state, datasets, runs, predictions, compute}`: the board document the run was started from, its plan with every step's parameters, the fold at its head, what its steps bound and where it ran, for comparing runs                                                                                                                                             |
| GET    | `/api/v1/projects/{id}/pipeline-runs/{run}/deletion`               | `{run, archived, removes: {events, work_files}, outputs_kept: [{kind, id, version, referenced_by}], outputs_unused: [{kind, id, name}], continued_by, blocked}`                                                                                                                                                                                                                                                  |
| DELETE | `/api/v1/projects/{id}/pipeline-runs/{run}`                        | `?outputs=unused`. Needs `runs.delete`; 409 unless archived. Removes the log and work files and leaves a tombstone; keeps what the run produced, or with `outputs=unused` also deletes `outputs_unused` of the preview                                                                                                                                                                                           |
| GET    | `/api/v1/projects/{id}/boards/{board}/runs/deleted`                | the board's tombstones: `[{run, state, created_by, created_at, deleted_by, at, parent, removed}]`                                                                                                                                                                                                                                                                                                                |
| POST   | `/api/v1/projects/{id}/pipeline-runs/{run}/restore-board`          | -> `{board, run}`: the board document the run was started from becomes the board's newest version; 404 when it is no longer kept                                                                                                                                                                                                                                                                                 |
| GET    | `/api/v1/projects/{id}/pipeline-runs/{run}/steps/{step}/log`       | `?tail=200` -> `{lines, first_line, running}`: the step's part of the job's log (event lines left out), live while it runs. Needs `jobs.read`                                                                                                                                                                                                                                                                    |

Permissions: reads need `projects.read` in the project, submit, fork and re-run
`jobs.submit`, cancel `jobs.cancel`, restore-board `runs.write` (as restoring a board
version). The submit body may carry `document` (the board as the app has it), which the
run keeps for restore-board. A run of another project is 404 through this project's routes. Events are
not posted to these routes: they arrive with the job's log (`POST /api/v1/jobs/{id}/status`)
as `frontier-event: {…}` lines.

### Inference

Run a model on samples as a background job and keep the predictions. It does not score
and does not draw: an evaluation scores predictions, viewers show them. Every route is
under the project.

| Method | Path                                                      | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| POST   | `/api/v1/projects/{id}/inference`                         | `{model: {source: "run" \| "package", ref, checkpoint?: "best" \| "last"}, samples?: {dataset?, split?, ids?: [..], first?}, compute?, backend?, notify?: {url}, allow_ungated?, board?, node?}` -> 201 `Inference`. `model.ref` is a run id of the project (its `best.pt` or `last.pt` artifact) or `name@version` of the model registry (`version`: a version, a stage or `latest`; a `frontier-model` v1 package). `samples.dataset` is a dataset id or name; left out, the dataset the run was trained on. 400 for an unknown source, a run without a checkpoint or no dataset; 404 for a run of another project; 403 `blocked_by_policy` for `notify` while `policy.inference.webhooks` is off or the host is not in `policy.inference.webhook_hosts` |
| GET    | `/api/v1/projects/{id}/inference`                         | `?board=&node=&limit=50` -> `[Inference]`, newest first                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| GET    | `/api/v1/projects/{id}/inference/{inf}`                   | `Inference` (poll this, or use the stream)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| GET    | `/api/v1/projects/{id}/inference/{inf}/events`            | server-sent events: the `Inference` as `data` now and at each change of state, then `event: end`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| POST   | `/api/v1/projects/{id}/inference/{inf}/cancel`            | cancels the job -> `Inference`; 409 once it ended                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| GET    | `/api/v1/projects/{id}/inference/{inf}/predictions`       | `{inference, run_id, state, manifest, samples: [{sample_id, formats, url, viewer}]}`; `manifest` is the predictions' `manifest.json` (null until written)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| GET    | `/api/v1/projects/{id}/inference/{inf}/predictions/{sid}` | `?format=arrow` (default; the Arrow stream as written), `vtk` (legacy PolyData, ASCII: points, triangles, each `field:<name>` as point or cell scalars; opens in ParaView) or `zarr` (a tar of the sample's store, for transient predictions). 404 when the sample has no file in that format                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

`Inference`: `{id, project_id, run_id, job_id, state: queued | running | succeeded | failed |
cancelled, model, samples, cause: {kind: user | trigger, rule?, by?, board?, node?,
pipeline_run?, detail?}, board_id?, node_id?, pipeline_run?, error?, n_samples?, notify?,
webhook?, created_by, created_at, finished_at}`. `error` is one line.

**The predictions artifact.** An inference makes a run tagged `inference` (`run_id`; its
`config.inference` holds the request) and its job, `python -m frontier_train infer --json
<request>` queued like every job, attaches the predictions to it as `predictions/…`
artifacts: the layout an evaluation writes (`<sample>.arrow`, `manifest.json` with
`"format": "frontier-predictions"`). So one inference is one immutable version, the
existing reads work on it (`/api/v1/runs/{run_id}/predictions…`, `until_time`), and
`manifest.provenance` says what made it: `{model: {kind: checkpoint | package, ref, path,
sha256, name, version}, dataset: {path, version, manifest_hash, tags, gate}, samples:
{split, ids}, config, timing, inference, model_ref, cause}`. The model reference is the
package's own (`name`, `version`, `sha256`), so a serving deployment can name the same
model later. Predictions an automatic rule made are kept to the newest
`server.inference_keep_per_node` (20) per node; older ones are removed with their run and files
when a newer job of the node ends. The same number applies, counted on its own, to the
board runs the `schedule`, `upstream` and `dataset_version` rules start for a node: when
such a run ends, the predictions of the node's older rule-started runs go, except those a
kept run, the board's newest run or a run that has not ended still shows as the step's
output. The event logs of those runs stay; a viewer wired to a removed output says the
predictions are no longer kept. Predictions a person started are never removed by this. Normalisation comes from the checkpoint or package, never from the data; a
versioned dataset that is not train-ready is refused unless `allow_ungated`.

**Webhook.** With `notify.url` (or "Notify URL" on an Inference node) the server POSTs
once the job ended, unless it was cancelled: `{event: "predictions.ready" |
"inference.failed", delivery, at, inference: Inference, links: {self, predictions,
viewer}}`: ids and links, never arrays. Headers: `X-Frontier-Event`, and
`X-Frontier-Signature: sha256=<hex HMAC-SHA256(key, body)>` with the key in `<data
dir>/webhook.key` (made on first use, mode 0600; the administrator gives it to the
receiver). Off unless `policy.inference.webhooks` is on; the policy is checked at the
request and again at delivery; no redirects are followed; three tries, ten seconds each;
the result is `webhook: {ok, status?, attempts?, error?, at}` on the record.

**Automatic rules.** A step node carries `trigger` (`manual`, `checkpoint`, `upstream`,
`dataset_version`, `schedule`), `min_interval_s` and `every_min`. The server evaluates them
from the events it already receives, so they fire with no app open
(`docs/spec/pipeline-run-events.md`, section 7): per node at most one job in flight, a
request that arrives meanwhile replaces the one that waited, and starts are at least
`min_interval_s` apart.

Permissions: reads need `projects.read` in the project, the request `jobs.submit`, cancel
`jobs.cancel`; request and cancel are audited. Told to people as `predictions_ready` and
`inference_failed`.

### Services

A board with a Request node (`service.in`) and a Response node (`service.out`) can be
served: the server keeps a **snapshot** of the board (its Request, Inference, Adapt, Input
and Response nodes, their wiring, and the models the Inference steps name resolved to
files) and runs it as a long-lived job, `python -m frontier_deploy service --json
<snapshot>`, on a worker of the `inference` role through the allocator like any job. The
runtime loads the models once, binds a port and reports where it answers with one
`frontier-event: {"type": "service.ready", "data": {url, name, version, model, task}}` line
in its log. Editing the board changes nothing until "Deploy new version".

| Method | Path                                                         | Notes                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| POST   | `/api/v1/projects/{id}/services`                             | `{board?, name?, document?, config?, start?}` -> 201 `Service`. `board` an id or `default`; `document` the board as the app has it (absent: the stored board); `name` the service name (absent: the Request node's `name`, else the board's name as a slug; `[a-z0-9-]{1,63}`, unique on the server). 400 when the board has no Request or Response node or its Inference step names no model; 409 when the name is taken |
| GET    | `/api/v1/projects/{id}/services`, `/api/v1/services`         | `[Service]`; the server-wide list adds `latency: {p50_ms, p95_ms, in_flight, calls, errors, window}` and leaves the schema out                                                                                                                                                                                                                                                                                            |
| GET    | `/api/v1/projects/{id}/services/{svc}`                       | `Service`; `{svc}` an id or name                                                                                                                                                                                                                                                                                                                                                                                          |
| DELETE | `/api/v1/projects/{id}/services/{svc}`                       | 409 while it is starting, running or stopping                                                                                                                                                                                                                                                                                                                                                                             |
| POST   | `/api/v1/projects/{id}/services/{svc}/start`                 | queues the runtime's job -> `Service` with state `starting`; 409 when it is up                                                                                                                                                                                                                                                                                                                                            |
| POST   | `/api/v1/projects/{id}/services/{svc}/stop`                  | cancels the job and releases the worker -> `Service` (`stopped`)                                                                                                                                                                                                                                                                                                                                                          |
| POST   | `/api/v1/projects/{id}/services/{svc}/deploy`                | `{document?, config?}`: a new snapshot from the board as it is now, `version` + 1; a running service is restarted on it                                                                                                                                                                                                                                                                                                   |
| GET    | `/api/v1/projects/{id}/services/{svc}/status`                | the `Service` with `pending_reason` (why its job waits), `health` (the runtime's own, when it is up) and `latency`                                                                                                                                                                                                                                                                                                        |
| GET    | `/api/v1/projects/{id}/services/{svc}/metrics`               | `{requests, errors, p50_ms, p95_ms, in_flight, calls, window, last_request_at, state, version, started_at}`                                                                                                                                                                                                                                                                                                               |
| POST   | `/api/v1/projects/{id}/services/{svc}/tokens`                | `{name?, expires_in_days?}` -> 201 `{token, info, scope, endpoint}`: a token on the caller's account with the one scope `service:<name>`; 409 in single-user mode                                                                                                                                                                                                                                                         |
| POST   | `/svc/{name}[?format=json\|arrow\|vtk\|npz&async=1&files=1]` | one request -> one answer (below)                                                                                                                                                                                                                                                                                                                                                                                         |
| GET    | `/svc/{name}/schema`                                         | the request and response schema the runtime generated from the model (kept on the record, so it answers while the service sleeps, with `state`)                                                                                                                                                                                                                                                                           |
| POST   | `/svc/{name}/stream`                                         | server-sent events: one `data:` message per frame of a transient prediction (`{step, t, state}`) or per chunk of a surface (`{indices, fields, scalars, done, total}`), then `event: end`; `event: error` with `{status, message}` when it stops                                                                                                                                                                          |
| GET    | `/svc/{name}/requests/{rid}`, `…/result`                     | a request made with `?async=1`: `{request, state: running\|succeeded\|failed, error?, result_url?}`, then the answer itself                                                                                                                                                                                                                                                                                               |
| GET    | `/svc/{name}/health`                                         | the runtime's health, or `{status: <state>}` while it is not up                                                                                                                                                                                                                                                                                                                                                           |

`Service`: `{id, project_id, name, endpoint: "/svc/<name>", board_id, board_version,
node_id, version, state: stopped | starting | running | stopping | idle | failed, job_id?,
model, task, config, limits, error?, requests, errors, created_by, created_at,
updated_at, started_at?, last_request_at?, schema?}`. `config` is what was given
(`{rate_per_min?, max_request_mb?, warm?, idle_release_min?, backend?, compute?}`);
`limits` the effective values, the server's settings `server.service_rate_per_min` (600),
`server.service_max_request_mb` (64), `server.service_warm` (1: requests the runtime
answers at once) and `server.service_idle_release_min` (0: never) filling the gaps. A
running service that took no request for `idle_release_min` minutes is stopped with state
`idle`; the next call starts it again and waits up to 90 s for it.

**A request.** JSON `{parameters: {name: number}, geometry?: {points: [[x,y,z]…],
faces?, normals?, areas?}, options?: {chunk_size?}, window?, n_steps?}`, or multipart with
a file field `surface` (STL, VTP, VTK, OBJ, `.arrow`, `.npy`; `design_stl` as the DoMINO
NIM) next to a `parameters` field. `parameters` names the model's conditions (the Request
node may narrow or rename them); a missing one is a 400 `missing_parameters`, a missing
geometry a 400 `missing_geometry`. The answer: JSON `{request, service, version, model,
scalars, scalar_std, intervals, forces, in_envelope, ood_score, n_points, latency_ms,
fields: {name: [values]}, field_std?}` (the Response node's `fields` and `scalars` choose
what is returned), Arrow (one batch: `points`, `field:<name>`, the summary in the schema
metadata), VTK (legacy PolyData, ASCII) or NPZ; a transient model answers `{…, n_steps,
fields, dt, shape, states}`. Every answer carries `X-Frontier-Service`,
`X-Frontier-Service-Version` and `X-Frontier-Request`. With `?async=1` the answer is 202
`{request, state, status_url, result_url}`; with `?files=1` (or the Response node's
`files`) it is also written to `<data dir>/services/<name>/out/<request>/result.<format>`
with a `meta.json` next to it, for readers that poll; the header `X-Frontier-Files` names
them.

**Limits and records.** A call beyond `limits.rate_per_min` is 429 `{status:
"rate_limited", retry_after_s}` with `Retry-After`; a body beyond `limits.max_request_mb`
is 413 `too_large`; a call to a stopped or failed service is 503 `service_stopped`, to one
that is starting 503 `service_unavailable` after 90 s; a runtime that does not answer is
502 `service_unreachable`. Every call is an audit record with the SHA-256 of its body;
the body is not kept. `requests`, `errors` and `last_request_at` count on the record;
`p50_ms` and `p95_ms` are of the last 2000 calls since the server started (they survive a
restart of the runtime; the rate window starts over with it).

Permissions: reads `projects.read`; create, start, stop, deploy, delete and tokens
`services.write` (member); the `/svc/*` routes `services.call` (member). A token whose
scope is `service:<name>` may call that service's routes and nothing else; the server
checks the name before it looks the service up. The runtime itself answers only requests
that carry the secret the server gave it (`X-Frontier-Service-Secret`).

### Job and step logs as pages

`GET /api/v1/jobs/{id}/log` without parameters is the whole log as text, and `GET
…/pipeline-runs/{run}/steps/{step}/log[?tail=]` answers `{lines, first_line, running}` as
before. With `format=json` both answer a `Page` of records, read from the file in ranges
through an index the server keeps per log and extends as the log grows (a large log is
never loaded whole). With any other parameter and no `format`, the job route answers the
text of the selected records (`text/plain`), which is what "download what I filtered" needs.
No new route: the permissions are those of the two routes.

A record is a line and the lines that continue it (a traceback, indented text); the states
of a progress bar that follow each other are one record, the last state; blank lines are
none. Lines are read by `frontier_canvas::logs`: a JSON line (`{"ts", "level", "logger",
"step", "message", "fields"}`, and the usual shapes of other JSON loggers), a
`frontier-event: {…}` line (source `event`), or text, from which a leading timestamp, a
level word, a logger name and trailing `key=value` pairs are taken when present. Text that
names no level counts as `info`. A record belongs to the step that was running when it was
written (from the event lines), or to the `step` it names. The step route answers for the
step's part of the job's log without the event lines.

**Who writes records.** Frontier's own tools write their log as JSON lines when they run
as a job (`FRONTIER_JOB_ID` is in the environment, which a worker sets; the setting
`FRONTIER_LOG_FORMAT=json|text` decides either way) and as text at a terminal: the pipeline
runner, the trainer, data preparation, the hub and the evaluator through
`frontier_data.joblog` (their `logging` records, and each status line they print; `step` is
the pipeline step the process runs as), the worker and the server through
`frontier_core::joblog` for the lines they add to a job's log themselves (`started: …`,
`finished: …`, a dataset registered after a pull). Lines other programs read stay as they
are in both modes: `frontier-event:` lines, the `error:` and `hint:` lines a command ends
with, a `frontier-result:` line, and the JSON a command prints as its result. A reader that
looks for that result skips records (an object with `ts`, `level` and `message`).

| Parameter                                     | Meaning                                                                                                                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `offset`, `limit`                             | position of the first record among those that pass the filters (0 is the first), and how many (200; at most 2000)                                                                                                   |
| `tail`                                        | the last N records instead of `offset`                                                                                                                                                                              |
| `q`                                           | words a record must contain, all of them, any case; the folded lines are searched too                                                                                                                               |
| `level`                                       | `trace,debug,info,warn,error` (any of)                                                                                                                                                                              |
| `source`                                      | logger names (any of); `output` is text that names no logger, `event` the pipeline-run events                                                                                                                       |
| `step`                                        | pipeline step ids (any of)                                                                                                                                                                                          |
| `since`, `until`                              | RFC 3339 or milliseconds since the epoch, inclusive; a record without a time has the time of the last record before it that had one                                                                                 |
| `raw=1`                                       | every state of a progress bar                                                                                                                                                                                       |
| `histogram=<buckets>`                         | adds `histogram` (at most 400 buckets)                                                                                                                                                                              |
| `find`, `find_level`, `find_from`, `find_dir` | adds `found`: the next (`next`, default) or previous (`prev`) record from position `find_from`, or the `first` or `last`, that contains `find` and has one of `find_level`, among the records that pass the filters |
| `line`, `at`                                  | adds `located`: the position of the record that holds line `line` of the log, or of the first record at or after time `at`                                                                                          |

Without `offset` and `tail`, a page asked for with `find`, `line` or `at` starts a third of
a page above that place.

`Page` = `{lines: [Row], offset, total, total_lines, first_line, next_offset?, running,
matches?, levels, sources, steps, histogram?, found?, located?}`: `total` is how many records
pass the filters, `matches` the same when a filter was given, `total_lines` the lines of the
log (of the step's part on the step route), `first_line` the line of the first row,
`next_offset` the `offset` of the next page (absent at the end), `running` whether the log
may still grow (the job, or the step, has not ended). `levels` counts the records per level
(trace, debug, info, warn, error) without the `level` filter; `sources` and `steps` are the
names seen so far.

`Row` = `{n, text, more?, more_lines?, step?, level?, ts?}`: `n` is the line of the job's
log the record starts at (1 is the first line; also on the step route), `text` its first
line as written, `more` the lines that continue it (at most 400; `more_lines` says how many
there are), `level` its level when a continuation line raised it, `ts` its time in
milliseconds since the epoch (its own, or the last one before it).

`histogram` = `{by: "time" | "line", from, to, buckets: [[trace, debug, info, warn,
error]], offsets: [position of each bucket's first record]}`, over the records that pass
the filters other than `since` and `until`; by position when the log has no times.
`found` = `{at?: {offset, n}, index, total}` (`index` counts from 1; `at` is absent when
nothing matches in that direction). `located` = `{offset, n}`.

### Notifications

What happened while a person was looking elsewhere: one row per person and event, kept by
the server (the newest 500 per person) and sent live. The vocabulary is
`frontier_core::notification`, shared by the server and the app.

| Method | Path                           | Notes                                                                                                                                                                                                                                 |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/v1/notifications`        | `?since=<seq>&limit=100&unread=1` -> `{notifications: [Notification], unread, head, user}`, newest first. `head` is the caller's newest sequence number (what to resume the stream from), `user` the caller's login name              |
| GET    | `/api/v1/notifications/stream` | server-sent events, one message per notification (`id: <seq>`, the `Notification` as `data`), live. With `?since=<seq>` or `Last-Event-ID` the ones after that number come first, oldest first, so nothing is lost across a reconnect |
| POST   | `/api/v1/notifications/read`   | `{ids: [<id>]}` or `{all: true}` -> `{unread}`. 400 with neither. Ids of other people change nothing                                                                                                                                  |
| DELETE | `/api/v1/notifications`        | removes the caller's notifications -> `{deleted, unread: 0}`. Sequence numbers are not reused                                                                                                                                         |

`Notification`: `{id, seq, kind, level, title, body, project, subject, actor, created_at, read}`.
`id` is the sequence number as text; `level` is `info`, `success`, `warning` or `error`;
`title` says what happened and `body` is one more line (the reason of a failure, with
known secrets masked); neither carries ids or paths. `project` is a project id or null.
`subject` is the thing to go to, `{type, id, ...}`:

| `type`         | `id`        | other fields                                     |
| -------------- | ----------- | ------------------------------------------------ |
| `pipeline_run` | run id      | `board`, `job`, `name` (the board's)             |
| `step`         | step id     | `run`, `board`, `job`, `name` (the node's title) |
| `run`          | run id      | `name`                                           |
| `job`          | job id      | `name`                                           |
| `approval`     | approval id | `session`, `dataset`, `plan`                     |
| `model`        | model name  | `version`                                        |
| `worker`       | worker id   | `job`                                            |
| `credential`   | provider id |                                                  |

`actor` is the login name of who caused it (who started the run), or null; a client that
shows "only mine" compares it with `user`.

| `kind`                                                                                                                                                                      | Level            | Made when                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipeline_run_finished`, `pipeline_run_failed`                                                                                                                              | success, error   | the event log of a board's pipeline run gets `run.finished` (a cancelled run is not told)                                                                                                               |
| `step_failed`                                                                                                                                                               | error            | a step of a pipeline run goes to `failed`; `body` is its one-line reason                                                                                                                                |
| `run_finished`, `run_failed`                                                                                                                                                | success, error   | a training run changes to `finished` or `failed`                                                                                                                                                        |
| `job_failed`                                                                                                                                                                | error            | a job that is not a pipeline run's fails; `title` says what it was for                                                                                                                                  |
| `dataset_ready`                                                                                                                                                             | success          | a hub pull or a data-preparation job succeeds                                                                                                                                                           |
| `evaluation_ready`                                                                                                                                                          | success          | a benchmark job succeeds                                                                                                                                                                                |
| `predictions_ready`, `inference_failed`                                                                                                                                     | success, error   | the job of an inference ends (not when cancelled); the subject is the run that holds the predictions (`type: run`, with `job` and `inference`); `body` is the number of samples, or the one-line reason |
| `model_promoted`, `model_deployed`                                                                                                                                          | success          | a model version changes stage; a deployment is started                                                                                                                                                  |
| `approval_granted`, `approval_refused`                                                                                                                                      | success, warning | a person decides on an AI agent's plan or signs off a dataset version                                                                                                                                   |
| `worker_offline`                                                                                                                                                            | warning          | a worker stops renewing the lease of work it holds                                                                                                                                                      |
| `credential_test_failed`                                                                                                                                                    | error            | a stored credential is tested and refused                                                                                                                                                               |
| `job_queued_long`, `gate_failed`, `report_ready`, `approval_requested`, `pool_no_capacity`, `pod_idle_release`, `cost_threshold`, `credential_expiring`, `update_available` |                  | defined; no server event makes them yet                                                                                                                                                                 |
| `connection_lost`, `connection_restored`                                                                                                                                    | error, success   | made by the app, never by the server                                                                                                                                                                    |

Who is told: the people who may read the notification's project (everyone of the
organisation when it has none); `worker_offline` and `credential_test_failed` go to
administrators and maintainers. In single-user mode everything is the local
administrator's. A notification of a project the caller can no longer read, or of a
deleted project, is left out of the list and the stream. The four routes need a signed-in
caller (`auth.self`) and write no audit records. Which kinds a person sees, and how, is the
app's `notifications.*` settings.

## JobSpec

```json
{
  "id": "...",
  "name": "train-ahmed",
  "image": "ghcr.io/dimensionlab/frontier-trainer:dev",
  "command": [
    "python",
    "-m",
    "frontier_train",
    "--config",
    "configs/ahmed_transolver.yaml"
  ],
  "env": { "FRONTIER_SERVER": "...", "FRONTIER_RUN_ID": "..." },
  "resources": { "gpus": 1, "gpu_type": "any", "cpus": 8, "memory_gb": 32 },
  "mounts": [{ "uri": "s3://.../dataset", "path": "/data" }],
  "backend": "local|runpod|anyscale|slurm|k8s",
  "compute": {
    "pool": "training",
    "role": "training",
    "gpus": 2,
    "nodes": 2,
    "gpu_memory_gb": 40,
    "exclusive": false
  },
  "allocation": null
}
```

`compute` is optional (`{}` = automatic; `gpus` is a count per node or a list of device uuids);
`allocation` is filled in by the server when a worker claims the job:
`{id, holder, job_id, pool, role, nodes:[{node_id,hostname,node_rank,gpu_uuids,gpu_indices,claimed,cpus}],
rendezvous:{master_addr,master_port,nnodes,nproc_per_node,world_size,id}, exclusive, created_at, lease_until,
released_at, released_reason}`.

## Model registry (contract; server routes not implemented yet)

Versioned models, their variants and serving deployments. The documents below are what the
local file registry (`python -m frontier_deploy registry ... --root <dir>`) already writes to
its `registry.json`; the server stores the same documents and exposes them under
`/api/v1/models`. A version is created from a `frontier-model` v1 package
([model-package.md](model-package.md)); everything else is derived from its `model.json`.

```
GET    /api/v1/models                                   -> [ModelSummary]
POST   /api/v1/models/{name}/versions   (multipart: file=<package.tar>, stage=draft|candidate)
                                                        -> ModelVersion        (201; 409 if it exists)
GET    /api/v1/models/{name}                            -> Model (with versions[])
GET    /api/v1/models/{name}/versions                   -> [ModelVersion]
GET    /api/v1/models/{name}/versions/{version}         -> ModelVersion         ({version} may be a stage: "approved")
GET    /api/v1/models/{name}/versions/{version}/package -> the package tar (application/x-tar)
GET    /api/v1/models/{name}/versions/{version}/manifest -> model.json
POST   /api/v1/models/{name}/versions/{version}/promote {to, comment, override?}
                                                        -> ModelVersion         (409 on a forbidden transition)
GET    /api/v1/models/{name}/versions/{version}/variants -> [Variant]           (parity, accuracy, benchmarks)
POST   /api/v1/models/{name}/versions/{version}/deployments {variant?, device, target, port?}
                                                        -> Deployment           (runs as a Job, see JobSpec)
GET    /api/v1/deployments                              -> [Deployment]
GET    /api/v1/deployments/{id}                         -> Deployment
POST   /api/v1/deployments/{id}/stop                    -> Deployment
GET    /api/v1/models/events?since=<ts>                 -> [Event]               (audit log)
```

```jsonc
// ModelSummary
{"id": "ahmed-transolver-lite", "name": "...", "task": "surface", "description": "...",
 "created_at": "...", "n_versions": 2, "latest_version": "1.1.0", "latest_stage": "candidate",
 "approved_version": "1.0.0"}

// ModelVersion
{
  "model": "ahmed-transolver-lite", "version": "1.0.0",
  "created_at": "2026-10-11T01:00:00Z", "created_by": "alice",
  "stage": "candidate",                                   // draft | candidate | approved | archived
  "stage_history": [{"stage": "draft", "by": "alice", "at": "...", "comment": "registered"},
                    {"stage": "candidate", "by": "bob", "at": "...", "comment": "..."}],
  "task": "surface", "sample_kind": "surface", "architecture": "transolver_lite", "n_params": 1610054,
  "lineage": {"run_id": "...", "run_name": "...", "member_run_ids": ["..."],
              "dataset": {"name": "ahmedml-40-ph", "manifest_sha256": "..."},
              "eval_report": "<path or artifact of the credibility report>",
              "base_model": null | {"source", "kind", "model", "dataset", "run_id"},   // fine-tunes
              "git_sha": "..."},
  "evaluation": {"status": "evaluated", "verdict": {"credible": false, "tier": "below-acceptable"},
                 "benchmarks": [{"benchmark", "split", "primary_metric", "value", "tier", "passed"}],
                 "edge_recommendation": {"variant": "onnx-int8-dynamic", "bytes": 2102602, ...}},
  "license": {"model": "...", "dataset": "CC-BY-SA-4.0", ...},
  "envelope": true, "signed": false,
  "package": {"path": "models/<name>/<version>/package.tar", "artifact": "model/<name>-<version>.tar",
              "sha256": "...", "bytes": 47123456, "n_files": 21},
  "variants": [Variant],
  "deployments": [Deployment]
}

// Variant (summary of model.json variants[])
{"id": "onnx-fp16", "format": "onnx", "precision": "fp16", "default": false, "bytes": 3773126,
 "derived_from": "onnx-fp32",
 "parity": {"max_abs", "max_rel", "l2_rel", "passed"},
 "accuracy": {"metrics": {...}, "delta_vs_fp32": {...}, "within_tolerance": true},
 "benchmarks": [{"host": {"label", "cpu", "threads", ...}, "device", "providers", "latency": [...], "peak_rss_mb"}]}

// Deployment
{"id": "3f2a...", "model": "...", "version": "1.0.0", "variant": "onnx-fp32" | null,
 "target": "serve" | "triton", "device": "cpu" | "cuda" | "tensorrt",
 "endpoint": "http://host:8900", "state": "requested" | "starting" | "running" | "stopped" | "failed",
 "created_by": "...", "created_at": "...", "started_at": "...", "stopped_at": null,
 "job_id": "<Job id when the server runs it>", "command": ["python", "-m", "frontier_deploy", "serve", ...]}

// Event
{"at": "...", "by": "...", "action": "add" | "promote" | "deploy" | "stop", "model": "...",
 "version": "...", "to": "approved", "comment": "...", "override": false}
```

Rules:

- Stage transitions: draft -> candidate | archived; candidate -> approved | draft | archived;
  approved -> archived | candidate; archived -> draft. Anything else is 409.
- `approved` needs `evaluation.verdict.credible == true`, or `override: true` with a non-empty
  `comment`; the override is recorded in `stage_history`. An approved version is immutable
  (re-uploading the same version is 409; register a new version).
- Every mutation appends an Event with the authenticated user.
- Upload verifies the package (checksums, schema) and rejects it with 422 listing the problems.
- A deployment is a Job whose command is `python -m frontier_deploy serve --model <package>
--port <p> --device <d> [--variant <v>]`; its endpoint speaks the model server's inference
  API (`POST /v1/infer` in the request/response shape of NVIDIA's DoMINO NIM,
  `GET /v1/health/ready`, `GET /v1/models`, `GET /metrics`).
- Until these routes exist, packages are attached to their training run as the artifact
  `model/<name>-<version>.tar` (`POST /api/v1/runs/{id}/artifacts`), which the app already lists.

## Authentication

The administrator guide has the model; this is the wire contract. Every route of this document is
subject to it: one rule per `(method, route)` in `frontier_auth::routes::RULES`
(`GET /api/v1/auth/permissions`).

- Callers present `Authorization: Bearer <API token>` or the session cookie
  `frontier_session`. With a cookie, every non-GET request also carries
  `X-Frontier-Csrf: <csrf_token>`.
- `401` `{error, status: "unauthenticated", auth: {mode, config_url, login_url,
oidc_start_url?, local_session_url?}}` with `WWW-Authenticate: Bearer realm="frontier",
error="invalid_token", error_description="…", auth_mode="none|token|oidc|supabase",
auth_config="<url>/api/v1/auth/config"`.
- `403` `{error, status: "forbidden", permission, required_role}`; `{…, status: "csrf"}`
  for a cookie request without the CSRF header; `{…, status: "forbidden_origin"}` from the
  origin/host guard; `{…, status: "blocked_by_policy"}` stays what it was.
- Every response has `X-Request-Id`; audit records carry the same id.
- `X-Frontier-Project: <id or name>` (or `?project=`) names the project a created object
  belongs to. `X-Frontier-Actor` is ignored whenever the caller is established.
- With authentication on, a 500 answers `{error: "internal error", request_id}` to anyone
  but an administrator.

| Route                                                                                                                                         |                                                                                                                                                                                                              |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GET /api/v1/auth/config`                                                                                                                     | Public. `{mode, single_user, local_token_required, local_session_url, token_login, oidc: {issuer, start_url} \| null, login_url, whoami_url, device_start_url, session_cookie, csrf_header, project_header}` |
| `POST /api/v1/auth/local-session`                                                                                                             | Single-user mode, browser on an allowed origin. `{mode: "none", token, token_type: "Bearer", user, role}` + cookie. 403 without `Origin` / `Sec-Fetch-Site: same-origin`; 409 in other modes.                |
| `POST /api/v1/auth/login {"token"}`                                                                                                           | Opens a session: `Set-Cookie`, body as `whoami` plus `session_expires_at`. 401 for a bad token, 403 for a service account.                                                                                   |
| `POST /api/v1/auth/logout`                                                                                                                    | Ends the session of the cookie.                                                                                                                                                                              |
| `GET /api/v1/auth/whoami`                                                                                                                     | `{mode, user: {id, name, display_name, kind}, org, role, project_roles: {id: role}, scopes, method, token_id, permissions: [...], csrf_token}`                                                               |
| `GET /api/v1/auth/permissions`                                                                                                                | `{roles, permissions: [{id, min, title}], routes: [{method, pattern, permission, scope, audit}]}`                                                                                                            |
| `GET /api/v1/auth/oidc/start[?return_to=]`                                                                                                    | 303 to the provider (code flow, PKCE S256, state, nonce).                                                                                                                                                    |
| `GET /api/v1/auth/oidc/callback?code=&state=`                                                                                                 | Validates, provisions, sets the cookie, 303 to `return_to` or `/`. An HTML page with 403 when refused.                                                                                                       |
| `POST /api/v1/auth/device/start {"client_name"}`                                                                                              | `{device_code, user_code, verification_uri, verification_uri_complete, interval, expires_in}`                                                                                                                |
| `GET /api/v1/auth/device/verify?user_code=`                                                                                                   | HTML: sign in if needed, then Allow / Deny.                                                                                                                                                                  |
| `POST /api/v1/auth/device/approve {"user_code", "approve"}`                                                                                   | A signed-in person decides.                                                                                                                                                                                  |
| `POST /api/v1/auth/device/poll {"device_code"}`                                                                                               | `{status: pending \| denied \| expired}` or, once, `{status: "complete", token, token_id, expires_at, user}`                                                                                                 |
| `GET /api/v1/auth/tokens[?user=\|all=true]`                                                                                                   | `{tokens: [{id, user, name, scopes, created_at, created_by, expires_at, last_used_at, last_used_ip, revoked_at, state}]}`; others' need `tokens.manage`.                                                     |
| `POST /api/v1/auth/tokens {"name", "scopes"?, "expires_in_days"?, "user"?}`                                                                   | 201 `{token, info, note}`: the only time the token is shown.                                                                                                                                                 |
| `DELETE /api/v1/auth/tokens/{id}`                                                                                                             | Revoke.                                                                                                                                                                                                      |
| `GET /api/v1/users`, `POST /api/v1/users {"name", "role"?, "kind"?: "user"\|"service", "display_name"?, "email"?, "issue_token"?, "scopes"?}` | List (`users.read`); create (`users.manage`), 201 `{user, token?, token_info?}`                                                                                                                              |
| `GET\|PATCH\|DELETE /api/v1/users/{id or name}`                                                                                               | `PATCH {"role"?, "disabled"?, "display_name"?, "email"?}`; `DELETE` disables and revokes. 409 for the last administrator.                                                                                    |
| `GET /api/v1/orgs`                                                                                                                            | The caller's organisation.                                                                                                                                                                                   |
| `GET /api/v1/projects/{id}/members`                                                                                                           | `{project_id, project, visibility, members: [{user_id, user, role, source, added_by, added_at}]}`                                                                                                            |
| `PUT\|DELETE /api/v1/projects/{id}/members/{user}`                                                                                            | `PUT {"role"}` (`members.manage`).                                                                                                                                                                           |
| `PUT /api/v1/projects/{id}/visibility {"visibility": "org"\|"private"}`                                                                       |                                                                                                                                                                                                              |

## Start-up settings and restarting the server

Settings of scope `system` (the listen address, sign-in mode, limits, the secret store,
`compute.local_*`, ...) are read by the server when it starts. Since 1.0 they can be changed
through `PATCH /api/v1/settings` like the others, when `policy.settings.edit_startup`
allows it:

| `policy.settings.edit_startup` | Who may change start-up settings and restart the server                           |
| ------------------------------ | --------------------------------------------------------------------------------- |
| `local` (default)              | single-user mode (`auth.mode = "none"`, listening on loopback), from this machine |
| `admin`                        | any administrator (`settings.write`, `server.restart`)                            |
| `off`                          | nobody through the API: the config file only                                      |

- **Stored, not applied.** The value is checked like any other setting and then with the
  server's own start-up checks on the resulting configuration (a network address without
  sign-in, sign-in over plain HTTP, a sign-in mode that does not exist, a listen address
  that is not one, a secret store that cannot be opened): a change the server would not
  start with is refused with `422 {status: "would_not_start"}` and nothing is written. It
  is written to the system file when the policy is `admin` and that file is writable and
  the value is not already set in the user file, else to the user file of the server's
  account (`<FRONTIER_HOME>/config.toml`); the answer says where (`written`). The running
  process keeps the value it started with.
- **Pending.** While the stored value differs from the running one, the key has
  `pending_restart: {value, since, by}` (`since` and `by` are null for a change made in the
  file by hand) and is listed in `startup.pending` and in `GET /api/v1/server`.
- **Read-only, with the reason.** A start-up setting that cannot be changed here has
  `editable: false` and `read_only: {kind, message, variable, file}`: `locked` (a lock in the
  system file), `env` (an environment variable of the server process, named in
  `variable`), `flag` (its command line), `file_only` (`server.data_dir`, `server.db_path`,
  or a config file the server cannot write) or `policy`.
- **Typed confirmation.** `auth.mode`, `auth.allow_insecure`, `server.bind`,
  `server.allowed_hosts` and `secrets.*` (`dangerous: true`) are changed only when the
  request names them in `?confirm=`; otherwise `409 {status: "needs_confirmation", confirm}`.
- **Audit.** Every change is in the settings trail (scope `startup`) and in the audit log as
  `settings.startup` (or `settings.startup.discard`) with the file and the value before and
  after, known secret values redacted.
- **Restart.** `POST /api/v1/server/restart` stops accepting requests, lets requests in
  flight finish (ten seconds at most), stops the server's built-in worker and replaces the
  process with a new start of the same program and arguments (`restart.how = "self"`). The
  process id stays the same, so a systemd unit, a launch script or a terminal keep the
  same process and no supervisor is needed. Jobs on other workers keep running and report
  again once the server answers; jobs of the built-in worker are interrupted, which is why
  `when: "idle"` waits until it has none. Where the process cannot replace itself (Windows,
  `FRONTIER_RESTART=manual`) `restart.how` is `manual` and the answer is `409 {status:
"manual", command}`: `sudo systemctl restart <unit>` when the server runs as a systemd
  unit, else the command line it was started with. A client waits for a new `boot_id` in
  `GET /api/v1/server`.

## Audit log

`audit.read` (administrators, auditors).

- `GET /api/v1/audit[?actor=&action=&outcome=&object_kind=&object_id=&project=&request_id=&from=&to=&before=&limit=]`
  → `{entries: [Entry], next_before}` newest first; `action` takes a trailing `*`.
- `Entry`: `{seq, at, request_id, actor_id, actor_name, actor_kind, role, auth_method,
token_id, ip, http_method, path, action, outcome, status, org_id, project_id,
object_kind, object_id, before_hash, after_hash, request_hash, response_hash, detail,
prev_hash, hash}`; `hash = SHA-256(prev_hash + "\n" + canonical JSON of the entry
without prev_hash and hash)`, keys sorted, compact.
- `GET /api/v1/audit/export[?after=<seq>]` → `application/x-ndjson`, one `Entry` per line
  in chain order; header `X-Frontier-Audit-Anchor` is the hash the first line links to.
- `GET /api/v1/audit/verify` → `{ok, checked, anchor_seq, anchor_hash, head_seq,
head_hash, break: null | {seq, reason}}`.
- `GET /api/v1/settings/audit` (the older settings trail) needs `audit.read` too.

## Secret store

Where the control plane keeps the credentials stored on it. All three
routes need `credentials.write` (organisation administrators) and are audited. None returns
a secret value, a key or a store credential.

The backend is a start-up setting, not something these routes change:
`secrets.server_backend = "db" | "vault" | "aws" | "azure" | "gcp"` and `secrets.server_key = "file" |
"aws-kms"` in the system file, with `secrets.vault.*` / `secrets.aws.*` / `secrets.azure.*` / `secrets.gcp.*` (all in
`GET /api/v1/settings/schema`, section "Secrets"). A settings page shows them read-only with
their source and lock state, offers **Test** and, when `needs_migration` is true, **Move
stored credentials**.

- `GET /api/v1/secrets/backend` →

```json
{"backend": "db" | "vault" | "aws" | "azure" | "gcp",
 "location": "HashiCorp Vault https://vault.example.com:8200 (KV v2 mount `secret`, path `frontier/<provider>`, auth approle)",
 "key": {"protection": "file" | "aws-kms", "source": "key file /var/lib/frontier/secret.key"},
 "stored": {"total": 7, "by_backend": {"db": 5, "vault": 2}},
 "needs_migration": true,
 "available": [{"id": "db", "title": "...", "configured": true},
               {"id": "vault", "title": "...", "configured": false, "settings": ["secrets.vault.addr", "..."]},
               {"id": "aws", "title": "...", "configured": false, "settings": ["secrets.aws.region", "..."]}],
 "settings": {"secrets.server_backend": {"value": "vault", "source": "system", "locked": false},
              "secrets.server_key": {"value": "file", "source": "default", "locked": false}},
 "how_to_change": "..."}
```

`needs_migration`: some stored credential is not in the backend new ones go to.
`available[].configured`: the backend's address or region is set (not that it works: use
the test). `available[].verified`, when present, says how far the backend has been
verified (`"contract tests only"` for `azure` and `gcp`): show it.

- `POST /api/v1/secrets/backend/test [{"backend": "db" | "vault" | "aws"}]` → always 200:
  `{"ok": true, "backend", "detail"}` or `{"ok": false, "backend", "error"}`. Writes a
  marker, reads it back and removes it. Without `backend` the one in use is tested; naming
  another one tries its settings before switching to it.

- `POST /api/v1/secrets/backend/migrate [{"to": "db" | "<backend in use>", "dry_run": false}]`
  → `{"to", "dry_run", "moved": [{"credential": "huggingface/token (org)", "from": "db",
"to": "vault"}], "already_there": 3}`. Each value is written to the destination and read
  back before its row is switched and the old copy removed. 409 when `to` is an external
  backend that is not the configured one; 502 `{"error", "moved_before_the_failure"}` when
  a store fails part-way (what was moved stays moved; run it again).

Rows of the credentials API are unchanged: `GET /api/v1/credentials` still reports
`configured`, `source: {"store": "server"}` whichever backend holds the value. In an
external store one secret per provider is kept at `<prefix>/<provider>` (project and user
scopes: `<provider>+project.<id>`, `<provider>+user.<name>`), a map of field to value, so
it can be rotated in the secret manager itself.

## Lineage

The lineage graph is not stored. Each request joins what the server already has: the tables
`runs`, `artifacts`, `datasets`, `jobs`, `compute_allocations` and `approvals`, every dataset's
`.frontier/` version store, the `model.json` card inside each `model/<name>-<version>.tar` run
artifact, `<data_dir>/models/registry.json` and the hub catalogue. Runs and packages recorded
before this endpoint existed are therefore covered.

```
GET /api/v1/lineage?node=<kind>:<id>[&depth=3][&direction=both][&format=json][&limit=2000]
GET /api/v1/lineage/kinds      -> node kinds and relations with descriptions (the legend)
```

| Parameter   | Values                                     | Meaning                                             |
| ----------- | ------------------------------------------ | --------------------------------------------------- |
| `node`      | `<kind>:<id>`                              | Root of the walk. Required.                         |
| `depth`     | 0 to 8, default 3                          | Hops from the root; larger values are clamped to 8. |
| `direction` | `both` (default), `upstream`, `downstream` | Which edges the walk follows.                       |
| `format`    | `json` (default), `prov`                   | `prov` returns W3C PROV-JSON.                       |
| `limit`     | 1 to 2000, default 2000                    | Node cap; reaching it sets `truncated`.             |

```jsonc
{"root": "dataset:0199...", "depth": 3,
 "nodes": [{"id": "run:0199...", "kind": "run", "label": "base-run",
            "project": "<project id>" | null, "attrs": {...}, "at": "<RFC 3339>" | null}],
 "edges": [{"id": "<from>|<rel>|<to>", "from": "<upstream id>", "to": "<downstream id>",
            "rel": "used"}],
 "truncated": false, "warnings": ["..."]}
```

Nodes and edges are sorted by id, and ids are stable across calls. Status codes: 400 for a
missing or malformed `node`, an unknown `direction` or `format`; 404 when the root does not
exist. Anything else that is missing or unreadable (a card that does not parse, a dataset or
version a run names that is not registered) becomes an entry in `warnings` and, where the
reference is known, a node with `attrs.missing = true`.

### Node kinds

| Kind                                       | Id after `<kind>:`                                    | `project`                                                            | PROV     |
| ------------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------- | -------- |
| `hub_entry`                                | hub catalogue id                                      | null                                                                 | entity   |
| `connector`                                | connector id                                          | null                                                                 | entity   |
| `dataset`                                  | dataset id (`unregistered/<label>` for a placeholder) | `datasets.project_id` when the column exists                         | entity   |
| `dataset_version`                          | `<dataset id>@<version id>`                           | the dataset's                                                        | entity   |
| `prep_op`                                  | `<dataset id>@<provenance record id>`                 | the dataset's                                                        | activity |
| `run`                                      | run id                                                | `runs.project`                                                       | activity |
| `job`                                      | job id                                                | `jobs.project_id` when stamped, else the project of the run it names | activity |
| `allocation`                               | allocation id                                         | its job's                                                            | entity   |
| `code`                                     | git sha                                               | null                                                                 | entity   |
| `checkpoint`, `evaluation`, `uq`, `report` | `<run id>/<artifact name>`                            | the run's                                                            | entity   |
| `model`                                    | model name                                            | the one project all its versions share, else null                    | entity   |
| `model_version`                            | `<name>@<version>`                                    | the run the package is attached to                                   | entity   |
| `variant`                                  | `<name>@<version>/<variant id>`                       | the model version's                                                  | entity   |
| `deployment`                               | deployment id                                         | the model version's                                                  | entity   |

Artifacts are classified by name: `.pt`, `.pth`, `.ckpt`, `.mdlus` and `.safetensors` are
checkpoints; `eval-*.json`, `uq*.json` and `report.json` (by basename) are the evaluation kinds;
`model/<name>-<version>.tar` is a model version. People named on a node are listed in
`attrs.workers` as `[{"id", "role"}]` (roles: `submitter`, `reviewer`, `creator`, `promoter`,
`deployer`, or the provenance actor's kind). A run's submitter is `config.user` (also
`submitted_by`, `created_by`, `author`), else `runs.created_by`; a job's is `jobs.submitted_by`.

### Relations

Every edge points from the upstream cause (`from`) to the downstream effect (`to`). `reads`
says which end is the subject of the relation's name.

| `rel`             | Reads       | Edges (`from` -> `to`)                                                                                                                                 |
| ----------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pulled_from`     | to rel from | hub_entry -> dataset; hub_entry -> model_version                                                                                                       |
| `converted_by`    | to rel from | connector -> dataset                                                                                                                                   |
| `version_of`      | to rel from | dataset -> dataset_version; model -> model_version                                                                                                     |
| `derived_from`    | to rel from | parent dataset_version -> child; evaluation, uq -> report; variant -> variant; member run -> ensemble run; dataset or dataset_version -> model_version |
| `used`            | to rel from | dataset_version -> prep_op; dataset or dataset_version -> run; code -> run; evaluation, uq, report -> model_version                                    |
| `generated_by`    | to rel from | prep_op -> dataset_version; hub pull job -> dataset; serve job -> deployment                                                                           |
| `ran_on`          | to rel from | allocation -> job                                                                                                                                      |
| `fine_tuned_from` | to rel from | model_version, model, checkpoint, run or hub_entry -> run; the same -> model_version                                                                   |
| `produced`        | from rel to | job -> run; run -> checkpoint                                                                                                                          |
| `evaluated_by`    | from rel to | run -> evaluation, uq, report                                                                                                                          |
| `packaged_as`     | from rel to | run -> model_version; checkpoint -> model_version; model_version -> variant                                                                            |
| `deployed_as`     | from rel to | model_version -> deployment; variant -> deployment                                                                                                     |

### Where each link comes from

- dataset to hub entry and connector: `manifest.hub_id`, `manifest.source.connector`. A hub
  pull job is linked when its `FRONTIER_HUB_ID` matches and its `FRONTIER_HUB_TARGET` contains
  the dataset directory.
- dataset versions and prep ops: `.frontier/versions/<id>.json` (`parent`, `op_record`) and
  the `op` records of `.frontier/provenance.jsonl` (`input_version`). Reviewers come from
  `approvals` rows with `dataset` and `plan_id = <version id>`.
- run to dataset: `config.data.path`, `config.dataset` or `config.data.dataset` (an id, name
  or path, or an object with `id`, `name`, `path`, `version`, `manifest_sha256`),
  `config.dataset_id`, `summary.dataset_manifest_hash`, `summary.dataset_version`,
  `summary.dataset`. A registered dataset is matched by id, then by the SHA-256 of its
  `manifest.json`, then by path, then by a unique name, then by the one store holding the
  version id. With a version the edge starts at the dataset version, otherwise at the dataset.
- run to code: `git_sha` (also `git.sha`, `git.commit`, `code.git_sha`, `code.sha`,
  `code.commit`, `provenance.git_sha`) in the run config or summary, then the `git_sha` of the
  run's `model_card.json` artifact, then `provenance.git_sha` of a package attached to the run.
- job to run: `spec.run_id` or `spec.env.FRONTIER_RUN_ID`; allocation to job:
  `compute_allocations.job_id`.
- fine-tune base: `config.base_model`, `config.finetune.base_model`, `summary.base_model`,
  `summary.finetune.base` (the trainer's `{source, kind, run_id}`), `config.finetune.base`,
  `config.base_checkpoint`, and `provenance.base_model` of a package card. A base may be
  `<model>@<version>`, a model name, a hub id or `hub:<id>`, `run:<id or name>`, a run id or
  name, or a checkpoint path. Ensemble members come from `summary.ensemble_of`.
- model version: the run the package artifact is attached to; `provenance.run_id`,
  `member_run_ids` and `members[].run_id` when they are run ids on this server;
  `members[].checkpoint` matched to that run's checkpoint artifacts by file name (then by
  stem, so `best.mdlus` stands for `best.pt`); `evaluation.benchmarks[].benchmark` (and
  `split`), `evaluation.uq` and `evaluation.report` matched to those runs' evaluation
  artifacts; `provenance.dataset` (`manifest_sha256`, `name`); `variants[]` (`id`,
  `derived_from`). Stage, promotion actors, deployments and hub models come from
  `models/registry.json`.

### What is left out

No secret is copied into a node. A run contributes a fixed subset of its config (model and
architecture, the dataset reference, optimiser name, learning rate, weight decay and
schedule, epochs, steps, seed, batch size, git sha) and never the config as a whole. A job
contributes its name, backend, state, labels, image, times and exit code, never its command
or environment. A deployment omits its serve command. Prep op parameters are copied with
secret-looking keys removed.

### PROV-JSON

`format=prov` returns `{"prefix", "entity", "activity", "worker", "wasGeneratedBy", "used",
"wasDerivedFrom", "wasInformedBy", "wasAssociatedWith", "wasAttributedTo"}` with ids
`frontier:<node id>` and workers `frontier:worker/<name>`. An edge maps by the class of its ends:
entity -> activity is `used`, activity -> entity is `wasGeneratedBy`, entity -> entity is
`wasDerivedFrom`, activity -> activity is `wasInformedBy`; each keeps the Frontier relation as
`frontier:rel`. Workers of an activity are `wasAssociatedWith`, workers of an entity
`wasAttributedTo`. The response header `x-frontier-lineage-truncated` carries the node-cap flag.
