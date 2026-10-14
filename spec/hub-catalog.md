<!-- SPDX-License-Identifier: Apache-2.0 -->

# Datasets hub catalog schema

## Catalog entries

Source of truth: `packages/frontier-hub/src/frontier_hub/catalog/<domain>/<id>.yaml`, compiled
to `packages/frontier-hub/src/frontier_hub/catalog.json` (committed; the server embeds it).

```yaml
id: drivaernet-plus-plus # stable kebab-case
name: DrivAerNet++
kind: dataset | benchmark | collection
domain: cfd # same vocabulary as manifest.domain
tags: [automotive, external-aero, surface, volume, steady, rans]
physics:
  { equations: [incompressible-ns], fidelity: rans, temporal: steady, dims: 3 }
geometry: { sample_kind: surface, typical_cells: 8.8e6 }
size:
  {
    samples: 8121,
    bytes: 3.9e13,
    subsets: [{ id: surface-pressure, bytes: ... }],
  }
license: { spdx: CC-BY-NC-4.0, commercial_use: false, url: ... }
citation: { bibtex: '...', doi: ... }
homepage: ...
sources: # how to fetch; first usable wins
  - { type: huggingface, repo: ..., repo_type: dataset, patterns: [...] }
  - { type: http | s3 | zenodo | dataverse | globus | git, ... }
access: open | gated | request | manual | paid | generate | none # section "Access"
obtain: { url, credential, steps, files, command, connector, alternatives } # optional
formats: [vtk, hdf5, ...]
converter: frontier_hub.converters.drivaernet:convert # -> frontier-dataset v1.1, or null
benchmarks: [drivaernet-pp-cd] # ids of benchmark entries scored on it
reference_results: [{ model: ..., metric: ..., value: ..., source: ... }]
status: verified | listed # verified = we downloaded a subset and converted it
verified: { date: 2026-10-10, subset: '...', notes: '...' }
enterprise_notes: 'why a Fortune 500 team would use it (pretraining, validation, V&V baseline)'
```

CLI: `python -m frontier_hub list|search|info|pull|obtain|attach|mirror|convert|prep|verify|export`
(`frontier hub <command>` runs the same). `pull` supports
`--subset`, `--max-bytes`, resumable downloads, checksum + provenance, and `--to <dir>`.
Server: `GET /api/v1/hub/datasets[?domain=&tag=&q=]`, `GET /api/v1/hub/datasets/{id}`,
`POST /api/v1/hub/datasets/{id}/pull {subset, target}` (enqueues a JobSpec).

## Access

`access` says how the data of an entry is actually obtained. Every entry has exactly one mode.

| `access`   | Meaning                                                                                             | What Frontier offers                                |
| ---------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `open`     | A public URL or repository that Frontier downloads unattended.                                      | `pull`                                              |
| `gated`    | Click-through terms accepted with an account, then a credential.                                    | provider page, credential, `obtain --check`, `pull` |
| `request`  | The publisher approves each applicant (a form, an e-mail, a gated repository with manual approval). | where to ask, then `pull` or `attach`               |
| `manual`   | A portal, a file share, a vendor CLI or an API that Frontier has no fetcher for.                    | the page, the files, `attach`                       |
| `paid`     | A commercial licence.                                                                               | the vendor page, `attach`                           |
| `generate` | The data is produced by running a script or a solver.                                               | the command or the connector, `attach`              |
| `none`     | Nothing is published (announced, or code only).                                                     | comparable entries                                  |

Older catalogs may write `registration` (read as `gated`) and `direct` (read as `open`).

Rules used for the built-in catalog: an entry is `open` only if at least one of its sources
is fetched unattended (`huggingface`, `http` with file URLs, `zenodo`, `dataverse`, `s3`,
`gcs`, `git`, `github-release`). A source of type `api`, `kaggle` or `globus`, or an `http`
source that names only a landing page, makes the entry `manual`. A benchmark or collection
without files of its own takes the mode of the datasets it refers to (a benchmark the most
restrictive one, a collection the most common one).

### `obtain`

Optional block with what cannot be derived from `access`, `sources` and `homepage`:

```yaml
obtain:
  url: https://portal.example/dataset # where access is requested or files are downloaded
  credential: huggingface # provider id for `frontier auth login <id>`
  steps: ['Open ...', 'Download ...', 'Run `frontier hub attach ...`.'] # replaces the derived steps
  files: # files downloaded by hand; `attach` checks them
    - { name: fields_0001.h5, bytes: 1048576, sha256: '...' }
    - { name: 'run_*/boundary.vtu' } # a glob when the publisher numbers files
  command: python make_data.py --out data/ # generate: the command that produces the data
  connector: openfoam # generate: the connector that reads the results
  alternatives: [airfrans, ahmedml] # none: comparable entries (default: derived)
```

The compiled catalog carries the result for clients as `obtain_guide` on every entry:

```json
{
  "entry": "era5",
  "access": "manual",
  "steps": ["...", "..."],
  "actions": ["attach"],
  "url": "https://...",
  "credential": null,
  "files": [{ "name": "...", "bytes": 1 }],
  "command": null,
  "connector": null,
  "alternatives": [],
  "filled": true
}
```

`steps` are sentences with one action each, in order. `actions` is the subset of
`pull`, `check_access`, `attach`, `generate` a client offers as buttons. `filled` is false
when a user would have to search for what to do; the release gate fails on it.

`python -m frontier_hub obtain <id> [--check] [--subset s] [--json]` prints the guide.
`--check` lists the files without downloading and reports `{"ok": true, files, bytes}` or
`{"ok": false, "status": needs_auth | needs_user_action | blocked_by_policy | manual | failed,
"message", "next_steps"}` (exit 0, 5, 6, 3, 1).

## Attach: files the user already has

```
frontier hub attach <id> <source> [--subset s] [--mode copy|hardlink|reference] [--to raw-dir]
                                  [--out dataset-dir] [--no-convert] [--replace] [--json]
frontier hub attach --name <name> [--connector <id>] <source> [--out dataset-dir]
```

`<source>` is a directory or mounted share, an archive (zip, tar, tar.gz ...), an object-store
prefix (`s3://`, `r2://`, `gs://`, `az://`; credentials of the providers `s3`, `gcs`, `azure`
from the secret store, `r2://` uses the `endpoint_url` of the `s3` credential) or one
`https://` file. The files become the raw directory of the entry, where a pull would have put
them, and the entry's converter and the preparation tail run exactly as after a pull.

1. **Stage.** Files are brought into a staging directory next to the raw directory. Archives
   are unpacked with the same checks as downloaded ones (no path outside the target, no
   links, bounded size and ratio). Hidden directories and `provenance.json` of the source
   are not taken.
2. **Check.** Each file is hashed (SHA-256). It is compared with `obtain.files` of the entry
   (exact name first, then glob), with a manifest when the caller supplies one (a mirror),
   and with the file already recorded under that name in the raw directory. A size or
   checksum that differs refuses the whole attach: nothing is stored (exit 8). A stored file
   is replaced by a different one only with `--replace`. Where nothing is published the
   checksum seen is recorded.
3. **Store.** `copy` (default) copies; `hardlink` links on the same file system; `reference`
   stores one symbolic link per file, so a set too large to duplicate stays where it is.
   Object stores and URLs are always copied.
4. **Record.** `provenance.json` of the raw directory gets `source: {type: "user-supplied",
location, mode}`, `supplied: {by, at, from, mode, origin, kind, archive_sha256?}`, a
   `supplied_history` list, and per file `{bytes, sha256, origin, expected_checksum,
checksum_verified}`. `by` is the signed-in user (`FRONTIER_USER`, set by the server for a
   job) or the operating-system account. Locations are recorded without user info or query.
5. **Convert.** With a converter, the dataset is written to `--out` and the preparation tail
   runs; the dataset's provenance log gets a note `raw files user-supplied` with the
   `supplied` record, and every sample's `meta.provenance` names the source type.

The licence policy (`policy.allowed_dataset_licences`) applies as for a pull; the public-host
policy does not.

**Without a hub entry** (`--name`): the same staging, checks and provenance. A directory that
is a Frontier dataset is taken as it is; anything else is converted with
`frontier connectors convert` (`--connector`, default: detection). The preparation tail runs
on the result. This is the path for an organisation's own simulation data.

## Organisation mirrors

Setting `hub.mirrors` (list, editable by an administrator, lockable, environment
`FRONTIER_HUB__MIRRORS`): items `<entry id or glob>=<location>`, first match wins, for
example `["shift-*=s3://company-datasets/restricted", "*=/mnt/datasets/hub"]`. A location is
a directory (NFS, a mounted share) or an object-store prefix. `--mirror <location>` on
`pull` and `--to <location>` on `mirror push|status` name one for a single command.

Layout under a location:

```
<location>/<entry id>/<subset or _all>/mirror.json    {schema: "frontier-hub-mirror/1", entry, subset,
                                                       licence, access, files: {path: {bytes, sha256}},
                                                       bytes, obtained, pushed: {by, at}, acknowledgement}
<location>/<entry id>/<subset or _all>/LICENCE.txt    licence id, URL, terms, citation
<location>/<entry id>/<subset or _all>/files/...      the raw files
```

- `frontier hub pull <id>` asks the mirror first (unless `--no-mirror` or `--pattern`). Files
  are checked against `mirror.json`; a file that differs refuses the pull. Provenance records
  `source.type: mirror`. When the mirror does not hold the entry, or cannot be read, the
  public sources are used (a mirror that cannot be read is reported). In offline mode a
  mirror on a directory still works.
- `frontier hub mirror push <id> [--subset s] [--raw-dir d] [--to location]` prints the licence,
  verifies the raw directory against its provenance, uploads the files, the manifest and
  `LICENCE.txt`.
- `frontier hub mirror status <id>` and `frontier hub mirror list`.

**Licence gate.** A push needs `--i-have-the-right` when the entry's terms do not plainly
allow an organisation to keep a copy for its members: `license.redistribution: false`,
access `gated`, `request` or `paid` (terms accepted per person), or a licence id containing
`Unknown` or `Proprietary`. Without the flag the push is refused before anything is uploaded
(status `needs_acknowledgement`, exit 7). With it, the manifest records
`acknowledgement: {by, at, reason, statement}` and the audit log records the event.

**Audit.** `hub.mirror.push` and `hub.mirror.pull` are appended to
`<FRONTIER_HOME>/audit/hub.jsonl` with actor, entry, subset, location, files, bytes, licence
and the acknowledgement. The server writes the same events to its audit log for jobs it runs.

## Server routes

| Route                                                                                                                             | Role       | Permission        | Audit event                                             |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | ------------------------------------------------------- |
| `GET /api/v1/hub/datasets/{id}`                                                                                                   | viewer     | `hub.read`        | -                                                       |
| `POST /api/v1/hub/datasets/{id}/access-check` `{subset?}`                                                                         | member     | `hub.pull`        | -                                                       |
| `POST /api/v1/hub/datasets/{id}/attach` `{source, subset?, mode?, replace?, target?, connector?, cases?, case_params?, backend?}` | member     | `hub.attach`      | `hub.attach`                                            |
| `POST /api/v1/hub/attach` `{name, source, connector?, cases?, case_params?, mode?, target?, backend?}` (no hub entry)             | member     | `hub.attach`      | `hub.attach`                                            |
| `PUT /api/v1/hub/uploads/{name}` (the file as the request body)                                                                   | member     | `hub.attach`      | `hub.attach`                                            |
| `GET /api/v1/hub/mirror/{id}` `[?subset=]`                                                                                        | member     | `hub.pull`        | -                                                       |
| `POST /api/v1/hub/mirror/{id}/push` `{subset?, acknowledged?, backend?}`                                                          | maintainer | `hub.mirror.push` | `hub.mirror.push`, `hub.mirror.acknowledge`             |
| `POST /api/v1/hub/datasets/{id}/pull` (existing)                                                                                  | member     | `hub.pull`        | `hub.pull`; `hub.mirror.pull` when the mirror served it |
| `GET /api/v1/hub/mirrors`                                                                                                         | member     | `hub.pull`        | -                                                       |
| `PATCH /api/v1/settings` `{"hub.mirrors": [...]}` (the settings route)                                                            | admin      | `settings.write`  | `settings.write`                                        |

- `GET .../{id}` returns the entry with `obtain_guide`.
- `access-check` runs `frontier_hub obtain <id> --check --json` on the control plane and
  returns its `check` object with status 200, also when `ok` is false. The credential used is
  the caller's own stored one where they have one, else the organisation's; the same holds
  for `GET /hub/mirror/{id}`.
- `attach` and `push` enqueue a job on a worker (`frontier_hub attach ... --json`,
  `frontier_hub mirror push ... --to <location> --json`) with `FRONTIER_USER` set to the
  caller. The command's final JSON is the job's `result` (`GET /api/v1/jobs/{id}`), also
  for a pull; clients do not read the log for it. The answer is
  `201 {job, hub_id | name, subset, source, target}` and `201 {job, hub_id, subset, location}`.
- `source` is an absolute path, an `s3://`, `r2://`, `gs://` or `az://` prefix, or one
  `https://` file. A path must lie under the data directory, `storage.datasets_dir` or
  `storage.import_roots`, as for registering a dataset (403 otherwise); an object-store or
  `https` source needs no root. A location that carries a credential (a key and secret in
  front of the bucket or host) is refused: the job gets the stored credential of the provider `s3`, `gcs` or `azure`.
- **Solver cases.** `cases` (a pattern under the source such as `*/` or
  `run_*/case.foam`, or `auto`) makes one sample per case; `case_params` says where the case
  parameters come from (`names`, `none`, or a CSV or JSON table: a path under the source, or
  an absolute path under an import root); `connector` names the connector (`detect` or
  absent: detection). They are passed to the command as `--cases`, `--case-params` and
  `--connector`, which reads them for a dataset without a hub entry; an entry's own
  converter does not use them.
- **Upload.** `PUT /api/v1/hub/uploads/{name}` stores one file (an archive, or a single data
  file) and returns `201 {source, name, bytes, sha256}`; `source` is then given to `attach`.
  The size limit is `limits.max_artifact_bytes`. An uploaded file is always copied by the
  job and is removed when the job ends, or after a day when no job was started.
- The files go to `<target>/raw` and the dataset to `<target>/dataset`; `target` defaults to
  `<datasets root>/hub/<id>[-<subset>]`, where a pull puts them, and to
  `<datasets root>/attached/<name>` without an entry. A finished attach registers the dataset.
- `GET /hub/mirror/{id}` answers `{configured: false, held: false}` when no item of
  `hub.mirrors` matches, else what `frontier_hub mirror status` reports plus
  `configured: true` and the `location` (shown without user info or query). A mirror that
  cannot be read is `{configured: true, held: false, readable: false, message}`.
- `push` reads the raw files from `<datasets root>/hub/<id>[-<subset>]/raw`. Without a mirror
  for the entry it answers `409 {status: "no_mirror"}`. When the licence gate applies and
  `acknowledged` is not true it answers `409 {status: "needs_acknowledgement", message,
reason, license, location}` and no job is queued.
- `acknowledged: true` is accepted from a maintainer or administrator only (the route's
  permission). It is written to the audit log as `hub.mirror.acknowledge` with the caller,
  the entry, the location, the reason and the job. When the job has finished, a second
  record `hub.mirror.push` in the caller's name carries what the command reports: entry,
  subset, location, files, bytes, licence and `acknowledged`. The statement itself
  (`acknowledgement: {by, at, reason, statement}`) is in the mirror's `mirror.json`.
- A pull queued by the server gets the matching item of `hub.mirrors` as the setting
  (`FRONTIER_HUB__MIRRORS` in the job's environment) and that location's credential: the
  mirror is asked first and the public sources are used when it does not hold the entry.
  When the mirror served the pull (`result.origin` is `mirror`) the audit log gets
  `hub.mirror.pull` in the submitter's name with entry, subset, location, files and bytes.
- **Progress.** With `--json`, `pull`, `attach` and `mirror push` write progress to stderr,
  one JSON object per line: `{"ts", "level": "info", "logger": "frontier_hub", "message",
"fields": {"file", "bytes_done", "bytes_total", "files_done", "files_total"}}`. Every
  transfer has an opening record (what is moved, how many files, to or from where) and a
  closing record with the totals reached. In between, a record is written when a file is
  finished (not for each of many small files that finish within a fifth of a second of
  each other) and otherwise at most every two seconds, with the bytes so far, so one large
  file shows that it moves. This covers a download from a public source, an attach from
  an object store or a URL (`fetching ...`, then `checking ...` as the files are hashed),
  an upload to a mirror and a pull served by one, on a directory as on an object store.
  `bytes_total` and `files_total` are `null` when not known (one file from a URL whose
  size the host does not say). The job log holds these records next to the result; stdout
  carries the result alone, on one line inside a job.
- Refusals are the job's `result` and also its `error` as a JSON string: `needs_auth`, `needs_user_action` and
  `blocked_by_policy` as for a pull (exit 5 and 6), `needs_acknowledgement` (7), `mismatch`
  (8, `{status: "mismatch", message}`).
- When an attach or a push ends, its submitter is notified ("Dataset import finished" /
  "failed", "Dataset shared with your organisation" / "Sharing a dataset with your
  organisation failed").

## Contract for the app

Hub entry page:

- Show `obtain_guide.steps` as a numbered list under the title "How to get it". No prose
  around it. Show `obtain_guide.files` as a small table (name, size) when present.
- Buttons from `obtain_guide.actions`: `pull` "Pull", `check_access` "Check access" (calls
  `access-check`; show `message` and `next_steps` when `ok` is false), `attach` "Attach
  files", `generate` "Run" when `obtain_guide.command` is set (a job on a worker), else a
  link to the connector.
- "I already have the files" is offered for every entry of kind `dataset`, also `open` ones.
  It opens the attach form: source (path under an import root, or `s3://`, `r2://`, `gs://`,
  `az://`, `https://`), subset (when the entry has subsets), how to store local files (copy,
  hard link, leave in place). Submit calls `attach`.
- A refused attach (`mismatch`) shows the command's message; nothing was stored.
- When a mirror is configured for the entry (`GET /hub/mirror/{id}`): show "On your
  organisation's mirror" with who pushed it and when; "Pull" then needs no credential. For a
  maintainer, "Share with my organisation" calls `push`; when the answer is
  `needs_acknowledgement`, show the reason and the licence link and ask for the confirmation
  "My organisation may keep a copy of this dataset" before sending `acknowledged: true`.
- `access: none`: show the first step and `alternatives` as links to those entries.

Import node:

- Sources: "A bundled example", "A dataset on this server", "From the hub" (entry picker,
  the entry's subsets, what the pull downloads), "Files I already have for a hub dataset"
  (entry picker, subset, where the files are, how local files are stored), "My own data"
  (where the results are, connector picker with detection as the default, cases, name, how
  local files are stored). The fields of the chosen source are in the node. One line under
  the choice, for "My own data": "Your own simulation results: pick a connector and point at
  the files."
- For an entry that is not `open`: `obtain_guide.steps` under "How to get it", and "Check
  access" when `obtain_guide.actions` has `check_access`.
- **The node does not call the attach routes.** An import on a board is a step of a
  pipeline run (`docs/spec/pipeline-run-events.md`, "Steps"): the run's one job runs the
  same commands (`frontier_hub pull`, `frontier_hub attach`, `frontier_connectors convert`),
  so the import has a state, progress and a log in the run, is reused by its fingerprint
  when the board runs again, is cancelled with the run, and writes into the run's project
  (`<data dir>/projects/<project>/datasets`) instead of the shared `<datasets root>/hub`.
  The routes above start a job of their own outside any run; they serve the hub entry page.
  What the routes check, the submit of a board run checks for its import steps:

  | Rule                                                                                         | Attach and pull routes   | Board run (`POST …/boards/{board}/runs`)                                  |
  | -------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------- |
  | A local path lies under the data directory, `storage.datasets_dir` or `storage.import_roots` | 403                      | 403, naming the step                                                      |
  | A location that carries a credential                                                         | 400                      | 400, naming the step                                                      |
  | Stored credential of the provider of an object-store location (`s3`, `gcs`, `azure`)         | given to the job         | given to the run's job, for the import steps that run                     |
  | Hugging Face token and the organisation's mirror for a hub entry                             | given to a pull          | given to the run's job (`FRONTIER_HUB__MIRRORS`, the mirror's credential) |
  | Who supplied the files (`FRONTIER_USER`, written to the provenance)                          | the caller               | the caller who started the run                                            |
  | Audit                                                                                        | `hub.attach`, `hub.pull` | the pipeline run's own records; no separate `hub.*` record                |
  | Result                                                                                       | the job's `result`       | the step's `dataset` output, registered in the run's project              |

  A hub entry's organisation mirror is not written by a board run: sharing with the
  organisation is `push`, from the hub entry page.
