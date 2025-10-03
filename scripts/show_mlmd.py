"""Show MLMD (SQLite) metadata contents for recent pipeline runs.

### EXPLAINER
Connects to `outputs/mlmd/metadata.db` and prints contexts (pipelines/runs),
executions (component runs), artifacts (with URIs), and events linking them.
Use this for assignment parts requesting metadata store evidence.
"""

from __future__ import annotations

import os
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


def main() -> None:
    db_path = os.path.abspath(os.path.join("outputs", "mlmd", "metadata.db"))
    if not os.path.exists(db_path):
        raise SystemExit(f"MLMD not found at {db_path}. Run the pipeline first.")

    config = metadata_store_pb2.ConnectionConfig()
    config.sqlite.filename_uri = db_path
    config.sqlite.connection_mode = metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE

    store = metadata_store.MetadataStore(config)

    print("— Contexts (pipelines, runs) —")
    for c in store.get_contexts():
        print(dict(id=c.id, type_id=c.type_id, name=c.name))

    print("\n— Executions (component runs) —")
    for e in store.get_executions():
        print(dict(id=e.id, type_id=e.type_id, state=e.last_known_state, name=e.name, created=e.create_time_since_epoch))

    print("\n— Artifacts (by type) —")
    types = {t.id: t.name for t in store.get_artifact_types()}
    for a in store.get_artifacts():
        print(dict(id=a.id, type=types.get(a.type_id), uri=a.uri, state=a.state, name=a.name))

    print("\n— Events (execution↔artifact) —")
    for ev in store.get_events():
        print(dict(artifact_id=ev.artifact_id, execution_id=ev.execution_id, type=ev.type))


if __name__ == "__main__":
    main()
