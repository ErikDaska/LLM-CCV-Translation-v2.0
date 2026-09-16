"""Persist local experiment records without GPU, Transformers or W&B dependencies."""
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunTracking:
    """Manage a unique run directory and JSON records for a single writer."""
    def __init__(self, base_dir, mode):
        """Create a timestamp/UUID directory under base_dir with running status.

        Args:
            base_dir: Parent directory for experiment or trial records.
            mode: Execution label, such as manual, hpo or hpo_trial.
        """
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex
        self.path = Path(base_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.mode = mode
        self.status('running')

    def write(self, name, data):
        """Atomically replace a JSON record using a temporary file in the same folder.

        Args:
            name: Record filename within the run directory.
            data: JSON-serializable content; other values are converted to strings.

        This prevents partial replacement but is not a power-loss durability or
        concurrent-writer guarantee.
        """
        target = self.path / name
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        os.replace(temporary, target)

    def status(self, state, **details):
        """Replace status.json with the state, UTC timestamps and optional details."""
        self.write('status.json', {
            'run_id': self.run_id, 'mode': self.mode, 'state': state,
            'started_at': self.started_at,
            'updated_at': datetime.now(timezone.utc).isoformat(), **details,
        })

    def fail(self, error):
        """Record a failure; call inside an exception handler to capture its traceback."""
        self.status('failed', error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc())
