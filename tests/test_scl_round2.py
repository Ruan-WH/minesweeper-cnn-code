"""Failure and isolation checks for the sequential SCL orchestrator."""
import json
from pathlib import Path
import contextlib
import shutil
import uuid
import unittest
from unittest.mock import patch

import scl_round2 as scl


@contextlib.contextmanager
def work_directory():
    path = scl.ROOT / "output" / ("scl_unit_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        assert path.resolve().parent == (scl.ROOT / "output").resolve()
        shutil.rmtree(path)


class SequentialTrainingTests(unittest.TestCase):
    def config(self):
        return json.loads((scl.ROOT / "configs/scl_round2.json").read_text())

    def test_formal_budget_and_seed_validation(self):
        c = self.config()
        scl.validate_config(c)
        self.assertEqual(sum(s["budget"] for s in c["stages"]), 7850000)
        c["seeds"][1] = c["seeds"][0]
        with self.assertRaises(ValueError):
            scl.validate_config(c)

    def test_worker_failure_stops_chain(self):
        class FailedProcess:
            pid = 123
            def wait(self):
                return 7
        with work_directory() as tmp, patch.object(scl.subprocess, "Popen", return_value=FailedProcess()) as launch:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Stage failed"):
                scl.run_chain(root, self.config(), {}, True, "cpu", {}, [])
            self.assertEqual(launch.call_count, 1)
            self.assertFalse((root / "smoke/271828/14").exists())
            report = json.loads((root / "smoke/271828/10/result.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_missing_checkpoint_never_allows_transfer(self):
        class FakeProcess:
            pid = 456
            def wait(self):
                return 0
        with work_directory() as tmp:
            root = Path(tmp)
            def launch(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                scl.write_json(Path(job["output"]) / "result.json", {
                    "status": "completed", "actual_steps": 1100, "checkpoint_verified": True,
                    "best": {"path": str(root / "missing.dat"), "sha256": "bad"}})
                return FakeProcess()
            with patch.object(scl.subprocess, "Popen", side_effect=launch) as called:
                with self.assertRaises(FileNotFoundError):
                    scl.run_chain(root, self.config(), {}, True, "cpu", {}, [])
                self.assertEqual(called.call_count, 1)

    def test_six_stage_order_and_seed_isolation(self):
        jobs = []
        class FakeProcess:
            pid = 789
            def wait(self):
                return 0
        with work_directory() as tmp:
            root = Path(tmp)
            def launch(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                jobs.append(job)
                out = Path(job["output"])
                checkpoint = out / "best.dat"
                checkpoint.write_bytes(b"fake verified tensor data")
                scl.write_json(out / "result.json", {
                    "status": "completed", "actual_steps": 1100, "checkpoint_verified": True,
                    "best": {"path": str(checkpoint), "sha256": scl.digest(checkpoint)}})
                return FakeProcess()
            with patch.object(scl.subprocess, "Popen", side_effect=launch):
                scl.run_chain(root, self.config(), {}, True, "cpu", {}, [])
            self.assertEqual([(j["seed"], j["stage"]["mines"]) for j in jobs],
                             [(s, m) for s in self.config()["seeds"] for m in (10, 14, 18)])
            self.assertIsNone(jobs[0]["initial"])
            self.assertIsNone(jobs[3]["initial"])
            for i in (1, 2, 4, 5):
                self.assertEqual(Path(jobs[i]["initial"]["path"]).parent, Path(jobs[i - 1]["output"]))


if __name__ == "__main__":
    unittest.main()
