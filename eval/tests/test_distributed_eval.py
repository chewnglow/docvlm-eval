from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


EVAL_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = EVAL_DIR / "distributed"
sys.path[:0] = [str(EVAL_DIR), str(DIST_DIR)]

import benchmark_common
import build_work_queue
import device_plan
import merge_predictions
import queue_worker
import ray_orchestrator
import run_benchmarks


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class DistributedEvalTests(unittest.TestCase):
    def test_ray_discovers_only_live_npu_nodes(self) -> None:
        nodes = ray_orchestrator.discover_compute_nodes(
            [
                {
                    "Alive": True,
                    "NodeID": "worker-b",
                    "NodeManagerAddress": "10.0.0.2",
                    "Resources": {"CPU": 16, "NPU": 8},
                },
                {
                    "Alive": True,
                    "NodeID": "head",
                    "NodeManagerAddress": "10.0.0.1",
                    "Resources": {"CPU": 4},
                },
                {
                    "Alive": False,
                    "NodeID": "dead-worker",
                    "NodeManagerAddress": "10.0.0.3",
                    "Resources": {"NPU": 8},
                },
                {
                    "Alive": True,
                    "NodeID": "worker-a",
                    "NodeManagerAddress": "10.0.0.4",
                    "Resources": {"NPU": 8},
                },
            ],
            "NPU",
        )
        self.assertEqual([node.node_id for node in nodes], ["worker-b", "worker-a"])
        self.assertEqual([node.capacity for node in nodes], [8, 8])

    def test_ray_assigns_distinct_nodes_to_both_model_groups(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode(f"node-{index}", f"10.0.0.{index}", 8)
            for index in range(1, 17)
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            qwen_devices=32,
            step_devices=64,
            node_capacity=8,
            qwen_model_key="qwen3.5-9b",
            step_model_key="step3vl-10b",
        )
        self.assertEqual(len(assignments), 12)
        self.assertEqual([item.family for item in assignments[:4]], ["qwen"] * 4)
        self.assertEqual([item.family for item in assignments[4:]], ["step"] * 8)
        self.assertEqual([item.node_rank for item in assignments[:4]], list(range(4)))
        self.assertEqual([item.node_rank for item in assignments[4:]], list(range(8)))
        self.assertEqual(len({item.node_id for item in assignments}), 12)

    def test_ray_assignment_supports_partial_final_node(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode(f"node-{index}", f"10.0.0.{index}", 8)
            for index in range(1, 6)
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            qwen_devices=10,
            step_devices=10,
            node_capacity=8,
            qwen_model_key="qwen3.5-9b",
            step_model_key="step3vl-10b",
        )
        self.assertEqual(
            [(item.family, item.local_devices) for item in assignments],
            [("qwen", 8), ("qwen", 2), ("step", 8), ("step", 2)],
        )
        self.assertTrue(all(item.reserved_devices == 8 for item in assignments))

    def test_ray_reserves_all_advertised_devices_on_selected_node(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode("node-1", "10.0.0.1", 16),
            ray_orchestrator.ClusterNode("node-2", "10.0.0.2", 16),
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            qwen_devices=8,
            step_devices=8,
            node_capacity=8,
            qwen_model_key="qwen3.5-9b",
            step_model_key="step3vl-10b",
        )
        self.assertEqual([item.reserved_devices for item in assignments], [16, 16])

    def test_ray_node_environment_sets_rank_and_model_without_leaking_env(self) -> None:
        assignment = ray_orchestrator.NodeAssignment(
            family="qwen",
            model_key="qwen3.5-9b",
            node_id="node-1",
            node_ip="10.0.0.1",
            node_rank=2,
            local_devices=4,
            reserved_devices=8,
        )
        common = ray_orchestrator.forwarded_environment(
            {
                "BENCHMARKS": "slidevqa:val",
                "OPENAI_API_KEY": "dummy",
                "UNRELATED_SECRET": "do-not-copy",
            }
        )
        env = ray_orchestrator.build_node_environment(
            assignment,
            total_devices=20,
            qwen_devices=10,
            step_devices=10,
            node_capacity=8,
            output_root="/shared/results",
            dataset_root="/data/dataset",
            python="/opt/venv/bin/python",
            common_env=common,
            model_path="/models/qwen",
        )
        self.assertEqual(env["NODE_RANK"], "2")
        self.assertEqual(env["DEVICES_PER_NODE"], "4")
        self.assertEqual(env["MODEL_PATH"], "/models/qwen")
        self.assertEqual(env["BENCHMARKS"], "slidevqa:val")
        self.assertTrue(env["LOG_DIR"].startswith("/shared/results/_ray_nodes/"))
        self.assertTrue(env["PID_DIR"].startswith("/shared/results/_ray_nodes/"))
        self.assertNotIn("UNRELATED_SECRET", env)

    def test_ray_rejects_relative_data_paths(self) -> None:
        with self.assertRaises(ValueError):
            ray_orchestrator.validate_data_paths(
                "outputs",
                "/data/dataset",
                ("/models/qwen", "/models/step"),
            )
        with self.assertRaises(ValueError):
            ray_orchestrator.validate_data_paths(
                "/shared/outputs",
                "/data/dataset",
                ("models/qwen", "/models/step"),
            )

    def test_ray_node_suite_cleans_up_servers(self) -> None:
        assignment = ray_orchestrator.NodeAssignment(
            family="step",
            model_key="step3vl-10b",
            node_id="node-1",
            node_ip="10.0.0.1",
            node_rank=0,
            local_devices=8,
            reserved_devices=8,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts = root / "eval" / "distributed"
            scripts.mkdir(parents=True)
            (scripts / "run_ascend_suite.sh").write_text("#!/usr/bin/env bash\n")
            (scripts / "stop_vllm_servers.sh").write_text("#!/usr/bin/env bash\n")
            success = SimpleNamespace(returncode=0)
            with mock.patch.object(
                ray_orchestrator.subprocess,
                "run",
                side_effect=[success, success],
            ) as run:
                result = ray_orchestrator.run_node_suite(
                    vars(assignment),
                    {"MODEL_KEY": assignment.model_key},
                    str(root),
                    keep_servers=False,
                )
        self.assertEqual(result["node_ip"], assignment.node_ip)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(run.call_args_list[1].args[0][1].endswith("stop_vllm_servers.sh"))

    def test_ray_node_suite_cleans_up_after_failure(self) -> None:
        assignment = ray_orchestrator.NodeAssignment(
            family="qwen",
            model_key="qwen3.5-9b",
            node_id="node-1",
            node_ip="10.0.0.1",
            node_rank=0,
            local_devices=8,
            reserved_devices=8,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts = root / "eval" / "distributed"
            scripts.mkdir(parents=True)
            (scripts / "run_ascend_suite.sh").write_text("#!/usr/bin/env bash\n")
            (scripts / "stop_vllm_servers.sh").write_text("#!/usr/bin/env bash\n")
            failure = SimpleNamespace(returncode=7)
            success = SimpleNamespace(returncode=0)
            with mock.patch.object(
                ray_orchestrator.subprocess,
                "run",
                side_effect=[failure, success],
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    ray_orchestrator.run_node_suite(
                        vars(assignment),
                        {"MODEL_KEY": assignment.model_key},
                        str(root),
                        keep_servers=False,
                    )
        self.assertEqual(run.call_count, 2)

    def test_device_plan_scales_from_16_to_128(self) -> None:
        for total in (16, 32, 48, 96, 127, 128):
            qwen = device_plan.make_plan(
                total_devices=total,
                model_key="qwen3.5-9b",
                node_rank=0,
                node_capacity=8,
            )
            step = device_plan.make_plan(
                total_devices=total,
                model_key="step3vl-10b",
                node_rank=0,
                node_capacity=8,
            )
            self.assertEqual(qwen["qwen_devices"] + qwen["step_devices"], total)
            self.assertEqual(qwen["group_devices"], qwen["qwen_devices"])
            self.assertEqual(step["group_devices"], step["step_devices"])
            self.assertGreaterEqual(qwen["local_devices"], 1)
            self.assertLessEqual(qwen["local_devices"], 8)

    def test_device_plan_handles_custom_split_and_partial_last_node(self) -> None:
        plan = device_plan.make_plan(
            total_devices=128,
            model_key="qwen3.5-9b",
            node_rank=9,
            node_capacity=8,
            qwen_devices=76,
            step_devices=52,
        )
        self.assertEqual(plan["group_nodes"], 10)
        self.assertEqual(plan["local_devices"], 4)

    def test_device_plan_rejects_out_of_range_fleet(self) -> None:
        for total in (15, 129):
            with self.assertRaises(ValueError):
                device_plan.make_plan(
                    total_devices=total,
                    model_key="qwen3.5-9b",
                    node_rank=0,
                    node_capacity=8,
                )

    def test_multimodal_prompt_keeps_question_after_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "page.png"
            image.write_bytes(b"fake-image")
            completions = FakeCompletions()
            client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            args = SimpleNamespace(
                model="model",
                max_images=None,
                max_tokens=8,
                temperature=0,
                image_data_cache_mb=1,
            )
            record = {
                "dataset": "longdocurl",
                "id": "q1",
                "question": "Where is the answer?",
                "image_paths": [str(image)],
            }
            with mock.patch.object(run_benchmarks, "get_openai_client", return_value=client):
                self.assertEqual(run_benchmarks.call_openai(args, record), "ok")
            content = completions.kwargs["messages"][0]["content"]
            self.assertEqual([part["type"] for part in content], ["text", "image_url", "text"])
            self.assertNotIn(record["question"], content[0]["text"])
            self.assertIn(record["question"], content[-1]["text"])

    def test_mmlong_questions_get_unique_ids_but_share_document(self) -> None:
        rows = [
            {"doc_id": "same.pdf", "question": "first?", "answer": "a"},
            {"doc_id": "same.pdf", "question": "second?", "answer": "b"},
        ]
        with mock.patch.object(benchmark_common, "parquet_rows", return_value=iter(rows)):
            with mock.patch.object(benchmark_common, "list_image_files", return_value=[]):
                records = list(benchmark_common.load_mmlongbench(Path("unused")))
        self.assertNotEqual(records[0]["id"], records[1]["id"])
        self.assertEqual(records[0]["doc_id"], records[1]["doc_id"])
        self.assertEqual(
            build_work_queue.document_key(records[0]),
            build_work_queue.document_key(records[1]),
        )

    def test_queue_claims_longest_first_and_recovers_stale_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp)
            for state in ("pending", "claimed", "completed", "failed"):
                (queue / state).mkdir()
            for task_id in (2, 0, 1):
                path = queue / "pending" / f"task-{task_id:06d}.json"
                path.write_text(json.dumps({"task_id": task_id}))
            claimed = queue_worker.claim_task(queue)
            self.assertEqual(claimed.name, "task-000000.json")
            old = time.time() - 100
            claimed.touch()
            import os

            os.utime(claimed, (old, old))
            self.assertEqual(queue_worker.recover_stale_claims(queue, lease_seconds=10), 1)
            self.assertTrue((queue / "pending" / "task-000000.json").exists())

    def test_queue_builder_groups_documents_and_sorts_cost_descending(self) -> None:
        records = [
            {
                "dataset": "longdocurl",
                "id": "short-1",
                "doc_no": "short",
                "image_paths": ["a", "b"],
            },
            {
                "dataset": "longdocurl",
                "id": "short-2",
                "doc_no": "short",
                "image_paths": ["a", "b"],
            },
            {
                "dataset": "longdocurl",
                "id": "long-1",
                "doc_no": "long",
                "image_paths": [str(i) for i in range(10)],
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp) / "queue"
            argv = [
                "build_work_queue.py",
                "--dataset",
                "longdocurl",
                "--queue-dir",
                str(queue),
                "--max-records-per-task",
                "2",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(build_work_queue, "iter_dataset_records", return_value=iter(records)):
                    build_work_queue.main()
            first = json.loads((queue / "pending" / "task-000000.json").read_text())
            second = json.loads((queue / "pending" / "task-000001.json").read_text())
            self.assertEqual(first["record_ids"], ["long-1"])
            self.assertEqual(second["record_ids"], ["short-1", "short-2"])
            self.assertTrue((queue / "READY").exists())

    def test_merge_accepts_queue_worker_outputs_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shards = root / "shards"
            shards.mkdir()
            (shards / "worker-a.jsonl").write_text('{"id":"q1","response":"first"}\n')
            (shards / "worker-b.jsonl").write_text(
                '{"id":"q1","response":"duplicate"}\n{"id":"q2","response":"second"}\n'
            )
            output = root / "predictions.jsonl"
            argv = ["merge_predictions.py", "--shard-dir", str(shards), "--output", str(output)]
            with mock.patch.object(sys, "argv", argv):
                merge_predictions.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
