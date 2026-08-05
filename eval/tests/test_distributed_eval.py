from __future__ import annotations

import json
import os
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
import device_memory_monitor
import merge_predictions
import queue_worker
import queue_status
import ray_orchestrator
import run_benchmarks
import score_benchmarks


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class DistributedEvalTests(unittest.TestCase):
    def test_parses_nvidia_and_ascend_device_memory(self) -> None:
        nvidia = device_memory_monitor.parse_nvidia_csv("0, 1024, 16384\n1, 2048, 16384\n")
        self.assertEqual(nvidia[1], {"index": 1, "used_mb": 2048.0, "total_mb": 16384.0})

        ascend = device_memory_monitor.parse_ascend_info(
            """
| NPU   Chip | Bus-Id        | AICore(%)   HBM-Usage(MB) |
| 0     0    | 0000:01:00.0  | 72          32768 / 65536 |
| 1     0    | 0000:02:00.0  | 65          16384 / 65536 |
"""
        )
        self.assertEqual(
            ascend,
            [
                {"index": 0, "used_mb": 32768.0, "total_mb": 65536.0},
                {"index": 1, "used_mb": 16384.0, "total_mb": 65536.0},
            ],
        )

    def test_queue_status_aggregates_fresh_device_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            current = Path(temp) / "current"
            current.mkdir()
            for node, used in (("node-a", 1000), ("node-b", 3000)):
                (current / f"{node}.json").write_text(
                    json.dumps(
                        {
                            "timestamp": 1000,
                            "active": True,
                            "backend": "ascend",
                            "devices": [{"index": 0, "used_mb": used, "total_mb": 8000}],
                        }
                    )
                )
            status = queue_status.memory_status(Path(temp), max_age=120, now=1050)
            self.assertIn("device_memory=4000/16000MB (25.0%)", status)
            self.assertIn("per_device=12.5-37.5%", status)
            self.assertIn("devices=2 nodes=2", status)

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

    def test_ray_defaults_to_all_four_models(self) -> None:
        allocations = ray_orchestrator.resolve_allocations(
            128,
            None,
            None,
            None,
            None,
            qwen_model_key="qwen3.5-9b",
            qwen_base_model_key="qwen3.5-9b-base",
            step_model_key="step3vl-10b",
            step_base_model_key="step3vl-10b-base",
        )
        self.assertEqual(
            [(allocation.group, allocation.devices) for allocation in allocations],
            [
                ("qwen", 32),
                ("qwen-base", 32),
                ("step", 32),
                ("step-base", 32),
            ],
        )
        small_allocations = ray_orchestrator.resolve_allocations(
            16,
            None,
            None,
            None,
            None,
            qwen_model_key="qwen3.5-9b",
            qwen_base_model_key="qwen3.5-9b-base",
            step_model_key="step3vl-10b",
            step_base_model_key="step3vl-10b-base",
        )
        packed = ray_orchestrator.build_assignments(
            [
                ray_orchestrator.ClusterNode("node-1", "10.0.0.1", 8),
                ray_orchestrator.ClusterNode("node-2", "10.0.0.2", 8),
            ],
            allocations=small_allocations,
            node_capacity=8,
        )
        self.assertEqual(len({item.node_id for item in packed}), 2)
        self.assertEqual(
            [(item.local_devices, item.device_offset) for item in packed],
            [(4, 0), (4, 4), (4, 0), (4, 4)],
        )

    def test_ray_preserves_explicit_legacy_two_model_allocation(self) -> None:
        allocations = ray_orchestrator.resolve_allocations(
            96,
            32,
            None,
            64,
            None,
            qwen_model_key="qwen3.5-9b",
            qwen_base_model_key="qwen3.5-9b-base",
            step_model_key="step3vl-10b",
            step_base_model_key="step3vl-10b-base",
        )
        self.assertEqual(
            [(allocation.group, allocation.devices) for allocation in allocations],
            [("qwen", 32), ("step", 64)],
        )

    def test_ray_accepts_custom_four_model_allocation(self) -> None:
        allocations = ray_orchestrator.resolve_allocations(
            128,
            24,
            24,
            40,
            40,
            qwen_model_key="qwen3.5-9b",
            qwen_base_model_key="qwen3.5-9b-base",
            step_model_key="step3vl-10b",
            step_base_model_key="step3vl-10b-base",
        )
        self.assertEqual(
            [allocation.devices for allocation in allocations],
            [24, 24, 40, 40],
        )
        with self.assertRaises(ValueError):
            ray_orchestrator.resolve_allocations(
                128,
                24,
                24,
                40,
                None,
                qwen_model_key="qwen3.5-9b",
                qwen_base_model_key="qwen3.5-9b-base",
                step_model_key="step3vl-10b",
                step_base_model_key="step3vl-10b-base",
            )

    def test_ray_assigns_distinct_nodes_to_all_model_groups(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode(f"node-{index}", f"10.0.0.{index}", 8)
            for index in range(1, 17)
        ]
        allocations = [
            ray_orchestrator.ModelAllocation("qwen", "qwen3.5-9b", 32),
            ray_orchestrator.ModelAllocation("qwen-base", "qwen3.5-9b-base", 32),
            ray_orchestrator.ModelAllocation("step", "step3vl-10b", 32),
            ray_orchestrator.ModelAllocation("step-base", "step3vl-10b-base", 32),
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            allocations=allocations,
            node_capacity=8,
        )
        self.assertEqual(len(assignments), 16)
        self.assertEqual([item.family for item in assignments[:4]], ["qwen"] * 4)
        self.assertEqual([item.family for item in assignments[4:8]], ["qwen-base"] * 4)
        self.assertEqual([item.family for item in assignments[8:12]], ["step"] * 4)
        self.assertEqual([item.family for item in assignments[12:]], ["step-base"] * 4)
        self.assertEqual([item.node_rank for item in assignments[:4]], list(range(4)))
        self.assertEqual([item.node_rank for item in assignments[12:]], list(range(4)))
        self.assertEqual(len({item.node_id for item in assignments}), 16)

    def test_ray_assignment_supports_partial_final_node(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode(f"node-{index}", f"10.0.0.{index}", 8)
            for index in range(1, 6)
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            allocations=[
                ray_orchestrator.ModelAllocation("qwen", "qwen3.5-9b", 10),
                ray_orchestrator.ModelAllocation("step", "step3vl-10b", 10),
            ],
            node_capacity=8,
        )
        self.assertEqual(
            [(item.family, item.local_devices, item.device_offset) for item in assignments],
            [
                ("qwen", 8, 0),
                ("qwen", 2, 0),
                ("step", 6, 2),
                ("step", 4, 0),
            ],
        )
        self.assertEqual(len({item.node_id for item in assignments}), 3)
        self.assertEqual(
            [item.reserved_devices for item in assignments],
            [8, 2, 6, 4],
        )

    def test_ray_reserves_only_devices_used_by_each_launcher(self) -> None:
        nodes = [
            ray_orchestrator.ClusterNode("node-1", "10.0.0.1", 16),
            ray_orchestrator.ClusterNode("node-2", "10.0.0.2", 16),
        ]
        assignments = ray_orchestrator.build_assignments(
            nodes,
            allocations=[
                ray_orchestrator.ModelAllocation("qwen", "qwen3.5-9b", 8),
                ray_orchestrator.ModelAllocation("step", "step3vl-10b", 8),
            ],
            node_capacity=8,
        )
        self.assertEqual([item.reserved_devices for item in assignments], [8, 8])

    def test_ray_node_environment_sets_rank_and_model_without_leaking_env(self) -> None:
        assignment = ray_orchestrator.NodeAssignment(
            family="qwen",
            model_key="qwen3.5-9b",
            node_id="node-1",
            node_ip="10.0.0.1",
            node_rank=2,
            group_devices=20,
            group_nodes=3,
            local_devices=4,
            device_offset=2,
            reserved_devices=8,
        )
        common = ray_orchestrator.forwarded_environment(
            {
                "BENCHMARKS": "slidevqa:val",
                "OPENAI_API_KEY": "dummy",
                "DEVICE_MEMORY_INTERVAL": "10",
                "UNRELATED_SECRET": "do-not-copy",
            }
        )
        env = ray_orchestrator.build_node_environment(
            assignment,
            total_devices=20,
            node_capacity=8,
            output_root="/shared/results",
            dataset_root="/data/dataset",
            python="/opt/venv/bin/python",
            common_env=common,
            model_path="/models/qwen",
        )
        self.assertEqual(env["NODE_RANK"], "2")
        self.assertEqual(env["MODEL_GROUP_DEVICE_COUNT"], "20")
        self.assertEqual(env["MODEL_GROUP_NODE_COUNT"], "3")
        self.assertEqual(env["DEVICES_PER_NODE"], "4")
        self.assertEqual(env["DEVICE_OFFSET"], "2")
        self.assertEqual(env["BASE_PORT"], "8002")
        self.assertEqual(env["MODEL_PATH"], "/models/qwen")
        self.assertEqual(env["BENCHMARKS"], "slidevqa:val")
        self.assertEqual(env["DEVICE_MEMORY_INTERVAL"], "10")
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
            group_devices=8,
            group_nodes=1,
            local_devices=8,
            device_offset=0,
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
            group_devices=8,
            group_nodes=1,
            local_devices=8,
            device_offset=0,
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

    def test_ascend_launcher_accepts_direct_base_model_group_size(self) -> None:
        script = DIST_DIR / "run_ascend_queue_eval.sh"
        env = os.environ.copy()
        env.update(
            {
                "MODEL_KEY": "qwen3.5-9b-base",
                "DATASET": "slidevqa",
                "MODEL_GROUP_DEVICE_COUNT": "4",
                "MODEL_GROUP_NODE_COUNT": "1",
                "LOCAL_DEVICE_COUNT": "4",
                "NODE_DEVICE_CAPACITY": "8",
                "NODE_RANK": "1",
                "TOTAL_DEVICES": "16",
            }
        )
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=EVAL_DIR.parent,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("nothing to run", completed.stdout)

    def test_base_model_presets_have_distinct_served_names(self) -> None:
        preset = DIST_DIR / "model_presets.sh"
        with tempfile.TemporaryDirectory() as temp:
            served_names = []
            for model_key in ("qwen3.5-9b-base", "step3vl-10b-base"):
                command = (
                    f"source {preset!s}; "
                    f"MODEL_PATH={temp!s}; "
                    f"resolve_model_preset {model_key}; "
                    'echo "$SERVED_MODEL_NAME"'
                )
                completed = subprocess.run(
                    ["bash", "-c", command],
                    cwd=EVAL_DIR.parent,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                served_names.append(completed.stdout.strip())
        self.assertEqual(served_names, ["Qwen3.5-9B-Base", "Step3VL10B-Base"])

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

    def test_distributed_coverage_excludes_only_terminal_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp)
            (queue / "failed").mkdir()
            (queue / "manifest.json").write_text(json.dumps({"record_count": 3}))
            (queue / "records.jsonl").write_text(
                '\n'.join(
                    json.dumps({"id": record_id, "question": f"question-{record_id}"})
                    for record_id in ("q1", "q2", "q3")
                )
                + "\n"
            )
            (queue / "failed" / "task-000001.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-000001",
                        "record_ids": ["q2", "q3"],
                        "attempts": 3,
                        "last_error": "context length exceeded",
                    }
                )
            )

            coverage, failed = score_benchmarks.distributed_coverage(
                queue,
                [{"id": "q1"}, {"id": "q2"}],
            )

            self.assertEqual(coverage["status"], "partial")
            self.assertEqual(coverage["expected_count"], 3)
            self.assertEqual(coverage["scored_count"], 2)
            self.assertEqual(coverage["excluded_failed_count"], 1)
            self.assertAlmostEqual(coverage["score_coverage"], 2 / 3)
            self.assertEqual([record["id"] for record in failed], ["q3"])
            self.assertEqual(failed[0]["queue_failure"]["attempts"], 3)

    def test_distributed_coverage_rejects_empty_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp)
            (queue / "failed").mkdir()
            (queue / "manifest.json").write_text(json.dumps({"record_count": 0}))
            (queue / "records.jsonl").write_text("")
            with self.assertRaises(SystemExit):
                score_benchmarks.distributed_coverage(queue, [])

    def test_distributed_coverage_rejects_unaccounted_missing_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queue = Path(temp)
            (queue / "failed").mkdir()
            (queue / "manifest.json").write_text(json.dumps({"record_count": 2}))
            (queue / "records.jsonl").write_text('{"id":"q1"}\n{"id":"q2"}\n')
            with self.assertRaises(SystemExit):
                score_benchmarks.distributed_coverage(queue, [{"id": "q1"}])


if __name__ == "__main__":
    unittest.main()
