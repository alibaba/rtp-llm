#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("create_kmonitor_dashboard.py")


class CreateKmonitorDashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        output = Path(cls._temp_dir.name) / "dashboard.json"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--generate-only",
                "--output",
                str(output),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        cls.dashboard = json.loads(output.read_text(encoding="utf-8"))["dashboard"]

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_all_graphs_belong_to_one_nonempty_collapsed_row(self):
        rows = self.dashboard["panels"]
        self.assertTrue(rows)
        self.assertTrue(all(panel["type"] == "row" for panel in rows))

        graph_ids = []
        for row in rows:
            self.assertTrue(row["collapsed"], row["title"])
            self.assertTrue(row["panels"], row["title"])
            self.assertTrue(
                all(panel["type"] == "graph" for panel in row["panels"]),
                row["title"],
            )
            graph_ids.extend(panel["id"] for panel in row["panels"])

        self.assertEqual(len(graph_ids), len(set(graph_ids)))

    def test_dashboard_uses_shared_crosshair(self):
        self.assertEqual(1, self.dashboard["graphTooltip"])

    def test_all_selectors_aggregate_by_default_and_expose_wildcard(self):
        variables = {
            variable["name"]: variable
            for variable in self.dashboard["templating"]["list"]
        }
        for name in (
            "deployment",
            "master_deployment",
            "host",
            "bucket",
            "residual_bucket",
        ):
            with self.subTest(name=name):
                self.assertTrue(variables[name]["includeAll"])
                self.assertEqual("None", variables[name]["allValue"])
                self.assertIn(variables[name]["current"]["text"], ("All", ["All"]))
                self.assertIn(
                    {"selected": True, "text": "All", "value": "$__all"},
                    variables[name]["options"],
                )
                self.assertIn(
                    {"selected": False, "text": "*", "value": "*"},
                    variables[name]["options"],
                )

        for name in ("host", "bucket", "residual_bucket"):
            self.assertTrue(variables[name]["multi"])

        for row in self.dashboard["panels"]:
            for panel in row["panels"]:
                for target in panel["targets"]:
                    role = target["tags"]["role"]
                    if role in ("agent", "client"):
                        self.assertEqual("[[host]]", target["tags"]["host_ip"])

    def test_agent_and_master_use_independent_deployment_selectors(self):
        variables = {
            variable["name"]: variable
            for variable in self.dashboard["templating"]["list"]
        }
        self.assertEqual(
            "suggest_tagv(navi_sched.role.alive,deployment_name,role=agent,hippo_app=[[app]])(1h-ago)(default)",
            variables["deployment"]["query"],
        )
        self.assertEqual(
            "suggest_tagv(navi_sched.role.alive,deployment_name,role=master,hippo_app=[[app]])(1h-ago)(default)",
            variables["master_deployment"]["query"],
        )
        self.assertEqual(
            "suggest_tagv(navi_sched.agent.latency_model_revision,host_ip,hippo_app=[[app]],deployment_name=[[deployment]])(5m-ago)(default)",
            variables["host"]["query"],
        )

    def test_deployment_and_host_selectors_narrow_bucket_candidates(self):
        variables = {
            variable["name"]: variable
            for variable in self.dashboard["templating"]["list"]
        }
        self.assertIn("deployment_name=[[deployment]]", variables["host"]["query"])
        self.assertIn("deployment_name=[[deployment]]", variables["bucket"]["query"])
        self.assertIn("host_ip=[[host]]", variables["bucket"]["query"])
        self.assertIn(
            "deployment_name=[[master_deployment]]",
            variables["residual_bucket"]["query"],
        )
        self.assertIn("host_ip=[[host]]", variables["residual_bucket"]["query"])

        for row in self.dashboard["panels"]:
            for panel in row["panels"]:
                for target in panel["targets"]:
                    role = target["tags"]["role"]
                    expected = (
                        "[[master_deployment]]"
                        if role == "master"
                        else "[[deployment]]"
                    )
                    self.assertEqual(expected, target["tags"]["deployment_name"])

    def test_every_metric_and_selector_filters_by_application(self):
        variables = {
            variable["name"]: variable
            for variable in self.dashboard["templating"]["list"]
        }
        app = variables["app"]
        self.assertEqual(
            "suggest_tagv(navi_sched.role.alive,hippo_app)(1h-ago)(default)",
            app["query"],
        )
        self.assertEqual(
            "whale_prod_Qwen3-235B-A22B-Instruct-2507_for_omega_64K",
            app["current"]["value"],
        )
        for name in ("deployment", "master_deployment", "host", "bucket", "residual_bucket"):
            self.assertIn("hippo_app=[[app]]", variables[name]["query"])
        for row in self.dashboard["panels"]:
            for panel in row["panels"]:
                for target in panel["targets"]:
                    self.assertEqual("[[app]]", target["tags"]["hippo_app"])

    def test_online_model_parameter_panel_matches_capacity_split_model(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        parameter_targets = [
            target
            for title in (
                "Execution 在线模型：线性参数",
                "Execution 在线模型：Batch 与非线性参数",
                "Execution 在线模型：Capacity Split 参数",
            )
            for target in panels[title]["targets"]
        ]
        self.assertEqual(
            [f"navi_sched.agent.latency_model_parameter_{index}" for index in range(19)],
            [target["metric"] for target in parameter_targets],
        )
        self.assertEqual(
            [
                "P9 总 Prompt Tokens",
                "P10 Waiting 请求数",
                "P11 Waiting Cache Hit",
                "P12 Waiting Compute",
                "P13 Waiting Compute 平方",
                "P14 Waiting Hit x Compute",
                "P15 前序请求数",
            ],
            [target["alias"].split(" / ")[0] for target in parameter_targets[9:16]],
        )
        self.assertEqual(
            [
                "P16 Running 已运行时间",
                "P17 Running Decode 请求数",
                "P18 Running Decode 已生成 Tokens",
            ],
            [
                target["alias"].split(" / ", 1)[0]
                for target in parameter_targets[-3:]
            ],
        )

    def test_adam_diagnostics_are_grouped_by_parameter_index(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        expected = {
            "Adam 每步原始梯度": "navi_sched.agent.latency_model_raw_gradient",
            "Adam 每步一阶矩": "navi_sched.agent.latency_model_adam_moment_1",
            "Adam 每步二阶矩": "navi_sched.agent.latency_model_adam_moment_2",
            "Adam 每步归一化梯度": (
                "navi_sched.agent.latency_model_adam_normalized_gradient"
            ),
            "Adam 每步参数更新：投影前": (
                "navi_sched.agent.latency_model_adam_candidate_update"
            ),
            "Adam 每步参数更新：实际": (
                "navi_sched.agent.latency_model_actual_update"
            ),
            "Adam 每步单参数预测扰动": (
                "navi_sched.agent.latency_model_parameter_prediction_shift_ms"
            ),
        }
        for title, metric in expected.items():
            target = panels[title]["targets"][0]
            self.assertEqual(metric, target["metric"])
            self.assertEqual("*", target["tags"]["parameter_index"])
            self.assertIn("$tag_parameter_index", target["alias"])

    def test_waiting_cost_safety_metrics_are_visible(self):
        panels = {
            panel["title"]: panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        self.assertEqual(
            [
                "navi_sched.agent.latency_model_unsafe_waiting_candidate_recovery_qps",
                "navi_sched.agent.latency_model_unsafe_waiting_state_reset_qps",
            ],
            [
                target["metric"]
                for target in panels["Execution 模型安全恢复 QPS"]["targets"]
            ],
        )
        self.assertIn(
            "navi_sched.master.cold_start_unsafe_waiting_marginal_node_observation_qps",
            [
                target["metric"]
                for target in panels["Master 冷启动路由 QPS"]["targets"]
            ],
        )
        self.assertIn(
            "navi_sched.master.cold_start_latest_unsafe_waiting_marginal_node_count",
            [
                target["metric"]
                for target in panels["Master 冷启动节点与容量"]["targets"]
            ],
        )

    def test_nonlinear_operating_region_metrics_are_visible(self):
        panels = {
            panel["title"]: panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        self.assertEqual(
            ["navi_sched.agent.latency_nonlinear_r"],
            [target["metric"] for target in
             panels["Execution 在线模型：非线性工作区段 r"]["targets"]],
        )
        self.assertEqual(
            ["navi_sched.agent.latency_nonlinear_u",
             "navi_sched.agent.latency_nonlinear_curvature"],
            [target["metric"] for target in
             panels["Execution 在线模型：非线性输入与曲率"]["targets"]],
        )

    def test_running_decode_progress_metrics_are_visible(self):
        panels = {
            panel["title"]: panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        self.assertEqual(
            [
                "navi_sched.agent.workload_running_decode_request_count",
                "navi_sched.agent.workload_running_decode_generated_tokens",
                "navi_sched.agent.workload_running_decode_remaining_token_budget",
            ],
            [
                target["metric"]
                for target in panels["单机 Running Decode 进度"]["targets"]
            ],
        )
        self.assertEqual(
            [
                "navi_sched.master.workload_running_decode_request_count",
                "navi_sched.master.workload_running_decode_generated_tokens",
                "navi_sched.master.workload_running_decode_remaining_token_budget",
            ],
            [
                target["metric"]
                for target in panels["Master 聚合 Running Decode 进度"]["targets"]
            ],
        )

    def test_vllm_pure_wait_metrics_are_visible(self):
        panels = {
            panel["title"]: panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        self.assertEqual(
            [
                "navi_sched.agent.vllm_admission_to_first_execution_window_mean_ms",
                "navi_sched.agent.vllm_admission_to_first_execution_window_max_ms",
            ],
            [target["metric"] for target in panels["单机 vLLM 纯等待（ms）"]["targets"]],
        )
        self.assertEqual(
            ["navi_sched.agent.vllm_admission_to_first_execution_window_sample_count"],
            [target["metric"] for target in panels["单机 vLLM 纯等待样本"]["targets"]],
        )

    def test_vllm_execution_step_metrics_are_visible(self):
        panels = {
            panel["title"]: panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        self.assertEqual(
            [
                "navi_sched.agent.vllm_execution_step_window_mean_ms",
                "navi_sched.agent.vllm_execution_step_window_max_ms",
                "navi_sched.agent.vllm_pure_decode_step_window_mean_ms",
                "navi_sched.agent.vllm_pure_decode_step_window_max_ms",
            ],
            [target["metric"] for target in panels["单机 vLLM Future 回调跨度（非单步）（ms）"]["targets"]],
        )
        self.assertEqual(
            [
                "navi_sched.agent.vllm_execution_step_window_count",
                "navi_sched.agent.vllm_pure_decode_step_window_count",
            ],
            [target["metric"] for target in panels["单机 vLLM Future 回调观测数"]["targets"]],
        )

        self.assertEqual(
            [
                "navi_sched.agent.vllm_core_step_window_mean_ms",
                "navi_sched.agent.vllm_core_step_window_max_ms",
                "navi_sched.agent.vllm_future_wait_window_mean_ms",
                "navi_sched.agent.vllm_future_wait_window_max_ms",
            ],
            [target["metric"] for target in panels["单机 vLLM EngineCore 单步耗时（ms）"]["targets"]],
        )

    def test_freshness_and_global_zero_point_availability_are_visible(self):
        metrics = {
            target["metric"]
            for row in self.dashboard["panels"]
            for panel in row["panels"]
            for target in panel["targets"]
        }
        expected = {
            "navi_sched.agent.latency_validation_enabled",
            "navi_sched.agent.validation_window_new_sample_count",
            "navi_sched.agent.validation_last_update_time_s",
            "navi_sched.agent.validation_last_update_age_ms",
            "navi_sched.agent.validation_execution_bucket_enabled",
            "navi_sched.agent.validation_execution_bucket_new_sample_count",
            "navi_sched.agent.validation_execution_bucket_last_update_time_s",
            "navi_sched.agent.validation_execution_bucket_last_update_age_ms",
            "navi_sched.master.latency_model_global_zero_point_available_10",
            "navi_sched.master.latency_model_global_zero_point_available_19",
            "navi_sched.master.latency_model_global_zero_point_invalid_reason_10",
            "navi_sched.master.latency_model_global_zero_point_invalid_reason_19",
        }
        self.assertTrue(expected.issubset(metrics), expected - metrics)
        self.assertFalse(
            any(
                metric.startswith("navi_sched.master.latency_model_global_zero_point_")
                and metric.endswith(("_11", "_18"))
                for metric in metrics
            )
        )

    def test_master_request_panel_distinguishes_arrival_and_scheduling(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        panel = panels["Master 入口与调度 QPS"]
        self.assertEqual(
            [
                "navi_sched.master.client_request_received_qps",
                "navi_sched.master.schedule_round_qps",
                "navi_sched.master.schedule_request_qps",
            ],
            [target["metric"] for target in panel["targets"]],
        )

    def test_client_local_fallback_is_visible(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        panel = panels["Client 本机回退 QPS"]
        self.assertEqual(
            [
                "navi_sched.client.master_error_qps",
                "navi_sched.client.local_fallback_qps",
                "navi_sched.client.master_success_qps",
            ],
            [target["metric"] for target in panel["targets"]],
        )

    def test_cache_hit_panels_distinguish_physical_pending_and_final_error(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        comparison = panels["单机 Cache Hit 四段对比"]
        self.assertEqual(
            [
                "Master 调度预测",
                "Agent 到达时物理快照",
                "Agent 到达时 Pending-aware 预测",
                "RTP 实际复用",
            ],
            [target["alias"].split(" / ", 1)[0] for target in comparison["targets"]],
        )

        bias_metrics = {
            target["metric"]
            for target in panels["单机 Cache Hit 最终预测 Bias"]["targets"]
        }
        mae_metrics = {
            target["metric"]
            for target in panels["单机 Cache Hit 最终预测 MAE"]["targets"]
        }
        self.assertIn(
            "navi_sched.agent.cache_hit_path_agent_pending_aware_to_rtp_error_tokens",
            bias_metrics,
        )
        self.assertIn(
            "navi_sched.agent.cache_hit_path_master_to_rtp_error_tokens",
            bias_metrics,
        )
        self.assertIn(
            "navi_sched.agent.cache_hit_path_agent_pending_aware_to_rtp_absolute_error_tokens",
            mae_metrics,
        )
        self.assertIn(
            "navi_sched.agent.cache_hit_path_master_to_rtp_absolute_error_tokens",
            mae_metrics,
        )
        physical_bias_metrics = {
            target["metric"]
            for target in panels["单机 Cache 物理化时序 Bias"]["targets"]
        }
        physical_mae_metrics = {
            target["metric"]
            for target in panels["单机 Cache 物理化时序 MAE"]["targets"]
        }
        self.assertEqual(
            {
                "navi_sched.agent.cache_hit_path_master_to_agent_error_tokens",
                "navi_sched.agent.cache_hit_path_agent_to_rtp_error_tokens",
            },
            physical_bias_metrics,
        )
        self.assertEqual(
            {
                "navi_sched.agent.cache_hit_path_master_to_agent_absolute_error_tokens",
                "navi_sched.agent.cache_hit_path_agent_to_rtp_absolute_error_tokens",
            },
            physical_mae_metrics,
        )
        self.assertTrue(
            bias_metrics.isdisjoint(physical_bias_metrics),
            "final prediction and physical snapshot diagnostics must stay separate",
        )
        self.assertTrue(
            mae_metrics.isdisjoint(physical_mae_metrics),
            "final prediction and physical snapshot diagnostics must stay separate",
        )
        error_metrics = bias_metrics | mae_metrics | physical_bias_metrics | physical_mae_metrics
        self.assertTrue(
            all("_window_" not in metric and "_mean_" not in metric
                for metric in error_metrics),
            "cache error panels must aggregate raw C++ request samples",
        )

    def test_agent_error_panels_only_use_direct_cpp_samples(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        error_panel_titles = {
            "单机 Queue 预测误差",
            "Queue 在线模型：Raw / 拟合误差",
            "单机 Execution 预测误差",
            "单机 Execution 相对误差",
            "Execution 相对误差分布",
            "Execution 分桶绝对误差",
            "Execution 分桶相对误差",
            "单机端到端预测误差",
        }
        metrics = {
            target["metric"]
            for title in error_panel_titles
            for target in panels[title]["targets"]
        }
        self.assertTrue(metrics)
        self.assertTrue(
            all(
                metric.startswith("navi_sched.agent.")
                and "_window_" not in metric
                and "_mean_" not in metric
                for metric in metrics
            ),
            "Agent error panels must aggregate direct C++ batch samples",
        )
        self.assertIn("navi_sched.agent.validation_queue_error_ms", metrics)
        self.assertIn(
            "navi_sched.agent.validation_execution_master_error_ms", metrics
        )
        self.assertIn(
            "navi_sched.agent.validation_execution_agent_error_ms", metrics
        )
        self.assertIn("navi_sched.agent.validation_total_error_ms", metrics)

    def test_cluster_error_panels_aggregate_direct_agent_samples(self):
        panels = {
            panel["title"].replace("（ms）", ""): panel
            for row in self.dashboard["panels"]
            for panel in row["panels"]
        }
        error_panel_titles = {
            "Queue 预测误差",
            "Agent Queue 模型：Raw / 拟合误差",
            "Execution 预测误差",
            "端到端预测误差",
            "Execution 与端到端相对误差",
            "Cache Hit 偏差与 MAE",
        }
        metrics = {
            target["metric"]
            for title in error_panel_titles
            for target in panels[title]["targets"]
        }
        self.assertTrue(metrics)
        self.assertTrue(
            all(
                metric.startswith("navi_sched.agent.")
                and "_window_" not in metric
                and "_mean_" not in metric
                for metric in metrics
            ),
            "cluster error panels must aggregate direct Agent C++ samples",
        )


if __name__ == "__main__":
    unittest.main()
