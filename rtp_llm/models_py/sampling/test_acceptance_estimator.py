import unittest

import torch

from rtp_llm.models_py.sampling.acceptance_estimator import OnlineAcceptanceEstimator


def reference_features(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    temperatures = temperatures.cpu().double()
    scale = torch.where(temperatures > 0, temperatures, 1.0)
    scaled = logits.cpu().double() / scale[:, None]
    maximum, indices = scaled.max(dim=-1)
    complement = scaled.scatter(1, indices[:, None], -torch.inf)
    return (maximum - complement.logsumexp(dim=-1)).clamp(-40.0, 40.0)


def graded_fixture() -> tuple[OnlineAcceptanceEstimator, torch.Tensor, torch.Tensor]:
    estimator = OnlineAcceptanceEstimator(16, 4, torch.device("cuda"))
    estimator.features.copy_(torch.linspace(-2, 2, 64, device="cuda").view(16, 4))
    estimator.predictions.copy_(
        torch.sigmoid(estimator.slope * estimator.features + estimator.intercepts)
    )
    slots = torch.tensor([7, 2, 11, 5, 1, 4], dtype=torch.int32, device="cuda")
    # Sampled counts include the bonus; rejected counts include the untested suffix.
    grades = torch.tensor([[1, 4], [2, 2], [3, 0], [5, 0], [1, 0], [2, 0]])
    return estimator, slots, grades


def reference_observations(
    estimator: OnlineAcceptanceEstimator, slots: torch.Tensor, grades: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, predictions = estimator.features.cpu(), estimator.predictions.cpu()
    rows, labels, probabilities = [], [], []
    for slot, (sampled, rejected) in zip(slots.cpu().tolist(), grades.tolist()):
        accepted = max(sampled - 1, 0)
        observed_labels = [1.0] * accepted + ([0.0] if rejected else [])
        for step, label in enumerate(observed_labels):
            row = torch.zeros(1 + estimator.num_speculative_steps, dtype=torch.float64)
            row[0], row[step + 1] = features[slot, step], 1.0
            rows.append(row)
            labels.append(label)
            probabilities.append(predictions[slot, step].double())
    return torch.stack(rows), torch.tensor(labels).double(), torch.stack(probabilities)


class AcceptanceEstimatorTest(unittest.TestCase):
    def test_predict_matches_reference_and_preserves_unmapped_slots(self) -> None:
        for vocab in (1, 37, 8193):
            for block_steps in (False, True):
                with self.subTest(vocab=vocab, block_steps=block_steps):
                    estimator = OnlineAcceptanceEstimator(16, 4, torch.device("cuda"))
                    estimator.intercepts.copy_(torch.tensor([1.5, 1.0, 0.5, 0.0]))
                    storage = torch.tensor(
                        [7, 99, 2, 99, -1, 99, 11, 99], device="cuda"
                    )
                    slots = storage[::2]
                    mapping = slots.repeat_interleave(4) if block_steps else slots
                    steps = (
                        torch.arange(4, device="cuda").repeat(4)
                        if block_steps
                        else torch.tensor(2, device="cuda")
                    )
                    generator = torch.Generator(device="cuda").manual_seed(20261109)
                    logits = torch.randn(
                        2 * mapping.numel(), vocab, device="cuda", generator=generator
                    )[::2]
                    temperatures = torch.linspace(0.4, 1.7, 16, device="cuda")
                    temperatures[7] = 0
                    confidence = torch.full((4, 4), -99.0, device="cuda")
                    expected_features = torch.zeros_like(estimator.features)
                    expected_predictions = torch.zeros_like(estimator.predictions)
                    expected_confidence = confidence.clone()
                    valid = mapping >= 0
                    row_steps = (
                        steps.expand(mapping.numel()) if steps.dim() == 0 else steps
                    )
                    valid_steps, valid_slots = row_steps[valid], mapping[valid]
                    features = reference_features(logits[valid], temperatures[valid_slots])
                    probabilities = torch.sigmoid(
                        estimator.slope.cpu().double() * features
                        + estimator.intercepts[valid_steps].cpu().double()
                    ).float().cuda()
                    expected_features[valid_slots, valid_steps] = features.float().cuda()
                    expected_predictions[valid_slots, valid_steps] = probabilities
                    batch = torch.arange(mapping.numel(), device="cuda")
                    batch = batch // 4 if block_steps else batch
                    expected_confidence[batch[valid], valid_steps] = probabilities

                    estimator.predict(logits, mapping, steps, confidence, temperatures)

                    torch.testing.assert_close(
                        estimator.features, expected_features, atol=2e-4, rtol=2e-4
                    )
                    torch.testing.assert_close(
                        estimator.predictions, expected_predictions, atol=2e-4, rtol=2e-4
                    )
                    torch.testing.assert_close(
                        confidence, expected_confidence, atol=2e-4, rtol=2e-4
                    )

    def test_step_labels_only_observed_prefix(self) -> None:
        estimator, slots, grades = graded_fixture()
        design, labels, probabilities = reference_observations(estimator, slots, grades)
        weights = probabilities * (1 - probabilities)
        residual = labels - probabilities
        expected_info = torch.zeros(4, 3, dtype=torch.float64)
        expected_grad = torch.zeros(4, 2, dtype=torch.float64)
        for step in range(4):
            selected = design[:, step + 1].bool()
            local_design = design[selected][:, [0, step + 1]]
            hessian = local_design.T @ (weights[selected, None] * local_design)
            expected_info[step] = torch.stack(
                (hessian[0, 0], hessian[0, 1], hessian[1, 1])
            )
            expected_grad[step] = local_design.T @ residual[selected]

        estimator.step(slots, grades[:, 0].int().cuda(), grades[:, 1].int().cuda())

        torch.testing.assert_close(
            estimator.info.cpu().double(), expected_info, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            estimator.grad.cpu().double(), expected_grad, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            estimator.counts.cpu(), torch.tensor([5.0, 3.0, 1.0, 1.0])
        )

    def test_refit_matches_dense_newton_solve_and_respects_cadence(self) -> None:
        estimator, slots, grades = graded_fixture()
        grades = torch.tensor([[1, 2], [2, 1], [3, 0], [3, 0], [1, 0], [2, 0]])
        estimator.REFIT_INTERVAL = 2
        design, labels, probabilities = reference_observations(estimator, slots, grades)
        weights = probabilities * (1 - probabilities)
        hessian = 2 * design.T @ (weights[:, None] * design)
        hessian += estimator.L2 * torch.eye(5, dtype=torch.float64)
        gradient = 2 * design.T @ (labels - probabilities)
        delta = torch.linalg.solve(hessian, gradient)
        counts = 2 * design[:, 1:].sum(0)
        slope_delta = (
            delta[0] * counts.sum() / (counts.sum() + estimator.DAMPING_OBSERVATIONS)
        )
        initial_slope = estimator.slope.cpu().double()
        if initial_slope.item() + slope_delta.item() < 0:
            slope_delta.zero_()
        bias_delta = (
            torch.linalg.solve(
                hessian[1:, 1:], gradient[1:] - hessian[1:, 0] * slope_delta
            )
            * counts
            / (counts + estimator.DAMPING_OBSERVATIONS)
        )
        initial_bias = estimator.intercepts.cpu().double()
        sampled, rejected = grades[:, 0].int().cuda(), grades[:, 1].int().cuda()

        estimator.step(slots, sampled, rejected)
        torch.testing.assert_close(
            estimator.slope.cpu().double(), initial_slope, rtol=0, atol=0
        )
        self.assertEqual(estimator._refits, 0)
        estimator.step(slots, sampled, rejected)

        torch.testing.assert_close(
            estimator.slope.cpu().double(), initial_slope + slope_delta, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            estimator.intercepts.cpu().double(), initial_bias + bias_delta, atol=1e-5, rtol=1e-5
        )
        self.assertEqual(estimator._refits, 1)
        self.assertEqual(estimator._steps_since_refit, 0)
        for stats in (estimator.info, estimator.grad, estimator.counts):
            self.assertEqual(torch.count_nonzero(stats).item(), 0)

    def test_warmup_drops_statistics_without_resetting_coefficients(self) -> None:
        estimator, slots, grades = graded_fixture()
        estimator.step(slots, grades[:, 0].int().cuda(), grades[:, 1].int().cuda())
        estimator.slope.fill_(0.8)
        estimator.intercepts.fill_(1.3)
        slope, intercepts = estimator.slope.clone(), estimator.intercepts.clone()

        estimator.warmup()

        torch.testing.assert_close(estimator.slope, slope, rtol=0, atol=0)
        torch.testing.assert_close(estimator.intercepts, intercepts, rtol=0, atol=0)
        self.assertEqual(estimator._steps_since_refit, 0)
        for stats in (estimator.info, estimator.grad, estimator.counts):
            self.assertEqual(torch.count_nonzero(stats).item(), 0)

    def test_graph_replay_reads_changed_inputs_metadata_and_coefficients(self) -> None:
        estimator = OnlineAcceptanceEstimator(16, 4, torch.device("cuda"))
        eager = OnlineAcceptanceEstimator(16, 4, torch.device("cuda"))
        logits = torch.zeros(4, 8193, device="cuda")
        slots = torch.tensor([7, 2, 11, -1], device="cuda")
        step = torch.tensor(0, device="cuda")
        temperatures = torch.ones(16, device="cuda")
        confidence = torch.zeros(4, 4, device="cuda")
        for _ in range(3):
            estimator.predict(logits, slots, step, confidence, temperatures)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            estimator.predict(logits, slots, step, confidence, temperatures)

        for iteration, mapping in enumerate(([3, -1, 7, -1], [11, 1, 4, 2])):
            generator = torch.Generator(device="cuda").manual_seed(20261109 + iteration)
            logits.normal_(generator=generator)
            slots.copy_(torch.tensor(mapping, device="cuda"))
            step.fill_(iteration + 1)
            temperatures.copy_(torch.linspace(0.3, 1.6, 16, device="cuda"))
            temperatures[mapping[0]] = 0
            estimator.slope.fill_(0.6 + iteration * 0.2)
            estimator.intercepts.copy_(torch.arange(4, device="cuda") * (iteration + 1))
            eager.slope.copy_(estimator.slope)
            eager.intercepts.copy_(estimator.intercepts)
            estimator.features.zero_()
            estimator.predictions.zero_()
            eager.features.zero_()
            eager.predictions.zero_()
            confidence.fill_(-99)
            eager_confidence = torch.full_like(confidence, -99)

            graph.replay()
            eager.predict(logits, slots, step, eager_confidence, temperatures)

            torch.testing.assert_close(confidence, eager_confidence, atol=0, rtol=0)
            torch.testing.assert_close(estimator.features, eager.features, atol=0, rtol=0)
            torch.testing.assert_close(
                estimator.predictions, eager.predictions, atol=0, rtol=0
            )
            valid = slots >= 0
            expected_features = reference_features(
                logits[valid], temperatures[slots[valid]]
            )
            expected = torch.sigmoid(
                estimator.slope.cpu().double() * expected_features
                + estimator.intercepts[iteration + 1].cpu().double()
            )
            torch.testing.assert_close(
                confidence[valid, iteration + 1].cpu().double(), expected, atol=2e-4, rtol=2e-4
            )
            self.assertTrue((confidence[~valid] == -99).all().item())


if __name__ == "__main__":
    unittest.main()
