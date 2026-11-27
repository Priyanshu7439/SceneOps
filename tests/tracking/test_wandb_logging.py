from sceneops.tracking.wandb_logging import WandbTracker


class FakeRun:
    def __init__(self, config):
        self.config = dict(config)
        self.logged = []
        self.finished = False

    def log(self, metrics):
        self.logged.append(dict(metrics))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.calls = []
        self.run = None

    def init(self, **kwargs):
        self.calls.append(kwargs)
        self.run = FakeRun(kwargs["config"])
        return self.run


def test_disabled_tracking_does_not_initialize_wandb():
    fake = FakeWandb()
    tracker = WandbTracker(
        project="sceneops",
        run_name="test",
        config={"model": "base"},
        mode="disabled",
        wandb_module=fake,
    )
    tracker.log_metrics({"eval/f1": 0.5})
    tracker.finish()
    assert fake.calls == []


def test_offline_tracking_logs_metadata_metrics_and_finishes():
    fake = FakeWandb()
    tracker = WandbTracker(
        project="sceneops",
        run_name="offline-test",
        config={"model": "base"},
        mode="offline",
        wandb_module=fake,
    )
    tracker.log_dataset_metadata(
        split_counts={"train": 3554, "validation": 1121, "test": 1181},
        dataset_metadata={"sha256": "test"},
    )
    tracker.log_model_metadata({"adapter": "lora", "rank": 16})
    tracker.log_metrics({"train/loss": 1.2, "eval/f1": 0.5})
    tracker.finish()

    assert fake.calls[0]["mode"] == "offline"
    assert fake.run.finished is True
    assert fake.run.logged == [{"train/loss": 1.2, "eval/f1": 0.5}]
    assert fake.run.config["dataset_split_counts"]["test"] == 1181
    assert fake.run.config["model_metadata"]["rank"] == 16
