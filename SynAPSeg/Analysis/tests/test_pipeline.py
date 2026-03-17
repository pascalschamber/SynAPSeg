#!/usr/bin/env python3

from pipeline import Pipeline, PipelineStage
import logging

# Optional: configure root logger for demo
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


class DummyStage(PipelineStage):
    def run(self, data: dict, config: dict) -> dict:
        # Example: add a new key with a dummy value
        data["dummy"] = "processed"
        return data


class AnotherDummyStage(PipelineStage):
    def run(self, data: dict, config: dict) -> dict:
        # Example: add another dummy value based on config
        data["another_dummy"] = config.get("example_param", "default")
        return data


if __name__ == '__main__':
    # Define dummy stages and build pipeline
    stages = [DummyStage(), AnotherDummyStage()]
    pipeline = Pipeline(stages)

    # Input data and config
    initial_data = {"start": True}
    dummy_config = {"example_param": "custom_value"}

    # Run the pipeline
    final_data = pipeline.run(initial_data, dummy_config)
    print("Final data state:", final_data)
