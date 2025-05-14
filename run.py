# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os
from transformers import set_seed

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent, infer
from android_world.agents import m3a
from android_world.agents import vdroid
from android_world.env import env_launcher
from android_world.env import interface
import subprocess

logging.set_verbosity(logging.WARNING)


os.environ['GRPC_VERBOSITY'] = 'NONE'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing

set_seed(42)


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.expanduser('/Users/daigaole/Documents/platform-tools/adb'),
        os.path.expanduser('/root/.android/platform-tools/adb'),
        os.path.expanduser('/home/emdl/.android/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories. If it's"
        ' already installed, point to the installed location.'
    )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    True,
    # False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)

_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_DEVICE_NAME = flags.DEFINE_string(
    'device_name',
    'emulator-5554',
    'The name of device to be connected.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
        # The other two benchmarks used in our paper, AndroidLab and MobileAgentBench
        # We would release them in the future.
        # registry.TaskRegistry.ANDROID_LAB_FAMILY,
        # registry.TaskRegistry.mobile_agent_bench_family,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)


# Agent specific.
_AGENT_NAME = flags.DEFINE_string(
    'agent_name', 't3a_llama', help='Agent name.')
_LLM_NAME = flags.DEFINE_string(
    'llm_name', 'gpt-4o', help='LLM name for action completion and working memory construction, [gpt-4o, gpt-4].')
_SAVE_NAME = flags.DEFINE_string(
    'save_name', 'test', help='Path to store the results.')
_LORA_DIR = flags.DEFINE_string(
    'lora_dir', 'RewardModel_Ori', help='The path to the lora module.')
_ITERATION = flags.DEFINE_string(
    'iteration', '1', help='The search iteration.')
_SUMMARY = flags.DEFINE_string('summary', 'llm', help='The summary mode.')
_NUM_GPUS = flags.DEFINE_integer(
    'num_gpus', 2, help='The num of gpu for parallel execution of verifier.')


_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    8554,
    'The port for the gprc communication.',
)


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
    """Gets agent."""
    print('Initializing agent...')
    agent = None

    # We use llama-3.1-8B-Instruct as the base model for V-Droid.
    base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    if _AGENT_NAME.value == "VDroid":
        agent = vdroid.VDroidAgent(env, base_model_name, adapter_dir=_LORA_DIR.value, llm_name=_LLM_NAME.value, n_iters=int(
            _ITERATION.value), family=family, summary_mode=_SUMMARY.value, num_actors=_NUM_GPUS.value)

    if not agent:
        raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

    agent.name = _AGENT_NAME.value

    return agent


def disable_key_board(emulator_id="emulator-5554"):
    commands = [
        f"adb -s {emulator_id} shell pm disable-user com.google.android.inputmethod.latin",
        f"adb -s {emulator_id} shell pm disable-user com.google.android.tts"
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")

    return


def _main() -> None:
    """Runs eval suite and gets rewards back."""
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
        grpc_port=_GRPC_PORT.value,
        device_name=_DEVICE_NAME.value,
        family=_SUITE_FAMILY.value,
    )

    if _EMULATOR_SETUP.value:
        disable_key_board(emulator_id=_DEVICE_NAME.value)

    env_launcher.verify_api_level(env)

    n_task_combinations = _N_TASK_COMBINATIONS.value
    task_registry = registry.TaskRegistry()
    suite = suite_utils.create_suite(
        task_registry.get_registry(family=_SUITE_FAMILY.value),
        n_task_combinations=n_task_combinations,
        seed=_TASK_RANDOM_SEED.value,
        tasks=_TASKS.value,
        use_identical_params=_FIXED_TASK_SEED.value,
    )

    suite.suite_family = _SUITE_FAMILY.value

    agent = _get_agent(env, _SUITE_FAMILY.value)

    if _SUITE_FAMILY.value.startswith('miniwob'):
        agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
    else:
        agent.transition_pause = None

    checkpoint_dir = f"./saved/" + agent.name + \
        '_' + _SAVE_NAME.value + '/task_info/'

    print(
        f'Starting eval with agent {_AGENT_NAME.value} and writing to'
        f' {checkpoint_dir}'
    )

    suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
        save_name=_SAVE_NAME.value,
    )

    print(
        f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
        f' family. Wrote to {checkpoint_dir}.'
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
