from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class Zoom_Lab(task_eval.TaskEval):
    """Base class for Zoom tasks."""

    app_names = ('us.zoom.videomeetings',)
    complexity = 2
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    template = ''

    device_time = device_constants.DT_LAB

    @classmethod
    def set_device_time(cls, env: interface.AsyncEnv) -> None:
        """Sets the device time."""
        del env
        cls.device_time = device_constants.DT_LAB

    def is_successful(self, env: interface.AsyncEnv) -> float:
        return 0.0

    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:
        return {}

    def initialize_task(self, env: interface.AsyncEnv):
        """(Optional) If you need to open Zoom or navigate somewhere first."""
        load_snapshots(env)
        env.reset(go_home=True)
        super().initialize_task(env)
        
        command = 'adb -s emulator-5554 shell settings put system time_12_24 12'
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")

        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'us.zoom.videomeetings', '1'], env.controller)
        # time.sleep(5.0)

        


class ZoomJoinMeeting(Zoom_Lab):
    """
    Task to join meeting ID 1234567890.
    You should not tap 'Join' button, just fill in the meeting ID field.
    """
    template = 'Type in the meeting ID 1234567890 in Zoom. You should not click join button. Leave it to user.'

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # If you want to check the Zoom app is on the meeting screen, you can do so.
        # For now, just return 0.0 or your own logic.
        return 0.0


class ZoomJoinMeetingSetName(Zoom_Lab):
    """
    Task to join meeting ID 0987654321 with the name 'Alice'.
    You should not tap 'Join' button, just fill in meeting ID and name.
    """
    template = "Type in the meeting ID 0987654321, and set my name as 'Alice' in Zoom. You should not click join button. Leave it to user."

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # Possibly check if 'Alice' is typed in the Name field
        # Return 0.0 or your own logic
        return 0.0


class ZoomJoinMeetingWithoutAudioVideo(Zoom_Lab):
    """
    Task to join meeting 1234567890, disabling audio/video prior to joining.
    """
    template = 'I need to join meeting 1234567890 without audio and video in Zoom. You should not click join button. Leave it to user.'

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # Possibly verify that audio/video toggles are OFF
        return 0.0


class ZoomSetAutoConnectAudio(Zoom_Lab):
    """
    Task to set auto connect to audio when on Wi-Fi in Zoom settings.
    """
    template = 'Set auto connect to audio when wifi is connected in Zoom settings.'

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # Potentially check if the setting is toggled in Zoom's shared preferences or UI
        return 0.0


class ZoomChangeReactionSkin(Zoom_Lab):
    """
    Task to change the reaction skin tone to Medium-light in Zoom settings.
    """
    template = 'Change my reaction skin to Medium-light in Zoom settings.'

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # Possibly detect the chosen reaction skin in Zoom's settings or UI
        return 0.0
