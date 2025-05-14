from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class Clock_Lab(task_eval.TaskEval):
    """Base class for Clock tasks."""

    app_names = ('clock',)
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
        
        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.google.android.deskclock', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
        # time.sleep(5.0)
        return

    
class ClockSetAlarmWithLabel(Clock_Lab):
    """Task for setting an alarm with a label."""

    app_names = ("clock",)
    template = "Set an alarm for 3PM with the label 'meeting' using Clock."


class ClockSetAlarmCustomRingtone(Clock_Lab):
    """Task for setting an alarm with custom settings (disable vibrate, change ringtone)."""

    app_names = ("clock",)
    template = "Set an alarm for 6:45AM, disable vibrate and change ring song to Argon in Clock."


class ClockSetRecurringAlarm(Clock_Lab):
    """Task for setting a recurring alarm from Monday to Friday."""

    app_names = ("clock",)
    template = "Help me set an alarm every Monday to Friday, 7AM in the morning in Clock."


class ClockSetDailyAlarm(Clock_Lab):
    """Task for setting a daily alarm."""

    app_names = ("clock",)
    template = "Change my clock at 9AM, make it ring every day in Clock."


class ClockSetTomorrowAlarm(Clock_Lab):
    """Task for setting an alarm for the next day."""

    app_names = ("clock",)
    template = "Help me set an alarm at 10:30AM tomorrow in Clock."


class ClockSetWeekendAlarm(Clock_Lab):
    """Task for setting a weekend alarm with a label."""

    app_names = ("clock",)
    template = "I need to set a 10:30PM clock every weekend, and label it as 'Watch Football Games' to remind me in Clock."


class ClockTurnOffAllAlarms(Clock_Lab):
    """Task for turning off all alarms."""

    app_names = ("clock",)
    template = "Turn off all alarms in Clock."


class ClockDeleteAlarmsAfterTime(Clock_Lab):
    """Task for deleting alarms set after a specific time."""

    app_names = ("clock",)
    template = "Delete all alarms after 2PM in Clock."


class ClockTurnOffSpecificAlarm(Clock_Lab):
    """Task for turning off a specific alarm."""

    app_names = ("clock",)
    template = "Turn off the alarm at 4PM in Clock."


class ClockQueryEarliestAlarm(Clock_Lab):
    """Task for querying the earliest active alarm."""

    app_names = ("clock",)
    template = "What is my earliest alarm which is open in Clock?"


class ClockQueryAlarmAtTime(Clock_Lab):
    """Task for checking if an alarm exists at a specific time."""

    app_names = ("clock",)
    template = "Is there an alarm set on 4PM every day in Clock?"


class ClockQueryAlarmVibration(Clock_Lab):
    """Task for checking if an alarm has vibration enabled."""

    app_names = ("clock",)
    template = "Does my alarm at 4PM turn on vibrate in Clock?"


class ClockQueryActiveAlarms(Clock_Lab):
    """Task for counting the number of active alarms."""

    app_names = ("clock",)
    template = "How many alarms have been turned on in Clock?"


class ClockQueryAlarmStatus(Clock_Lab):
    """Task for checking if a specific alarm is active."""

    app_names = ("clock",)
    template = "Does my alarm at 9AM turn on in Clock?"


class ClockAddWorldClock(Clock_Lab):
    """Task for adding multiple time zones to the world clock."""

    app_names = ("clock",)
    template = "Add London and Barcelona time in Clock."


class ClockQueryWorldTime(Clock_Lab):
    """Task for querying the current time and time difference in a city."""

    app_names = ("clock",)
    template = "What is the current time in Barcelona and the time difference with local time in Clock?"


class ClockDeleteWorldClock(Clock_Lab):
    """Task for deleting a world clock entry."""

    app_names = ("clock",)
    template = "Delete Barcelona time from clock."


class ClockSetTimerWithoutStarting(Clock_Lab):
    """Task for setting a countdown timer without starting it."""

    app_names = ("clock",)
    template = "Set a countdown timer for 1 hour 15 minutes but do not start it in Clock."


class ClockSetBedtime(Clock_Lab):
    """Task for setting a bedtime schedule."""

    app_names = ("clock",)
    template = "Set bedtime for 10PM to sleep, wake up at 7AM in Clock."


class ClockSetSleepSounds(Clock_Lab):
    """Task for configuring sleep sounds."""

    app_names = ("clock",)
    template = "Set sleep sounds to deep space in Clock."


class ClockEnableWakeupAlarm(Clock_Lab):
    """Task for turning on the wake-up alarm in bedtime mode."""

    app_names = ("clock",)
    template = "Turn on the Wake-up alarm in Bedtime in Clock."


class ClockSetAnalogAlarmStyle(Clock_Lab):
    """Task for changing the alarm display style to Analog."""

    app_names = ("clock",)
    template = "Set alarm style to Analog in Clock."


class ClockChangeTimeZone(Clock_Lab):
    """Task for changing the home time zone setting."""

    app_names = ("clock",)
    template = "Change home time zone to Tokyo in Clock."


class ClockModifySilenceDuration(Clock_Lab):
    """Task for modifying the silence duration of an alarm."""

    app_names = ("clock",)
    template = "Modify silence after to 5 minutes in Clock."


class ClockOpenClockApp(Clock_Lab):
    """Task for opening the Clock application."""

    app_names = ("clock",)
    template = "Open Clock app."


class ClockTurnOffSpecificAlarmByTime(Clock_Lab):
    """Task for turning off a specific alarm by its time."""

    app_names = ("clock",)
    template = "Close my 7:30AM alarm in Clock."


class ClockSetSimpleAlarm(Clock_Lab):
    """Task for setting a basic alarm."""

    app_names = ("clock",)
    template = "Set an alarm at 3PM in Clock."
