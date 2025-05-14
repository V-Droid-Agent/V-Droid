from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class Calendar_Lab(task_eval.TaskEval):
    """Base class for Calendar tasks."""

    app_names = ("calendar",)
    complexity = 2
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    template = ""
    
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
        """Load the necessary emulator snapshot before testing."""
        load_snapshots(env)
        env.reset(go_home=True)
        super().initialize_task(env)

        command = 'adb -s emulator-5554 shell settings put system time_12_24 12'
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
        
        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.skuld.calendario', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
        # time.sleep(5.0)


# Operation Tasks
class CalendarAddEventAtTime(Calendar_Lab):
    """Task for adding an event at a specific time."""

    template = 'I want to add an event at 5:00PM today, whose Title is "work" in Calendar.'


class CalendarAddEventWithNotification(Calendar_Lab):
    """Task for adding an event with a specific notification time."""

    template = 'Arrange an event titled "homework" for me at May 21st, and set the notification time to be 10 minutes before in Calendar.'


class CalendarAddEventWithNoteAndDate(Calendar_Lab):
    """Task for adding an event with a note and specific date."""

    template = 'Help me arrange an event titled "meeting" at May 13th with note "conference room B202" in Calendar.'


class CalendarAddRecurringEvent(Calendar_Lab):
    """Task for arranging a recurring event."""

    template = 'Arrange an event which starts at 2024/6/1 and repeats monthly in Calendar.'


class CalendarEditEventEndTime(Calendar_Lab):
    """Task for editing an event's end time."""

    template = 'Edit the event with title "work", change the end time to be 7:00 PM in Calendar.'


class CalendarAddNoteToEvent(Calendar_Lab):
    """Task for adding a note to an event."""

    template = 'Add the note "classroom 101" to the event "homework" in Calendar.'


class CalendarEditNotificationTime(Calendar_Lab):
    """Task for editing the notification time of an event."""

    template = 'Change the notification time of event "meeting" to be 5 minutes before and 10 minutes before in Calendar.'


class CalendarEditEventAndAddNote(Calendar_Lab):
    """Task for editing an event and adding a note."""

    template = 'Edit the event titled "work" and add a Note "computer" to it in Calendar.'


class CalendarSetRecurrenceToDaily(Calendar_Lab):
    """Task for setting the recurrence of an event to daily."""

    template = 'For the event titled "work", please help me set recurrence to be daily in Calendar.'


class CalendarArrangeEventForToday(Calendar_Lab):
    """Task for arranging an event titled 'this day'."""

    template = 'Arrange an event "this day" in Calendar.'


class CalendarEditEventAndSetWeeklyRecurrence(Calendar_Lab):
    """Task for editing an event and setting it to repeat weekly."""

    template = 'Edit the event titled "this day", and make it repeat weekly in Calendar.'


class CalendarAddNoteToEventToday(Calendar_Lab):
    """Task for adding a note to an event titled 'Today'."""

    template = 'Help me add a note "Hello" to the event titled "Today" in Calendar.'


class CalendarArrangeEventForExam(Calendar_Lab):
    """Task for arranging an event titled 'exam'."""

    template = 'Arrange an event titled "exam" in Calendar.'


class CalendarEditEventAndSetAllDay(Calendar_Lab):
    """Task for editing an event and making it an all-day event."""

    template = 'Edit the event titled "exam" and make it an all-day event in Calendar.'
