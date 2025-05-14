from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class MapMe_Lab(task_eval.TaskEval):
    """Base class for Map.me tasks."""

    app_names = ("mapme",)
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

        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.mapswithme.maps.pro', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
        # time.sleep(5.0)



# Query Tasks
class MapMeQueryWalkingDistance(MapMe_Lab):
    """Task for checking walking distance and time between two locations."""

    template = "Check the walking distance and time between Bus Stop of Stanford Campus Oval and Bus Stop of Oxford Street & University Avenue in MAPS.ME."


class MapMeQueryDrivingDistance(MapMe_Lab):
    """Task for checking driving distance and time between two locations."""

    template = "Check the driving distance and time between Bus stop of 2700 Coast Avenue and Bus Stop Route 51 in MAPS.ME."


class MapMeQueryRidingTime(MapMe_Lab):
    """Task for checking the riding time between two locations."""

    template = "Check the riding time between Bus Stop of Stanford Campus Oval and Bus Stop of Oxford Street & University Avenue in MAPS.ME."


class MapMeQueryPublicTransportRoute(MapMe_Lab):
    """Task for checking a public transportation route between two locations."""

    template = "Check the route by public transportation between Bus stop of 2700 Coast Avenue and Bus Stop Route 51 in MAPS.ME."


class MapMeCompareTravelTimeRideVsTransport1(MapMe_Lab):
    """Task for comparing travel times between riding and public transport for a specific route."""

    template = "Compare which takes less time to travel between Bus stop of 2700 Coast Avenue and Bus Stop Route 51, by riding or by public transportation in MAPS.ME?"


class MapMeCompareTravelTimeRideVsTransport2(MapMe_Lab):
    """Task for comparing travel times between riding and public transport for another route."""

    template = "Compare which takes less time to travel between Bus Stop of Stanford Campus Oval and Bus Stop of Oxford Street & University Avenue, by riding or by public transportation in MAPS.ME?"


class MapMeQueryNearestRestaurant(MapMe_Lab):
    """Task for finding the nearest restaurant."""

    template = "Check the nearest restaurant and tell me what is it in MAPS.ME."


class MapMeQueryNearestRestaurantWalkingTime(MapMe_Lab):
    """Task for finding the nearest restaurant and walking time to it."""

    template = "Check the nearest restaurant, and tell me the time it will take to walk to the restaurant in MAPS.ME."


class MapMeQueryNearestHotel(MapMe_Lab):
    """Task for finding the nearest hotel."""

    template = "Check the nearest hotel, tell me what is it in MAPS.ME."


class MapMeQueryNearestIKEA(MapMe_Lab):
    """Task for finding the nearest IKEA and driving time to it."""

    template = "Check the nearest IKEA, and tell me how long it will take to drive to the IKEA in MAPS.ME."


# Operation Tasks
class MapMeAddWorkAddress(MapMe_Lab):
    """Task for adding an address to the Work place."""

    template = "Add the address of OpenAI to my Work place in MAPS.ME."


class MapMeNavigateToStanford(MapMe_Lab):
    """Task for navigating to Stanford University."""

    template = "Navigate from my location to Stanford University in MAPS.ME."


class MapMeNavigateToUniversitySouth(MapMe_Lab):
    """Task for navigating to University South."""

    template = "Navigate from my location to University South in MAPS.ME."


class MapMeNavigateToOpenAI(MapMe_Lab):
    """Task for navigating to OpenAI."""

    template = "Navigate from my location to OpenAI in MAPS.ME."


class MapMeNavigateToUC_Berkeley(MapMe_Lab):
    """Task for navigating to University of California, Berkeley."""

    template = "Navigate from my location to University of California, Berkeley in MAPS.ME."
