import dataclasses
import random
from typing import Any
from absl import logging
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.utils import fuzzy_match_lib
import immutabledict
import subprocess
import time
from util import load_snapshots

class System_Lab(task_eval.TaskEval):
  """Validator for checking that a contact has been added."""

  app_names = ()
  complexity = 1
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
    load_snapshots(env)
    env.reset(go_home=True)
    super().initialize_task(env)
    command = 'adb -s emulator-5554 shell settings put system time_12_24 12'
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")

    # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.android.settings', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
    # time.sleep(5.0)


class SystemAirplaneMode(System_Lab):
  """Task for checking that airplane has been turned {on_or_off}."""

  app_names = ('settings',)
  template = 'Turn off airplane mode of my phone in the Settings App.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'airplane_mode_on'], env.controller
    )
    bluetooth_status = res.generic.output.decode().strip()
    expected_status = '0'
    return 1.0 if bluetooth_status == expected_status else 0.0


class SystemTurnWifiOffAuto(System_Lab):
    app_names = ('settings',)
    template = 'I do not want turn on wifi automatically, turn it off in the Settings App.'


class SystemSetDNS(System_Lab):
    app_names = ('settings',)
    template = 'set private DNS to dns.google in the Settings App.'


class SystemTurnBluetoothOff(System_Lab):
    app_names = ('settings',)
    template = 'Turn off my bluetooth in the Settings App.'

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        res = adb_utils.issue_generic_request(
            ['shell', 'settings', 'get', 'global', 'bluetooth_on'], env.controller
        )
        bluetooth_status = res.generic.output.decode().strip()
        expected_status = '0'
        return 1.0 if bluetooth_status == expected_status else 0.0

class SystemChangeBluetoothName(System_Lab):
    app_names = ('settings',)
    template = 'Change my bluetooth device name to "my AVD" in the Settings App.'

class SystemShowBatteryPercentage(System_Lab):
    app_names = ('settings',)
    template = 'Show battery percentage in status bar in the Settings App.'


class SystemStorage(System_Lab):
    app_names = ('settings',)
    template = 'How much storage does Apps use. Please check in the Settings App.'


class SystemDarkTheme(System_Lab):
    app_names = ('settings',)
    template = 'Turn my phone to Dark theme in the Settings App.'

    def tear_down(self, env):
        command = 'adb shell "cmd uimode night no"'
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
        return super().tear_down(env)

class SystemBrightnessLevel(System_Lab):
    app_names = ('settings',)
    template = 'Change my Brightness level to 0% in the Settings App.'

class SystemRingVolume(System_Lab):
    app_names = ('settings',)
    template = 'I need to close down my Ring & notification volume to 0% in the Settings App.'

class SystemAlarmVolum(System_Lab):
    app_names = ('settings',)
    template = 'Set my alarm volume to max in the Settings App.'

class SystemText2Speech(System_Lab):
    app_names = ('settings',)
    template = 'Change text-to-speech language to Chinese in the Settings App.'

class SystemSetTime(System_Lab):
    app_names = ('settings',)
    template = 'Set current time of my phone to 2024-5-1 in the Settings App.'

#
# setting_13
# category: Sound
# task: Turn off Ring vibration
#
class SystemTurnOffRingVibration(System_Lab):
    app_names = ('settings',)
    template = 'Turn off Ring vibration in the Settings App.'

    # def is_successful(self, env: interface.AsyncEnv) -> float:
    #     super().is_successful(env)
    #     # Check 'vibrate_when_ringing' setting; expected to be '0' if off.
    #     res = adb_utils.issue_generic_request(
    #         ['shell', 'settings', 'get', 'system', 'vibrate_when_ringing'],
    #         env.controller
    #     )
    #     vibrate_status = res.generic.output.decode().strip()
    #     return 1.0 if vibrate_status == '0' else 0.0


#
# setting_14
# category: Time
# task: What is my time zone
#
class SystemCheckTimeZone(System_Lab):
    app_names = ('settings',)
    template = 'What is my time zone? Please check in the Settings App.'

    # def is_successful(self, env: interface.AsyncEnv) -> float:
    #     super().is_successful(env)
    #     res = adb_utils.issue_generic_request(
    #         ['shell', 'getprop', 'persist.sys.timezone'],
    #         env.controller
    #     )
    #     timezone = res.generic.output.decode().strip()
    #     # If we can retrieve a non-empty timezone, we consider it successful.
    #     return 1.0 if timezone else 0.0


#
# setting_15
# category: Language
# task: Add Español (Estados Unidos) as second favorite language
#
class SystemAddEspanolLanguage(System_Lab):
    app_names = ('settings',)
    template = 'Add Español (Estados Unidos) as second favorite language in the Settings App.'


#
# setting_16
# category: Language
# task: What is the primary language of my phone
#
class SystemCheckPrimaryLanguage(System_Lab):
    app_names = ('settings',)
    template = 'What is the primary language of my phone? Please check in the Settings App.'

    # def is_successful(self, env: interface.AsyncEnv) -> float:
    #     super().is_successful(env)
    #     # On many devices, persist.sys.locale or ro.product.locale might store the default language.
    #     # The property name can vary by device/Android version.
    #     res = adb_utils.issue_generic_request(
    #         ['shell', 'getprop', 'persist.sys.locale'],
    #         env.controller
    #     )
    #     locale = res.generic.output.decode().strip()
    #     return 1.0 if locale else 0.0


#
# setting_17
# category: Language
# task: Check Android Version
#
class SystemCheckAndroidVersion(System_Lab):
    app_names = ('settings',)
    template = 'Check the Android Version on my phone in the Settings App.'

    # def is_successful(self, env: interface.AsyncEnv) -> float:
    #     super().is_successful(env)
    #     # Checking ro.build.version.release
    #     res = adb_utils.issue_generic_request(
    #         ['shell', 'getprop', 'ro.build.version.release'],
    #         env.controller
    #     )
    #     version = res.generic.output.decode().strip()
    #     return 1.0 if version else 0.0


#
# setting_18
# category: App notifications
# task: Disable Contacts' APP notifications
#
class SystemDisableContactsNotifications(System_Lab):
    app_names = ('settings',)
    template = "Disable the 'Contacts' app notifications in the Settings App."

#
# setting_19
# category: APP
# task: Check my default browser and change it to firefox
#
class SystemChangeDefaultBrowserToFirefox(System_Lab):
    app_names = ('settings',)
    template = 'Check my default browser and change it to Firefox in the Settings App.'


#
# setting_20
# category: APP
# task: uninstall booking app

class SystemUninstallBookingApp(System_Lab):
    app_names = ('settings',)
    template = 'Uninstall Booking app in the Settings App.'

    def initialize_task(self, env: interface.AsyncEnv):
        super().initialize_task(env)
        booking_apk_path = './lab_app/Booking.apk'

        res = adb_utils.issue_generic_request(
            ['shell', 'pm', 'list', 'packages'],
            env.controller
        )
        packages = res.generic.output.decode()
        if 'com.booking' not in packages:
            adb_utils.issue_generic_request(
                ['install', booking_apk_path],
                env.controller
            )

    def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        # Check if com.booking is still installed
        res = adb_utils.issue_generic_request(
            ['shell', 'pm', 'list', 'packages'],
            env.controller
        )
        packages = res.generic.output.decode()
        
        
        # If not found, success
        if 'com.booking' not in packages:
            return 1.0
        else:
            adb_utils.issue_generic_request(
                ['shell', 'pm', 'uninstall', 'com.booking'],
                env.controller
            )
            return 0.0


#
# setting_21
# category: APP
# task: Open settings
#
class SystemOpenSettings(System_Lab):
    app_names = ('settings',)
    template = 'Open the Settings app.'

#
# setting_22
# category: APP
# task: Does my airplane mode open or not
#
class SystemCheckAirplaneModeOn(System_Lab):
    app_names = ('settings',)
    template = 'Does my airplane mode open or not? Please check in the Settings App.'

    # def is_successful(self, env: interface.AsyncEnv) -> float:
    #     super().is_successful(env)
    #     res = adb_utils.issue_generic_request(
    #         ['shell', 'settings', 'get', 'global', 'airplane_mode_on'],
    #         env.controller
    #     )
    #     airplane_status = res.generic.output.decode().strip()
    #     # Return 1.0 if we can detect either on (1) or off (0).
    #     # In a real scenario, you might want more nuanced logic.
    #     if airplane_status in ['0', '1']:
    #         return 1.0
    #     return 0.0