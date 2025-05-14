
"""Task for contacts apps."""

import random
import re
from absl import logging
from typing import Any
from android_world.env import device_constants, interface, tools
from android_world.env import representation_utils
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import contacts_validators
from android_world.task_evals.utils import user_data_generation
from android_world.utils import contacts_utils, fuzzy_match_lib
from android_world.env import adb_utils
import time
import subprocess
import pdb

from util import load_snapshots, refresh_env_with_retries

class AddContact_Lab(task_eval.TaskEval):
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

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    load_snapshots(env)
    env.reset(go_home=True)
    super().initialize_task(env)
    command = 'adb -s emulator-5554 shell settings put system time_12_24 12'
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Command succeeded: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")

      
    try:
      controller = tools.AndroidToolController(env=env.controller)
      adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.google.android.contacts', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
      time.sleep(2.0)
      controller.click_element('Contacts')
      time.sleep(2.0)
    except:
       print("fail to clear ")

    # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.google.android.contacts', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
    # time.sleep(5.0)


  def _has_contact(self, contacts: list[contacts_utils.Contact]) -> bool:
    return (
        contacts_utils.Contact(
            self.params['name'],
            contacts_utils.clean_phone_number(self.params['number']),
        )
        in contacts
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    # contact_found = self._has_contact(
    #     contacts_utils.list_contacts(env.controller)
    # )
    # return super().is_successful(env) if contact_found else 0.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {}

class ContactsAddSimpleContact(AddContact_Lab):
  """Task for adding a new contact."""

  app_names = ("contacts",)
  template = 'Add "John" as a contacts and set his mobile phone number to be "12345678" in Contacts.'


class ContactsAddContactEmail(AddContact_Lab):
  """Task for adding a new contact."""

  app_names = ("contacts",)
  template = "Add a contacts whose first name is John, last name is Smith, mobile phone number is 12345678, and working email as 123456@gmail.com in Contacts."


class ContactsAddTwoPhone(AddContact_Lab):
  """Task for adding a new contact."""

  app_names = ("contacts",)
  template = "Add a contacts whose name is Xu, set the working phone number to be 12345678 and mobile phone number to be 87654321 in Contacts."

class ContactsAddNameCompany(AddContact_Lab):
  """Task for adding a new contact."""

  app_names = ("contacts",)
  template = "Add a contacts named Chen, whose company is Tsinghua University in Contacts."
  
class ContactsAddLabel(AddContact_Lab):
  """Task for adding a new contact."""

  app_names = ("contacts",)
  template = "Create a new label as work, and add AAA、ABC into it in Contacts."


class ContactsAddWorkPhone(AddContact_Lab):
    """Task for modifying a contact by adding a work phone number."""

    app_names = ("contacts",)
    template = "Add a work phone number 00112233 to contacts ABC in Contacts."


class ContactsAddBirthday(AddContact_Lab):
    """Task for modifying a contact by adding a birthday."""

    app_names = ("contacts",)
    template = "Add birthday to AAA as 1996/10/24 in Contacts."


class ContactsSetWebsite(AddContact_Lab):
    """Task for modifying a contact by setting their website."""

    app_names = ("contacts",)
    template = "Set contacts ABC's website to be abc.github.com in Contacts."


class ContactsEditMessage(AddContact_Lab):
    """Task for composing a message without sending it."""

    app_names = ("contacts",)
    template = "Edit a message to ABC in Contacts, whose content is 'Nice to meet you', but do not send it."


class ContactsCallContact(AddContact_Lab):
    """Task for calling a contact."""

    app_names = ("contacts",)
    template = "Call ABC in Contact"
      
    def tear_down(self, env):
       adb_utils.end_call_if_active(env.controller)
       return super().tear_down(env)

class ContactsDeleteContact(AddContact_Lab):
    """Task for deleting a contact."""

    app_names = ("contacts",)
    template = "Delete contacts AAA in Contacts."


class ContactsCheckPhoneNumber(AddContact_Lab):
    """Task for checking a contact's phone number."""

    app_names = ("contacts",)
    template = "What is ABC's phone number in Contacts?"


class ContactsCheckEmail(AddContact_Lab):
    """Task for checking a contact's working email."""

    app_names = ("contacts",)
    template = "What is Li's working email in Contacts?"


class ContactsCheckBirthday(AddContact_Lab):
    """Task for checking a contact's birthday."""

    app_names = ("contacts",)
    template = "When is ABC's birthday in Contacts?"


class ContactsCheckCompany(AddContact_Lab):
    """Task for checking a contact's company."""

    app_names = ("contacts",)
    template = "What is AAA's company in Contacts?"

