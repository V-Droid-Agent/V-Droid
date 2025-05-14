from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class Cantook_Lab(task_eval.TaskEval):
    """Base class for Cantook tasks."""

    app_names = ("cantook",)
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
        
        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.aldiko.android', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
        # time.sleep(5.0)
            


# Query Tasks
class CantookQueryBookOnShelf(Cantook_Lab):
    """Task for checking if a book exists on the bookshelf."""

    template = "Do I have Pride and Prejudice on my bookshelf in Cantook?"


class CantookQueryLastReadBook(Cantook_Lab):
    """Task for retrieving the last recently read book."""

    template = "What was the last book I recently read in Cantook?"


class CantookQueryAuthorOfRecentlyAdded(Cantook_Lab):
    """Task for retrieving the author of the second book in the recently added section."""

    template = "Who is the author of the second book in my recently added in Cantook?"


class CantookQueryAuthorBookCount(Cantook_Lab):
    """Task for counting the number of books by a specific author."""

    template = "How many Charles Dickens books do I have in Cantook?"


class CantookQueryReadingProgress(Cantook_Lab):
    """Task for checking the reading progress of a specific book."""

    template = "What is my progress in reading Romeo and Juliet in Cantook?"


# Operation Tasks
class CantookImportBook(Cantook_Lab):
    """Task for importing an eBook from a specific folder."""

    template = "Import Alice's Adventures in Wonderland from folder /Download/Ebooks/ in Cantook."


class CantookDeleteBook(Cantook_Lab):
    """Task for deleting a book from the collection."""

    template = "Delete Don Quixote from my books in Cantook."


class CantookMarkBookAsRead(Cantook_Lab):
    """Task for marking a book as read."""

    template = "Mark Hamlet as read in Cantook."


class CantookMarkBookAsUnread(Cantook_Lab):
    """Task for marking a recently read book as unread."""

    template = "Mark the second book I recently read as unread in Cantook."


class CantookOpenBook(Cantook_Lab):
    """Task for opening a specific book."""

    template = "Open Romeo and Juliet in Cantook."


class CantookOpenCategory(Cantook_Lab):
    """Task for opening a book category."""

    template = "Open the category named 'Tragedies' in Cantook."


class CantookCreateCollection(Cantook_Lab):
    """Task for creating a new book collection."""

    template = 'Create a new collection called "Favorite" in Cantook.'
