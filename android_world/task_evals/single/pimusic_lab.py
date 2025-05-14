from typing import Any
from android_world.env import adb_utils, device_constants, tools
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class PiMusic_Lab(task_eval.TaskEval):
    """Base class for Pi Music Player tasks."""

    app_names = ("Pi Music Player",)
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
        # env.reset(go_home=True)
        super().initialize_task(env)
        command = 'adb -s emulator-5554 shell settings put system time_12_24 12'
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command succeeded: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")

        try:
            controller = tools.AndroidToolController(env=env.controller)
            adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.Project100Pi.themusicplayer', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
            time.sleep(2.0)
            controller.click_element('Allow')
            time.sleep(2.0)
        except:
            print("fail to clear ")
        
        env.reset(go_home=True)
        time.sleep(2.0)

# Query Tasks
class PiMusicQueryTotalSongs(PiMusic_Lab):
    """Task for checking the total number of songs in the library."""

    template = "Tell me how many songs do I have in total in Pi Music Player?"


class PiMusicQueryArtistSongCount(PiMusic_Lab):
    """Task for checking the number of songs by a specific artist."""

    template = "Help me check how many Pink Floyd's songs do I have in Pi Music Player?"


class PiMusicQueryAlbumBySong(PiMusic_Lab):
    """Task for retrieving the album name of a specific song."""

    template = "What is the album name of the song Wish You Were Here in Pi Music Player?"


class PiMusicQueryLongestSongDuration(PiMusic_Lab):
    """Task for checking the duration of the longest song by a specific artist."""

    template = "What is the duration time of the longest song by Pink Floyd in Pi Music Player?"


class PiMusicQuerySortedSongTitles(PiMusic_Lab):
    """Task for sorting songs by title and retrieving specific positions."""

    template = "Sort the songs by title in ascending order. What are the second and fourth songs in Pi Music Player?"


class PiMusicQueryTotalDurationByArtist(PiMusic_Lab):
    """Task for checking the total duration of all songs by a specific artist."""

    template = "What is the total duration time of all of Eason Chan's songs in Pi Music Player?"


# Operation Tasks
class PiMusicPlayFirstFavoriteSong(PiMusic_Lab):
    """Task for playing the first song in a specific playlist."""

    template = "Play the first song in 'Favorite' playlist in Pi Music Player."


class PiMusicSortSongsByDurationDescending(PiMusic_Lab):
    """Task for sorting songs by duration in descending order."""

    template = "Sort Pink Floyd's songs by duration time in descending order in Pi Music Player."


class PiMusicCreatePlaylist(PiMusic_Lab):
    """Task for creating a new playlist."""

    template = "Create a playlist named 'Creepy' for me in Pi Music Player."


class PiMusicPauseAndSeek(PiMusic_Lab):
    """Task for pausing a song and seeking to a specific timestamp."""

    template = "Pause the currently playing song and seek to 1 minute and 27 seconds in Pi Music Player."


class PiMusicPlaySpecificSong(PiMusic_Lab):
    """Task for playing a specific song by an artist."""

    template = "Play Lightship by Sonny Boy in Pi Music Player."


class PiMusicSortSongsByDurationAscending(PiMusic_Lab):
    """Task for sorting songs by duration in ascending order."""

    template = "Sort the songs by duration time in ascending order in Pi Music Player."



