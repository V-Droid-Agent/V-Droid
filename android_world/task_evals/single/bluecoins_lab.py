from typing import Any
from android_world.env import adb_utils, device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
import pdb
import time
import subprocess

from util import load_snapshots

class Bluecoins_Lab(task_eval.TaskEval):
    """Base class for Bluecoins tasks."""

    app_names = ("bluecoins",)
    complexity = 2
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    template = ""

    device_time = device_constants.DT_LAB

    def is_successful(self, env: interface.AsyncEnv) -> float:
        return 0.0

    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def set_device_time(cls, env: interface.AsyncEnv) -> None:
        """Sets the device time."""
        del env
        cls.device_time = device_constants.DT_LAB

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
        
        # adb_utils.issue_generic_request(['shell', 'monkey', '-p', 'com.rammigsoftware.bluecoins', '-c', 'android.intent.category.LAUNCHER', '1'], env.controller)
        # time.sleep(5.0)
        
        

# Query Tasks
class BluecoinsQueryDailySpending(Bluecoins_Lab):
    """Task for querying the total spending on a specific date."""

    template = "Could you tell me how much I spent on May 10, 2024 in Bluecoins?"


class BluecoinsQuerySpendingReason(Bluecoins_Lab):
    """Task for querying the reason behind a specific transaction."""

    template = "What was the reason behind the 388.88 CNY I spent on May 3, 2024 in Bluecoins?"


class BluecoinsQueryTotalSpending(Bluecoins_Lab):
    """Task for querying the total expenditure on a date."""

    template = "How much did I shell out in total on May 6, 2024 in Bluecoins?"


class BluecoinsQueryTransactionCount(Bluecoins_Lab):
    """Task for querying the total number of transactions on a specific date."""

    template = "How many transactions did I make all together on May 6, 2024 in Bluecoins?"


class BluecoinsQueryCategorySpending(Bluecoins_Lab):
    """Task for querying the total amount spent on a category."""

    template = "What's the total amount I spent on taxis this week in Bluecoins?"


# Operation (Create) Tasks
class BluecoinsLogExpenditure(Bluecoins_Lab):
    """Task for logging a new expenditure."""

    template = "Log an expenditure of 512 CNY in the books in Bluecoins."


class BluecoinsLogIncomeWithLabel(Bluecoins_Lab):
    """Task for recording an income and labeling it."""

    template = "Record an income of 8000 CNY in the books, and mark it as 'salary' in Bluecoins."


class BluecoinsLogDatedExpense(Bluecoins_Lab):
    """Task for noting down an expense on a specific date."""

    template = "Note down an expense of 768 CNY for May 11, 2024 in Bluecoins."


class BluecoinsLogDatedIncomeWithNote(Bluecoins_Lab):
    """Task for recording an income entry on a specific date with a note."""

    template = "For March 8, 2024, jot down an income of 3.14 CNY with 'Weixin red packet' as the note in Bluecoins."


class BluecoinsLogLabeledExpenditure(Bluecoins_Lab):
    """Task for recording an expenditure with a label."""

    template = "For May 14, 2024, record an expenditure of 256 CNY, marked as 'eating' in Bluecoins."


# Operation (Edit) Tasks
class BluecoinsEditExpenditure(Bluecoins_Lab):
    """Task for modifying an expenditure entry."""

    template = "Adjust the expenditure on May 15, 2024, to 500 CNY in Bluecoins."


class BluecoinsEditIncomeDateAndAmount(Bluecoins_Lab):
    """Task for shifting an income entry to another date and updating its amount."""

    template = "Shift the income entry from May 12th, 2024, to May 10th, 2024, and update the amount to 18,250 CNY in Bluecoins."


class BluecoinsEditTransactionType(Bluecoins_Lab):
    """Task for changing the type of a transaction and adding a note."""

    template = "Switch the May 13, 2024, transaction from 'expense' to 'income' and add 'Gift' as the note in Bluecoins."


class BluecoinsEditTransactionCategory(Bluecoins_Lab):
    """Task for changing the transaction type, adjusting the amount, and modifying the note."""

    template = "Change the type of the transaction on May 2, 2024, from 'income' to 'expense', adjust the amount to 520 CNY, and change the note to 'Wrong Operation' in Bluecoins."


class BluecoinsEditExpenseEntry(Bluecoins_Lab):
    """Task for modifying an expense entry's date, amount, and note."""

    template = "Move the expense entry from May 12, 2024, to May 13, 2024, adjust the amount to 936.02 CNY, and update the note to 'Grocery Shopping' in Bluecoins."
