from tasks.alert_dedup import AlertDedupTask
from tasks.expense_validator import ExpenseValidatorTask
from tasks.scheduler import SchedulerTask
from tasks.dependency_resolver import DependencyResolverTask
from tasks.log_classifier import LogClassifierTask
from tasks.contract_checker import ContractCheckerTask

ALL_TASKS = [
    AlertDedupTask(),
    ExpenseValidatorTask(),
    # Below are stubs — uncomment when implemented:
    # SchedulerTask(),
    # DependencyResolverTask(),
    # LogClassifierTask(),
    # ContractCheckerTask(),
]
