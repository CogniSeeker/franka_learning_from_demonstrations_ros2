import time
from typing import Tuple

class RiskPolicy:
    def __init__(self, patience: int = 2):
        self.patience = patience
        self.prev_risky_estimations = 0

    def do(self, lfd, clf_risk: bool, human_risk: bool) -> Tuple[str, float]:
        return "continue", lfd.time_phase
    
    def merge_risks(self, clf_risk: bool, human_risk: bool) -> bool:
        """Either risk activates risk policy.

        Args:
            clf_risk (bool): Risk Estimator Risk
            human_risk (bool): Human Gated Estimator (Keyboard or Franka Button, etc..)

        Returns:
            bool: Merged Risk
        """        
        return clf_risk or human_risk

    def apply_patience(self, risk):
        if risk:
            self.prev_risky_estimations += 1
            if self.prev_risky_estimations >= self.patience:
                return True
        else:
            self.prev_risky_estimations = 0

    def wait_for_feedback(self, lfd):
        while True:  # Waiting for flag
            time.sleep(0.5)
            print("Stopped on risk, waiting for flag")
            if lfd.end:
                return "quit", 1.0

            if lfd.risk_flag:
                alpha = 0.0
                return "repeat", alpha
            elif lfd.safe_flag:
                return "continue", lfd.time_phase
            elif lfd.recovery_phase != -1.0:
                return "continue", lfd.recovery_phase


class ContinueRiskPolicy(RiskPolicy):
    """Execution continues on Risk detection."""
    pass


class AbortRiskPolicy(RiskPolicy):
    """Execution is stopped on Risk detection. No repeating."""

    def do(self, lfd, clf_risk: bool, human_risk: bool) -> Tuple[str, float]:
        if self.apply_patience(self.merge_risks(clf_risk, human_risk)):
            lfd.communicate_risk()
            return "quit", 1.0 # point to end
        else:
            return "continue", lfd.time_phase


class WaitForFeedbackRiskPolicy(RiskPolicy):
    """Execution is stopped on Risk detection.
    When Human Safe flag is observed, execution continues.
    When Human Risk flag is observed, demonstration is repeated.
    """

    def do(self, lfd, clf_risk: bool, human_risk: bool) -> Tuple[str, float]:
        if self.apply_patience(self.merge_risks(clf_risk, human_risk)):
            lfd.communicate_risk_detected()
            return self.wait_for_feedback(lfd)
        else:
            return "continue", lfd.time_phase

