import pandas as pd
from pydts.utils import get_expanded_df


class BaseFitter:
    """
    This class implements the basic fitter methods and attributes api
    """

    def __init__(self):
        self.event_models = {}
        self.expanded_df = pd.DataFrame()
        self.event_type_col = None
        self.duration_col = None
        self.pid_col = None
        self.events = None
        self.covariates = None
        self.formula = None

    def fit(self, df: pd.DataFrame, event_type_col: str = 'J', duration_col: str = 'X', pid_col: str = 'pid',
            **kwargs) -> dict:
        raise NotImplemented

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplemented

    def evaluate(self, test_df: pd.DataFrame, oracle_col: str = 'T', **kwargs) -> float:
        raise NotImplemented

    def print_summary(self, **kwargs) -> None:
        raise NotImplemented


class ExpansionBasedFitter(BaseFitter):
    """
    This class implements the data expansion method which is common for the existing fitters
    """

    @staticmethod
    def _expand_data(df: pd.DataFrame, event_type_col: str, duration_col: str, pid_col: str) -> pd.DataFrame:
        """
        This method expands the raw data as explained in Lee et al. 2018

        Args:
            df (pandas.DataFrame): Dataframe to expand.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).

        Returns:
            Expanded df (pandas.DataFrame): the expanded dataframe.
        """
        return get_expanded_df(df=df, event_type_col=event_type_col, duration_col=duration_col, pid_col=pid_col)
