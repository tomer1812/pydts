import numpy as np
from scipy.special import expit
from typing import Union
import pandas as pd


class EventTimesSampler(object):

    def __init__(self, d_times: int, j_event_types: int):
        """
        This class implements sampling procedure for discrete event times and censoring times for given observations.
        Args:
            d_times (int): number of possible event times
            j_event_types (int): number of possible event types
        """

        self.d_times = d_times
        self.times = range(1, self.d_times + 2)  # d + 1 for administrative censoring
        self.j_event_types = j_event_types
        self.events = range(1, self.j_event_types + 1)

    def _validate_prob_dfs_list(self, dfs_list: list, numerical_error_tolerance: float = 0.001) -> list:
        for df in dfs_list:
            if (((df < (0-numerical_error_tolerance)).any().any()) or ((df > (1+numerical_error_tolerance)).any().any())):
                raise ValueError("The chosen sampling parameters result in invalid probabilities for event j at time t")
            # Only fixes numerical errors smaller than the tolerance size
            df.clip(0, 1, inplace=True)
        return dfs_list

    def calculate_hazards(self, observations_df: pd.DataFrame, hazard_coefs: dict, events: list = None) -> list:
        """
        Calculates the hazard function for the observations given the hazard coefficients.

        Args:
            observations_df (pd.DataFrame): Dataframe with observations covariates.
            coefficients (dict): time coefficients and covariates coefficients for each event type.

        Returns:
            hazards_dfs (list): A list of dataframes, one for each event type, with the hazard function at time t to each of the observations.
        """
        events = events if events is not None else self.events
        a_t = {}
        for event in events:
            if callable(hazard_coefs['alpha'][event]):
                a_t[event] = {t: hazard_coefs['alpha'][event](t) for t in range(1, self.d_times + 1)}
            else:
                a_t[event] = {t: hazard_coefs['alpha'][event][t-1] for t in range(1, self.d_times + 1)}
        b = pd.concat([observations_df.dot(hazard_coefs['beta'][j]) for j in events], axis=1, keys=events)
        hazards_dfs = [pd.concat([expit((a_t[j][t] + b[j]).astype(float)) for t in range(1, self.d_times + 1)],
                                  axis=1, keys=(range(1, self.d_times + 1))) for j in events]
        return hazards_dfs

    def calculate_overall_survival(self, hazards: list, numerical_error_tolerance: float = 0.001) -> pd.DataFrame:
        """
        Calculates the overall survival function given the hazards
        Args:
            hazards (list): A list of hazards dataframes for each event type (as returned from EventTimesSampler.calculate_hazards function)
            numerical_error_tolerance (float): Tolerate numerical errors of probabilities up to this value.
        Returns:
            overall_survival (pd.Dataframe): The overall survival functions
        """
        if (((sum(hazards)) > (1 + numerical_error_tolerance)).sum().sum() > 0):
            raise ValueError("The chosen sampling parameters result in negative values of the overall survival function")
        sum_hazards = sum(hazards).clip(0, 1)
        overall_survival = pd.concat([pd.Series(1, index=hazards[0].index),
                                      (1 - sum_hazards).cumprod(axis=1).iloc[:, :-1]], axis=1)
        overall_survival.columns += 1
        return overall_survival

    def calculate_prob_event_at_t(self, hazards: list, overall_survival: pd.DataFrame,
                                  numerical_error_tolerance: float = 0.001) -> list:
        """
        Calculates the probability for event j at time t.

        Args:
            hazards (list): A list of hazards dataframes for each event type (as returned from EventTimesSampler.calculate_hazards function)
            overall_survival (pd.Dataframe): The overall survival functions
            numerical_error_tolerance (float): Tolerate numerical errors of probabilities up to this value.
        Returns:
            prob_event_at_t (list): A list of dataframes, one for each event type, with the probability of event occurrance at time t to each of the observations.
        """
        prob_event_at_t = [hazard * overall_survival for hazard in hazards]
        prob_event_at_t = self._validate_prob_dfs_list(prob_event_at_t, numerical_error_tolerance)
        return prob_event_at_t

    def calculate_prob_event_j(self, prob_j_at_t: list, numerical_error_tolerance: float = 0.001) -> list:
        """
        Calculates the total probability for event j.

        Args:
            prob_j_at_t (list): A list of dataframes, one for each event type, with the probability of event occurrance at time t to each of the observations.
            numerical_error_tolerance (float): Tolerate numerical errors of probabilities up to this value.

        Returns:
            total_prob_j (list): A list of dataframes, one for each event type, with the total probability of event occurrance to each of the observations.
        """
        total_prob_j = [prob.sum(axis=1) for prob in prob_j_at_t]
        total_prob_j = self._validate_prob_dfs_list(total_prob_j, numerical_error_tolerance)
        return total_prob_j

    def calc_prob_t_given_j(self, prob_j_at_t, total_prob_j, numerical_error_tolerance=0.001):
        """
        Calculates the conditional probability for event occurrance at time t given J_i=j
        Args:
            prob_j_at_t (list): A list of dataframes, one for each event type, with the probability of event occurrance at time t to each of the observations.
            total_prob_j (list): A list of dataframes, one for each event type, with the total probability of event occurrance to each of the observations.
            numerical_error_tolerance (float): Tolerate numerical errors of probabilities up to this value.
        Returns:
            conditional_prob (list): A list of dataframes, one for each event type, with the conditional probability of event occurrance at t given event type j to each of the observations.
        """
        conditional_prob = [prob.div(sumj, axis=0) for prob, sumj in zip(prob_j_at_t, total_prob_j)]
        conditional_prob = self._validate_prob_dfs_list(conditional_prob, numerical_error_tolerance)
        return conditional_prob

    def sample_event_times(self, observations_df: pd.DataFrame,
                                 hazard_coefs: dict,
                                 covariates: Union[list, None] = None,
                                 events: Union[list, None] = None,
                                 seed: Union[int, None] = None) -> pd.DataFrame:
        """
        Sample event type and event occurance times
        Args:
            observations_df (pd.DataFrame): Dataframe with observations covariates.
            covariates (list): list of covariates name, must be a subset of observations_df.columns
            coefficients (dict): time coefficients and covariates coefficients for each event type.
            seed (int, None): numpy seed number for pseudo random sampling.
        Returns:
            observations_df (pd.DataFrame): Dataframe with additional columns for sampled event time (T) and event type (J).
        """
        np.random.seed(seed)
        if covariates is None:
            covariates = [c for c in observations_df.columns if c not in ['X', 'T', 'C', 'J']]
        events = events if events is not None else self.events
        cov_df = observations_df[covariates]
        hazards = self.calculate_hazards(cov_df, hazard_coefs, events=events)
        overall_survival = self.calculate_overall_survival(hazards)
        probs_j_at_t = self.calculate_prob_event_at_t(hazards, overall_survival)
        total_prob_j = self.calculate_prob_event_j(probs_j_at_t)
        probs_t_given_j = self.calc_prob_t_given_j(probs_j_at_t, total_prob_j)
        sampled_jt = self.sample_jt(total_prob_j, probs_t_given_j)
        if 'J' in observations_df.columns:
            observations_df.drop('J', axis=1, inplace=True)
        if 'T' in observations_df.columns:
            observations_df.drop('T', axis=1, inplace=True)
        observations_df = pd.concat([observations_df, sampled_jt], axis=1)
        return observations_df

    def sample_jt(self, total_prob_j: list, probs_t_given_j: list, numerical_error_tolerance: float = 0.001) -> pd.DataFrame:
        """
        Sample event type and event time for each observation
        Args:
            total_prob_j (list): A list of dataframes, one for each event type, with the total probability of event occurrance to each of the observations.
            probs_t_given_j (list): A list of dataframes, one for each event type, with the conditional probability of event occurrance at t given event type j to each of the observations.

        Returns:
            sampled_df (pd.DataFrame): A dataframe with sampled event time and event type for each observation.
        """

        total_prob_j = self._validate_prob_dfs_list(total_prob_j, numerical_error_tolerance)
        probs_t_given_j = self._validate_prob_dfs_list(probs_t_given_j, numerical_error_tolerance)

        # Add administrative censoring (no event occured until Tmax) probability as J=0
        temp_sums = pd.concat([1 - sum(total_prob_j), *total_prob_j], axis=1, keys=[0, *self.events])
        if (((temp_sums < (0 - numerical_error_tolerance)).any().any()) or \
                ((temp_sums > (1 + numerical_error_tolerance)).any().any())):
            raise ValueError("The chosen sampling parameters result in invalid probabilities")
        # Only fixes numerical errors smaller than the tolerance size
        temp_sums.clip(0, 1, inplace=True)

        # Efficient way to sample j for each observation with different event probabilities
        sampled_df = (temp_sums.cumsum(1) > np.random.rand(temp_sums.shape[0])[:, None]).idxmax(axis=1).to_frame('J')

        temp_ts = []
        for j in self.events:
            # Get the index of the observations with J_i = j
            rel_j = sampled_df.query("J==@j").index

            # Get probs dataframe from the dfs list
            prob_df = probs_t_given_j[j - 1]  # the prob j to sample from

            # Sample time of occurrence given J_i = j
            temp_ts.append((prob_df.loc[rel_j].cumsum(1) >= np.random.rand(rel_j.shape[0])[:, None]).idxmax(axis=1))

        # Add Tmax+1 for observations with J_i = 0
        temp_ts.append(pd.Series(self.d_times + 1, index=sampled_df.query('J==0').index))
        sampled_df["T"] = pd.concat(temp_ts).sort_index()
        return sampled_df

    def sample_independent_lof_censoring(self, observations_df: pd.DataFrame,
                                         prob_lof_at_t: np.array, seed: Union[int, None] = None) -> pd.DataFrame:
        """
        Samples loss of follow-up censoring time from probabilities independent of covariates.
        Args:
            observations_df (pd.DataFrame): Dataframe with observations covariates.
            prob_lof_at_t (np.array): Array of probabilities for sampling each of the possible times.
            seed (int): pseudo random seed number for numpy.random.seed()

        Returns:
            observations_df (pd.DataFrame): Upadted dataframe including sampled censoring time.
        """
        np.random.seed(seed)
        administrative_censoring_prob = (1 - sum(prob_lof_at_t))
        assert (administrative_censoring_prob >= 0), "Check the sum of prob_lof_at_t argument."
        assert (administrative_censoring_prob <= 1), "Check the sum of prob_lof_at_t argument."

        prob_lof_at_t = np.append(prob_lof_at_t, administrative_censoring_prob)
        sampled_df = pd.DataFrame(np.random.choice(a=self.times, size=len(observations_df), p=prob_lof_at_t),
                                  index=observations_df.index, columns=['C'])
        # No follow-up censoring, C=d+2 such that T wins when building X column:
        #sampled_df.loc[sampled_df['C'] == self.times[-1], 'C'] = self.d_times + 2
        if 'C' in observations_df.columns:
            observations_df.drop('C', axis=1, inplace=True)
        observations_df = pd.concat([observations_df, sampled_df], axis=1)
        return observations_df

    def sample_hazard_lof_censoring(self, observations_df: pd.DataFrame, censoring_hazard_coefs: dict,
                                    seed: Union[int, None] = None,
                                    covariates: Union[list, None] = None) -> pd.DataFrame:
        """
        Samples loss of follow-up censoring time from hazard coefficients.
        Args:
            observations_df (pd.DataFrame): Dataframe with observations covariates.
            censoring_hazard_coefs (dict): time coefficients and covariates coefficients for the censoring hazard.
            seed (int): pseudo random seed number for numpy.random.seed()
            covariates (list): list of covariates names, must be a subset of observations_df.columns
        Returns:
            observations_df (pd.DataFrame): Upadted dataframe including sampled censoring time.
        """
        if covariates is None:
            covariates = [c for c in observations_df.columns if c not in ['X', 'T', 'C', 'J']]
        cov_df = observations_df[covariates]
        tmp_ets = EventTimesSampler(d_times=self.d_times, j_event_types=1)
        sampled_df = tmp_ets.sample_event_times(cov_df, censoring_hazard_coefs, seed=seed, covariates=covariates,
                                                events=[0])

        # No follow-up censoring, C=d+2 such that T wins when building X column:
        #sampled_df.loc[sampled_df['J'] == 0, 'T'] = self.d_times + 2
        sampled_df = sampled_df[['T']]
        sampled_df.columns = ['C']
        if 'C' in observations_df.columns:
            observations_df.drop('C', axis=1, inplace=True)
        observations_df = pd.concat([observations_df, sampled_df], axis=1)
        return observations_df

    def update_event_or_lof(self, observations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Updates time column 'X' to be the minimum between event time column 'T' and censoring time column 'C'.
        Event type 'J' will be changed to 0 for observation with 'C' < 'T'.

        Args:
            observations_df (pd.DataFrame): Dataframe with observations after sampling event times 'T' and censoring time 'C'.

        Returns:
            observations_df (pd.DataFrame): Dataframe with updated time column 'X' and event type column 'J'
        """
        assert ('T' in observations_df.columns), "Trying to update event or censoring before sampling event times"
        assert ('C' in observations_df.columns), "Trying to update event or censoring before sampling censoring time"
        observations_df['X'] = observations_df[['T', 'C']].min(axis=1)
        observations_df.loc[observations_df.loc[(observations_df['C'] < observations_df['T'])].index, 'J'] = 0
        return observations_df
