# Introduction

Based on "PyDTS: A Python Package for Discrete Time Survival-analysis with Competing Risks" 
Tomer Meir*, Rom Gutman*, and Malka Gorfine (2022).


Discrete-data survival analysis refers to the case where data can only take values over a discrete grid. Sometimes, events can only occur at regular, discrete points in time. For example, in the United States a change in party controlling the presidency only occurs quadrennially in the month of January \cite{allison_discrete-time_1982}. In other situations events may occur at any point in time, but available data record only the particular interval of time in which each event occurs. For example, death from cancer measured by months since time of diagnosis \cite{lee_analysis_2018}, or length of stay in hospital recorded on a daily basis. It is well-known that naively using standard continuous-time models (even after correcting for ties) with discrete-time data may result in biased estimators for the discrete time models.

Competing events arise when individuals are susceptible to several types of events but can experience at most one event. For example, competing risks for hospital length of stay are discharge and in-hospital death. Occurrence of one of these events precludes us from observing the other event on this patient. Another classical example of competing risks is cause-specific mortality, such as death from heart disease, death from cancer and death from other causes \cite{kalbfleisch_statistical_2011,klein_survival_2003}. 

PyDTS is an open source Python package which implements tools for discrete-time survival analysis with competing risks.


## References


