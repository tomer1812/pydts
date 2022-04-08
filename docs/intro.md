# Introduction

Based on 

"PyDTS: A Python Package for Discrete Time Survival-analysis with Competing Risks"

Tomer Meir\*, Rom Gutman\*, and Malka Gorfine (2022) [[1]](#1).

## Discrete-data survival analysis
Discrete-data survival analysis refers to the case where data can only take values over a discrete grid. Sometimes, events can only occur at regular, discrete points in time. For example, in the United States a change in party controlling the presidency only occurs quadrennially in the month of January [[2]](#2). In other situations events may occur at any point in time, but available data record only the particular interval of time in which each event occurs. For example, death from cancer measured by months since time of diagnosis [[3]](#3), or length of stay in hospital recorded on a daily basis. It is well-known that naively using standard continuous-time models (even after correcting for ties) with discrete-time data may result in biased estimators for the discrete time models.

## Competing events

Competing events arise when individuals are susceptible to several types of events but can experience at most one event. For example, competing risks for hospital length of stay are discharge and in-hospital death. Occurrence of one of these events precludes us from observing the other event on this patient. Another classical example of competing risks is cause-specific mortality, such as death from heart disease, death from cancer and death from other causes [[4, 5]](#4#5). 


PyDTS is an open source Python package which implements tools for discrete-time survival analysis with competing risks.


## References
<a id="1">[1]</a> 
Tomer Meir\*, Rom Gutman\*, and Malka Gorfine 
"PyDTS: A Python Package for Discrete Time Survival-analysis with Competing Risks"
(2022)

<a id="2">[2]</a> 
Allison, Paul D.
"Discrete-Time Methods for the Analysis of Event Histories"
Sociological Methodology (1982),
doi: 10.2307/270718

<a id="3">[3]</a> 
Lee, Minjung and Feuer, Eric J. and Fine, Jason P.
"On the analysis of discrete time competing risks data"
Biometrics (2018)
doi: 10.1111/biom.12881

<a id="4">[4]</a> 
Kalbfleisch, John D. and Prentice, Ross L.
"The Statistical Analysis of Failure Time Data" 2nd Ed.,
Wiley (2011)
ISBN: 978-1-118-03123-0

<a id="5">[5]</a> 
Klein, John P. and Moeschberger, Melvin L.
"Survival Analysis",
Springer (2003)
ISBN: 978-0-387-95399-1
