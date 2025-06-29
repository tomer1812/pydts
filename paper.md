[---
title: 'PyDTS: A Python Package for Discrete-Time Survival Analysis with Competing Risks and Optional Penalization' 
tags:
  - Python
  - Discrete-time
  - Survival Analysis
  - Competing Events
  - Regularized Regression
  - Sure Independence Screening
authors:
  - name: Tomer Meir^[Corresponding author]
    orcid: 0000-0002-3208-8460
    affiliation: "1" 
  - name: Rom Gutman
    orcid: 0000-0002-4125-7545
    affiliation: "1"
  - name: Malka Gorfine
    orcid: 0000-0002-1577-6624
    affiliation: "2"
affiliations:
 - name: Technion - Israel Institute of Technology, Haifa, 3200003, Israel
   index: 1
 - name: Department of Statistics and Operations Research, Tel Aviv University, Tel Aviv, 6997801, Israel
   index: 2
date: 29 June 2025
bibliography: paper.bib
---

# Summary
Time-to-event (survival) analysis models  the time until a pre-specified event occurs. When time is measured in  discrete units  or rounded into intervals, standard continuous-time models can yield biased estimators. In addition, the event of interest may belong to one of several mutually exclusive types, referred to as competing risks, where the occurrence of one event prevents the occurrence or observation of the others.  *PyDTS* is an open-source Python package for analyzing discrete-time survival data with competing-risks. It provides regularized estimation methods, model evaluation metrics, variable screening tools, and a simulation module to support research and development.

# Statement of need

Time-to-event analysis is applied when the outcome of interest is the time until a pre-specified event occurs. In some settings, the time variable is inherently or effectively discrete, for example, when time is measured in weeks or months, or when event times are rounded or grouped into intervals. Competing risks arise when observations are at risk of experiencing multiple mutually exclusive event types, such that the occurrence of one event precludes the occurrence or observation of the others. Discrete-time survival data with competing risks are encountered across a wide range of scientific disciplines. For instance, in healthcare, the time to death from cancer is often recorded in months, with death from other causes considered a competing event. 

While excellent Python packages for continuous-time survival-analysis exist 


# Acknowledgments
T.M. is supported by the Israeli Council for Higher Education (Vatat) fellowship in data science via the Technion; M.G. work was supported by the ISF 767/21 grant and Malag competitive grant in data science (DS).

# References