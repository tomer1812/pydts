# Methods


## Definitions

We let $T$  denote a discrete event time that can take on only the values $\{1,2,...,d\}$ and $J$ denote the type of event, $J \in \{1,\ldots,M\}$.  Consider a $p \times 1$ vector of baseline covariates $Z$. A general discrete cause-specific hazard function is of the form
$$
\lambda_j(t|Z) = \Pr(T=t,J=j|T\geq t, Z)  \hspace{0.3cm} t \in \{1,2,...,d\} \hspace{0.3cm} j=1,\ldots,M  \, .
$$
A popular semi-parametric model of the above hazard function based on a transformation regression model is of the form
$$
h(\lambda_{j}(t|Z))  = \alpha_{jt} +Z^T \beta_j \hspace{0.3cm} t \in \{1,2,...,d\} \hspace{0.3cm} j=1, \ldots,M  
$$
such that $h$ is a known function [[2, and reference therein]](#2). The total number of parameters in the model is $M(d+p)$. The logit function $h(a)=\log \{ a/(1-a) \}$ yields 
\begin{equation}\label{eq:logis}
\lambda_j(t|Z)=\frac{\exp(\alpha_{jt}+Z^T\beta_j)}{1+\exp(\alpha_{jt}+Z^T\beta_j)} \, .
\end{equation}
It should be noted that leaving $\alpha_{jt}$ unspecified is analogous to having an unspecified baseline hazard function in the Cox proportional hazard model [[3]](#3), and thus we consider the above as a semi-parametric model.


Let $S(t|Z) = \Pr(T>t|Z)$ be the overall survival given $Z$. Then, the probability of experiencing event of type $j$ at time $t$ equals
$$
\Pr(T=t,J=j|Z)=\lambda_j(t|Z) \prod_{k=1}^{t-1} \left\lbrace 1-
\sum_{j'=1}^M\lambda_{j'}(k|Z) \right\rbrace  
$$
and the cumulative incident function (CIF) of cause $j$ is given by
$$
F_j(t|Z) = \Pr(T \leq t, J=j|Z) = \sum_{m=1}^{t} \lambda_j(m|Z) S(m-1|Z) = \sum_{m=1}^{t}\lambda_j(m|Z) \prod_{k=1}^{m-1} \left\lbrace 1-\sum_{j'=1}^M\lambda_{j'}(k|Z) \right\rbrace \, .
$$
Finally, the marginal probability of event type $j$ (marginally with respect to the time of event), given $Z$, equals
$$
\Pr(J=j|Z) = \sum_{m=1}^{d} \lambda_j(m|Z) \prod_{k=1}^{m-1} \left\lbrace 1-\sum_{j'=1}^M\lambda_{j'}(k|Z) \right\rbrace \, .
$$
In the next section we provide a fast estimation technique of the parameters $\{\alpha_{j1},\ldots,\alpha_{jd},\beta_j^T \, ; \, j=1,\ldots,M\}$.


## The Collapsed Log-Likelihood Approach and the Proposed Estimators
For simplicity of presentation, we assume two competing events, i.e., $M=2$ and our goal is estimating $\{\alpha_{11},\ldots,\alpha_{1d},\beta_1^T,\alpha_{21},\ldots,\alpha_{2d},\beta_2^T\}$ along with the standard error of the estimators. The data at hand consist of $n$ independent observations, each with $(X_i,\delta_i,J_i,Z_i)$ where $X_i=\min(C_i,T_i)$, $C_i$ is a right-censoring time, 
$\delta_i=I(X_i=T_i)$ is the event indicator and $J_i\in\{0,1,2\}$, where $J_i=0$ if and only if $\delta_i=0$. Assume that given the covariates, the censoring and failure time are independent and non-informative. Then, the likelihood function is proportional to 
$$
L = \prod_{i=1}^n  \left\lbrace\frac{\lambda_1(X_i|Z_i)}{1-\lambda_1(X_i|Z_i)-\lambda_2(X_i|Z_i)}\right\rbrace^{I(\delta_{1i}=1)} \left\lbrace\frac{\lambda_2(X_i|Z_i)}{1-\lambda_1(X_i|Z_i)-\lambda_2(X_i|Z_i)}\right\rbrace^{I(\delta_{2i}=1)} \prod_{k=1}^{X_i}\lbrace 1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\rbrace
$$
or, equivalently,
$$
L = \prod_{i=1}^n \left\[ \prod_{j=1}^2 \prod_{m=1}^{X_i} \left\lbrace \frac{\lambda_j(m|Z_i)}{1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)}\right\rbrace^{\delta_{jim}}\right] \prod_{k=1}^{X_i}\lbrace 1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\rbrace
$$
where $\delta_{jim}$ equals one if subject $i$ experienced event type $j$ at time $m$; and 0 otherwise. Clearly $L$ cannot be decomposed into separate likelihoods for each cause-specific
hazard function $\lambda_j$.
The log likelihood becomes
$$
\log L = \sum_{i=1}^n \left\[ \sum_{j=1}^2 \sum_{m=1}^{X_i} \left\[ \delta_{jim} \log \lambda_j(m|Z_i) - \delta_{jim}\{1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)\}\right\] \right\.  
+\left\.\sum_{k=1}^{X_i}\log \lbrace 1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\rbrace \right\]
$$
$$
	= \sum_{i=1}^n \sum_{m=1}^{X_i} \left\[  \delta_{1im} \log \lambda_1(m|Z_i)+\delta_{2im} \log \lambda_2(m|Z_i) \right\. +\left\. \lbrace 1-\delta_{1im}-\delta_{2im}\rbrace \log\lbrace 1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)\rbrace \right\] \, .
$$

Instead of maximizing the $M(d+p)$ parameters simultaneously based on the above log-likelihood, the collapsed log-likelihood of Lee et al. [[4]](#4) can be adopted. Specifically, the data are expanded  such that for each observation $i$ the expanded dataset includes $X_i$ rows, one row for each time $t$, $t \leq X_i$. At each time point $t$ the expanded data are conditionally multinomial with one of three possible outcomes $\{\delta_{1it},\delta_{2it},1-\delta_{1it}-\delta_{2it}\}$, as in [Table 1](#tbl:expanded).

<a id="tbl:expanded">Table 1:</a>  Original and expanded datasets with $M = 2$ competing events [[Lee et al. (2018)]](#4). 


| $i$ | $X_i$ | $\delta_i$ | $Z_i$ | $i$ | $\tilde{X}_i$ | $\delta_{1it}$ | $\delta_{2it}$ | $1 - \delta_{1it} - \delta_{2it}$ | $Z_i$ |
|-----|--------|--------------|-------|-----|----------------|----------------|----------------|-------------------------------|-------|
| 1   | 2      | 1            | $Z_1$ | 1   | 1              | 0              | 0              | 1                             | $Z_1$ |
|     |        |              |       | 1   | 2              | 1              | 0              | 0                             | $Z_1$ |
| 2   | 3      | 2            | $Z_2$ | 2   | 1              | 0              | 0              | 1                             | $Z_2$ |
|     |        |              |       | 2   | 2              | 0              | 0              | 1                             | $Z_2$ |
|     |        |              |       | 2   | 3              | 0              | 1              | 0                             | $Z_2$ |
| 3   | 3      | 0            | $Z_3$ | 3   | 1              | 0              | 0              | 1                             | $Z_3$ |
|     |        |              |       | 3   | 2              | 0              | 0              | 1                             | $Z_3$ |
|     |        |              |       | 3   | 3              | 0              | 0              | 1                             | $Z_3$ |




Then, for estimating $\{\alpha_{11},\ldots,\alpha_{1d},\beta_1^T\}$, we combine $\delta_{2it}$ and $1-\delta_{1it}-\delta_{2it}$, and the collapsed log-likelihood for cause $J=1$ based on a binary regression model with $\delta_{1it}$ as the outcome is given by
$$
\log L_1 = \sum_{i=1}^n \sum_{m=1}^{X_i}\left\[ \delta_{1im} \log \lambda_1(m|Z_i)+(1-\delta_{1im})\log \lbrace 1-\lambda_1(m|Z_i)\rbrace \right\] \, .
$$
Similarly, the collapsed log-likelihood for cause $J=2$ based on a binary regression model with $\delta_{2it}$ as the outcome becomes
$$
\log L_2 = \sum_{i=1}^n \sum_{m=1}^{X_i}\left\[ \delta_{2im} \log \lambda_2(m|Z_i)+(1-\delta_{2im})\log \lbrace 1-\lambda_2(m|Z_i)\rbrace \right\]
$$
and one can fit the two models, separately.  

In general, for $M$ competing events, 
the estimators of $\{\alpha_{j1},\ldots,\alpha_{jd},\beta_j^T\}$, $j=1,\ldots,M$, are the respective values that maximize  
$$
\log L_j = \sum_{i=1}^n \sum_{m=1}^{X_i}\left[ \delta_{jim} \log \lambda_j(m|Z_i)+(1-\delta_{jim})\log \{1-\lambda_j(m|Z_i)\} \right] \, .
$$
Namely, each maximization $j$, $j=1,\ldots,M$, consists of  maximizing $d + p$ parameters simultaneously. 

### Proposed Estimators
Alternatively, we propose the following simpler and faster estimation procedure, with a negligible efficiency loss, if any. Our idea exploits the close relationship between conditional logistic regression analysis and stratified Cox regression analysis [[5]](#5). We propose to estimate each $\beta_j$ separately, and given $\beta_j$, $\alpha_{jt}$, $t=1\ldots,d$, are separately estimated. In particular, the proposed estimating procedure consists of the following two speedy steps:
#### Step 1.
Use the expanded dataset and estimate each vector $\beta_j$, $j \in \{1,\ldots, M\}$, by a simple conditional logistic regression, conditioning on the event time $X$, using a stratified Cox analysis.

#### Step 2.
Given the estimators $\widehat{\beta}_j$ , $j \in \{1,\ldots, M\}$, of Step 1, use the original (un-expanded) data and estimate each $\alpha_{jt}$, $j \in \{1,\ldots,M\}$, $t=1,\ldots,d$, separately, by

$$\widehat{ \alpha }_{jt} = argmin_{a} \left\lbrace \frac{1}{y_t} \sum_{i=1}^n I(X_i \geq t) \frac{ \exp(a+Z_i^T \widehat{\beta}_j)}{1 + \exp(a + Z_i^T \widehat{\beta}_j)} - \frac{n_{tj}}{y_t} \right\rbrace ^2 $$

where $y_t=\sum_{i=1}^n I(X_i \geq t)$ and $n_{tj}=\sum_{i=1}^n I(X_i = t, J_i=j)$.

The above equation consists minimizing the squared distance between the observed proportion of failures of type $j$ at time $t$ 
($n_{tj}/y_t$) and the expected proportion of failures given model defined above for $\lambda_j$ and $\widehat{\beta}_j$. 
The simulation results of section Simple Simulation reveals that the above two-step procedure performs well in terms of bias, and provides similar standard error of that of [[3]](#3). However, the improvement in computational time, by using our procedure, could be improved by a factor of 1.5-3.5 depending on d. Standard errors of $\widehat{\beta}_j$, $j \in \{1,\ldots,M\}$, can be derived directly from the stratified Cox analysis.

### Time-dependent covariates

Similarly to the continuous-time Cox model, the simplest way to code time-dependent covariates uses intervals of time [[Therneau et al. (2000)]](#6). Then, the data is encoded by breaking the individual’s time into multiple time intervals, with one row of data for each interval. Hence combining this data expansion step with the expansion demonstrated in [Table 1](#tbl:expanded) is straightforward.

### Regularized regression models

Penalized regression methods, such as lasso, adaptive lasso, and elastic net [[Hastie et al. 2009]](#7), place a constraint on the size of the regression coefficients. The estimation procedure of [[Meir and Gorfine (2023)]](#8) that separates the estimation of $\beta_j$ and $\alpha_{jt}$ can easily incorporate such constraints in Lagrangian form by minimizing

$$
-\log L_j^c(\beta_j)  + \eta_j P(\beta_j) \, , \quad j=1,\ldots,M \, ,
$$

where $P$ is a penalty function and $\eta_j \geq 0$ are shrinkage tuning parameters. The parameters $\alpha_{jt}$ are estimated once the regularization step is completed and $\beta_j$ are estimated.

Clearly, any regularized Cox regression model routines can be used for estimating $\beta_j$, $j=1,\ldots,M$, based on the above equation, for example, the `CoxPHFitter` of the `lifelines` Python package [[Davidson-Pilon (2019)]](#9) with penalization.


### Sure Independence Screening

When the number of available covariates greatly exceeds the number of observations (as common in genetic datasets, for example), i.e., the ultra-high setting, most regularized methods suffer from the curse of dimensionality, high variance, and overfitting [[Hastie et al. (2009)]](#7). 
Sure Independent Screening (SIS) is a marginal screening technique designed to filter out uninformative covariates. 
Penalized regression methods can be applied after the marginal screening process to the remaining covariates.

We start the SIS procedure by ranking all the covariates using a utility measure between the response and each covariate, and then retain only covariates with estimated coefficients that exceeds a threshold value. 
We focus on SIS and SIS followed by lasso (SIS-L) [[Fan et al. (2010); Saldana and Feng (2018)]](#10) within the proposed two-step procedure.

We start by fitting a marginal regression for each covariate by maximizing:

$$
L_j^{\mathcal{C}}(\beta_{jr}) \quad \text{for } j=1,\ldots,M, \quad r=1,\ldots,p 
$$

where $\boldsymbol{\beta}_j = (\beta_{j1},\ldots,\beta_{jp})^T$. 
Then we rank the features based on the magnitude of their marginal regression coefficients. 
The selected sets of variables are given by:

$$
\widehat{\mathcal{M}}_{j,w_n} = \left\{1 \leq k \leq p \, : \, |\widehat{\beta}_{jk}| \geq w_n \right\}, \quad j=1,\ldots,M,
$$

where $w_n$ is a threshold value. 
We adopt the data-driven threshold of [[Saldana and Feng (2018)]](#11). 
Given data of the form $\{X_i, \delta_i, J_i, \mathbf{Z}_i \, ; \, i = 1, \ldots, n\}$, a random permutation $\pi$ of $\{1,\ldots,n\}$ is used to decouple $\mathbf{Z}_i$ and $(X_i, \delta_i, J_i)$, so that the permuted data $\{X_i, \delta_i, J_i, \mathbf{Z}_{\pi(i)}\}$ follow a model where the covariates have no predictive power over the survival time of any event type.

For the permuted data, we re-estimate individual regression coefficients and obtain $\widehat{\beta}^*_{jr}$. The data-driven threshold is defined as:

$$
w_n = \max_{1 \leq j \leq M, \, 1 \leq k \leq p} |\widehat{\beta}^*_{jk}|.
$$

For the SIS-L procedure, lasso regularization is then applied in the first step of the two-step procedure to the set of covariates selected by SIS. 



## References
<a id="1">[1]</a> 
Meir, Tomer\*, Gutman, Rom\*, and Gorfine, Malka 
"PyDTS: A Python Package for Discrete Time Survival-analysis with Competing Risks"
(2022)

<a id="2">[2]</a> 
Allison, Paul D.
"Discrete-Time Methods for the Analysis of Event Histories"
Sociological Methodology (1982),
doi: 10.2307/270718

<a id="3">[3]</a> 
Cox, D. R.
"Regression Models and Life-Tables"
Journal of the Royal Statistical Society: Series B (Methodological) (1972)
doi: 10.1111/j.2517-6161.1972.tb00899.x

<a id="4">[4]</a> 
Lee, Minjung and Feuer, Eric J. and Fine, Jason P.
"On the analysis of discrete time competing risks data"
Biometrics (2018)
doi: 10.1111/biom.12881

<a id="5">[5]</a> 
Prentice, Ross L and Breslow, Norman E
"Retrospective studies and failure time models"
Biometrika (1978)
doi: 10.1111/j.2517-6161.1972.tb00899.x

<a id="6">[6]</a> 
Therneau, Terry M and Grambsch, Patricia M,
"Modeling Survival Data: Extending the Cox Model", Springer-Verlag,
(2000)

<a id="7">[7]</a> 
Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome H,
"The Elements of Statistical Learning: Data Mining, Inference, and Prediction.", Springer-Verlag,
(2009)

<a id="8">[8]</a> 
Meir, Tomer and Gorfine, Malka, 
"Discrete-time Competing-Risks Regression with or without Penalization"
(2023)

<a id="9">[9]</a> 
Davidson-Pilon, Cameron,
"lifelines: Survival Analysis in Python", Journal of Open Source Software,
(2019)

<a id="10">[10]</a> 
Fan, J and Feng, Y and Wu, Y,
"High-dimensional variable selection for Cox’s proportional hazards model", 
Institute of Mathematical Statistics,
(2010)

<a id="11">[11]</a> 
Saldana, DF and Feng, Y,
"SIS: An R package for sure independence screening in ultrahigh-dimensional statistical models", 
Journal of Statistical Software,
(2018)
