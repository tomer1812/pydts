# Introduction

Based on "PyDTS: A Python Package for Discrete Time Survival-analysis with Competing Risks" 
Tomer Meir*, Rom Gutman, and Malka Gorfine.


Time-to-event analysis (survival analysis) is used when we would like to estimate how much time is left until the occurrence of the event of interest. For this kind of task, removing from the dataset the observations for which the event has not yet happened (censored observations) during the training process introduces bias, and ignores the information that this observation was free of the event until censorship. Time-to-event models, such as Cox proportional hazards model, accelerated failure time model and others, can take into account censored observations and avoid this bias. 
These models are suitable for continuous time, in which the probability of an event to occur at the same time for more than one observation in our dataset is neglectable. In discrete time, however, we are likely to observe many observations with the same event occurrence time. Examples for such cases can be found, for example, in healthcare when discussing patients hospitalization length of stay (LOS). In this case we usually count number of days at the hospital, and we have many patients who are released on the same day of their hospitalization. Other examples can be number of months until developing a curtain disease, weeks until birth during pregnancy etc. Similarly, many examples can be found in other fields such as finance, biology, physics, social sciences etc. Some continuous time models use ties correction methods presented by Breslow, Efron, and Kalbfleisch to give an approximation for the discrete model.
In many practical cases, the observations are at risk for one of several distinct event types. For example, when trying to estimate survival time for a cause-specific death, such as heart failure, cancer and other causes. In these cases, competing risks models should be in use. The competing risks could be terminal states type, i.e. once one of the events occurs, the observation is no longer at risks for other risks, or multi-state competing risks type, in which transition between states is possible even after events occur, or semi-competing risks - a mix of terminal and non-terminal states. Here, we discuss terminal states competing risks.



# Methods

## Definitions

Let $T$ be a discrete event time that takes the values $\{1,2,...,d\}$, and $J$ be the event, $J \in \{1,\ldots,M\}$. 
Let $Z$ be the baseline vector of covariates. A general discrete cause-specific hazard model is of the form
$$
\lambda_j(t|Z) = \Pr(T=t,J=j|T\geq t, Z)  \hspace{0.3cm} t \in \{1,2,...,d\} \hspace{0.3cm} j=1,\ldots,M  \, .
$$
A popular model of the above hazard function based on a transformation regression model is of the form
$$
h(\lambda_{j}(t|Z))  = \alpha_{jt} +Z^T \beta_j \hspace{0.3cm} t \in \{1,2,...,d\} \hspace{0.3cm} j=1, \ldots,M  
$$
such that $h$ is a known function.  \\
The logit function $h(a)=\log \{ a/(1-a) \}$ yields 
$$
\lambda_j(t|Z)=\frac{\exp(\alpha_{jt}+Z^T\beta_j)}{1+\exp(\alpha_{jt}+Z^T\beta_j)} \, .
$$

Let $S(t|Z) = \Pr(T>t|Z)$ be the overall survival given $Z$ and assume for simplicity of presentation two competing events, i.e., $M=2$. 
Then, the probability of event of type $j$ at time $t$ equals
$$
\Pr(T=t,J=j|Z)=\lambda_j(t|Z) \prod_{k=1}^{t-1} \left\{1-\lambda_1(k|Z) - \lambda_2(k|Z)\right\}  \, ,
$$
the cumulative incident function of cause $j$ is given by
$$
	\Pr(T \leq t, J=j|Z)  & = & \sum_{m=1}^{t} \lambda_j(m|Z) S(m-1|Z) \\ 
	& = & \sum_{m=1}^{t}\lambda_j(m|Z) \prod_{k=1}^{m-1} \left\{1-\lambda_1(k|Z) - \lambda_2(k|Z) \right\}
$$
and the marginal probability of event type $j$ equals
$$
\Pr(J=j|Z) = \sum_{m=1}^{d} \lambda_j(m|Z) \prod_{k=1}^{m-1} \left\{1-\lambda_1(k|Z) - \lambda_2(k|Z) \right\} \, .
$$


## Collapsed log-likelihood approach

The data for estimating $\{\alpha_{11},\ldots,\alpha_{1d},\beta_1^T,\alpha_{21},\ldots,\alpha_{2d},\beta_2^T\}$ 
consist of $n$ independent observations, each with $(X_i,\delta_i,J_i,Z_i)$ where $X_i=\min(C_i,T_i)$, $C_i$ is a right-censoring time, 
$\delta_i=I(X_i=T_i)$ is the event indicator and $J\in\{0,1,2\}$, where $J_i=0$ if $\delta_i=0$. 

Then, the likelihood function is proportional to 
$$
	L &=& \prod_{i=1}^n  \left\{\frac{\lambda_1(X_i|Z_i)}{1-\lambda_1(X_i|Z_i)-\lambda_2(X_i|Z_i)}\right\}^{I(\delta_{1i}=1)}\\ 
	& & \left\{\frac{\lambda_2(X_i|Z_i)}{1-\lambda_1(X_i|Z_i)-\lambda_2(X_i|Z_i)}\right\}^{I(\delta_{2i}=1)}\\ 
	& & \prod_{k=1}^{X_i}\{1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\}\
$$
or, equivalently,
$$
L &=& \prod_{i=1}^n \left[ \prod_{j=1}^2 \prod_{m=1}^{X_i} \left\{  \frac{\lambda_j(m|Z_i)}{1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)}\right\}^{\delta_{jim}}\right]\\ &&\prod_{k=1}^{X_i}\{1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\}
$$
where $\delta_{jim}$ equals one if subject $i$ experienced event type $j$ at time $m$; and 0 otherwise. Then, the log likelihood becomes
$$
	\log L &=& \sum_{i=1}^n \left[\sum_{j=1}^2 \sum_{m=1}^{X_i} \left[ \delta_{jim} \log \lambda_j(m|Z_i) - \delta_{jim}\{1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)\}\right] \right. \\ 
	&& +\left.\sum_{k=1}^{X_i}\log \{1-\lambda_1(k|Z_i)-\lambda_2(k|Z_i)\}\right] \\ 
	&=& \sum_{i=1}^n \sum_{m=1}^{X_i} \left[  \delta_{1im} \log \lambda_1(m|Z_i)+\delta_{2im} \log \lambda_2(m|Z_i) \right. \\ 
	&& +\left. \{ 1-\delta_{1im}-\delta_{2im}\}\log\{ 1-\lambda_1(m|Z_i)-\lambda_2(m|Z_i)\}\right] \, .
$$

Instead of maximizing the above likelihood directly, the collapsed log-likelihood of Lee et al. 2018 can be adopted. Specifically, we expand the data such that for each observation we have rows for each of the times until the event occurrence or censoring of the observation. At each time point $t$ the data are conditionally multinomial with one of three possible outcomes $\{\delta_{1it},\delta_{2it},1-\delta_{1it}-\delta_{2it}\}$. Then, for estimating $\{\alpha_{11},\ldots,\alpha_{1d},\beta_1^T\}$, we combine $\delta_{2it}$ and $1-\delta_{1it}-\delta_{2it}$, and the collapsed log-likelihood for cause $J=1$ based on a binary regression model with $\delta_{1it}$ as the outcome is given by
$$
\log L_1 = \sum_{i=1}^n \sum_{m=1}^{X_i}\left[ \delta_{1im} \log \lambda_1(m|Z_i)+(1-\delta_{1im})\log \{1-\lambda_1(m|Z_i)\} \right]
$$
Similarly, the collapsed log-likelihood for cause $J=2$ based on a binary regression model with $\delta_{2it}$ as the outcome is
$$
\log L_2 = \sum_{i=1}^n \sum_{m=1}^{X_i}\left[ \delta_{2im} \log \lambda_2(m|Z_i)+(1-\delta_{2im})\log \{1-\lambda_2(m|Z_i)\} \right]
$$
and one can fit the two models, separately. Further details about this model can be found in Lee et al. 2018.
In this approach, we estimate one model for each event, with D $\alpha_{jt}$ parameters as covariates in addition to the Z covariates. When the total number of discrete times and covariates D + Z is large, we essentially fit a high dimensional model, and the estimation procedure can become challenging. 

## Our approach
Given that the estimation procedure includes $DM$ parameters of $\alpha_{jt}$, we propose the following simpler estimation procedure. 
Start with estimating the vectors $\beta_j$, $j=1,2$, by a simple conditional logistic regression, conditioning on the event time (denoted by $X$), based on the expanded dataset, using stratified Cox analysis.
Given the estimators $\widehat{\beta}_j$ , $j=1,2$, we propose to go back to original unexpanded dataset and run $DM$ separate optimization procedures, for each $\alpha_{jt}$, and these are extremely simple (and thus very fast) optimization problems. Specifically, let $y_t=\sum_{i=1}^n I(X_i \geq t)$ and $n_{tj}=\sum_{i=1}^n I(X_i = t, J_i=j)$. Then, for each $(j,t)$, find 

$$
\widehat{\alpha}_{jt} = 
\mbox{argmin}_{a} \left\{ \frac{1}{y_t} \sum_{i=1}^n I(X_i \geq t)\frac{\exp(a+Z_i^T\widehat{\beta}_j)}{1+\exp(a+Z_i^T\widehat{\beta}_j)} - \frac{n_{tj}}{y_t}\right\}^2
$$


