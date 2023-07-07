# Evaluation Measures

$$
\pi_{ij}(t) = \widehat{\Pr}(T_i=t, J_i=j \mid Z_i) = \widehat{\lambda}_j (t \mid Z_i) \widehat{S}(t-1 \mid Z_i)
$$
and
$$
D_{ij} (t) = I(T_i = t, J_i = j) \, .
$$
The cause-specific incidence/dynamic area under the receiver operating characteristics curve (AUC) is defined and estimated in the spirit of Heagerty and Zheng (2005) \cite{heagerty2005survival} and Blanche et al. (2015) \cite{blanche2015quantifying} as the probability of a random observation with observed event $j$ at time $t$ having a higher risk prediction for cause $j$ than a randomly selected observation $m$, at risk at time $t$, without the observed event $j$ at time $t$. Namely,
$$
\mbox{AUC}_j(t) = \Pr (\pi_{ij}(t) > \pi_{mj}(t) \mid D_{ij} (t) = 1, D_{mj} (t) = 0, T_m \geq t) \, .
$$

In the presence of censored data and under the assumption that the censoring is independent of the failure time and observed covariates, an inverse probability censoring weighting (IPCW) estimator of  $\mbox{AUC}_j(t)$ becomes
$$
\widehat{\mbox{AUC}}_j (t) &=&  \frac{\sum_{i=1}^{n}\sum_{m=1}^{n} D_{ij}(t)(1-D_{mj}(t))I(X_m \geq t) W_{ij}(t) W_{mj}(t)
\{I(\pi_{ij}(t) > \pi_{mj}(t))+0.5I(\pi_{ij}(t)=\pi_{mj}(t))\}}{\sum_{i=1}^{n}\sum_{m=1}^{n}  D_{ij}(t)(1-D_{mj}(t))I(X_m \geq t) W_{ij}(t) W_{mj}(t)} \nonumber \\
&= & \frac{\sum_{i=1}^{n}\sum_{m=1}^{n} D_{ij}(t)(1-D_{mj}(t))I(X_m \geq t) 
\{I(\pi_{ij}(t) > \pi_{mj}(t))+0.5I(\pi_{ij}(t)=\pi_{mj}(t))\}}{\sum_{i=1}^{n}\sum_{m=1}^{n} D_{ij}(t)(1-D_{mj}(t))I(X_m \geq t)}
$$
where  
$$
W_{ij}(t) & = &  \frac{D_{ij}(t)}{\widehat{G}_C(T_i)} + I(X_i \geq t)\frac{1-D_{ij}(t)}{\widehat{G}_C(t)} = \frac{D_{ij}(t)}{\widehat{G}_C(t)} + I(X_i \geq t)\frac{1-D_{ij}(t)}{\widehat{G}_C(t)}   \\
& = & I(X_i \geq t) / \widehat{G}_C(t)
$$
and 
$\widehat{G}_C(\cdot)$ is the estimated survival function of the censoring (e.g., the Kaplan-Meier estimator). Interestingly, the IPCWs  have no effect on $\widehat{\mbox{AUC}}_j (t)$.
An integrated cause-specific AUC can be estimated as a weighted sum  by
$$
\widehat{\mbox{AUC}}_j = \sum_t \widehat{\mbox{AUC}}_j (t) w_j (t) \, ,
$$
and we adopt a simple  data-driven weight function of the form 
$$
w_j(t) = \frac{N_j(t)}{\sum_t N_j(t)} \, .
$$  
A global AUC can be defined as
$$
\widehat{\mbox{AUC}} = \sum_j \widehat{\mbox{AUC}}_j v_j
$$
where 
$$
v_j = \frac{\sum_{t} N_j(t)}{ \sum_{j=1}^M \sum_{t} N_j(t) } \, .
$$
Another well-known performance measure is the Brier Score (BS). In the spirit of Blanche et al. (2015) \cite{blanche2015quantifying} we define
$$
  \widehat{\mbox{BS}}_{j}(t) = \frac{1}{Y_{\cdot}(t)} {\sum_{i=1}^n W_{ij}(t) \left( D_{ij}(t) - \pi_{ij}(t)\right)^2} \, . 
$$
An integrated cause-specific BS can be estimated by the weighted sum
$$
 \widehat{\mbox{BS}}_{j} = \sum_t \widehat{\mbox{BS}}_{j}(t) w_j(t) 
$$
and an estimated global BS is given by 
$$
 \widehat{\mbox{BS}} = \sum_j \widehat{\mbox{BS}}_{j} v_j \, .
$$
