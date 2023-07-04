# Nested Design
[Julian Faraway](https://julianfaraway.github.io/)
2023-06-30

- [Data](#data)
- [Mixed Effect Model](#mixed-effect-model)
- [INLA](#inla)
  - [Informative Gamma priors on the
    precisions](#informative-gamma-priors-on-the-precisions)
  - [Penalized Complexity Prior](#penalized-complexity-prior)
- [STAN](#stan)
  - [Diagnostics](#diagnostics)
  - [Output summaries](#output-summaries)
  - [Posterior Distributions](#posterior-distributions)
- [BRMS](#brms)
- [MGCV](#mgcv)
- [GINLA](#ginla)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](index.md) for an overview.

This example is discussed in more detail in my book [Extending the
Linear Model with R](https://julianfaraway.github.io/faraway/ELM/)

Required libraries:

``` r
library(faraway)
library(ggplot2)
library(lme4)
library(pbkrtest)
library(RLRsim)
library(INLA)
library(knitr)
library(cmdstanr)
register_knitr_engine(override = FALSE)
library(brms)
library(mgcv)
```

# Data

When the levels of one factor vary only within the levels of another
factor, that factor is said to be *nested*. Here is an example to
illustrate nesting. Consistency between laboratory tests is important
and yet the results may depend on who did the test and where the test
was performed. In an experiment to test levels of consistency, a large
jar of dried egg powder was divided up into a number of samples. Because
the powder was homogenized, the fat content of the samples is the same,
but this fact is withheld from the laboratories. Four samples were sent
to each of six laboratories. Two of the samples were labeled as G and
two as H, although in fact they were identical. The laboratories were
instructed to give two samples to two different technicians. The
technicians were then instructed to divide their samples into two parts
and measure the fat content of each. So each laboratory reported eight
measures, each technician four measures, that is, two replicated
measures on each of two samples.

Load in and plot the data:

``` r
data(eggs, package="faraway")
summary(eggs)
```

          Fat         Lab    Technician Sample
     Min.   :0.060   I  :8   one:24     G:24  
     1st Qu.:0.307   II :8   two:24     H:24  
     Median :0.370   III:8                    
     Mean   :0.388   IV :8                    
     3rd Qu.:0.430   V  :8                    
     Max.   :0.800   VI :8                    

``` r
ggplot(eggs, aes(y=Fat, x=Lab, color=Technician, shape=Sample)) + geom_point(position = position_jitter(width=0.1, height=0.0))
```

![](figs/eggplot-1..svg)

# Mixed Effect Model

The model is
$$y_{ijkl} = \mu + L_i + T_{ij} + S_{ijk} + \epsilon_{ijkl}$$ where
laboratories (L), technicians (T) and samples (S) are all random
effects:

``` r
cmod = lmer(Fat ~ 1 + (1|Lab) + (1|Lab:Technician) +  (1|Lab:Technician:Sample), data=eggs)
faraway::sumary(cmod)
```

    Fixed Effects:
    coef.est  coef.se 
        0.39     0.04 

    Random Effects:
     Groups                Name        Std.Dev.
     Lab:Technician:Sample (Intercept) 0.06    
     Lab:Technician        (Intercept) 0.08    
     Lab                   (Intercept) 0.08    
     Residual                          0.08    
    ---
    number of obs: 48, groups: Lab:Technician:Sample, 24; Lab:Technician, 12; Lab, 6
    AIC = -54.2, DIC = -73.3
    deviance = -68.8 

Is there a difference between samples? The `exactRLRT` function requires
not only the specification of a null model without the random effect of
interest but also one where only that random effect is present. Note
that because of the way the samples are coded, we need to specify this a
three-way interaction. Otherwise `G` from one lab would be linked to `G`
from another lab (which is not the case).

``` r
cmodr <- lmer(Fat ~ 1 + (1|Lab) + (1|Lab:Technician), data=eggs)
cmods <- lmer(Fat ~ 1 + (1|Lab:Technician:Sample), data=eggs)
exactRLRT(cmods, cmod, cmodr)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 1.6, p-value = 0.11

We can remove the sample random effect from the model. But consider the
confidence intervals:

``` r
confint(cmod, method="boot")
```

                   2.5 %   97.5 %
    .sig01      0.000000 0.095625
    .sig02      0.000000 0.141366
    .sig03      0.000000 0.153524
    .sigma      0.059987 0.106322
    (Intercept) 0.302717 0.468338

We see that all three random effects include zero at the lower end,
indicating that we might equally have disposed of the lab or technician
random effects first. There is considerable uncertainty in the
apportioning of variation due the three effects.

# INLA

Integrated nested Laplace approximation is a method of Bayesian
computation which uses approximation rather than simulation. More can be
found on this topic in [Bayesian Regression Modeling with
INLA](http://julianfaraway.github.io/brinla/) and the [chapter on
GLMMs](https://julianfaraway.github.io/brinlabook/chaglmm.html)

Use the most recent computational methodology:

``` r
inla.setOption(inla.mode="experimental")
inla.setOption("short.summary",TRUE)
```

Need to construct unique labels for nested factor levels. Don’t really
care which technician and sample is which otherwise would take more care
with the labeling.

``` r
eggs$labtech <- factor(paste0(eggs$Lab,eggs$Technician))
eggs$labtechsamp <- factor(paste0(eggs$Lab,eggs$Technician,eggs$Sample))
```

``` r
formula <- Fat ~ 1 + f(Lab, model="iid") + f(labtech, model="iid") + f(labtechsamp, model="iid")
result <- inla(formula, family="gaussian", data=eggs)
summary(result)
```

    Fixed effects:
                 mean    sd 0.025quant 0.5quant 0.975quant  mode kld
    (Intercept) 0.388 0.035      0.318    0.388      0.457 0.388   0

    Model hyperparameters:
                                                mean       sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations   114.86    27.02      69.99   112.08     175.72 34.71
    Precision for Lab                       21230.87 22842.36    1438.12 14065.24   81875.80 74.74
    Precision for labtech                     105.62    54.89      35.45    93.81     245.11  9.37
    Precision for labtechsamp               20398.62 22494.59    1221.25 13240.41   80054.53 57.80

     is computed 

The lab and sample precisions look far too high. Need to change the
default prior

## Informative Gamma priors on the precisions

Now try more informative gamma priors for the precisions. Define it so
the mean value of gamma prior is set to the inverse of the variance of
the residuals of the fixed-effects only model. We expect the error
variances to be lower than this variance so this is an overestimate. The
variance of the gamma prior (for the precision) is controlled by the
`apar` shape parameter in the code.

``` r
apar <- 0.5
bpar <- apar*var(eggs$Fat)
lgprior <- list(prec = list(prior="loggamma", param = c(apar,bpar)))
formula = Fat ~ 1+f(Lab, model="iid", hyper = lgprior)+f(labtech, model="iid", hyper = lgprior)+f(labtechsamp, model="iid", hyper = lgprior)
result <- inla(formula, family="gaussian", data=eggs)
summary(result)
```

    Fixed effects:
                 mean    sd 0.025quant 0.5quant 0.975quant  mode kld
    (Intercept) 0.388 0.062      0.263    0.388      0.512 0.388   0

    Model hyperparameters:
                                              mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 160.24 41.99      91.05   155.74     255.10 41.81
    Precision for Lab                       116.02 95.66      20.41    89.62     369.07  5.01
    Precision for labtech                   131.28 87.55      31.91   109.44     359.68  6.92
    Precision for labtechsamp               184.67 93.60      63.59   164.90     421.68 14.89

     is computed 

Looks more credible.

Compute the transforms to an SD scale for the field and error. Make a
table of summary statistics for the posteriors:

``` r
sigmaLab <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaTech <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[3]])
sigmaSample <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[4]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaLab,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaTech,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaSample,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","Lab","Technician","Sample","epsilon")
data.frame(restab)
```

                     mu      Lab Technician   Sample  epsilon
    mean         0.3875  0.11342    0.10044 0.080197 0.081028
    sd         0.061899 0.043168   0.031582 0.019438  0.01066
    quant0.025  0.26332 0.052272   0.052928 0.048858 0.062716
    quant0.25   0.34857 0.082539   0.077746 0.066235 0.073444
    quant0.5    0.38735  0.10544    0.09547 0.077825 0.080055
    quant0.75   0.42613  0.13551     0.1177  0.09156 0.087617
    quant0.975  0.51138  0.21971    0.17594  0.12483   0.1045

Also construct a plot the SD posteriors:

``` r
ddf <- data.frame(rbind(sigmaLab,sigmaTech,sigmaSample,sigmaepsilon),errterm=gl(4,nrow(sigmaLab),labels = c("Lab","Tech","Samp","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("Fat")+ylab("density")+xlim(0,0.25)
```

![](figs/plotsdseggs-1..svg)

Posteriors look OK. Notice that they are all well bounded away from
zero.

## Penalized Complexity Prior

In [Simpson et al (2015)](http://arxiv.org/abs/1403.4630v3), penalized
complexity priors are proposed. This requires that we specify a scaling
for the SDs of the random effects. We use the SD of the residuals of the
fixed effects only model (what might be called the base model in the
paper) to provide this scaling.

``` r
sdres <- sd(eggs$Fat)
pcprior <- list(prec = list(prior="pc.prec", param = c(3*sdres,0.01)))
formula = Fat ~ 1+f(Lab, model="iid", hyper = pcprior)+f(labtech, model="iid", hyper = pcprior)+f(labtechsamp,model="iid", hyper = pcprior)
result <- inla(formula, family="gaussian", data=eggs, control.family=list(hyper=pcprior))
summary(result)
```

    Fixed effects:
                 mean    sd 0.025quant 0.5quant 0.975quant  mode kld
    (Intercept) 0.388 0.051      0.284    0.387      0.491 0.387   0

    Model hyperparameters:
                                              mean     sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 143.19  40.34      78.74   138.22     236.18 34.12
    Precision for Lab                       443.83 904.24      27.87   203.09    2395.42  4.37
    Precision for labtech                   187.11 175.01      25.59   136.62     650.38  4.54
    Precision for labtechsamp               427.59 434.84      63.94   299.90    1575.77  6.54

     is computed 

Compute the summaries as before:

``` r
sigmaLab <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaTech <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[3]])
sigmaSample <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[4]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaLab,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaTech,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaSample,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","Lab","Technician","Sample","epsilon")
data.frame(restab)
```

                     mu      Lab Technician   Sample  epsilon
    mean         0.3875 0.079257   0.093748 0.062163 0.086023
    sd         0.051365  0.04387   0.040553 0.025554 0.012034
    quant0.025  0.28411  0.02059   0.039368 0.025348 0.065191
    quant0.25   0.35565 0.047053   0.065006 0.043688 0.077471
    quant0.5    0.38738 0.070706   0.085317 0.057801    0.085
    quant0.75    0.4191  0.10159    0.11318 0.075658 0.093497
    quant0.975  0.49064  0.18796    0.19599  0.12427  0.11237

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaLab,sigmaTech,sigmaSample,sigmaepsilon),errterm=gl(4,nrow(sigmaLab),labels = c("Lab","Tech","Samp","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("Fat")+ylab("density")+xlim(0,0.25)
```

![](figs/eggspc-1..svg)

Posteriors have generally smaller values for the three random effects
and the possibility of values closer to zero is given greater weight.

# STAN

[STAN](https://mc-stan.org/) performs Bayesian inference using MCMC. I
use `cmdstanr` to access Stan from R.

You see below the Stan code to fit our model. Rmarkdown allows the use
of Stan chunks (elsewhere I have R chunks). The chunk header looks like
this.

STAN chunk will be compiled to ‘mod’. Chunk header is:

    cmdstan, output.var="mod", override = FALSE

``` stan
data {
     int<lower=0> Nobs;
     int<lower=0> Nlev1;
     int<lower=0> Nlev2;
     int<lower=0> Nlev3;
     array[Nobs] real y;
     array[Nobs] int<lower=1,upper=Nlev1> levind1;
     array[Nobs] int<lower=1,upper=Nlev2> levind2;
     array[Nobs] int<lower=1,upper=Nlev3> levind3;
     real<lower=0> sdscal;
}
parameters {
           real mu;
           real<lower=0> sigmalev1;
           real<lower=0> sigmalev2;
           real<lower=0> sigmalev3;
           real<lower=0> sigmaeps;

           vector[Nlev1] eta1;
           vector[Nlev2] eta2;
           vector[Nlev3] eta3;
}
transformed parameters {
  vector[Nlev1] ran1;
  vector[Nlev2] ran2;
  vector[Nlev3] ran3;
  vector[Nobs] yhat;

  ran1  = sigmalev1 * eta1;
  ran2  = sigmalev2 * eta2;
  ran3  = sigmalev3 * eta3;

  for (i in 1:Nobs)
    yhat[i] = mu+ran1[levind1[i]]+ran2[levind2[i]]+ran3[levind3[i]];

}
model {
  eta1 ~ normal(0, 1);
  eta2 ~ normal(0, 1);
  eta3 ~ normal(0, 1);
  sigmalev1 ~ cauchy(0, 2.5*sdscal);
  sigmalev2 ~ cauchy(0, 2.5*sdscal);
  sigmalev3 ~ cauchy(0, 2.5*sdscal);
  sigmaeps ~ cauchy(0, 2.5*sdscal);
  y ~ normal(yhat, sigmaeps);
}
```

``` r
levind1 <- as.numeric(eggs$Lab)
levind2 <- as.numeric(eggs$labtech)
levind3 <- as.numeric(eggs$labtechsamp)
sdscal <- sd(eggs$Fat)
eggdat <- list(Nobs=nrow(eggs),
               Nlev1=max(levind1),
               Nlev2=max(levind2),
               Nlev3=max(levind3),
               y=eggs$Fat,
               levind1=levind1,
               levind2=levind2,
               levind3=levind3,
               sdscal=sdscal)
```

Do the MCMC sampling:

``` r
fit <- mod$sample(
  data = eggdat, 
  seed = 123, 
  chains = 4, 
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)
```

    Running MCMC with 4 parallel chains...

    Chain 1 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 2 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 3 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 4 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 1 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 2 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 3 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 4 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 1 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 2 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 3 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 4 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 3 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 4 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 3 finished in 0.6 seconds.
    Chain 4 finished in 0.6 seconds.
    Chain 1 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 2 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 1 finished in 0.6 seconds.
    Chain 2 finished in 0.6 seconds.

    All 4 chains finished successfully.
    Mean chain execution time: 0.6 seconds.
    Total execution time: 0.8 seconds.

## Diagnostics

Extract the draws into a convenient dataframe format:

``` r
draws_df <- fit$draws(format = "df")
```

For the error SD:

``` r
ggplot(draws_df,
       aes(x=.iteration,y=sigmaeps,color=factor(.chain))) + geom_line() +
  labs(color = 'Chain', x="Iteration")
```

![](figs/eggssigmaeps-1..svg)

Looks OK

For the Lab SD

``` r
ggplot(draws_df,
       aes(x=.iteration,y=sigmalev1,color=factor(.chain))) + geom_line() +
  labs(color = 'Chain', x="Iteration")
```

![](figs/eggssigmalev1-1..svg)

Looks OK

For the technician SD

``` r
ggplot(draws_df,
       aes(x=.iteration,y=sigmalev2,color=factor(.chain))) + geom_line() +
  labs(color = 'Chain', x="Iteration")
```

![](figs/eggssigmalev2-1..svg)

For the sample SD

``` reggssigmalev3
ggplot(draws_df,
       aes(x=.iteration,y=sigmalev3,color=factor(.chain))) + geom_line() +
  labs(color = 'Chain', x="Iteration")
```

All these are satisfactory.

## Output summaries

Display the parameters of interest:

``` r
fit$summary(c("mu","sigmalev1","sigmalev2","sigmalev3","sigmaeps"))
```

    # A tibble: 5 × 10
      variable    mean median     sd    mad      q5   q95  rhat ess_bulk ess_tail
      <chr>      <num>  <num>  <num>  <num>   <num> <num> <num>    <num>    <num>
    1 mu        0.383  0.384  0.0575 0.0506 0.292   0.476  1.00    1299.     844.
    2 sigmalev1 0.0947 0.0829 0.0673 0.0565 0.0108  0.216  1.00    1112.    1597.
    3 sigmalev2 0.0942 0.0905 0.0460 0.0407 0.0212  0.176  1.00     814.     632.
    4 sigmalev3 0.0591 0.0580 0.0308 0.0303 0.00931 0.113  1.01     699.     745.
    5 sigmaeps  0.0903 0.0891 0.0133 0.0127 0.0710  0.115  1.00    1421.    2206.

About what we expect:

## Posterior Distributions

We can use extract to get at various components of the STAN fit.

``` r
sdf = stack(draws_df[,c("sigmalev1","sigmalev2","sigmalev3","sigmaeps")])
colnames(sdf) = c("Fat","SD")
levels(sdf$SD) = c("Lab","Technician","Sample","Error")
ggplot(sdf, aes(x=Fat,color=SD)) + geom_density() +xlim(0,0.3)
```

![](figs/eggsstanhypsd-1..svg)

We see that the error SD can be localized much more than the other SDs.
The technician SD looks to be the largest of the three. We see non-zero
density at zero in contrast with the INLA posteriors.

# BRMS

[BRMS](https://paul-buerkner.github.io/brms/) stands for Bayesian
Regression Models with STAN. It provides a convenient wrapper to STAN
functionality.

The form of the model specification is important in this example. If we
use
`Fat ~ 1 + (1|Lab) + (1|Lab:Technician) +  (1|Lab:Technician:Sample)` as
in the `lmer` fit earlier, we get poor convergence whereas the
supposedly equivalent specification below does far better. In the form
below, the nesting is signalled by the form of the model specification
which may be essential to achieve the best results.

``` r
suppressMessages(bmod <- brm(Fat ~ 1 + (1|Lab/Technician/Sample), data=eggs,iter=10000, cores=4,backend = "cmdstanr"))
```

We get some warnings. We can obtain some posterior densities and
diagnostics with:

``` r
plot(bmod, variable = "^s", regex=TRUE)
```

![](figs/eggsbrmsdiag-1..svg)

We have chosen only the random effect hyperparameters since this is
where problems will appear first. Looks OK. We can see some weight is
given to values of the random effect SDs close to zero.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.19.0
    functions {
      
    }
    data {
      int<lower=1> N; // total number of observations
      vector[N] Y; // response variable
      // data for group-level effects of ID 1
      int<lower=1> N_1; // number of grouping levels
      int<lower=1> M_1; // number of coefficients per level
      array[N] int<lower=1> J_1; // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_1_1;
      // data for group-level effects of ID 2
      int<lower=1> N_2; // number of grouping levels
      int<lower=1> M_2; // number of coefficients per level
      array[N] int<lower=1> J_2; // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_2_1;
      // data for group-level effects of ID 3
      int<lower=1> N_3; // number of grouping levels
      int<lower=1> M_3; // number of coefficients per level
      array[N] int<lower=1> J_3; // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_3_1;
      int prior_only; // should the likelihood be ignored?
    }
    transformed data {
      
    }
    parameters {
      real Intercept; // temporary intercept for centered predictors
      real<lower=0> sigma; // dispersion parameter
      vector<lower=0>[M_1] sd_1; // group-level standard deviations
      array[M_1] vector[N_1] z_1; // standardized group-level effects
      vector<lower=0>[M_2] sd_2; // group-level standard deviations
      array[M_2] vector[N_2] z_2; // standardized group-level effects
      vector<lower=0>[M_3] sd_3; // group-level standard deviations
      array[M_3] vector[N_3] z_3; // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1; // actual group-level effects
      vector[N_2] r_2_1; // actual group-level effects
      vector[N_3] r_3_1; // actual group-level effects
      real lprior = 0; // prior contributions to the log posterior
      r_1_1 = sd_1[1] * z_1[1];
      r_2_1 = sd_2[1] * z_2[1];
      r_3_1 = sd_3[1] * z_3[1];
      lprior += student_t_lpdf(Intercept | 3, 0.4, 2.5);
      lprior += student_t_lpdf(sigma | 3, 0, 2.5)
                - 1 * student_t_lccdf(0 | 3, 0, 2.5);
      lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
                - 1 * student_t_lccdf(0 | 3, 0, 2.5);
      lprior += student_t_lpdf(sd_2 | 3, 0, 2.5)
                - 1 * student_t_lccdf(0 | 3, 0, 2.5);
      lprior += student_t_lpdf(sd_3 | 3, 0, 2.5)
                - 1 * student_t_lccdf(0 | 3, 0, 2.5);
    }
    model {
      // likelihood including constants
      if (!prior_only) {
        // initialize linear predictor term
        vector[N] mu = rep_vector(0.0, N);
        mu += Intercept;
        for (n in 1 : N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n] + r_2_1[J_2[n]] * Z_2_1[n]
                   + r_3_1[J_3[n]] * Z_3_1[n];
        }
        target += normal_lpdf(Y | mu, sigma);
      }
      // priors including constants
      target += lprior;
      target += std_normal_lpdf(z_1[1]);
      target += std_normal_lpdf(z_2[1]);
      target += std_normal_lpdf(z_3[1]);
    }
    generated quantities {
      // actual population-level intercept
      real b_Intercept = Intercept;
    }

We see that `brms` is using student t distributions with 3 degrees of
freedom for the priors. For the two error SDs, this will be truncated at
zero to form half-t distributions. You can get a more explicit
description of the priors with `prior_summary(bmod)`. These are
qualitatively similar to the the PC prior used in the INLA fit.

We examine the fit:

``` r
summary(bmod)
```

     Family: gaussian 
      Links: mu = identity; sigma = identity 
    Formula: Fat ~ 1 + (1 | Lab/Technician/Sample) 
       Data: eggs (Number of observations: 48) 
      Draws: 4 chains, each with iter = 10000; warmup = 5000; thin = 1;
             total post-warmup draws = 20000

    Group-Level Effects: 
    ~Lab (Number of levels: 6) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.10      0.08     0.01     0.31 1.00     3165     2649

    ~Lab:Technician (Number of levels: 12) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.10      0.05     0.01     0.20 1.00     4240     4486

    ~Lab:Technician:Sample (Number of levels: 24) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.06      0.03     0.00     0.12 1.00     4184     7004

    Population-Level Effects: 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept     0.39      0.06     0.26     0.52 1.00     3694     2551

    Family Specific Parameters: 
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma     0.09      0.01     0.07     0.12 1.00     7406    11378

    Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

The results are consistent with those seen previously.

# MGCV

It is possible to fit some GLMMs within the GAM framework of the `mgcv`
package. An explanation of this can be found in this
[blog](https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/)

``` r
gmod = gam(Fat ~ 1 + s(Lab,bs="re") + s(Lab,Technician,bs="re") + 
             s(Lab,Technician,Sample,bs="re"),
           data=eggs, method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    Fat ~ 1 + s(Lab, bs = "re") + s(Lab, Technician, bs = "re") + 
        s(Lab, Technician, Sample, bs = "re")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)    0.388      0.043    9.02  2.7e-10

    Approximate significance of smooth terms:
                              edf Ref.df     F p-value
    s(Lab)                   2.67      5 20.70   0.085
    s(Lab,Technician)        5.64     11  7.85   0.107
    s(Lab,Technician,Sample) 6.76     23  0.81   0.170

    R-sq.(adj) =  0.669   Deviance explained = 77.5%
    -REML = -32.118  Scale est. = 0.0071958  n = 48

We get the fixed effect estimate. We also get tests on the random
effects (as described in this
[article](https://doi.org/10.1093/biomet/ast038). The hypothesis of no
variation is not rejected for any of the three sources of variation.
This is consistent with earlier findings.

We can get an estimate of the operator and error SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

                              std.dev    lower   upper
    s(Lab)                   0.076942 0.021827 0.27123
    s(Lab,Technician)        0.083547 0.035448 0.19691
    s(Lab,Technician,Sample) 0.055359 0.021819 0.14045
    scale                    0.084828 0.063926 0.11256

    Rank: 4/4

The point estimates are the same as the REML estimates from `lmer`
earlier. The confidence intervals are different. A bootstrep method was
used for the `lmer` fit whereas `gam` is using an asymptotic
approximation resulting in substantially different results. Given the
problems of parameters on the boundary present in this example, the
bootstrap results appear more trustworthy.

The random effect estimates for the fields can be found with:

``` r
coef(gmod)
```

                    (Intercept)                    s(Lab).1                    s(Lab).2                    s(Lab).3 
                      0.3875000                   0.1028924                  -0.0253890                   0.0106901 
                       s(Lab).4                    s(Lab).5                    s(Lab).6         s(Lab,Technician).1 
                     -0.0060132                  -0.0180396                  -0.0641407                  -0.0358047 
            s(Lab,Technician).2         s(Lab,Technician).3         s(Lab,Technician).4         s(Lab,Technician).5 
                      0.0019557                  -0.0190829                  -0.0043911                  -0.0064041 
            s(Lab,Technician).6         s(Lab,Technician).7         s(Lab,Technician).8         s(Lab,Technician).9 
                      0.0248034                   0.1571217                  -0.0318910                   0.0316872 
           s(Lab,Technician).10        s(Lab,Technician).11        s(Lab,Technician).12  s(Lab,Technician,Sample).1 
                     -0.0026988                  -0.0148658                  -0.1004295                   0.0599864 
     s(Lab,Technician,Sample).2  s(Lab,Technician,Sample).3  s(Lab,Technician,Sample).4  s(Lab,Technician,Sample).5 
                     -0.0064703                   0.0188096                  -0.0239627                   0.0031939 
     s(Lab,Technician,Sample).6  s(Lab,Technician,Sample).7  s(Lab,Technician,Sample).8  s(Lab,Technician,Sample).9 
                      0.0238439                   0.0425412                   0.0297972                  -0.0160427 
    s(Lab,Technician,Sample).10 s(Lab,Technician,Sample).11 s(Lab,Technician,Sample).12 s(Lab,Technician,Sample).13 
                      0.0028574                   0.0162856                  -0.0151469                  -0.0757062 
    s(Lab,Technician,Sample).14 s(Lab,Technician,Sample).15 s(Lab,Technician,Sample).16 s(Lab,Technician,Sample).17 
                      0.0073289                  -0.0271879                   0.0220348                  -0.0060056 
    s(Lab,Technician,Sample).18 s(Lab,Technician,Sample).19 s(Lab,Technician,Sample).20 s(Lab,Technician,Sample).21 
                     -0.0129541                   0.0264421                  -0.0437988                   0.0299548 
    s(Lab,Technician,Sample).22 s(Lab,Technician,Sample).23 s(Lab,Technician,Sample).24 
                     -0.0040423                  -0.0228123                  -0.0289461 

although these have not been centered in contrast with that found from
the `lmer` fit.

# GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(Fat ~ 1 + s(Lab,bs="re") + s(Lab,Technician,bs="re") + 
             s(Lab,Technician,Sample,bs="re"),
           data=eggs, fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior density for the intercept as:

``` r
plot(gimod$beta[1,],gimod$density[1,],type="l",xlab="yield",ylab="fat")
```

![](figs/eggsginlaint-1..svg)

and for the laboratory effects as:

``` r
xmat = t(gimod$beta[2:7,])
ymat = t(gimod$density[2:7,])
matplot(xmat, ymat,type="l",xlab="fat",ylab="density")
legend("right",paste0("Lab",1:6),col=1:6,lty=1:6)
```

![](figs/eggsginlaleff-1..svg)

We can see the first lab tends to be higher but still substantial
overlap with the other labs.

The random effects for the technicians are:

``` r
sel = 8:19
xmat = t(gimod$beta[sel,])
ymat = t(gimod$density[sel,])
matplot(xmat, ymat,type="l",xlab="fat",ylab="density")
legend("right",row.names(coef(cmod)[[2]]),col=1:length(sel),lty=1:length(sel))
```

![](figs/eggsginlateff-1..svg)

There are a couple of technicians which stick out from the others. Not
overwhelming evidence that they are different but certainly worth
further investigation.

There are too many of the sample random effects to make plotting
helpful.

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments.

- As with some of the other analyses, we see that the INLA-produced
  posterior densities for the random effect SDs are well-bounded away
  from zero. We see that the choice of prior does make an important
  difference and that the default choice is a clear failure.

- The default STAN priors produce a credible result and posteriors do
  give some weight to values close to zero. There is no ground truth
  here but given the experience in the `lmer` analysis, there does
  appear to be some suggestion that any of the three sources of
  variation could be very small. INLA is the odd-one-out in this
  instance.

- The `mgcv` based analysis is mostly the same as the `lme4` fit
  excepting the confidence intervals where a different method has been
  used.

- The `ginla` does not readily produce posterior densities for the
  hyperparameters so we cannot compare on that basis. The other
  posteriors were produced very rapidly.

# Package version info

``` r
sessionInfo()
```

    R version 4.3.1 (2023-06-16)
    Platform: x86_64-apple-darwin20 (64-bit)
    Running under: macOS Ventura 13.4.1

    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/lib/libRblas.0.dylib 
    LAPACK: /Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/lib/libRlapack.dylib;  LAPACK version 3.11.0

    locale:
    [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

    time zone: Europe/London
    tzcode source: internal

    attached base packages:
    [1] parallel  stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] mgcv_1.8-42     nlme_3.1-162    brms_2.19.0     Rcpp_1.0.10     cmdstanr_0.5.3  knitr_1.43      INLA_23.05.30-1
     [8] sp_2.0-0        foreach_1.5.2   RLRsim_3.1-8    pbkrtest_0.5.2  lme4_1.1-33     Matrix_1.5-4.1  ggplot2_3.4.2  
    [15] faraway_1.0.8  

    loaded via a namespace (and not attached):
      [1] gridExtra_2.3        inline_0.3.19        rlang_1.1.1          magrittr_2.0.3       matrixStats_1.0.0   
      [6] compiler_4.3.1       loo_2.6.0            systemfonts_1.0.4    callr_3.7.3          vctrs_0.6.3         
     [11] reshape2_1.4.4       stringr_1.5.0        crayon_1.5.2         pkgconfig_2.0.3      fastmap_1.1.1       
     [16] backports_1.4.1      ellipsis_0.3.2       labeling_0.4.2       utf8_1.2.3           threejs_0.3.3       
     [21] promises_1.2.0.1     rmarkdown_2.22       markdown_1.7         ps_1.7.5             nloptr_2.0.3        
     [26] MatrixModels_0.5-1   purrr_1.0.1          xfun_0.39            jsonlite_1.8.5       later_1.3.1         
     [31] Deriv_4.1.3          prettyunits_1.1.1    broom_1.0.5          R6_2.5.1             dygraphs_1.1.1.6    
     [36] StanHeaders_2.26.27  stringi_1.7.12       boot_1.3-28.1        rstan_2.21.8         iterators_1.0.14    
     [41] zoo_1.8-12           base64enc_0.1-3      bayesplot_1.10.0     httpuv_1.6.11        splines_4.3.1       
     [46] igraph_1.5.0         tidyselect_1.2.0     rstudioapi_0.14      abind_1.4-5          yaml_2.3.7          
     [51] codetools_0.2-19     miniUI_0.1.1.1       processx_3.8.1       pkgbuild_1.4.1       lattice_0.21-8      
     [56] tibble_3.2.1         plyr_1.8.8           shiny_1.7.4          withr_2.5.0          bridgesampling_1.1-2
     [61] posterior_1.4.1      coda_0.19-4          evaluate_0.21        RcppParallel_5.1.7   xts_0.13.1          
     [66] pillar_1.9.0         tensorA_0.36.2       stats4_4.3.1         checkmate_2.2.0      DT_0.28             
     [71] shinyjs_2.1.0        distributional_0.3.2 generics_0.1.3       rstantools_2.3.1     munsell_0.5.0       
     [76] scales_1.2.1         minqa_1.2.5          gtools_3.9.4         xtable_1.8-4         glue_1.6.2          
     [81] tools_4.3.1          shinystan_2.6.0      data.table_1.14.8    colourpicker_1.2.0   mvtnorm_1.2-2       
     [86] grid_4.3.1           tidyr_1.3.0          crosstalk_1.2.0      colorspace_2.1-0     cli_3.6.1           
     [91] fansi_1.0.4          svglite_2.1.1        Brobdingnag_1.2-9    dplyr_1.1.2          gtable_0.3.3        
     [96] digest_0.6.31        htmlwidgets_1.6.2    farver_2.1.1         htmltools_0.5.5      lifecycle_1.0.3     
    [101] mime_0.12            shinythemes_1.2.0    MASS_7.3-60         
