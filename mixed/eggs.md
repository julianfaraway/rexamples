Nested Design
================
[Julian Faraway](https://julianfaraway.github.io/)
12 July 2022

-   <a href="#data" id="toc-data">Data</a>
-   <a href="#mixed-effect-model" id="toc-mixed-effect-model">Mixed Effect
    Model</a>
-   <a href="#inla" id="toc-inla">INLA</a>
    -   <a href="#informative-gamma-priors-on-the-precisions"
        id="toc-informative-gamma-priors-on-the-precisions">Informative Gamma
        priors on the precisions</a>
    -   <a href="#penalized-complexity-prior"
        id="toc-penalized-complexity-prior">Penalized Complexity Prior</a>
-   <a href="#stan" id="toc-stan">STAN</a>
    -   <a href="#diagnostics" id="toc-diagnostics">Diagnostics</a>
    -   <a href="#output-summaries" id="toc-output-summaries">Output
        summaries</a>
    -   <a href="#posterior-distributions"
        id="toc-posterior-distributions">Posterior Distributions</a>
-   <a href="#brms" id="toc-brms">BRMS</a>
-   <a href="#mgcv" id="toc-mgcv">MGCV</a>
-   <a href="#ginla" id="toc-ginla">GINLA</a>
-   <a href="#discussion" id="toc-discussion">Discussion</a>
-   <a href="#package-version-info" id="toc-package-version-info">Package
    version info</a>

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
library(rstan, quietly=TRUE)
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

![](figs/eggplot-1..svg)<!-- -->

# Mixed Effect Model

The model is

![y\_{ijkl} = \mu + L_i + T\_{ij} + S\_{ijk} + \epsilon\_{ijkl}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_%7Bijkl%7D%20%3D%20%5Cmu%20%2B%20L_i%20%2B%20T_%7Bij%7D%20%2B%20S_%7Bijk%7D%20%2B%20%5Cepsilon_%7Bijkl%7D "y_{ijkl} = \mu + L_i + T_{ij} + S_{ijk} + \epsilon_{ijkl}")

where laboratories (L), technicians (T) and samples (S) are all random
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
    RLRT = 1.6, p-value = 0.1

We can remove the sample random effect from the model. But consider the
confidence intervals:

``` r
confint(cmod, method="boot")
```

                   2.5 %  97.5 %
    .sig01      0.000000 0.09550
    .sig02      0.000000 0.13596
    .sig03      0.000000 0.15252
    .sigma      0.061713 0.10653
    (Intercept) 0.305194 0.47580

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
    (Intercept) 0.387 0.035      0.319    0.387      0.456 0.387   0

    Model hyperparameters:
                                                mean       sd 0.025quant 0.5quant 0.975quant    mode
    Precision for the Gaussian observations   113.88    26.68      69.11   111.35     173.63  106.75
    Precision for Lab                       17961.01 18068.82    1101.34 12434.27   66150.63 2949.93
    Precision for labtech                     105.78    55.94      34.76    93.64     248.32   73.14
    Precision for labtechsamp               17016.99 17638.78     883.53 11498.84   64109.17 2232.62

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
    (Intercept) 0.387 0.062      0.264    0.387      0.511 0.387   0

    Model hyperparameters:
                                              mean    sd 0.025quant 0.5quant 0.975quant   mode
    Precision for the Gaussian observations 159.30 41.57      90.36   155.12     252.95 147.50
    Precision for Lab                       109.40 87.04      19.04    86.01     339.67  49.76
    Precision for labtech                   129.40 85.39      31.45   108.41     352.29  74.54
    Precision for labtechsamp               183.45 92.64      62.99   164.09     417.96 130.72

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
    mean         0.3875  0.11636    0.10105 0.080452 0.081259
    sd         0.061483 0.044614   0.031763 0.019522 0.010692
    quant0.025  0.26415 0.054544   0.053528 0.049098 0.063008
    quant0.25   0.34892 0.084603    0.07824  0.06643 0.073643
    quant0.5    0.38735   0.1076   0.095938 0.078018 0.080236
    quant0.75   0.42578  0.13859    0.11829 0.091821 0.087838
    quant0.975  0.51055  0.22741    0.17723   0.1254   0.1049

Also construct a plot the SD posteriors:

``` r
ddf <- data.frame(rbind(sigmaLab,sigmaTech,sigmaSample,sigmaepsilon),errterm=gl(4,nrow(sigmaLab),labels = c("Lab","Tech","Samp","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("Fat")+ylab("density")+xlim(0,0.25)
```

![](figs/plotsdseggs-1..svg)<!-- -->

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
    (Intercept) 0.388 0.052      0.284    0.388      0.491 0.388   0

    Model hyperparameters:
                                              mean      sd 0.025quant 0.5quant 0.975quant   mode
    Precision for the Gaussian observations 141.85   39.79      77.66   137.24     233.17 128.64
    Precision for Lab                       490.87 1111.06      28.46   209.63    2723.75  67.44
    Precision for labtech                   185.97  175.16      24.77   135.59     651.27  66.47
    Precision for labtechsamp               445.57  469.18      65.44   306.69    1677.07 159.04

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
    mean         0.3875 0.077772   0.094461 0.061346 0.086422
    sd         0.051499 0.043738   0.041417 0.025354 0.012121
    quant0.025  0.28382 0.019296   0.039391 0.024569 0.065639
    quant0.25   0.35559 0.045507   0.065163 0.042957 0.077795
    quant0.5    0.38737 0.069276   0.085701  0.05713  0.08531
    quant0.75   0.41916  0.10027    0.11409  0.07489 0.093898
    quant0.975  0.49092  0.18586    0.19924  0.12267  0.11315

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaLab,sigmaTech,sigmaSample,sigmaepsilon),errterm=gl(4,nrow(sigmaLab),labels = c("Lab","Tech","Samp","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("Fat")+ylab("density")+xlim(0,0.25)
```

![](figs/eggspc-1..svg)<!-- -->

Posteriors have generally smaller values for the three random effects
and the possibility of values closer to zero is given greater weight.

# STAN

[STAN](https://mc-stan.org/) performs Bayesian inference using MCMC. Set
up STAN to use multiple cores. Set the random number seed for
reproducibility.

``` r
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(123)
```

Requires use of STAN command file [eggs.stan](../stancode/eggs.stan). We
have used half-cauchy priors for four variances using the SD of the
response to help with scaling. We view the code here:

``` r
writeLines(readLines("../stancode/eggs.stan"))
```

    data {
         int<lower=0> Nobs;
         int<lower=0> Nlev1;
         int<lower=0> Nlev2;
         int<lower=0> Nlev3;
         vector[Nobs] y;
         int<lower=1,upper=Nlev1> levind1[Nobs];
         int<lower=1,upper=Nlev2> levind2[Nobs];
         int<lower=1,upper=Nlev3> levind3[Nobs];
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

``` r
rt <- stanc(file="../stancode/eggs.stan")
sm <- stan_model(stanc_ret = rt, verbose=FALSE)
```

    Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    clang -mmacosx-version-min=10.13 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppParallel/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -DBOOST_NO_AUTO_PTR  -include '/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include   -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:88:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:1: error: unknown type name 'namespace'
    namespace Eigen {
    ^
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:16: error: expected ';' after top level declarator
    namespace Eigen {
                   ^
                   ;
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    #include <complex>
             ^~~~~~~~~
    3 errors generated.
    make: *** [foo.o] Error 1

``` r
system.time(fit <- sampling(sm, data=eggdat, iter=10000))
```

       user  system elapsed 
     20.269   0.934   8.576 

## Diagnostics

For the error SD:

``` r
pname <- "sigmaeps"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/eggssigmaeps-1..svg)<!-- -->

For the Lab SD

``` r
pname <- "sigmalev1"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/eggssigmalev1-1..svg)<!-- -->

For the technician SD

``` r
pname <- "sigmalev2"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/eggssigmalev2-1..svg)<!-- -->

For the sample SD

``` reggssigmalev3
pname <- "sigmalev3"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

All these are satisfactory.

## Output summaries

Display the parameters of interest:

``` r
print(fit,pars=c("mu","sigmalev1","sigmalev2","sigmalev3","sigmaeps"))
```

    Inference for Stan model: eggs.
    4 chains, each with iter=10000; warmup=5000; thin=1; 
    post-warmup draws per chain=5000, total post-warmup draws=20000.

              mean se_mean   sd 2.5%  25%  50%  75% 97.5% n_eff Rhat
    mu        0.40    0.01 0.08 0.27 0.36 0.39 0.42  0.67    85 1.05
    sigmalev1 0.11    0.01 0.09 0.01 0.05 0.09 0.13  0.43    74 1.06
    sigmalev2 0.09    0.00 0.05 0.01 0.06 0.09 0.12  0.19   540 1.01
    sigmalev3 0.06    0.00 0.03 0.00 0.04 0.06 0.08  0.12  2887 1.00
    sigmaeps  0.09    0.00 0.01 0.07 0.08 0.09 0.10  0.12  5143 1.00

    Samples were drawn using NUTS(diag_e) at Tue Jul 12 13:20:13 2022.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).

We see the posterior mean, SE and SD, quantiles of the samples. The
posterior means are somewhat different from the REML estimates seen in
the `lmer` output. The `n_eff` is a rough measure of the sample size
taking into account the correlation in the samples. The effective sample
sizes for the primary parameters are respectable. The
![\hat R](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%20R "\hat R")
statistics are good.

## Posterior Distributions

We can use extract to get at various components of the STAN fit.

``` r
postsig <- rstan::extract(fit, pars=c("sigmalev1","sigmalev2","sigmalev3","sigmaeps"))
ref <- reshape2::melt(postsig)
colnames(ref)[2:3] <- c("Fat","SD")
ggplot(data=ref,aes(x=Fat, color=SD))+geom_density()
```

![](figs/eggsstanhypsd-1..svg)<!-- -->

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
suppressMessages(bmod <- brm(Fat ~ 1 + (1|Lab/Technician/Sample), data=eggs,iter=10000, cores=4))
```

    Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    clang -mmacosx-version-min=10.13 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppParallel/include/"  -I"/Library/Frameworks/R.framework/Versions/4.2/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -DBOOST_DISABLE_ASSERTS  -DBOOST_PENDING_INTEGER_LOG2_HPP  -DSTAN_THREADS  -DUSE_STANC3 -DSTRICT_R_HEADERS  -DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION  -DBOOST_NO_AUTO_PTR  -include '/Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp'  -D_REENTRANT -DRCPP_PARALLEL_USE_TBB=1   -I/usr/local/include   -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:88:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:1: error: unknown type name 'namespace'
    namespace Eigen {
    ^
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:628:16: error: expected ';' after top level declarator
    namespace Eigen {
                   ^
                   ;
    In file included from <built-in>:1:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/StanHeaders/include/stan/math/prim/fun/Eigen.hpp:22:
    In file included from /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Dense:1:
    /Library/Frameworks/R.framework/Versions/4.2/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    #include <complex>
             ^~~~~~~~~
    3 errors generated.
    make: *** [foo.o] Error 1

We get some warnings. We can obtain some posterior densities and
diagnostics with:

``` r
plot(bmod, variable = "^s", regex=TRUE)
```

![](figs/eggsbrmsdiag-1..svg)<!-- -->

We have chosen only the random effect hyperparameters since this is
where problems will appear first. Looks OK. We can see some weight is
given to values of the random effect SDs close to zero.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.17.0
    functions {
    }
    data {
      int<lower=1> N;  // total number of observations
      vector[N] Y;  // response variable
      // data for group-level effects of ID 1
      int<lower=1> N_1;  // number of grouping levels
      int<lower=1> M_1;  // number of coefficients per level
      int<lower=1> J_1[N];  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_1_1;
      // data for group-level effects of ID 2
      int<lower=1> N_2;  // number of grouping levels
      int<lower=1> M_2;  // number of coefficients per level
      int<lower=1> J_2[N];  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_2_1;
      // data for group-level effects of ID 3
      int<lower=1> N_3;  // number of grouping levels
      int<lower=1> M_3;  // number of coefficients per level
      int<lower=1> J_3[N];  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_3_1;
      int prior_only;  // should the likelihood be ignored?
    }
    transformed data {
    }
    parameters {
      real Intercept;  // temporary intercept for centered predictors
      real<lower=0> sigma;  // dispersion parameter
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      vector[N_1] z_1[M_1];  // standardized group-level effects
      vector<lower=0>[M_2] sd_2;  // group-level standard deviations
      vector[N_2] z_2[M_2];  // standardized group-level effects
      vector<lower=0>[M_3] sd_3;  // group-level standard deviations
      vector[N_3] z_3[M_3];  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      vector[N_2] r_2_1;  // actual group-level effects
      vector[N_3] r_3_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      r_2_1 = (sd_2[1] * (z_2[1]));
      r_3_1 = (sd_3[1] * (z_3[1]));
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
        vector[N] mu = Intercept + rep_vector(0.0, N);
        for (n in 1:N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n] + r_2_1[J_2[n]] * Z_2_1[n] + r_3_1[J_3[n]] * Z_3_1[n];
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
    sd(Intercept)     0.10      0.07     0.01     0.28 1.00     5668     8636

    ~Lab:Technician (Number of levels: 12) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.09      0.05     0.01     0.20 1.00     4132     3716

    ~Lab:Technician:Sample (Number of levels: 24) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.06      0.03     0.00     0.13 1.00     3703     6289

    Population-Level Effects: 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept     0.39      0.06     0.26     0.51 1.00     9195     7542

    Family Specific Parameters: 
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma     0.09      0.01     0.07     0.12 1.00     7833    11806

    Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
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

![](figs/eggsginlaint-1..svg)<!-- -->

and for the laboratory effects as:

``` r
xmat = t(gimod$beta[2:7,])
ymat = t(gimod$density[2:7,])
matplot(xmat, ymat,type="l",xlab="fat",ylab="density")
legend("right",paste0("Lab",1:6),col=1:6,lty=1:6)
```

![](figs/eggsginlaleff-1..svg)<!-- -->

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

![](figs/eggsginlateff-1..svg)<!-- -->

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

-   As with some of the other analyses, we see that the INLA-produced
    posterior densities for the random effect SDs are well-bounded away
    from zero. We see that the choice of prior does make an important
    difference and that the default choice is a clear failure.

-   The default STAN priors produce a credible result and posteriors do
    give some weight to values close to zero. There is no ground truth
    here but given the experience in the `lmer` analysis, there does
    appear to be some suggestion that any of the three sources of
    variation could be very small. INLA is the odd-one-out in this
    instance.

-   The `mgcv` based analysis is mostly the same as the `lme4` fit
    excepting the confidence intervals where a different method has been
    used.

-   The `ginla` does not readily produce posterior densities for the
    hyperparameters so we cannot compare on that basis. The other
    posteriors were produced very rapidly.

# Package version info

``` r
sessionInfo()
```

    R version 4.2.1 (2022-06-23)
    Platform: x86_64-apple-darwin17.0 (64-bit)
    Running under: macOS Big Sur ... 10.16

    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.2/Resources/lib/libRblas.0.dylib
    LAPACK: /Library/Frameworks/R.framework/Versions/4.2/Resources/lib/libRlapack.dylib

    locale:
    [1] en_GB.UTF-8/en_GB.UTF-8/en_GB.UTF-8/C/en_GB.UTF-8/en_GB.UTF-8

    attached base packages:
    [1] parallel  stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] mgcv_1.8-40         nlme_3.1-157        brms_2.17.0         Rcpp_1.0.8.3        rstan_2.26.13      
     [6] StanHeaders_2.26.13 knitr_1.39          INLA_22.06.20-2     sp_1.4-7            foreach_1.5.2      
    [11] RLRsim_3.1-8        pbkrtest_0.5.1      lme4_1.1-29         Matrix_1.4-1        ggplot2_3.3.6      
    [16] faraway_1.0.8      

    loaded via a namespace (and not attached):
      [1] minqa_1.2.4          colorspace_2.0-3     ellipsis_0.3.2       ggridges_0.5.3       markdown_1.1        
      [6] base64enc_0.1-3      rstudioapi_0.13      Deriv_4.1.3          farver_2.1.0         DT_0.23             
     [11] fansi_1.0.3          mvtnorm_1.1-3        bridgesampling_1.1-2 codetools_0.2-18     splines_4.2.1       
     [16] shinythemes_1.2.0    bayesplot_1.9.0      jsonlite_1.8.0       nloptr_2.0.3         broom_0.8.0         
     [21] shiny_1.7.1          compiler_4.2.1       backports_1.4.1      assertthat_0.2.1     fastmap_1.1.0       
     [26] cli_3.3.0            later_1.3.0          htmltools_0.5.2      prettyunits_1.1.1    tools_4.2.1         
     [31] igraph_1.3.1         coda_0.19-4          gtable_0.3.0         glue_1.6.2           reshape2_1.4.4      
     [36] dplyr_1.0.9          posterior_1.2.2      V8_4.2.0             vctrs_0.4.1          svglite_2.1.0       
     [41] iterators_1.0.14     crosstalk_1.2.0      tensorA_0.36.2       xfun_0.31            stringr_1.4.0       
     [46] ps_1.7.0             mime_0.12            miniUI_0.1.1.1       lifecycle_1.0.1      gtools_3.9.2.1      
     [51] MASS_7.3-57          zoo_1.8-10           scales_1.2.0         colourpicker_1.1.1   promises_1.2.0.1    
     [56] Brobdingnag_1.2-7    inline_0.3.19        shinystan_2.6.0      yaml_2.3.5           curl_4.3.2          
     [61] gridExtra_2.3        loo_2.5.1            stringi_1.7.6        highr_0.9            dygraphs_1.1.1.6    
     [66] checkmate_2.1.0      boot_1.3-28          pkgbuild_1.3.1       systemfonts_1.0.4    rlang_1.0.2         
     [71] pkgconfig_2.0.3      matrixStats_0.62.0   distributional_0.3.0 evaluate_0.15        lattice_0.20-45     
     [76] purrr_0.3.4          labeling_0.4.2       rstantools_2.2.0     htmlwidgets_1.5.4    processx_3.5.3      
     [81] tidyselect_1.1.2     plyr_1.8.7           magrittr_2.0.3       R6_2.5.1             generics_0.1.2      
     [86] DBI_1.1.2            pillar_1.7.0         withr_2.5.0          xts_0.12.1           abind_1.4-5         
     [91] tibble_3.1.7         crayon_1.5.1         utf8_1.2.2           rmarkdown_2.14       grid_4.2.1          
     [96] callr_3.7.0          threejs_0.3.3        digest_0.6.29        xtable_1.8-4         tidyr_1.2.0         
    [101] httpuv_1.6.5         RcppParallel_5.1.5   stats4_4.2.1         munsell_0.5.0        shinyjs_2.1.0       
