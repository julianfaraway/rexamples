Crossed Effects Design
================
[Julian Faraway](https://julianfaraway.github.io/)
27 September 2022

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
    -   <a href="#output-summary" id="toc-output-summary">Output Summary</a>
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

Effects are said to be crossed when they are not nested. When at least
some crossing occurs, methods for nested designs cannot be used. We
consider a latin square example.

In an experiment, four materials, A, B, C and D, were fed into a
wear-testing machine. The response is the loss of weight in 0.1 mm over
the testing period. The machine could process four samples at a time and
past experience indicated that there were some differences due to the
position of these four samples. Also some differences were suspected
from run to run. Four runs were made. The latin square structure of the
design may be observed:

``` r
data(abrasion, package="faraway")
matrix(abrasion$material,4,4)
```

         [,1] [,2] [,3] [,4]
    [1,] "C"  "A"  "D"  "B" 
    [2,] "D"  "B"  "C"  "A" 
    [3,] "B"  "D"  "A"  "C" 
    [4,] "A"  "C"  "B"  "D" 

We can plot the data

``` r
ggplot(abrasion,aes(x=material, y=wear, shape=run, color=position))+geom_point(position = position_jitter(width=0.1, height=0.0))
```

![](figs/abrasionplot-1..svg)<!-- -->

# Mixed Effect Model

Since we are most interested in the choice of material, treating this as
a fixed effect is natural. We must account for variation due to the run
and the position but were not interested in their specific values
because we believe these may vary between experiments. We treat these as
random effects.

``` r
mmod <- lmer(wear ~ material + (1|run) + (1|position), abrasion)
faraway::sumary(mmod)
```

    Fixed Effects:
                coef.est coef.se
    (Intercept) 265.75     7.67 
    materialB   -45.75     5.53 
    materialC   -24.00     5.53 
    materialD   -35.25     5.53 

    Random Effects:
     Groups   Name        Std.Dev.
     run      (Intercept)  8.18   
     position (Intercept) 10.35   
     Residual              7.83   
    ---
    number of obs: 16, groups: run, 4; position, 4
    AIC = 114.3, DIC = 140.4
    deviance = 120.3 

We test the random effects:

``` r
mmodp <- lmer(wear ~ material + (1|position), abrasion)
mmodr <- lmer(wear ~ material + (1|run), abrasion)
exactRLRT(mmodp, mmod, mmodr)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 4.59, p-value = 0.014

``` r
exactRLRT(mmodr, mmod, mmodp)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 3.05, p-value = 0.034

We see both are statistically significant.

We can test the fixed effect:

``` r
mmod <- lmer(wear ~ material + (1|run) + (1|position), abrasion,REML=FALSE)
nmod <- lmer(wear ~ 1+ (1|run) + (1|position), abrasion,REML=FALSE)
KRmodcomp(mmod, nmod)
```

    large : wear ~ material + (1 | run) + (1 | position)
    small : wear ~ 1 + (1 | run) + (1 | position)
          stat  ndf  ddf F.scaling p.value
    Ftest 25.1  3.0  6.0         1 0.00085

We see the fixed effect is significant.

We can compute confidence intervals for the parameters:

``` r
confint(mmod, method="boot")
```

                      2.5 %   97.5 %
    .sig01       5.7029e-07  13.5882
    .sig02       1.3238e+00  15.0680
    .sigma       2.0629e+00   7.9128
    (Intercept)  2.5143e+02 278.2622
    materialB   -5.4698e+01 -37.4614
    materialC   -3.2946e+01 -15.8291
    materialD   -4.4933e+01 -25.2836

The lower ends of the confidence intervals for the random effect SDs are
zero (or close).

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

``` r
formula <- wear ~ material + f(run, model="iid") + f(position, model="iid")
result <- inla(formula, family="gaussian", data=abrasion)
summary(result)
```

    Fixed effects:
                   mean     sd 0.025quant 0.5quant 0.975quant    mode kld
    (Intercept) 261.031  7.079    246.269  261.275    274.357 261.702   0
    materialB   -38.888 10.013    -57.699  -39.247    -17.975 -39.874   0
    materialC   -18.290  9.923    -37.127  -18.579      2.240 -19.083   0
    materialD   -28.944  9.967    -47.763  -29.269     -8.221 -29.837   0

    Model hyperparameters:
                                                mean       sd 0.025quant 0.5quant 0.975quant     mode
    Precision for the Gaussian observations 5.00e-03 2.00e-03      0.002 5.00e-03   9.00e-03    0.004
    Precision for run                       2.31e+04 2.47e+04   1672.931 1.56e+04   8.87e+04 4595.388
    Precision for position                  2.56e+04 3.03e+04   2328.346 1.65e+04   1.05e+05 6256.278

     is computed 

The run and position precisions look far too high. Need to change the
default prior.

## Informative Gamma priors on the precisions

Now try more informative gamma priors for the random effect precisions.
Define it so the mean value of gamma prior is set to the inverse of the
variance of the residuals of the fixed-effects only model. We expect the
error variances to be lower than this variance so this is an
overestimate. The variance of the gamma prior (for the precision) is
controlled by the `apar` shape parameter.

``` r
apar <- 0.5
lmod <- lm(wear ~ material, abrasion)
bpar <- apar*var(residuals(lmod))
lgprior <- list(prec = list(prior="loggamma", param = c(apar,bpar)))
formula = wear ~ material+f(run, model="iid", hyper = lgprior)+f(position, model="iid", hyper = lgprior)
result <- inla(formula, family="gaussian", data=abrasion)
summary(result)
```

    Fixed effects:
                   mean     sd 0.025quant 0.5quant 0.975quant    mode kld
    (Intercept) 264.421 10.100    244.072  264.440    284.700 264.495   0
    materialB   -43.835  5.164    -53.548  -44.052    -32.785 -44.378   0
    materialC   -22.374  5.137    -32.138  -22.557    -11.483 -22.832   0
    materialD   -33.474  5.151    -43.211  -33.675    -22.502 -33.976   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.022 0.011      0.007    0.020      0.049 0.016
    Precision for run                       0.011 0.009      0.002    0.008      0.034 0.005
    Precision for position                  0.009 0.008      0.002    0.007      0.029 0.004

     is computed 

Results are more credible.

Compute the transforms to an SD scale for the random effect terms. Make
a table of summary statistics for the posteriors:

``` r
sigmarun <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmapos <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[3]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmarun,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmapos,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","B -- A","C -- A","D -- A","run","position","epsilon")
data.frame(restab) |> kable()
```

|            | mu     | B….A    | C….A    | D….A    | run    | position | epsilon |
|:-----------|:-------|:--------|:--------|:--------|:-------|:---------|:--------|
| mean       | 264.42 | -43.838 | -22.377 | -33.477 | 11.941 | 12.817   | 7.351   |
| sd         | 10.09  | 5.159   | 5.1322  | 5.1455  | 4.6763 | 4.8929   | 1.9492  |
| quant0.025 | 244.07 | -53.55  | -32.14  | -43.213 | 5.4351 | 5.8539   | 4.5347  |
| quant0.25  | 258.05 | -47.191 | -25.696 | -36.814 | 8.6067 | 9.3124   | 5.9524  |
| quant0.5   | 264.42 | -44.066 | -22.571 | -33.689 | 11.037 | 11.926   | 6.9914  |
| quant0.75  | 270.75 | -40.759 | -19.293 | -30.396 | 14.283 | 15.336   | 8.3808  |
| quant0.975 | 284.65 | -32.814 | -11.512 | -22.531 | 23.554 | 24.834   | 12.107  |

Also construct a plot the SD posteriors:

``` r
ddf <- data.frame(rbind(sigmarun,sigmapos,sigmaepsilon),errterm=gl(3,nrow(sigmarun),labels = c("run","position","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("wear")+ylab("density")+xlim(0,35)
```

![](figs/plotsdsab-1..svg)<!-- -->

Posteriors look OK although no weight given to smaller values.

## Penalized Complexity Prior

In [Simpson et al (2015)](http://arxiv.org/abs/1403.4630v3), penalized
complexity priors are proposed. This requires that we specify a scaling
for the SDs of the random effects. We use the SD of the residuals of the
fixed effects only model (what might be called the base model in the
paper) to provide this scaling.

``` r
lmod <- lm(wear ~ material, abrasion)
sdres <- sd(residuals(lmod))
pcprior <- list(prec = list(prior="pc.prec", param = c(3*sdres,0.01)))
formula = wear ~ material+f(run, model="iid", hyper = pcprior)+f(position, model="iid", hyper = pcprior)
result <- inla(formula, family="gaussian", data=abrasion)
summary(result)
```

    Fixed effects:
                   mean    sd 0.025quant 0.5quant 0.975quant    mode kld
    (Intercept) 264.347 8.770    246.651  264.370    281.961 264.437   0
    materialB   -43.728 5.319    -53.696  -43.975    -32.258 -44.348   0
    materialC   -22.284 5.288    -32.310  -22.492    -10.991 -22.805   0
    materialD   -33.376 5.303    -43.370  -33.604    -21.993 -33.948   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.022 0.011      0.006    0.020      0.048 0.016
    Precision for run                       0.020 0.021      0.003    0.014      0.075 0.007
    Precision for position                  0.012 0.012      0.002    0.009      0.044 0.005

     is computed 

Compute the summaries as before:

``` r
sigmarun <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmapos <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[3]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmarun,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmapos,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","B -- A","C -- A","D -- A","run","position","epsilon")
data.frame(restab) |> kable()
```

|            | mu     | B….A    | C….A    | D….A    | run    | position | epsilon |
|:-----------|:-------|:--------|:--------|:--------|:-------|:---------|:--------|
| mean       | 264.35 | -43.732 | -22.288 | -33.379 | 9.1605 | 11.481   | 7.4515  |
| sd         | 8.7617 | 5.313   | 5.2824  | 5.2976  | 4.0675 | 4.8646   | 2.0689  |
| quant0.025 | 246.65 | -53.698 | -32.312 | -43.372 | 3.6785 | 4.7703   | 4.5631  |
| quant0.25  | 258.85 | -47.177 | -25.694 | -36.805 | 6.2692 | 8.0072   | 5.9679  |
| quant0.5   | 264.35 | -43.99  | -22.507 | -33.619 | 8.331  | 10.537   | 7.0373  |
| quant0.75  | 269.82 | -40.59  | -19.142 | -30.236 | 11.126 | 13.901   | 8.5105  |
| quant0.975 | 281.92 | -32.289 | -11.022 | -22.023 | 19.383 | 23.586   | 12.575  |

Make the plots:

``` r
ddf <- data.frame(rbind(sigmarun,sigmapos,sigmaepsilon),errterm=gl(3,nrow(sigmarun),labels = c("run","position","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("wear")+ylab("density")+xlim(0,35)
```

![](figs/abrapc-1..svg)<!-- -->

Posteriors put more weight on lower values compared to gamma prior.

# STAN

[STAN](https://mc-stan.org/) performs Bayesian inference using MCMC. Set
up STAN to use multiple cores. Set the random number seed for
reproducibility.

``` r
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(123)
```

Fit the model. Requires use of STAN command file
[abrasion.stan](../stancode/abrasion.stan). We have used uninformative
priors for the fixed effects and half-cauchy priors for the three
variances. We view the code here:

``` r
writeLines(readLines("../stancode/abrasion.stan"))
```

    /*
    Latin square style design
    */
    data {
      int<lower=0> N;
      int<lower=0> Nt;
      int<lower=1,upper=Nt> treat[N];
      int<lower=1,upper=Nt> blk1[N];
      int<lower=1,upper=Nt> blk2[N];
      vector[N] y;
      real<lower=0> sdscal;
    }
    parameters {
      vector[Nt] eta1;
      vector[Nt] eta2;
      vector[Nt] trt;
      real<lower=0> sigmab1;
      real<lower=0> sigmab2;
      real<lower=0> sigmaeps;
    }
    transformed parameters {
      vector[Nt] bld1;
      vector[Nt] bld2;
      vector[N] yhat;

      bld1 = sigmab1 * eta1;
      bld2 = sigmab2 * eta2;

      for (i in 1:N)
        yhat[i] = trt[treat[i]] + bld1[blk1[i]] + bld2[blk2[i]];

    }
    model {
      eta1 ~ normal(0, 1);
      eta2 ~ normal(0, 1);
      sigmab1 ~ cauchy(0, 2.5*sdscal);
      sigmab2 ~ cauchy(0, 2.5*sdscal);
      sigmaeps ~ cauchy(0, 2.5*sdscal);

      y ~ normal(yhat, sigmaeps);
    }

Prepare data in a format consistent with the command file. Needs to be a
list:

``` r
sdscal <- sd(residuals(lm(wear ~ material, abrasion)))
abrdat <- list(N=16, Nt=4, treat=as.numeric(abrasion$material), blk1=as.numeric(abrasion$run), blk2=as.numeric(abrasion$position), y=abrasion$wear, sdscal=sdscal)
```

Run in three steps:

``` r
rt <- stanc(file="../stancode/abrasion.stan")
sm <- stan_model(stanc_ret = rt, verbose=FALSE)
system.time(fit <- sampling(sm, data=abrdat))
```

       user  system elapsed 
      3.829   0.272   1.698 

We have not used an overall mean. If we want an overall mean parameter,
we have to set up dummy variables. We can do this but it requires more
work.

## Diagnostics

For the error SD:

``` r
pname <- "sigmaeps"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/abrasigmaepsdiag-1..svg)<!-- -->

For the block one (run) SD:

``` r
pname <- "sigmab1"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/abrasigmab1diag-1..svg)<!-- -->

For the block two (position) SD:

``` r
pname <- "sigmab2"
muc <- rstan::extract(fit, pars=pname,  permuted=FALSE, inc_warmup=FALSE)
mdf <- reshape2::melt(muc)
ggplot(mdf,aes(x=iterations,y=value,color=chains)) + geom_line() + ylab(mdf$parameters[1])
```

![](figs/abrasigmab2-1..svg)<!-- -->

Everything looks reasonable.

## Output Summary

``` r
print(fit,pars=c("trt","sigmaeps","sigmab1","sigmab2"))
```

    Inference for Stan model: abrasion.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.

               mean se_mean    sd   2.5%    25%    50%    75%  97.5% n_eff Rhat
    trt[1]   266.66    0.58 13.20 240.48 258.65 266.51 274.04 297.03   515 1.00
    trt[2]   221.02    0.59 13.37 194.36 213.08 220.68 228.22 252.05   514 1.00
    trt[3]   242.72    0.59 13.38 216.90 234.41 242.53 250.26 272.98   515 1.00
    trt[4]   231.50    0.59 13.21 204.98 223.76 231.00 238.84 261.86   503 1.00
    sigmaeps  10.67    0.17  3.88   5.56   7.85   9.90  12.61  20.08   517 1.01
    sigmab1   11.77    0.33  9.00   0.94   6.09   9.84  14.54  36.34   731 1.00
    sigmab2   15.06    0.46 10.88   2.19   8.25  12.36  18.54  44.85   563 1.01

    Samples were drawn using NUTS(diag_e) at Thu Aug  4 15:00:32 2022.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).

The effective sample sizes are a bit low so we might want to rerun this
with more iterations if we care about the tails in particular.

## Posterior Distributions

We can use extract to get at various components of the STAN fit. First
consider the SDs for random components:

``` r
postsig <- rstan::extract(fit, pars=c("sigmaeps","sigmab1","sigmab2"))
ref <- reshape2::melt(postsig)
colnames(ref)[2:3] <- c("wear","SD")
ggplot(data=ref,aes(x=wear, color=SD))+geom_density()
```

![](figs/abrapostsig-1..svg)<!-- -->

As usual the error SD distribution is a bit more concentrated. We can
see some weight at zero for the random effects in contrast to the INLA
posteriors.

Now the treatment effects:

``` r
postsig <- rstan::extract(fit, pars="trt")
ref <- reshape2::melt(postsig)
colnames(ref)[2:3] <- c("material","wear")
ref$material <- LETTERS[1:4][ref$material]
ggplot(data=ref,aes(x=wear, color=material))+geom_density()
```

![](figs/abraposttrt-1..svg)<!-- -->

We can see that material A shows some separation from the other levels.

# BRMS

[BRMS](https://paul-buerkner.github.io/brms/) stands for Bayesian
Regression Models with STAN. It provides a convenient wrapper to STAN
functionality. We specify the model as in `lmer()` above. I have used
more than the standard number of iterations because this reduces some
problems and does not cost much computationally.

``` r
suppressMessages(bmod <- brm(wear ~ material + (1|run) + (1|position), data=abrasion,iter=10000, cores=4))
```

We get some minor warnings. We can obtain some posterior densities and
diagnostics with:

``` r
plot(bmod, variable = "^s", regex=TRUE)
```

![](figs/abrabrmsdiag-1..svg)<!-- -->

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
      int<lower=1> K;  // number of population-level effects
      matrix[N, K] X;  // population-level design matrix
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
      int prior_only;  // should the likelihood be ignored?
    }
    transformed data {
      int Kc = K - 1;
      matrix[N, Kc] Xc;  // centered version of X without an intercept
      vector[Kc] means_X;  // column means of X before centering
      for (i in 2:K) {
        means_X[i - 1] = mean(X[, i]);
        Xc[, i - 1] = X[, i] - means_X[i - 1];
      }
    }
    parameters {
      vector[Kc] b;  // population-level effects
      real Intercept;  // temporary intercept for centered predictors
      real<lower=0> sigma;  // dispersion parameter
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      vector[N_1] z_1[M_1];  // standardized group-level effects
      vector<lower=0>[M_2] sd_2;  // group-level standard deviations
      vector[N_2] z_2[M_2];  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      vector[N_2] r_2_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      r_2_1 = (sd_2[1] * (z_2[1]));
      lprior += student_t_lpdf(Intercept | 3, 234.5, 13.3);
      lprior += student_t_lpdf(sigma | 3, 0, 13.3)
        - 1 * student_t_lccdf(0 | 3, 0, 13.3);
      lprior += student_t_lpdf(sd_1 | 3, 0, 13.3)
        - 1 * student_t_lccdf(0 | 3, 0, 13.3);
      lprior += student_t_lpdf(sd_2 | 3, 0, 13.3)
        - 1 * student_t_lccdf(0 | 3, 0, 13.3);
    }
    model {
      // likelihood including constants
      if (!prior_only) {
        // initialize linear predictor term
        vector[N] mu = Intercept + rep_vector(0.0, N);
        for (n in 1:N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n] + r_2_1[J_2[n]] * Z_2_1[n];
        }
        target += normal_id_glm_lpdf(Y | Xc, mu, b, sigma);
      }
      // priors including constants
      target += lprior;
      target += std_normal_lpdf(z_1[1]);
      target += std_normal_lpdf(z_2[1]);
    }
    generated quantities {
      // actual population-level intercept
      real b_Intercept = Intercept - dot_product(means_X, b);
    }

We see that `brms` is using student t distributions with 3 degrees of
freedom for the priors. For the three error SDs, this will be truncated
at zero to form half-t distributions. You can get a more explicit
description of the priors with `prior_summary(bmod)`. These are
qualitatively similar to the the PC prior used in the INLA fit.

We examine the fit:

``` r
summary(bmod)
```

     Family: gaussian 
      Links: mu = identity; sigma = identity 
    Formula: wear ~ material + (1 | run) + (1 | position) 
       Data: abrasion (Number of observations: 16) 
      Draws: 4 chains, each with iter = 10000; warmup = 5000; thin = 1;
             total post-warmup draws = 20000

    Group-Level Effects: 
    ~position (Number of levels: 4) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)    11.12      6.06     2.01    26.38 1.00     5235     5130

    ~run (Number of levels: 4) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     9.06      5.48     0.86    22.64 1.00     4925     5356

    Population-Level Effects: 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept   264.49      8.31   247.80   280.88 1.00     7743     8259
    materialB   -45.78      7.56   -61.09   -31.02 1.00    11543    10011
    materialC   -24.08      7.51   -39.57    -8.94 1.00    12114    10247
    materialD   -35.23      7.62   -50.63   -20.28 1.00    11908     9705

    Family Specific Parameters: 
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma    10.15      3.41     5.45    18.50 1.00     3533     8074

    Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

The results are consistent with those seen previously.

# MGCV

It is possible to fit some GLMMs within the GAM framework of the `mgcv`
package. An explanation of this can be found in this
[blog](https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/)

``` r
gmod = gam(wear ~ material + s(run,bs="re") + s(position,bs="re"),
           data=abrasion, method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    wear ~ material + s(run, bs = "re") + s(position, bs = "re")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)   265.75       7.67   34.66  5.0e-09
    materialB     -45.75       5.53   -8.27  7.8e-05
    materialC     -24.00       5.53   -4.34  0.00349
    materialD     -35.25       5.53   -6.37  0.00039

    Approximate significance of smooth terms:
                 edf Ref.df    F p-value
    s(run)      2.44      3 4.37   0.031
    s(position) 2.62      3 6.99   0.012

    R-sq.(adj) =  0.877   Deviance explained = 94.3%
    -REML = 50.128  Scale est. = 61.25     n = 16

We get the fixed effect estimates. We also get tests on the random
effects (as described in this
[article](https://doi.org/10.1093/biomet/ast038). The hypothesis of no
variation is rejected for both the run and the position. This is
consistent with earlier findings.

We can get an estimate of the operator and error SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

                std.dev  lower  upper
    s(run)       8.1790 3.0337 22.051
    s(position) 10.3471 4.1311 25.916
    scale        7.8262 4.4446 13.781

    Rank: 3/3

The point estimates are the same as the REML estimates from `lmer`
earlier. The confidence intervals are different. A bootstrap method was
used for the `lmer` fit whereas `gam` is using an asymptotic
approximation resulting in substantially different results. Given the
problems of parameters on the boundary present in this example, the
bootstrap results appear more trustworthy.

The random effect estimates for the fields can be found with:

``` r
coef(gmod)
```

      (Intercept)     materialB     materialC     materialD      s(run).1      s(run).2      s(run).3      s(run).4 
        265.75000     -45.75000     -24.00000     -35.25000      -0.20343      -2.03434       9.96826      -7.73049 
    s(position).1 s(position).2 s(position).3 s(position).4 
         -9.40487      13.56051      -1.96846      -2.18718 

# GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(wear ~ material + s(run,bs="re") + s(position,bs="re"),
           data=abrasion, fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior density for the intercept as:

``` r
plot(gimod$beta[1,],gimod$density[1,],type="l",xlab="yield",ylab="wear")
```

![](figs/abraginlaint-1..svg)<!-- -->

and for the material effects as:

``` r
xmat = t(gimod$beta[2:4,])
ymat = t(gimod$density[2:4,])
matplot(xmat, ymat,type="l",xlab="wear",ylab="density")
legend("right",paste0("Material",LETTERS[2:4]),col=1:3,lty=1:3)
```

![](figs/abraginlalmat-1..svg)<!-- -->

We can see some overlap between the effects but clear separation from
zero.

The run effects are:

``` r
xmat = t(gimod$beta[5:8,])
ymat = t(gimod$density[5:8,])
matplot(xmat, ymat,type="l",xlab="wear",ylab="density")
legend("right",paste0("run",1:4),col=1:4,lty=1:4)
```

![](figs/abraginlalrun-1..svg)<!-- -->

All the run effects overlap with zero but runs 3 and 4 are more distinct
than 1 and 2.

The position effects are:

``` r
xmat = t(gimod$beta[9:12,])
ymat = t(gimod$density[9:12,])
matplot(xmat, ymat,type="l",xlab="wear",ylab="density")
legend("right",paste0("position",1:4),col=1:4,lty=1:4)
```

![](figs/abraginlalpos-1..svg)<!-- -->

Here positions 1 and 2 are more distinct in their effects.

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments.

-   All but the INLA analysis have some ambiguity about whether there is
    much difference in the run and position effects. In the mixed effect
    model, the effects are just statistically significant while the
    Bayesian analyses also suggest these effects have some impact while
    still expressing some chance that they do not. The INLA analysis
    does not give any weight to the no effect claim.

-   None of the analysis had any problem with the crossed effects.

-   There were no major computational issue with the analyses (in
    contrast with some of the other examples)

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
     [1] mgcv_1.8-40         nlme_3.1-159        brms_2.18.0         Rcpp_1.0.9          rstan_2.26.13      
     [6] StanHeaders_2.26.13 knitr_1.40          INLA_22.09.26       sp_1.5-0            foreach_1.5.2      
    [11] RLRsim_3.1-8        pbkrtest_0.5.1      lme4_1.1-30         Matrix_1.5-1        ggplot2_3.3.6      
    [16] faraway_1.0.9      

    loaded via a namespace (and not attached):
      [1] minqa_1.2.4          colorspace_2.0-3     ellipsis_0.3.2       ggridges_0.5.4       markdown_1.1        
      [6] base64enc_0.1-3      rstudioapi_0.14      Deriv_4.1.3          farver_2.1.1         DT_0.25             
     [11] fansi_1.0.3          mvtnorm_1.1-3        bridgesampling_1.1-2 codetools_0.2-18     splines_4.2.1       
     [16] shinythemes_1.2.0    bayesplot_1.9.0      jsonlite_1.8.0       nloptr_2.0.3         broom_1.0.1         
     [21] shiny_1.7.2          compiler_4.2.1       backports_1.4.1      assertthat_0.2.1     fastmap_1.1.0       
     [26] cli_3.4.1            later_1.3.0          htmltools_0.5.3      prettyunits_1.1.1    tools_4.2.1         
     [31] igraph_1.3.5         coda_0.19-4          gtable_0.3.1         glue_1.6.2           reshape2_1.4.4      
     [36] dplyr_1.0.10         posterior_1.3.1      V8_4.2.1             vctrs_0.4.1          svglite_2.1.0       
     [41] iterators_1.0.14     crosstalk_1.2.0      tensorA_0.36.2       xfun_0.33            stringr_1.4.1       
     [46] ps_1.7.1             mime_0.12            miniUI_0.1.1.1       lifecycle_1.0.2      gtools_3.9.3        
     [51] MASS_7.3-58.1        zoo_1.8-11           scales_1.2.1         colourpicker_1.1.1   promises_1.2.0.1    
     [56] Brobdingnag_1.2-7    inline_0.3.19        shinystan_2.6.0      yaml_2.3.5           curl_4.3.2          
     [61] gridExtra_2.3        loo_2.5.1            stringi_1.7.8        highr_0.9            dygraphs_1.1.1.6    
     [66] checkmate_2.1.0      boot_1.3-28          pkgbuild_1.3.1       systemfonts_1.0.4    rlang_1.0.6         
     [71] pkgconfig_2.0.3      matrixStats_0.62.0   distributional_0.3.1 evaluate_0.16        lattice_0.20-45     
     [76] purrr_0.3.4          labeling_0.4.2       rstantools_2.2.0     htmlwidgets_1.5.4    processx_3.7.0      
     [81] tidyselect_1.1.2     plyr_1.8.7           magrittr_2.0.3       R6_2.5.1             generics_0.1.3      
     [86] DBI_1.1.3            pillar_1.8.1         withr_2.5.0          xts_0.12.1           abind_1.4-5         
     [91] tibble_3.1.8         crayon_1.5.1         utf8_1.2.2           rmarkdown_2.16       grid_4.2.1          
     [96] callr_3.7.2          threejs_0.3.3        digest_0.6.29        xtable_1.8-4         tidyr_1.2.1         
    [101] httpuv_1.6.6         RcppParallel_5.1.5   stats4_4.2.1         munsell_0.5.0        shinyjs_2.1.0       
