# Split Plot Design
[Julian Faraway](https://julianfaraway.github.io/)
2024-08-22

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

In an agricultural field trial, the objective was to determine the
effects of two crop varieties and four different irrigation methods.
Eight fields were available, but only one type of irrigation may be
applied to each field. The fields may be divided into two parts with a
different variety planted in each half. The whole plot factor is the
method of irrigation, which should be randomly assigned to the fields.
Within each field, the variety is randomly assigned.

Load in and plot the data:

``` r
data(irrigation, package="faraway")
summary(irrigation)
```

         field   irrigation variety     yield     
     f1     :2   i1:4       v1:8    Min.   :34.8  
     f2     :2   i2:4       v2:8    1st Qu.:37.6  
     f3     :2   i3:4               Median :40.1  
     f4     :2   i4:4               Mean   :40.2  
     f5     :2                      3rd Qu.:42.7  
     f6     :2                      Max.   :47.6  
     (Other):4                                    

``` r
ggplot(irrigation, aes(y=yield, x=field, shape=variety, color=irrigation)) + geom_point()
```

![](figs/irriplot-1..svg)

# Mixed Effect Model

The irrigation and variety are fixed effects, but the field is a random
effect. We must also consider the interaction between field and variety,
which is necessarily also a random effect because one of the two
components is random. The fullest model that we might consider is:

$$y_{ijk} = \mu + i_i + v_j + (iv)_{ij} + f_k + (vf)_{jk} + \epsilon_{ijk}$$

where $\mu, i_i, v_j, (iv)_{ij}$ are fixed effects; the rest are random
having variances $\sigma^2_f$, $\sigma^2_{vf}$ and $\sigma^2_\epsilon$.
Note that we have no $(if)_{ik}$ term in this model. It would not be
possible to estimate such an effect since only one type of irrigation is
used on a given field; the factors are not crossed. Unfortunately, it is
not possible to distinguish the variety within the field variation. We
would need more than one observation per variety within each field for
us to separate the two variabilities. We resort to a simpler model that
omits the variety by field interaction random effect:

$$y_{ijk} = \mu + i_i + v_j + (iv)_{ij} + f_k +  \epsilon_{ijk}$$

We fit this model with:

``` r
lmod <- lme4::lmer(yield ~ irrigation * variety + (1|field), irrigation)
faraway::sumary(lmod)
```

    Fixed Effects:
                           coef.est coef.se
    (Intercept)            38.50     3.03  
    irrigationi2            1.20     4.28  
    irrigationi3            0.70     4.28  
    irrigationi4            3.50     4.28  
    varietyv2               0.60     1.45  
    irrigationi2:varietyv2 -0.40     2.05  
    irrigationi3:varietyv2 -0.20     2.05  
    irrigationi4:varietyv2  1.20     2.05  

    Random Effects:
     Groups   Name        Std.Dev.
     field    (Intercept) 4.02    
     Residual             1.45    
    ---
    number of obs: 16, groups: field, 8
    AIC = 65.4, DIC = 91.8
    deviance = 68.6 

The fixed effects don’t look very significant. We could use a parametric
bootstrap to test this but it’s less work to use the `pbkrtest` package
which implements the Kenward-Roger approximation. First test the
interaction:

``` r
lmoda <- lmer(yield ~ irrigation + variety + (1|field),data=irrigation)
faraway::sumary(lmoda)
```

    Fixed Effects:
                 coef.est coef.se
    (Intercept)  38.43     2.95  
    irrigationi2  1.00     4.15  
    irrigationi3  0.60     4.15  
    irrigationi4  4.10     4.15  
    varietyv2     0.75     0.60  

    Random Effects:
     Groups   Name        Std.Dev.
     field    (Intercept) 4.07    
     Residual             1.19    
    ---
    number of obs: 16, groups: field, 8
    AIC = 68.8, DIC = 85.1
    deviance = 70.0 

``` r
pbkrtest::KRmodcomp(lmod, lmoda)
```

    large : yield ~ irrigation * variety + (1 | field)
    small : yield ~ irrigation + variety + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.25 3.00 4.00         1    0.86

We can drop the interaction. Now test for a variety effect:

``` r
lmodi <- lmer(yield ~ irrigation + (1|field), irrigation)
KRmodcomp(lmoda, lmodi)
```

    large : yield ~ irrigation + variety + (1 | field)
    small : yield ~ irrigation + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 1.58 1.00 7.00         1    0.25

The variety can go also. Now check the irrigation method.

``` r
lmodv <- lmer(yield ~  variety + (1|field), irrigation)
KRmodcomp(lmoda, lmodv)
```

    large : yield ~ irrigation + variety + (1 | field)
    small : yield ~ variety + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.39 3.00 4.00         1    0.77

This can go also. As a final check, lets compare the null model with no
fixed effects to the full model.

``` r
lmodn <- lmer(yield ~  1 + (1|field), irrigation)
KRmodcomp(lmod, lmodn)
```

    large : yield ~ irrigation * variety + (1 | field)
    small : yield ~ 1 + (1 | field)
          stat  ndf  ddf F.scaling p.value
    Ftest 0.38 7.00 4.48     0.903    0.88

This confirms the lack of statistical significance for the variety and
irrigation factors.

We can check the significance of the random effect (field) term with:

``` r
RLRsim::exactRLRT(lmod)
```


        simulated finite sample distribution of RLRT.
        
        (p-value based on 10000 simulated values)

    data:  
    RLRT = 6.11, p-value = 0.0073

We can see that there is a significant variation among the fields.

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

Try the default INLA fit

``` r
formula <- yield ~ irrigation + variety +f(field, model="iid")
result <- inla(formula, family="gaussian", data=irrigation)
summary(result)
```

    Fixed effects:
                   mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept)  38.465 2.672     33.132   38.459     43.837 38.461   0
    irrigationi2  0.953 3.754     -6.594    0.960      8.455  0.958   0
    irrigationi3  0.556 3.754     -6.989    0.563      8.060  0.561   0
    irrigationi4  4.031 3.754     -3.528    4.041     11.522  4.038   0
    varietyv2     0.750 0.596     -0.439    0.750      1.938  0.750   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.920 0.450      0.307    0.833      2.032 0.673
    Precision for field                     0.102 0.065      0.024    0.086      0.269 0.060

     is computed 

Default looks more plausible than [one way](pulp.md) and
[RBD](penicillin.md) examples.

Compute the transforms to an SD scale for the field and error. Make a
table of summary statistics for the posteriors:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","ir2","ir3","ir4","v2","alpha","epsilon")
data.frame(restab)
```

                   mu     ir2     ir3     ir4       v2  alpha epsilon
    mean       38.465 0.95376 0.55665  4.0313  0.74975 3.5939  1.1349
    sd         2.6709  3.7521  3.7521  3.7522  0.59497 1.1382 0.27978
    quant0.025 33.135 -6.5897 -6.9853 -3.5239 -0.43837 1.9361 0.70356
    quant0.25  36.829 -1.3285 -1.7259  1.7513  0.37919 2.7792 0.93404
    quant0.5   38.452 0.95129 0.55372  4.0325   0.7483 3.3933  1.0928
    quant0.75   40.08  3.2259  2.8285  6.3058   1.1174 4.1944  1.2917
    quant0.975 43.822  8.4348  8.0392  11.501   1.9348 6.3636  1.7956

Also construct a plot the SD posteriors:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("alpha","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,10)
```

![](figs/plotsdsirri-1..svg)

Posteriors look OK.

## Informative Gamma priors on the precisions

Now try more informative gamma priors for the precisions. Define it so
the mean value of gamma prior is set to the inverse of the variance of
the residuals of the fixed-effects only model. We expect the two error
variances to be lower than this variance so this is an overestimate. The
variance of the gamma prior (for the precision) is controlled by the
`apar` shape parameter.

``` r
apar <- 0.5
lmod <- lm(yield ~ irrigation+variety, data=irrigation)
bpar <- apar*var(residuals(lmod))
lgprior <- list(prec = list(prior="loggamma", param = c(apar,bpar)))
formula = yield ~ irrigation+variety+f(field, model="iid", hyper = lgprior)
result <- inla(formula, family="gaussian", data=irrigation)
summary(result)
```

    Fixed effects:
                   mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept)  38.483 3.224     32.042   38.473     44.991 38.477   0
    irrigationi2  0.932 4.538     -8.226    0.944     10.013  0.940   0
    irrigationi3  0.537 4.538     -8.619    0.548      9.620  0.544   0
    irrigationi4  3.999 4.538     -5.181    4.017     13.060  4.010   0
    varietyv2     0.750 0.583     -0.411    0.750      1.911  0.750   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.925 0.446      0.312    0.840      2.025 0.682
    Precision for field                     0.071 0.048      0.015    0.059      0.196 0.039

     is computed 

Compute the summaries as before:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","ir2","ir3","ir4","v2","alpha","epsilon")
data.frame(restab)
```

                   mu     ir2     ir3     ir4       v2  alpha epsilon
    mean       38.484 0.93329 0.53759       4  0.74975 4.3831    1.13
    sd         3.2229  4.5366  4.5366  4.5369  0.58205 1.5068 0.27608
    quant0.025 32.047 -8.2185 -8.6115 -5.1733 -0.41103 2.2654 0.70463
    quant0.25  36.545 -1.7746 -2.1708  1.2961  0.38459 3.3089 0.93181
    quant0.5   38.465 0.93313 0.53662  4.0061  0.74834 4.0925  1.0883
    quant0.75  40.393  3.6326  3.2364  6.7035   1.1121 5.1472  1.2847
    quant0.975 44.974   9.988   9.595  13.035   1.9075 8.1126  1.7821

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("alpha","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,10)
```

![](figs/irrigam-1..svg)

Posteriors look OK.

## Penalized Complexity Prior

In [Simpson et al (2015)](http://arxiv.org/abs/1403.4630v3), penalized
complexity priors are proposed. This requires that we specify a scaling
for the SDs of the random effects. We use the SD of the residuals of the
fixed effects only model (what might be called the base model in the
paper) to provide this scaling.

``` r
lmod <- lm(yield ~ irrigation+variety, irrigation)
sdres <- sd(residuals(lmod))
pcprior <- list(prec = list(prior="pc.prec", param = c(3*sdres,0.01)))
formula <- yield ~ irrigation+variety+f(field, model="iid", hyper = pcprior)
result <- inla(formula, family="gaussian", data=irrigation)
summary(result)
```

    Fixed effects:
                   mean    sd 0.025quant 0.5quant 0.975quant   mode kld
    (Intercept)  38.473 2.922     32.642   38.466     44.343 38.468   0
    irrigationi2  0.944 4.110     -7.313    0.952      9.155  0.950   0
    irrigationi3  0.548 4.110     -7.708    0.555      8.760  0.553   0
    irrigationi4  4.017 4.110     -4.252    4.029     12.217  4.026   0
    varietyv2     0.750 0.585     -0.417    0.750      1.917  0.750   0

    Model hyperparameters:
                                             mean    sd 0.025quant 0.5quant 0.975quant  mode
    Precision for the Gaussian observations 0.924 0.447       0.31    0.838      2.025 0.680
    Precision for field                     0.081 0.052       0.02    0.069      0.217 0.048

     is computed 

Compute the summaries as before:

``` r
sigmaalpha <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[2]])
sigmaepsilon <- inla.tmarginal(function(x) 1/sqrt(exp(x)),result$internal.marginals.hyperpar[[1]])
restab=sapply(result$marginals.fixed, function(x) inla.zmarginal(x,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaalpha,silent=TRUE))
restab=cbind(restab, inla.zmarginal(sigmaepsilon,silent=TRUE))
colnames(restab) = c("mu","ir2","ir3","ir4","v2","alpha","epsilon")
data.frame(restab)
```

                   mu     ir2     ir3     ir4       v2  alpha epsilon
    mean       38.473 0.94443 0.54789  4.0176  0.74975 4.0204  1.1313
    sd         2.9192  4.1063  4.1063  4.1065  0.58494 1.2711   0.277
    quant0.025 32.644 -7.3097 -7.7046 -4.2485 -0.41708 2.1556 0.70471
    quant0.25  36.663 -1.5863 -1.9831  1.4891  0.38336 3.1098 0.93248
    quant0.5   38.459 0.94178 0.54476  4.0187  0.74834 3.8015  1.0895
    quant0.75   40.26  3.4639  3.0671  6.5394   1.1133 4.6961  1.2864
    quant0.975 44.327  9.1326  8.7376  12.194   1.9135  7.102  1.7858

Make the plots:

``` r
ddf <- data.frame(rbind(sigmaalpha,sigmaepsilon),errterm=gl(2,nrow(sigmaalpha),labels = c("alpha","epsilon")))
ggplot(ddf, aes(x,y, linetype=errterm))+geom_line()+xlab("yield")+ylab("density")+xlim(0,10)
```

![](figs/irripc-1..svg)

Posteriors look OK. Not much difference between the three priors tried
here.

# STAN

[STAN](https://mc-stan.org/) performs Bayesian inference using MCMC.

I use `cmdstanr` to access Stan from R.

You see below the Stan code to fit our model. Rmarkdown allows the use
of Stan chunks (elsewhere I have R chunks). The chunk header looks like
this.

STAN chunk will be compiled to ‘mod’. Chunk header is:

    cmdstan, output.var="mod", override = FALSE

``` stan
data {
  int<lower=0> N;
  array[N] int<lower=1,upper=8> field;
  array[N] int<lower=1,upper=4> irrigation;
  array[N] int<lower=1,upper=2> variety;
  array[N] real y;
}
transformed data { // need to manually create dummy variables
  vector[N] irmeth2;
  vector[N] irmeth3;
  vector[N] irmeth4;
  vector[N] var2;
  for (i in 1:N) {
    irmeth2[i] = irrigation[i] == 2;
    irmeth3[i] = irrigation[i] == 3;
    irmeth4[i] = irrigation[i] == 4;
    var2[i] = variety[i] == 2;
  }
}
parameters {
  vector[8] eta;
  real mu;
  real ir2;
  real ir3;
  real ir4;
  real va2;
  real<lower=0> sigmaf;
  real<lower=0> sigmay;
}
transformed parameters {
  vector[8] fld;
  vector[N] yhat;

  fld = sigmaf * eta;

  for (i in 1:N)
    yhat[i] = mu+ir2*irmeth2[i]+ir3*irmeth3[i]+ir4*irmeth4[i]+va2*var2[i]+fld[field[i]];

}
model {
  eta ~ normal(0, 1);

  y ~ normal(yhat, sigmay);
}
```

We have used uninformative priors for the fixed effects and the two
variances. Prepare data in a format consistent with the command file.
Needs to be a list.

``` r
irridat <- with(irrigation,list(N=length(yield), y=yield, field=as.numeric(field), irrigation=as.numeric(irrigation), variety=as.numeric(variety)))
```

Do the MCMC sampling:

``` r
fit <- mod$sample(
  data = irridat, 
  seed = 123, 
  chains = 4, 
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)
```

    Running MCMC with 4 parallel chains...

    Chain 1 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 2 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 2 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 3 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 4 Iteration:    1 / 2000 [  0%]  (Warmup) 
    Chain 4 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 1 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 2 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 3 Iteration:  500 / 2000 [ 25%]  (Warmup) 
    Chain 1 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 1000 / 2000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 1001 / 2000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 2 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 3 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 4 Iteration: 1500 / 2000 [ 75%]  (Sampling) 
    Chain 2 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 2 finished in 0.6 seconds.
    Chain 1 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 3 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 4 Iteration: 2000 / 2000 [100%]  (Sampling) 
    Chain 1 finished in 0.7 seconds.
    Chain 3 finished in 0.6 seconds.
    Chain 4 finished in 0.6 seconds.

    All 4 chains finished successfully.
    Mean chain execution time: 0.6 seconds.
    Total execution time: 0.9 seconds.

## Diagnostics

Extract the draws into a convenient dataframe format:

``` r
draws_df <- fit$draws(format = "df")
```

For the field SD:

``` r
ggplot(draws_df,
       aes(x=.iteration,y=sigmaf,color=factor(.chain))) + geom_line() +
  labs(color = 'Chain', x="Iteration")
```

![](figs/irristansigmaf-1..svg)

which also looks reasonable.

## Output summaries

Examine the output for the parameters we are mostly interested in:

``` r
fit$summary(c("mu","ir2","ir3","ir4","va2","sigmaf","sigmay","fld"))
```

    # A tibble: 15 × 10
       variable   mean median    sd   mad      q5   q95  rhat ess_bulk ess_tail
       <chr>     <dbl>  <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl>    <dbl>    <dbl>
     1 mu       38.6   38.6   4.64  3.67   31.5   45.7   1.00    1536.    1606.
     2 ir2       0.992  0.882 6.75  5.22   -9.27  11.3   1.00    1650.    1737.
     3 ir3       0.285  0.315 6.72  5.48  -10.8   10.8   1.00    1552.    1569.
     4 ir4       4.04   4.04  6.54  5.23   -6.04  14.1   1.00    1670.    1747.
     5 va2       0.734  0.742 0.817 0.712  -0.504  2.08  1.00    3451.    1813.
     6 sigmaf    6.13   5.18  3.51  2.30    2.63  12.8   1.00     902.    1423.
     7 sigmay    1.49   1.38  0.515 0.406   0.905  2.43  1.00    1059.    1439.
     8 fld[1]   -2.19  -2.23  4.62  3.67   -9.22   4.83  1.00    1540.    1541.
     9 fld[2]   -2.35  -2.24  4.93  3.82   -9.79   4.90  1.00    2345.    1584.
    10 fld[3]   -3.46  -3.40  4.77  3.74  -11.1    4.09  1.00    1975.    2020.
    11 fld[4]   -3.06  -2.96  4.97  3.76  -10.8    4.15  1.00    2508.    2157.
    12 fld[5]    1.89   1.83  4.62  3.59   -5.11   9.05  1.00    1565.    1490.
    13 fld[6]    2.10   2.17  4.93  3.85   -5.02   9.47  1.00    2395.    1890.
    14 fld[7]    3.74   3.65  4.77  3.72   -3.58  11.5   1.00    2028.    2034.
    15 fld[8]    2.88   2.94  4.96  3.66   -4.70  10.4   1.00    2494.    2102.

We see the posterior mean, median and SD, MAD of the samples. We see
some quantiles from which we could construct a 95% credible interval
(for example). The effective sample sizes for the primary parameters is
good enough for most purposes. The $\hat R$ statistics are good.

Notice that the posterior mean for field SD is substantially larger than
seen in the mixed effect model or the previous INLA models.

## Posterior Distributions

Plot the posteriors for the variance components

``` r
sdf = stack(draws_df[,startsWith(colnames(draws_df),"sigma")])
colnames(sdf) = c("yield","sigma")
levels(sdf$sigma) = c("field","epsilon")
ggplot(sdf, aes(x=yield,color=sigma)) + geom_density() +xlim(0,20)
```

![](figs/irristanvc-1..svg)

We see that the error SD can be localized much more than the field SD.
We can also look at the field effects:

``` r
sdf = stack(draws_df[,startsWith(colnames(draws_df),"fld")])
colnames(sdf) = c("yield","fld")
levels(sdf$fld) = 1:8
ggplot(sdf, aes(x=yield,color=fld)) + geom_density() + xlim(-25,25)
```

![](figs/irristanfld-1..svg)

We are looking at the differences from the overall mean. We see that all
eight field distributions clearly overlap zero. There is a distinction
between the first four and the second four fields. We can also look at
the “fixed” effects:

``` r
sdf = stack(draws_df[,c("ir2","ir3","ir4","va2")])
colnames(sdf) = c("yield","fixed")
levels(sdf$fixed) = c("ir2","ir3","ir4","va2")
ggplot(sdf, aes(x=yield,color=fixed)) + geom_density() + xlim(-15,15)
```

![](figs/irristanfixed-1..svg)

We are looking at the differences from the reference level. We see that
all four distributions clearly overlap zero although we are able to
locate the difference between the varieties more precisely than the
difference between the fields.

# BRMS

[BRMS](https://paul-buerkner.github.io/brms/) stands for Bayesian
Regression Models with STAN. It provides a convenient wrapper to STAN
functionality.

Fitting the model is very similar to `lmer` as seen above:

``` r
suppressMessages(bmod <- brm(yield ~ irrigation + variety + (1|field), 
                             irrigation, iter=10000, cores=4, backend = "cmdstanr"))
```

    Running MCMC with 4 parallel chains...

    Chain 1 Iteration:    1 / 10000 [  0%]  (Warmup) 
    Chain 1 Iteration:  100 / 10000 [  1%]  (Warmup) 
    Chain 1 Iteration:  200 / 10000 [  2%]  (Warmup) 
    Chain 1 Iteration:  300 / 10000 [  3%]  (Warmup) 
    Chain 1 Iteration:  400 / 10000 [  4%]  (Warmup) 
    Chain 1 Iteration:  500 / 10000 [  5%]  (Warmup) 
    Chain 1 Iteration:  600 / 10000 [  6%]  (Warmup) 
    Chain 1 Iteration:  700 / 10000 [  7%]  (Warmup) 
    Chain 1 Iteration:  800 / 10000 [  8%]  (Warmup) 
    Chain 1 Iteration:  900 / 10000 [  9%]  (Warmup) 
    Chain 1 Iteration: 1000 / 10000 [ 10%]  (Warmup) 
    Chain 1 Iteration: 1100 / 10000 [ 11%]  (Warmup) 
    Chain 2 Iteration:    1 / 10000 [  0%]  (Warmup) 
    Chain 2 Iteration:  100 / 10000 [  1%]  (Warmup) 
    Chain 2 Iteration:  200 / 10000 [  2%]  (Warmup) 
    Chain 2 Iteration:  300 / 10000 [  3%]  (Warmup) 
    Chain 2 Iteration:  400 / 10000 [  4%]  (Warmup) 
    Chain 2 Iteration:  500 / 10000 [  5%]  (Warmup) 
    Chain 2 Iteration:  600 / 10000 [  6%]  (Warmup) 
    Chain 2 Iteration:  700 / 10000 [  7%]  (Warmup) 
    Chain 2 Iteration:  800 / 10000 [  8%]  (Warmup) 
    Chain 3 Iteration:    1 / 10000 [  0%]  (Warmup) 
    Chain 3 Iteration:  100 / 10000 [  1%]  (Warmup) 
    Chain 3 Iteration:  200 / 10000 [  2%]  (Warmup) 
    Chain 3 Iteration:  300 / 10000 [  3%]  (Warmup) 
    Chain 3 Iteration:  400 / 10000 [  4%]  (Warmup) 
    Chain 3 Iteration:  500 / 10000 [  5%]  (Warmup) 
    Chain 3 Iteration:  600 / 10000 [  6%]  (Warmup) 
    Chain 3 Iteration:  700 / 10000 [  7%]  (Warmup) 
    Chain 3 Iteration:  800 / 10000 [  8%]  (Warmup) 
    Chain 4 Iteration:    1 / 10000 [  0%]  (Warmup) 
    Chain 4 Iteration:  100 / 10000 [  1%]  (Warmup) 
    Chain 4 Iteration:  200 / 10000 [  2%]  (Warmup) 
    Chain 4 Iteration:  300 / 10000 [  3%]  (Warmup) 
    Chain 4 Iteration:  400 / 10000 [  4%]  (Warmup) 
    Chain 1 Iteration: 1200 / 10000 [ 12%]  (Warmup) 
    Chain 1 Iteration: 1300 / 10000 [ 13%]  (Warmup) 
    Chain 1 Iteration: 1400 / 10000 [ 14%]  (Warmup) 
    Chain 1 Iteration: 1500 / 10000 [ 15%]  (Warmup) 
    Chain 1 Iteration: 1600 / 10000 [ 16%]  (Warmup) 
    Chain 1 Iteration: 1700 / 10000 [ 17%]  (Warmup) 
    Chain 1 Iteration: 1800 / 10000 [ 18%]  (Warmup) 
    Chain 1 Iteration: 1900 / 10000 [ 19%]  (Warmup) 
    Chain 1 Iteration: 2000 / 10000 [ 20%]  (Warmup) 
    Chain 1 Iteration: 2100 / 10000 [ 21%]  (Warmup) 
    Chain 1 Iteration: 2200 / 10000 [ 22%]  (Warmup) 
    Chain 2 Iteration:  900 / 10000 [  9%]  (Warmup) 
    Chain 2 Iteration: 1000 / 10000 [ 10%]  (Warmup) 
    Chain 2 Iteration: 1100 / 10000 [ 11%]  (Warmup) 
    Chain 2 Iteration: 1200 / 10000 [ 12%]  (Warmup) 
    Chain 2 Iteration: 1300 / 10000 [ 13%]  (Warmup) 
    Chain 2 Iteration: 1400 / 10000 [ 14%]  (Warmup) 
    Chain 2 Iteration: 1500 / 10000 [ 15%]  (Warmup) 
    Chain 2 Iteration: 1600 / 10000 [ 16%]  (Warmup) 
    Chain 2 Iteration: 1700 / 10000 [ 17%]  (Warmup) 
    Chain 3 Iteration:  900 / 10000 [  9%]  (Warmup) 
    Chain 3 Iteration: 1000 / 10000 [ 10%]  (Warmup) 
    Chain 3 Iteration: 1100 / 10000 [ 11%]  (Warmup) 
    Chain 3 Iteration: 1200 / 10000 [ 12%]  (Warmup) 
    Chain 3 Iteration: 1300 / 10000 [ 13%]  (Warmup) 
    Chain 3 Iteration: 1400 / 10000 [ 14%]  (Warmup) 
    Chain 3 Iteration: 1500 / 10000 [ 15%]  (Warmup) 
    Chain 3 Iteration: 1600 / 10000 [ 16%]  (Warmup) 
    Chain 4 Iteration:  500 / 10000 [  5%]  (Warmup) 
    Chain 4 Iteration:  600 / 10000 [  6%]  (Warmup) 
    Chain 4 Iteration:  700 / 10000 [  7%]  (Warmup) 
    Chain 4 Iteration:  800 / 10000 [  8%]  (Warmup) 
    Chain 4 Iteration:  900 / 10000 [  9%]  (Warmup) 
    Chain 4 Iteration: 1000 / 10000 [ 10%]  (Warmup) 
    Chain 4 Iteration: 1100 / 10000 [ 11%]  (Warmup) 
    Chain 4 Iteration: 1200 / 10000 [ 12%]  (Warmup) 
    Chain 4 Iteration: 1300 / 10000 [ 13%]  (Warmup) 
    Chain 1 Iteration: 2300 / 10000 [ 23%]  (Warmup) 
    Chain 1 Iteration: 2400 / 10000 [ 24%]  (Warmup) 
    Chain 1 Iteration: 2500 / 10000 [ 25%]  (Warmup) 
    Chain 1 Iteration: 2600 / 10000 [ 26%]  (Warmup) 
    Chain 1 Iteration: 2700 / 10000 [ 27%]  (Warmup) 
    Chain 1 Iteration: 2800 / 10000 [ 28%]  (Warmup) 
    Chain 1 Iteration: 2900 / 10000 [ 29%]  (Warmup) 
    Chain 1 Iteration: 3000 / 10000 [ 30%]  (Warmup) 
    Chain 1 Iteration: 3100 / 10000 [ 31%]  (Warmup) 
    Chain 2 Iteration: 1800 / 10000 [ 18%]  (Warmup) 
    Chain 2 Iteration: 1900 / 10000 [ 19%]  (Warmup) 
    Chain 2 Iteration: 2000 / 10000 [ 20%]  (Warmup) 
    Chain 2 Iteration: 2100 / 10000 [ 21%]  (Warmup) 
    Chain 2 Iteration: 2200 / 10000 [ 22%]  (Warmup) 
    Chain 2 Iteration: 2300 / 10000 [ 23%]  (Warmup) 
    Chain 2 Iteration: 2400 / 10000 [ 24%]  (Warmup) 
    Chain 2 Iteration: 2500 / 10000 [ 25%]  (Warmup) 
    Chain 2 Iteration: 2600 / 10000 [ 26%]  (Warmup) 
    Chain 3 Iteration: 1700 / 10000 [ 17%]  (Warmup) 
    Chain 3 Iteration: 1800 / 10000 [ 18%]  (Warmup) 
    Chain 3 Iteration: 1900 / 10000 [ 19%]  (Warmup) 
    Chain 3 Iteration: 2000 / 10000 [ 20%]  (Warmup) 
    Chain 3 Iteration: 2100 / 10000 [ 21%]  (Warmup) 
    Chain 3 Iteration: 2200 / 10000 [ 22%]  (Warmup) 
    Chain 3 Iteration: 2300 / 10000 [ 23%]  (Warmup) 
    Chain 3 Iteration: 2400 / 10000 [ 24%]  (Warmup) 
    Chain 3 Iteration: 2500 / 10000 [ 25%]  (Warmup) 
    Chain 4 Iteration: 1400 / 10000 [ 14%]  (Warmup) 
    Chain 4 Iteration: 1500 / 10000 [ 15%]  (Warmup) 
    Chain 4 Iteration: 1600 / 10000 [ 16%]  (Warmup) 
    Chain 4 Iteration: 1700 / 10000 [ 17%]  (Warmup) 
    Chain 4 Iteration: 1800 / 10000 [ 18%]  (Warmup) 
    Chain 4 Iteration: 1900 / 10000 [ 19%]  (Warmup) 
    Chain 4 Iteration: 2000 / 10000 [ 20%]  (Warmup) 
    Chain 4 Iteration: 2100 / 10000 [ 21%]  (Warmup) 
    Chain 4 Iteration: 2200 / 10000 [ 22%]  (Warmup) 
    Chain 1 Iteration: 3200 / 10000 [ 32%]  (Warmup) 
    Chain 1 Iteration: 3300 / 10000 [ 33%]  (Warmup) 
    Chain 1 Iteration: 3400 / 10000 [ 34%]  (Warmup) 
    Chain 1 Iteration: 3500 / 10000 [ 35%]  (Warmup) 
    Chain 1 Iteration: 3600 / 10000 [ 36%]  (Warmup) 
    Chain 1 Iteration: 3700 / 10000 [ 37%]  (Warmup) 
    Chain 1 Iteration: 3800 / 10000 [ 38%]  (Warmup) 
    Chain 1 Iteration: 3900 / 10000 [ 39%]  (Warmup) 
    Chain 1 Iteration: 4000 / 10000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 2700 / 10000 [ 27%]  (Warmup) 
    Chain 2 Iteration: 2800 / 10000 [ 28%]  (Warmup) 
    Chain 2 Iteration: 2900 / 10000 [ 29%]  (Warmup) 
    Chain 2 Iteration: 3000 / 10000 [ 30%]  (Warmup) 
    Chain 2 Iteration: 3100 / 10000 [ 31%]  (Warmup) 
    Chain 2 Iteration: 3200 / 10000 [ 32%]  (Warmup) 
    Chain 2 Iteration: 3300 / 10000 [ 33%]  (Warmup) 
    Chain 2 Iteration: 3400 / 10000 [ 34%]  (Warmup) 
    Chain 2 Iteration: 3500 / 10000 [ 35%]  (Warmup) 
    Chain 3 Iteration: 2600 / 10000 [ 26%]  (Warmup) 
    Chain 3 Iteration: 2700 / 10000 [ 27%]  (Warmup) 
    Chain 3 Iteration: 2800 / 10000 [ 28%]  (Warmup) 
    Chain 3 Iteration: 2900 / 10000 [ 29%]  (Warmup) 
    Chain 3 Iteration: 3000 / 10000 [ 30%]  (Warmup) 
    Chain 3 Iteration: 3100 / 10000 [ 31%]  (Warmup) 
    Chain 3 Iteration: 3200 / 10000 [ 32%]  (Warmup) 
    Chain 3 Iteration: 3300 / 10000 [ 33%]  (Warmup) 
    Chain 3 Iteration: 3400 / 10000 [ 34%]  (Warmup) 
    Chain 4 Iteration: 2300 / 10000 [ 23%]  (Warmup) 
    Chain 4 Iteration: 2400 / 10000 [ 24%]  (Warmup) 
    Chain 4 Iteration: 2500 / 10000 [ 25%]  (Warmup) 
    Chain 4 Iteration: 2600 / 10000 [ 26%]  (Warmup) 
    Chain 4 Iteration: 2700 / 10000 [ 27%]  (Warmup) 
    Chain 4 Iteration: 2800 / 10000 [ 28%]  (Warmup) 
    Chain 4 Iteration: 2900 / 10000 [ 29%]  (Warmup) 
    Chain 4 Iteration: 3000 / 10000 [ 30%]  (Warmup) 
    Chain 4 Iteration: 3100 / 10000 [ 31%]  (Warmup) 
    Chain 4 Iteration: 3200 / 10000 [ 32%]  (Warmup) 
    Chain 1 Iteration: 4100 / 10000 [ 41%]  (Warmup) 
    Chain 1 Iteration: 4200 / 10000 [ 42%]  (Warmup) 
    Chain 1 Iteration: 4300 / 10000 [ 43%]  (Warmup) 
    Chain 1 Iteration: 4400 / 10000 [ 44%]  (Warmup) 
    Chain 1 Iteration: 4500 / 10000 [ 45%]  (Warmup) 
    Chain 1 Iteration: 4600 / 10000 [ 46%]  (Warmup) 
    Chain 1 Iteration: 4700 / 10000 [ 47%]  (Warmup) 
    Chain 1 Iteration: 4800 / 10000 [ 48%]  (Warmup) 
    Chain 1 Iteration: 4900 / 10000 [ 49%]  (Warmup) 
    Chain 2 Iteration: 3600 / 10000 [ 36%]  (Warmup) 
    Chain 2 Iteration: 3700 / 10000 [ 37%]  (Warmup) 
    Chain 2 Iteration: 3800 / 10000 [ 38%]  (Warmup) 
    Chain 2 Iteration: 3900 / 10000 [ 39%]  (Warmup) 
    Chain 2 Iteration: 4000 / 10000 [ 40%]  (Warmup) 
    Chain 2 Iteration: 4100 / 10000 [ 41%]  (Warmup) 
    Chain 2 Iteration: 4200 / 10000 [ 42%]  (Warmup) 
    Chain 2 Iteration: 4300 / 10000 [ 43%]  (Warmup) 
    Chain 2 Iteration: 4400 / 10000 [ 44%]  (Warmup) 
    Chain 3 Iteration: 3500 / 10000 [ 35%]  (Warmup) 
    Chain 3 Iteration: 3600 / 10000 [ 36%]  (Warmup) 
    Chain 3 Iteration: 3700 / 10000 [ 37%]  (Warmup) 
    Chain 3 Iteration: 3800 / 10000 [ 38%]  (Warmup) 
    Chain 3 Iteration: 3900 / 10000 [ 39%]  (Warmup) 
    Chain 3 Iteration: 4000 / 10000 [ 40%]  (Warmup) 
    Chain 3 Iteration: 4100 / 10000 [ 41%]  (Warmup) 
    Chain 3 Iteration: 4200 / 10000 [ 42%]  (Warmup) 
    Chain 4 Iteration: 3300 / 10000 [ 33%]  (Warmup) 
    Chain 4 Iteration: 3400 / 10000 [ 34%]  (Warmup) 
    Chain 4 Iteration: 3500 / 10000 [ 35%]  (Warmup) 
    Chain 4 Iteration: 3600 / 10000 [ 36%]  (Warmup) 
    Chain 4 Iteration: 3700 / 10000 [ 37%]  (Warmup) 
    Chain 4 Iteration: 3800 / 10000 [ 38%]  (Warmup) 
    Chain 4 Iteration: 3900 / 10000 [ 39%]  (Warmup) 
    Chain 4 Iteration: 4000 / 10000 [ 40%]  (Warmup) 
    Chain 4 Iteration: 4100 / 10000 [ 41%]  (Warmup) 
    Chain 1 Iteration: 5000 / 10000 [ 50%]  (Warmup) 
    Chain 1 Iteration: 5001 / 10000 [ 50%]  (Sampling) 
    Chain 1 Iteration: 5100 / 10000 [ 51%]  (Sampling) 
    Chain 1 Iteration: 5200 / 10000 [ 52%]  (Sampling) 
    Chain 1 Iteration: 5300 / 10000 [ 53%]  (Sampling) 
    Chain 1 Iteration: 5400 / 10000 [ 54%]  (Sampling) 
    Chain 1 Iteration: 5500 / 10000 [ 55%]  (Sampling) 
    Chain 2 Iteration: 4500 / 10000 [ 45%]  (Warmup) 
    Chain 2 Iteration: 4600 / 10000 [ 46%]  (Warmup) 
    Chain 2 Iteration: 4700 / 10000 [ 47%]  (Warmup) 
    Chain 2 Iteration: 4800 / 10000 [ 48%]  (Warmup) 
    Chain 2 Iteration: 4900 / 10000 [ 49%]  (Warmup) 
    Chain 2 Iteration: 5000 / 10000 [ 50%]  (Warmup) 
    Chain 2 Iteration: 5001 / 10000 [ 50%]  (Sampling) 
    Chain 2 Iteration: 5100 / 10000 [ 51%]  (Sampling) 
    Chain 2 Iteration: 5200 / 10000 [ 52%]  (Sampling) 
    Chain 3 Iteration: 4300 / 10000 [ 43%]  (Warmup) 
    Chain 3 Iteration: 4400 / 10000 [ 44%]  (Warmup) 
    Chain 3 Iteration: 4500 / 10000 [ 45%]  (Warmup) 
    Chain 3 Iteration: 4600 / 10000 [ 46%]  (Warmup) 
    Chain 3 Iteration: 4700 / 10000 [ 47%]  (Warmup) 
    Chain 3 Iteration: 4800 / 10000 [ 48%]  (Warmup) 
    Chain 3 Iteration: 4900 / 10000 [ 49%]  (Warmup) 
    Chain 3 Iteration: 5000 / 10000 [ 50%]  (Warmup) 
    Chain 3 Iteration: 5001 / 10000 [ 50%]  (Sampling) 
    Chain 3 Iteration: 5100 / 10000 [ 51%]  (Sampling) 
    Chain 4 Iteration: 4200 / 10000 [ 42%]  (Warmup) 
    Chain 4 Iteration: 4300 / 10000 [ 43%]  (Warmup) 
    Chain 4 Iteration: 4400 / 10000 [ 44%]  (Warmup) 
    Chain 4 Iteration: 4500 / 10000 [ 45%]  (Warmup) 
    Chain 4 Iteration: 4600 / 10000 [ 46%]  (Warmup) 
    Chain 4 Iteration: 4700 / 10000 [ 47%]  (Warmup) 
    Chain 4 Iteration: 4800 / 10000 [ 48%]  (Warmup) 
    Chain 4 Iteration: 4900 / 10000 [ 49%]  (Warmup) 
    Chain 1 Iteration: 5600 / 10000 [ 56%]  (Sampling) 
    Chain 1 Iteration: 5700 / 10000 [ 57%]  (Sampling) 
    Chain 1 Iteration: 5800 / 10000 [ 58%]  (Sampling) 
    Chain 1 Iteration: 5900 / 10000 [ 59%]  (Sampling) 
    Chain 1 Iteration: 6000 / 10000 [ 60%]  (Sampling) 
    Chain 1 Iteration: 6100 / 10000 [ 61%]  (Sampling) 
    Chain 2 Iteration: 5300 / 10000 [ 53%]  (Sampling) 
    Chain 2 Iteration: 5400 / 10000 [ 54%]  (Sampling) 
    Chain 2 Iteration: 5500 / 10000 [ 55%]  (Sampling) 
    Chain 2 Iteration: 5600 / 10000 [ 56%]  (Sampling) 
    Chain 2 Iteration: 5700 / 10000 [ 57%]  (Sampling) 
    Chain 2 Iteration: 5800 / 10000 [ 58%]  (Sampling) 
    Chain 2 Iteration: 5900 / 10000 [ 59%]  (Sampling) 
    Chain 3 Iteration: 5200 / 10000 [ 52%]  (Sampling) 
    Chain 3 Iteration: 5300 / 10000 [ 53%]  (Sampling) 
    Chain 3 Iteration: 5400 / 10000 [ 54%]  (Sampling) 
    Chain 3 Iteration: 5500 / 10000 [ 55%]  (Sampling) 
    Chain 3 Iteration: 5600 / 10000 [ 56%]  (Sampling) 
    Chain 4 Iteration: 5000 / 10000 [ 50%]  (Warmup) 
    Chain 4 Iteration: 5001 / 10000 [ 50%]  (Sampling) 
    Chain 4 Iteration: 5100 / 10000 [ 51%]  (Sampling) 
    Chain 4 Iteration: 5200 / 10000 [ 52%]  (Sampling) 
    Chain 4 Iteration: 5300 / 10000 [ 53%]  (Sampling) 
    Chain 4 Iteration: 5400 / 10000 [ 54%]  (Sampling) 
    Chain 4 Iteration: 5500 / 10000 [ 55%]  (Sampling) 
    Chain 4 Iteration: 5600 / 10000 [ 56%]  (Sampling) 
    Chain 1 Iteration: 6200 / 10000 [ 62%]  (Sampling) 
    Chain 1 Iteration: 6300 / 10000 [ 63%]  (Sampling) 
    Chain 1 Iteration: 6400 / 10000 [ 64%]  (Sampling) 
    Chain 1 Iteration: 6500 / 10000 [ 65%]  (Sampling) 
    Chain 1 Iteration: 6600 / 10000 [ 66%]  (Sampling) 
    Chain 1 Iteration: 6700 / 10000 [ 67%]  (Sampling) 
    Chain 2 Iteration: 6000 / 10000 [ 60%]  (Sampling) 
    Chain 2 Iteration: 6100 / 10000 [ 61%]  (Sampling) 
    Chain 2 Iteration: 6200 / 10000 [ 62%]  (Sampling) 
    Chain 2 Iteration: 6300 / 10000 [ 63%]  (Sampling) 
    Chain 2 Iteration: 6400 / 10000 [ 64%]  (Sampling) 
    Chain 2 Iteration: 6500 / 10000 [ 65%]  (Sampling) 
    Chain 2 Iteration: 6600 / 10000 [ 66%]  (Sampling) 
    Chain 3 Iteration: 5700 / 10000 [ 57%]  (Sampling) 
    Chain 3 Iteration: 5800 / 10000 [ 58%]  (Sampling) 
    Chain 3 Iteration: 5900 / 10000 [ 59%]  (Sampling) 
    Chain 3 Iteration: 6000 / 10000 [ 60%]  (Sampling) 
    Chain 3 Iteration: 6100 / 10000 [ 61%]  (Sampling) 
    Chain 3 Iteration: 6200 / 10000 [ 62%]  (Sampling) 
    Chain 4 Iteration: 5700 / 10000 [ 57%]  (Sampling) 
    Chain 4 Iteration: 5800 / 10000 [ 58%]  (Sampling) 
    Chain 4 Iteration: 5900 / 10000 [ 59%]  (Sampling) 
    Chain 4 Iteration: 6000 / 10000 [ 60%]  (Sampling) 
    Chain 4 Iteration: 6100 / 10000 [ 61%]  (Sampling) 
    Chain 4 Iteration: 6200 / 10000 [ 62%]  (Sampling) 
    Chain 1 Iteration: 6800 / 10000 [ 68%]  (Sampling) 
    Chain 1 Iteration: 6900 / 10000 [ 69%]  (Sampling) 
    Chain 1 Iteration: 7000 / 10000 [ 70%]  (Sampling) 
    Chain 1 Iteration: 7100 / 10000 [ 71%]  (Sampling) 
    Chain 1 Iteration: 7200 / 10000 [ 72%]  (Sampling) 
    Chain 1 Iteration: 7300 / 10000 [ 73%]  (Sampling) 
    Chain 1 Iteration: 7400 / 10000 [ 74%]  (Sampling) 
    Chain 2 Iteration: 6700 / 10000 [ 67%]  (Sampling) 
    Chain 2 Iteration: 6800 / 10000 [ 68%]  (Sampling) 
    Chain 2 Iteration: 6900 / 10000 [ 69%]  (Sampling) 
    Chain 2 Iteration: 7000 / 10000 [ 70%]  (Sampling) 
    Chain 2 Iteration: 7100 / 10000 [ 71%]  (Sampling) 
    Chain 2 Iteration: 7200 / 10000 [ 72%]  (Sampling) 
    Chain 2 Iteration: 7300 / 10000 [ 73%]  (Sampling) 
    Chain 3 Iteration: 6300 / 10000 [ 63%]  (Sampling) 
    Chain 3 Iteration: 6400 / 10000 [ 64%]  (Sampling) 
    Chain 3 Iteration: 6500 / 10000 [ 65%]  (Sampling) 
    Chain 3 Iteration: 6600 / 10000 [ 66%]  (Sampling) 
    Chain 3 Iteration: 6700 / 10000 [ 67%]  (Sampling) 
    Chain 3 Iteration: 6800 / 10000 [ 68%]  (Sampling) 
    Chain 4 Iteration: 6300 / 10000 [ 63%]  (Sampling) 
    Chain 4 Iteration: 6400 / 10000 [ 64%]  (Sampling) 
    Chain 4 Iteration: 6500 / 10000 [ 65%]  (Sampling) 
    Chain 4 Iteration: 6600 / 10000 [ 66%]  (Sampling) 
    Chain 4 Iteration: 6700 / 10000 [ 67%]  (Sampling) 
    Chain 4 Iteration: 6800 / 10000 [ 68%]  (Sampling) 
    Chain 1 Iteration: 7500 / 10000 [ 75%]  (Sampling) 
    Chain 1 Iteration: 7600 / 10000 [ 76%]  (Sampling) 
    Chain 1 Iteration: 7700 / 10000 [ 77%]  (Sampling) 
    Chain 1 Iteration: 7800 / 10000 [ 78%]  (Sampling) 
    Chain 1 Iteration: 7900 / 10000 [ 79%]  (Sampling) 
    Chain 1 Iteration: 8000 / 10000 [ 80%]  (Sampling) 
    Chain 2 Iteration: 7400 / 10000 [ 74%]  (Sampling) 
    Chain 2 Iteration: 7500 / 10000 [ 75%]  (Sampling) 
    Chain 2 Iteration: 7600 / 10000 [ 76%]  (Sampling) 
    Chain 2 Iteration: 7700 / 10000 [ 77%]  (Sampling) 
    Chain 2 Iteration: 7800 / 10000 [ 78%]  (Sampling) 
    Chain 2 Iteration: 7900 / 10000 [ 79%]  (Sampling) 
    Chain 2 Iteration: 8000 / 10000 [ 80%]  (Sampling) 
    Chain 3 Iteration: 6900 / 10000 [ 69%]  (Sampling) 
    Chain 3 Iteration: 7000 / 10000 [ 70%]  (Sampling) 
    Chain 3 Iteration: 7100 / 10000 [ 71%]  (Sampling) 
    Chain 3 Iteration: 7200 / 10000 [ 72%]  (Sampling) 
    Chain 3 Iteration: 7300 / 10000 [ 73%]  (Sampling) 
    Chain 3 Iteration: 7400 / 10000 [ 74%]  (Sampling) 
    Chain 4 Iteration: 6900 / 10000 [ 69%]  (Sampling) 
    Chain 4 Iteration: 7000 / 10000 [ 70%]  (Sampling) 
    Chain 4 Iteration: 7100 / 10000 [ 71%]  (Sampling) 
    Chain 4 Iteration: 7200 / 10000 [ 72%]  (Sampling) 
    Chain 4 Iteration: 7300 / 10000 [ 73%]  (Sampling) 
    Chain 4 Iteration: 7400 / 10000 [ 74%]  (Sampling) 
    Chain 4 Iteration: 7500 / 10000 [ 75%]  (Sampling) 
    Chain 1 Iteration: 8100 / 10000 [ 81%]  (Sampling) 
    Chain 1 Iteration: 8200 / 10000 [ 82%]  (Sampling) 
    Chain 1 Iteration: 8300 / 10000 [ 83%]  (Sampling) 
    Chain 1 Iteration: 8400 / 10000 [ 84%]  (Sampling) 
    Chain 1 Iteration: 8500 / 10000 [ 85%]  (Sampling) 
    Chain 1 Iteration: 8600 / 10000 [ 86%]  (Sampling) 
    Chain 2 Iteration: 8100 / 10000 [ 81%]  (Sampling) 
    Chain 2 Iteration: 8200 / 10000 [ 82%]  (Sampling) 
    Chain 2 Iteration: 8300 / 10000 [ 83%]  (Sampling) 
    Chain 2 Iteration: 8400 / 10000 [ 84%]  (Sampling) 
    Chain 2 Iteration: 8500 / 10000 [ 85%]  (Sampling) 
    Chain 2 Iteration: 8600 / 10000 [ 86%]  (Sampling) 
    Chain 2 Iteration: 8700 / 10000 [ 87%]  (Sampling) 
    Chain 3 Iteration: 7500 / 10000 [ 75%]  (Sampling) 
    Chain 3 Iteration: 7600 / 10000 [ 76%]  (Sampling) 
    Chain 3 Iteration: 7700 / 10000 [ 77%]  (Sampling) 
    Chain 3 Iteration: 7800 / 10000 [ 78%]  (Sampling) 
    Chain 3 Iteration: 7900 / 10000 [ 79%]  (Sampling) 
    Chain 3 Iteration: 8000 / 10000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 7600 / 10000 [ 76%]  (Sampling) 
    Chain 4 Iteration: 7700 / 10000 [ 77%]  (Sampling) 
    Chain 4 Iteration: 7800 / 10000 [ 78%]  (Sampling) 
    Chain 4 Iteration: 7900 / 10000 [ 79%]  (Sampling) 
    Chain 4 Iteration: 8000 / 10000 [ 80%]  (Sampling) 
    Chain 4 Iteration: 8100 / 10000 [ 81%]  (Sampling) 
    Chain 1 Iteration: 8700 / 10000 [ 87%]  (Sampling) 
    Chain 1 Iteration: 8800 / 10000 [ 88%]  (Sampling) 
    Chain 1 Iteration: 8900 / 10000 [ 89%]  (Sampling) 
    Chain 1 Iteration: 9000 / 10000 [ 90%]  (Sampling) 
    Chain 1 Iteration: 9100 / 10000 [ 91%]  (Sampling) 
    Chain 1 Iteration: 9200 / 10000 [ 92%]  (Sampling) 
    Chain 2 Iteration: 8800 / 10000 [ 88%]  (Sampling) 
    Chain 2 Iteration: 8900 / 10000 [ 89%]  (Sampling) 
    Chain 2 Iteration: 9000 / 10000 [ 90%]  (Sampling) 
    Chain 2 Iteration: 9100 / 10000 [ 91%]  (Sampling) 
    Chain 2 Iteration: 9200 / 10000 [ 92%]  (Sampling) 
    Chain 2 Iteration: 9300 / 10000 [ 93%]  (Sampling) 
    Chain 2 Iteration: 9400 / 10000 [ 94%]  (Sampling) 
    Chain 3 Iteration: 8100 / 10000 [ 81%]  (Sampling) 
    Chain 3 Iteration: 8200 / 10000 [ 82%]  (Sampling) 
    Chain 3 Iteration: 8300 / 10000 [ 83%]  (Sampling) 
    Chain 3 Iteration: 8400 / 10000 [ 84%]  (Sampling) 
    Chain 3 Iteration: 8500 / 10000 [ 85%]  (Sampling) 
    Chain 4 Iteration: 8200 / 10000 [ 82%]  (Sampling) 
    Chain 4 Iteration: 8300 / 10000 [ 83%]  (Sampling) 
    Chain 4 Iteration: 8400 / 10000 [ 84%]  (Sampling) 
    Chain 4 Iteration: 8500 / 10000 [ 85%]  (Sampling) 
    Chain 4 Iteration: 8600 / 10000 [ 86%]  (Sampling) 
    Chain 4 Iteration: 8700 / 10000 [ 87%]  (Sampling) 
    Chain 1 Iteration: 9300 / 10000 [ 93%]  (Sampling) 
    Chain 1 Iteration: 9400 / 10000 [ 94%]  (Sampling) 
    Chain 1 Iteration: 9500 / 10000 [ 95%]  (Sampling) 
    Chain 1 Iteration: 9600 / 10000 [ 96%]  (Sampling) 
    Chain 1 Iteration: 9700 / 10000 [ 97%]  (Sampling) 
    Chain 1 Iteration: 9800 / 10000 [ 98%]  (Sampling) 
    Chain 2 Iteration: 9500 / 10000 [ 95%]  (Sampling) 
    Chain 2 Iteration: 9600 / 10000 [ 96%]  (Sampling) 
    Chain 2 Iteration: 9700 / 10000 [ 97%]  (Sampling) 
    Chain 2 Iteration: 9800 / 10000 [ 98%]  (Sampling) 
    Chain 2 Iteration: 9900 / 10000 [ 99%]  (Sampling) 
    Chain 2 Iteration: 10000 / 10000 [100%]  (Sampling) 
    Chain 3 Iteration: 8600 / 10000 [ 86%]  (Sampling) 
    Chain 3 Iteration: 8700 / 10000 [ 87%]  (Sampling) 
    Chain 3 Iteration: 8800 / 10000 [ 88%]  (Sampling) 
    Chain 3 Iteration: 8900 / 10000 [ 89%]  (Sampling) 
    Chain 3 Iteration: 9000 / 10000 [ 90%]  (Sampling) 
    Chain 3 Iteration: 9100 / 10000 [ 91%]  (Sampling) 
    Chain 4 Iteration: 8800 / 10000 [ 88%]  (Sampling) 
    Chain 4 Iteration: 8900 / 10000 [ 89%]  (Sampling) 
    Chain 4 Iteration: 9000 / 10000 [ 90%]  (Sampling) 
    Chain 4 Iteration: 9100 / 10000 [ 91%]  (Sampling) 
    Chain 4 Iteration: 9200 / 10000 [ 92%]  (Sampling) 
    Chain 4 Iteration: 9300 / 10000 [ 93%]  (Sampling) 
    Chain 2 finished in 1.5 seconds.
    Chain 1 Iteration: 9900 / 10000 [ 99%]  (Sampling) 
    Chain 1 Iteration: 10000 / 10000 [100%]  (Sampling) 
    Chain 3 Iteration: 9200 / 10000 [ 92%]  (Sampling) 
    Chain 3 Iteration: 9300 / 10000 [ 93%]  (Sampling) 
    Chain 3 Iteration: 9400 / 10000 [ 94%]  (Sampling) 
    Chain 3 Iteration: 9500 / 10000 [ 95%]  (Sampling) 
    Chain 3 Iteration: 9600 / 10000 [ 96%]  (Sampling) 
    Chain 4 Iteration: 9400 / 10000 [ 94%]  (Sampling) 
    Chain 4 Iteration: 9500 / 10000 [ 95%]  (Sampling) 
    Chain 4 Iteration: 9600 / 10000 [ 96%]  (Sampling) 
    Chain 4 Iteration: 9700 / 10000 [ 97%]  (Sampling) 
    Chain 4 Iteration: 9800 / 10000 [ 98%]  (Sampling) 
    Chain 4 Iteration: 9900 / 10000 [ 99%]  (Sampling) 
    Chain 4 Iteration: 10000 / 10000 [100%]  (Sampling) 
    Chain 1 finished in 1.5 seconds.
    Chain 4 finished in 1.5 seconds.
    Chain 3 Iteration: 9700 / 10000 [ 97%]  (Sampling) 
    Chain 3 Iteration: 9800 / 10000 [ 98%]  (Sampling) 
    Chain 3 Iteration: 9900 / 10000 [ 99%]  (Sampling) 
    Chain 3 Iteration: 10000 / 10000 [100%]  (Sampling) 
    Chain 3 finished in 1.6 seconds.

    All 4 chains finished successfully.
    Mean chain execution time: 1.5 seconds.
    Total execution time: 1.8 seconds.

We get some warnings but not as severe as seen with our STAN fit above.
We can obtain some posterior densities and diagnostics with:

``` r
plot(bmod, variable = "^s", regex=TRUE)
```

![](figs/irribrmsdiag-1..svg)

We have chosen only the random effect hyperparameters since this is
where problems will appear first. Looks OK.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.21.0
    functions {
    }
    data {
      int<lower=1> N;  // total number of observations
      vector[N] Y;  // response variable
      int<lower=1> K;  // number of population-level effects
      matrix[N, K] X;  // population-level design matrix
      int<lower=1> Kc;  // number of population-level effects after centering
      // data for group-level effects of ID 1
      int<lower=1> N_1;  // number of grouping levels
      int<lower=1> M_1;  // number of coefficients per level
      array[N] int<lower=1> J_1;  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_1_1;
      int prior_only;  // should the likelihood be ignored?
    }
    transformed data {
      matrix[N, Kc] Xc;  // centered version of X without an intercept
      vector[Kc] means_X;  // column means of X before centering
      for (i in 2:K) {
        means_X[i - 1] = mean(X[, i]);
        Xc[, i - 1] = X[, i] - means_X[i - 1];
      }
    }
    parameters {
      vector[Kc] b;  // regression coefficients
      real Intercept;  // temporary intercept for centered predictors
      real<lower=0> sigma;  // dispersion parameter
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      array[M_1] vector[N_1] z_1;  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      lprior += student_t_lpdf(Intercept | 3, 40.1, 3.9);
      lprior += student_t_lpdf(sigma | 3, 0, 3.9)
        - 1 * student_t_lccdf(0 | 3, 0, 3.9);
      lprior += student_t_lpdf(sd_1 | 3, 0, 3.9)
        - 1 * student_t_lccdf(0 | 3, 0, 3.9);
    }
    model {
      // likelihood including constants
      if (!prior_only) {
        // initialize linear predictor term
        vector[N] mu = rep_vector(0.0, N);
        mu += Intercept;
        for (n in 1:N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n];
        }
        target += normal_id_glm_lpdf(Y | Xc, mu, b, sigma);
      }
      // priors including constants
      target += lprior;
      target += std_normal_lpdf(z_1[1]);
    }
    generated quantities {
      // actual population-level intercept
      real b_Intercept = Intercept - dot_product(means_X, b);
    }

We see that `brms` is using student t distributions with 3 degrees of
freedom for the priors. For the two error SDs, this will be truncated at
zero to form half-t distributions. You can get a more explicit
description of the priors with `prior_summary(bmod)`. These are
qualitatively similar to the half-normal and the PC prior used in the
INLA fit.

We examine the fit:

``` r
summary(bmod)
```

     Family: gaussian 
      Links: mu = identity; sigma = identity 
    Formula: yield ~ irrigation + variety + (1 | field) 
       Data: irrigation (Number of observations: 16) 
      Draws: 4 chains, each with iter = 10000; warmup = 5000; thin = 1;
             total post-warmup draws = 20000

    Multilevel Hyperparameters:
    ~field (Number of levels: 8) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     4.35      1.65     2.10     8.39 1.00     5799     7831

    Regression Coefficients:
                 Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept       38.40      3.30    31.76    45.02 1.00     8312     9310
    irrigationi2     0.96      4.82    -8.56    10.74 1.00     8254     8960
    irrigationi3     0.61      4.83    -9.10    10.30 1.00     8405     7504
    irrigationi4     4.09      4.71    -5.25    13.58 1.00     9101     8709
    varietyv2        0.75      0.78    -0.83     2.35 1.00    16800    11317

    Further Distributional Parameters:
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sigma     1.48      0.53     0.83     2.85 1.00     4788     5628

    Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

The posterior mean for the field SD is more comparable to the mixed
model and INLA values seen earlier and smaller than the STAN fit. This
can be ascribed to the more informative prior used for the BRMS fit.

# MGCV

It is possible to fit some GLMMs within the GAM framework of the `mgcv`
package. An explanation of this can be found in this
[blog](https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/)

The `field` term must be a factor for this to work:

``` r
gmod = gam(yield ~ irrigation + variety + s(field,bs="re"), 
           data=irrigation, method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    yield ~ irrigation + variety + s(field, bs = "re")

    Parametric coefficients:
                 Estimate Std. Error t value Pr(>|t|)
    (Intercept)    38.425      2.952   13.02    3e-06
    irrigationi2    1.000      4.154    0.24     0.82
    irrigationi3    0.600      4.154    0.14     0.89
    irrigationi4    4.100      4.154    0.99     0.36
    varietyv2       0.750      0.597    1.26     0.25

    Approximate significance of smooth terms:
              edf Ref.df    F p-value
    s(field) 3.83      4 23.2 0.00034

    R-sq.(adj) =  0.888   Deviance explained = 94.6%
    -REML = 27.398  Scale est. = 1.4257    n = 16

We get the fixed effect estimates. We also get a test on the random
effect (as described in this
[article](https://doi.org/10.1093/biomet/ast038). The hypothesis of no
variation between the fields is rejected.

We can get an estimate of the operator and error SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

             std.dev   lower  upper
    s(field)   4.067 1.97338 8.3820
    scale      1.194 0.70717 2.0161

    Rank: 2/2

which is the same as the REML estimate from `lmer` earlier.

The random effect estimates for the fields can be found with:

``` r
coef(gmod)
```

     (Intercept) irrigationi2 irrigationi3 irrigationi4    varietyv2   s(field).1   s(field).2   s(field).3   s(field).4 
         38.4250       1.0000       0.6000       4.1000       0.7500      -2.0612      -2.2529      -3.6430      -3.0199 
      s(field).5   s(field).6   s(field).7   s(field).8 
          2.0612       2.2529       3.6430       3.0199 

# GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(yield ~ irrigation + variety + s(field,bs="re"), 
           data=irrigation, fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior density for the intercept as:

``` r
plot(gimod$beta[1,],gimod$density[1,],type="l",xlab="yield",ylab="density")
```

![](figs/irriginlaint-1..svg)

and for the treatment effects as:

``` r
xmat = t(gimod$beta[2:5,])
ymat = t(gimod$density[2:5,])
matplot(xmat, ymat,type="l",xlab="yield",ylab="density")
legend("right",c("i2","i3","i4","v2"),col=1:4,lty=1:4)
```

![](figs/irriginlateff-1..svg)

``` r
xmat = t(gimod$beta[6:13,])
ymat = t(gimod$density[6:13,])
matplot(xmat, ymat,type="l",xlab="yield",ylab="density")
legend("right",paste0("field",1:8),col=1:8,lty=1:8)
```

![](figs/irriginlareff-1..svg)

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

See the [Discussion of the single random effect
model](pulp.md#Discussion) for general comments. Given that the fixed
effects are not significant here, this example is not so different from
the single random effect example. This provides an illustration of why
we need to pay attention to the priors. In the `pulp` example, the
default priors resulted in unbelievable results from INLA and we were
prompted to consider alternatives. In this example, the default INLA
priors produce output that looked passable and, if we were feeling lazy,
we might have skipped a look at the alternatives. In this case, they do
not make much difference. Contrast this with the default STAN priors
used - the output looks reasonable but the answers are somewhat
different. Had we not been trying these other analyses, we would not be
aware of this. The minimal analyst might have stopped there. But BRMS
uses more informative priors and produces results closer to the other
methods.

STAN/BRMS put more weight on low values of the random effects SDs
whereas the INLA posteriors are clearly bounded away from zero. We saw a
similar effect in the `pulp` example. Although we have not matched up
the priors exactly, there does appear to be some structural difference.

Much of the worry above centers on the random effect SDs. The fixed
effects seem quite robust to these concerns. If we only care about
these, GINLA is giving us what we need with the minimum amount of effort
(we would not even need to install any packages beyond the default
distribution of R, though this is an historic advantage for `mgcv`).

# Package version info

``` r
sessionInfo()
```

    R version 4.4.1 (2024-06-14)
    Platform: x86_64-apple-darwin20
    Running under: macOS Sonoma 14.6.1

    Matrix products: default
    BLAS:   /Library/Frameworks/R.framework/Versions/4.4-x86_64/Resources/lib/libRblas.0.dylib 
    LAPACK: /Library/Frameworks/R.framework/Versions/4.4-x86_64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.0

    locale:
    [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

    time zone: Europe/London
    tzcode source: internal

    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] foreach_1.5.2  mgcv_1.9-1     nlme_3.1-165   brms_2.21.0    Rcpp_1.0.13    cmdstanr_0.8.1 knitr_1.48    
     [8] INLA_24.06.27  sp_2.1-4       RLRsim_3.1-8   pbkrtest_0.5.3 lme4_1.1-35.5  Matrix_1.7-0   ggplot2_3.5.1 
    [15] faraway_1.0.8 

    loaded via a namespace (and not attached):
     [1] tidyselect_1.2.1     farver_2.1.2         dplyr_1.1.4          loo_2.8.0            fastmap_1.2.0       
     [6] tensorA_0.36.2.1     digest_0.6.36        estimability_1.5.1   lifecycle_1.0.4      Deriv_4.1.3         
    [11] sf_1.0-16            StanHeaders_2.32.10  processx_3.8.4       magrittr_2.0.3       posterior_1.6.0     
    [16] compiler_4.4.1       rlang_1.1.4          tools_4.4.1          utf8_1.2.4           yaml_2.3.10         
    [21] data.table_1.15.4    labeling_0.4.3       bridgesampling_1.1-2 pkgbuild_1.4.4       classInt_0.4-10     
    [26] plyr_1.8.9           abind_1.4-5          KernSmooth_2.23-24   withr_3.0.1          purrr_1.0.2         
    [31] grid_4.4.1           stats4_4.4.1         fansi_1.0.6          xtable_1.8-4         e1071_1.7-14        
    [36] colorspace_2.1-1     inline_0.3.19        iterators_1.0.14     emmeans_1.10.3       scales_1.3.0        
    [41] MASS_7.3-61          cli_3.6.3            mvtnorm_1.2-5        rmarkdown_2.27       generics_0.1.3      
    [46] RcppParallel_5.1.8   rstudioapi_0.16.0    reshape2_1.4.4       minqa_1.2.7          DBI_1.2.3           
    [51] proxy_0.4-27         rstan_2.32.6         stringr_1.5.1        splines_4.4.1        bayesplot_1.11.1    
    [56] parallel_4.4.1       matrixStats_1.3.0    vctrs_0.6.5          boot_1.3-30          jsonlite_1.8.8      
    [61] systemfonts_1.1.0    tidyr_1.3.1          units_0.8-5          glue_1.7.0           nloptr_2.1.1        
    [66] codetools_0.2-20     ps_1.7.7             distributional_0.4.0 stringi_1.8.4        gtable_0.3.5        
    [71] QuickJSR_1.3.1       munsell_0.5.1        tibble_3.2.1         pillar_1.9.0         htmltools_0.5.8.1   
    [76] Brobdingnag_1.2-9    R6_2.5.1             fmesher_0.1.7        evaluate_0.24.0      lattice_0.22-6      
    [81] backports_1.5.0      broom_1.0.6          MatrixModels_0.5-3   rstantools_2.4.0     class_7.3-22        
    [86] gridExtra_2.3        svglite_2.1.3        coda_0.19-4.1        checkmate_2.3.2      xfun_0.46           
    [91] pkgconfig_2.0.3     
