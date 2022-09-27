Binary response GLMM
================
[Julian Faraway](https://julianfaraway.github.io/)
27 September 2022

- <a href="#data-and-model" id="toc-data-and-model">Data and Model</a>
- <a href="#lme4" id="toc-lme4">LME4</a>
- <a href="#inla" id="toc-inla">INLA</a>
- <a href="#brms" id="toc-brms">BRMS</a>
- <a href="#mgcv" id="toc-mgcv">MGCV</a>
- <a href="#ginla" id="toc-ginla">GINLA</a>
- <a href="#discussion" id="toc-discussion">Discussion</a>
- <a href="#package-version-info" id="toc-package-version-info">Package
  version info</a>

See the [introduction](../index.md) for an overview.

This example is discussed in more detail in the book [Bayesian
Regression Modeling with
INLA](https://julianfaraway.github.io/brinlabook/chaglmm.html)

Packages used:

``` r
library(ggplot2)
library(lme4)
library(INLA)
library(knitr)
library(brms)
library(mgcv)
```

# Data and Model

In [Fitzmaurice and Laird,
1993](https://doi.org/10.1093/biomet/80.1.141), data on 537 children
aged 7–10 in six Ohio cities are reported. The response is binary — does
the child suffer from wheezing (indication of a pulmonary problem) where
one indicates yes and zero no. This status is reported for each of four
years at ages 7, 8, 9 and 10. There is also an indicator variable for
whether the mother of the child is a smoker. Because we have four binary
responses for each child, we expect these to be correlated and our model
needs to reflect this.

We sum the number of smoking and non-smoking mothers:

``` r
data(ohio, package="brinla")
table(ohio$smoke)/4
```


      0   1 
    350 187 

We use this to produce the proportion of wheezing children classified by
age and maternal smoking status:

``` r
xtabs(resp ~ smoke + age, ohio)/c(350,187)
```

         age
    smoke      -2      -1       0       1
        0 0.16000 0.14857 0.14286 0.10571
        1 0.16578 0.20856 0.18717 0.13904

Age has been adjusted so that nine years old is zero. We see that
wheezing appears to decline with age and that there may be more wheezing
in children with mothers who smoke. But the effects are not clear and we
need modeling to be sure about these conclusions.

A plausible model uses a logit link with a linear predictor of the form:
$$
\eta_{ij} = \beta_0 + \beta_1 age_j + \beta_2 smoke_i + u_i, \quad i=1, \dots ,537, \quad j=1,2,3,4,
$$ with $$
P(Y_{ij} = 1) = {\exp(\eta_{ij}) \over 1+\exp(\eta_{ij})}.
$$ The random effect $u_i$ models the propensity of child $i$ to wheeze.
Children are likely to vary in their health condition and this effect
enables us to include this unknown variation in the model. Because $u_i$
is added to all four observations for a child, we induce a positive
correlation among the four responses as we might naturally expect. The
response is Bernoulli or, in other words, binomial with trial size one.

# LME4

Here is the model fit penalized quasi-likelihood using the `lme4`
package:

``` r
modagh <- glmer(resp ~ age + smoke + (1|id), nAGQ=25, 
              family=binomial, data=ohio)
summary(modagh, correlation = FALSE)
```

    Generalized linear mixed model fit by maximum likelihood (Adaptive Gauss-Hermite Quadrature, nAGQ = 25) ['glmerMod']
     Family: binomial  ( logit )
    Formula: resp ~ age + smoke + (1 | id)
       Data: ohio

         AIC      BIC   logLik deviance df.resid 
      1603.3   1626.0   -797.6   1595.3     2144 

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -1.373 -0.201 -0.177 -0.149  2.508 

    Random effects:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 4.69     2.16    
    Number of obs: 2148, groups:  id, 537

    Fixed effects:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  -3.1015     0.2191  -14.16   <2e-16
    age          -0.1756     0.0677   -2.60   0.0095
    smoke         0.3986     0.2731    1.46   0.1444

We see that there is no significant effect due to maternal smoking.

Suppose you do not take into account the correlated response within the
individuals and fit a GLM ignoring the ID random effect:

``` r
modglm <- glm(resp ~ age + smoke, family=binomial, data=ohio)
faraway::sumary(modglm)
```

                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  -1.8837     0.0838   -22.5   <2e-16
    age          -0.1134     0.0541    -2.1    0.036
    smoke         0.2721     0.1235     2.2    0.028

    n = 2148 p = 3
    Deviance = 1819.889 Null Deviance = 1829.089 (Difference = 9.199) 

We see that the effect of maternal smoking is significant (but this
would be the incorrect conclusion).

# INLA

Integrated nested Laplace approximation is a method of Bayesian
computation which uses approximation rather than simulation. More can be
found on this topic in [Bayesian Regression Modeling with
INLA](http://julianfaraway.github.io/brinla/) and the [chapter on
GLMMs](https://julianfaraway.github.io/brinlabook/chaglmm.html)

We can fit this model in INLA as:

``` r
formula <- resp ~ age + smoke + f(id, model="iid")
imod <- inla(formula, family="binomial", data=ohio)
```

The `id` variable represents the child and we use an `iid` model
indicating that the $u_i$ variables should be independent and
identically distributed between children. A summary of the posteriors
for the fixed effect components can be obtained as:

``` r
imod$summary.fixed |> kable()
```

|             |     mean |      sd | 0.025quant | 0.5quant | 0.975quant |     mode | kld |
|:------------|---------:|--------:|-----------:|---------:|-----------:|---------:|----:|
| (Intercept) | -2.98954 | 0.19446 |   -3.39128 | -2.98280 |   -2.62726 | -2.97029 |   0 |
| age         | -0.16652 | 0.06278 |   -0.29019 | -0.16632 |   -0.04394 | -0.16594 |   0 |
| smoke       |  0.39137 | 0.23904 |   -0.07678 |  0.39092 |    0.86206 |  0.39004 |   0 |

The posterior means are similar to the PQL estimates. We can get plots
of the posteriors of the fixed effects:

``` r
fnames = names(imod$marginals.fixed)
par(mfrow=c(1,2))
for(i in 2:3){
  plot(imod$marginals.fixed[[i]],
       type="l",
       ylab="density",
       xlab=fnames[i])
  abline(v=0)
}
par(mfrow=c(1,1))
```

<figure>
<img src="figs/fig-ohiofpd-1..svg" id="fig-ohiofpd"
alt="Figure 1: Posterior densities of the fixed effects model for the Ohio wheeze data." />
<figcaption aria-hidden="true">Figure 1: Posterior densities of the
fixed effects model for the Ohio wheeze data.</figcaption>
</figure>

We can also see the summary for the random effect SD:

``` r
hpd = inla.tmarginal(function(x) 1/sqrt(x), imod$marginals.hyperpar[[1]])
inla.zmarginal(hpd)
```

    Mean            1.92242 
    Stdev           0.153963 
    Quantile  0.025 1.62149 
    Quantile  0.25  1.82364 
    Quantile  0.5   1.91652 
    Quantile  0.75  2.01333 
    Quantile  0.975 2.26752 

Again the result is similar to the PQL output although notice that INLA
provides some assessment of uncertainty in this value in contrast to the
PQL result. We can also see the posterior density:

``` r
plot(hpd,type="l",xlab="linear predictor",ylab="density")
```

<figure>
<img src="figs/fig-ohiohyppd-1..svg" id="fig-ohiohyppd"
alt="Figure 2: Posterior density of the SD of id" />
<figcaption aria-hidden="true">Figure 2: Posterior density of the SD of
id</figcaption>
</figure>

# BRMS

[BRMS](https://paul-buerkner.github.io/brms/) stands for Bayesian
Regression Models with STAN. It provides a convenient wrapper to STAN
functionality.

Fitting the model is very similar to `lmer` as seen above. There is a
`bernoulli` option for the `family` which is appropriate for a 0-1
response.

``` r
bmod <- brm(resp ~ age + smoke + (1|id), family=bernoulli(), data=ohio, cores = 4)
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

We can check the MCMC diagnostics and the posterior densities with:

``` r
plot(bmod)
```

<img src="figs/fig-ohiobrmsdiag-1..svg" id="fig-ohiobrmsdiag" />

Looks quite similar to the INLA results.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.17.0
    functions {
    }
    data {
      int<lower=1> N;  // total number of observations
      int Y[N];  // response variable
      int<lower=1> K;  // number of population-level effects
      matrix[N, K] X;  // population-level design matrix
      // data for group-level effects of ID 1
      int<lower=1> N_1;  // number of grouping levels
      int<lower=1> M_1;  // number of coefficients per level
      int<lower=1> J_1[N];  // grouping indicator per observation
      // group-level predictor values
      vector[N] Z_1_1;
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
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      vector[N_1] z_1[M_1];  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      lprior += student_t_lpdf(Intercept | 3, 0, 2.5);
      lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
        - 1 * student_t_lccdf(0 | 3, 0, 2.5);
    }
    model {
      // likelihood including constants
      if (!prior_only) {
        // initialize linear predictor term
        vector[N] mu = Intercept + rep_vector(0.0, N);
        for (n in 1:N) {
          // add more terms to the linear predictor
          mu[n] += r_1_1[J_1[n]] * Z_1_1[n];
        }
        target += bernoulli_logit_glm_lpmf(Y | Xc, mu, b);
      }
      // priors including constants
      target += lprior;
      target += std_normal_lpdf(z_1[1]);
    }
    generated quantities {
      // actual population-level intercept
      real b_Intercept = Intercept - dot_product(means_X, b);
    }

We can see that some half-t distributions are used as priors for the
hyperparameters.

We examine the fit:

``` r
summary(bmod)
```

     Family: bernoulli 
      Links: mu = logit 
    Formula: resp ~ age + smoke + (1 | id) 
       Data: ohio (Number of observations: 2148) 
      Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
             total post-warmup draws = 4000

    Group-Level Effects: 
    ~id (Number of levels: 537) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     2.19      0.18     1.84     2.56 1.00     1044     1856

    Population-Level Effects: 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept    -3.11      0.22    -3.56    -2.70 1.00     1331     1942
    age          -0.17      0.07    -0.31    -0.04 1.01     5813     2587
    smoke         0.39      0.28    -0.16     0.94 1.00     1452     2279

    Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
    and Tail_ESS are effective sample size measures, and Rhat is the potential
    scale reduction factor on split chains (at convergence, Rhat = 1).

The results are consistent with previous results.

# MGCV

It is possible to fit some GLMMs within the GAM framework of the `mgcv`
package. An explanation of this can be found in this
[blog](https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/)

We need to make a factor version of id otherwise it gets treated as a
numerical variable.

``` r
ohio$fid = factor(ohio$id)
gmod = gam(resp ~ age + smoke + s(fid,bs="re"), 
           family=binomial, data=ohio, method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: binomial 
    Link function: logit 

    Formula:
    resp ~ age + smoke + s(fid, bs = "re")

    Parametric coefficients:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)  -2.3690     0.1508  -15.71   <2e-16
    age          -0.1523     0.0627   -2.43    0.015
    smoke         0.2956     0.2405    1.23    0.219

    Approximate significance of smooth terms:
           edf Ref.df Chi.sq p-value
    s(fid) 282    535    548  <2e-16

    R-sq.(adj) =  0.393   Deviance explained = 46.1%
    -REML = 814.01  Scale est. = 1         n = 2148

We get the fixed effect estimates. We also get a test on the random
effect (as described in this
[article](https://doi.org/10.1093/biomet/ast038)). The hypothesis of no
variation between the ids is rejected.

We can get an estimate of the id SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

           std.dev  lower  upper
    s(fid)  1.9486 1.6539 2.2957

    Rank: 1/1

which is the same as the REML estimate from `lmer` earlier.

The random effect estimates for the fields can be found with:

``` r
head(coef(gmod))
```

    (Intercept)         age       smoke    s(fid).1    s(fid).2    s(fid).3 
       -2.36900    -0.15226     0.29557    -0.72026    -0.72026    -0.72026 

# GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(resp ~ age + smoke + s(fid,bs="re"), 
           family=binomial, data=ohio, fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior densities for the fixed effects as:

``` r
par(mfrow=c(1,2))
for(i in 2:3){
plot(gimod$beta[i,],gimod$density[i,],type="l",
     xlab=gmod$term.names[i],ylab="density")
}
par(mfrow=c(1,1))
```

<figure>
<img src="figs/fig-ohioginlareff-1..svg" id="fig-ohioginlareff"
alt="Figure 3: Posteriors of the fixed effects" />
<figcaption aria-hidden="true">Figure 3: Posteriors of the fixed
effects</figcaption>
</figure>

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

- No strong differences in the results between the different methods. In
  all cases, we do not find strong evidence of an effect for maternal
  smoking.

- LME4 was very fast. INLA was fast. BRMS, MGCV and GINLA were slower.
  We have a large number of subject random effects which slows down the
  `mgcv` approach considerably.

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
     [1] mgcv_1.8-40   nlme_3.1-159  brms_2.18.0   Rcpp_1.0.9    knitr_1.40    INLA_22.09.26 sp_1.5-0      foreach_1.5.2
     [9] lme4_1.1-30   Matrix_1.5-1  ggplot2_3.3.6

    loaded via a namespace (and not attached):
      [1] minqa_1.2.4          colorspace_2.0-3     ellipsis_0.3.2       ggridges_0.5.4       markdown_1.1        
      [6] base64enc_0.1-3      rstudioapi_0.14      Deriv_4.1.3          farver_2.1.1         rstan_2.26.13       
     [11] DT_0.25              fansi_1.0.3          mvtnorm_1.1-3        bridgesampling_1.1-2 codetools_0.2-18    
     [16] splines_4.2.1        shinythemes_1.2.0    bayesplot_1.9.0      jsonlite_1.8.0       nloptr_2.0.3        
     [21] shiny_1.7.2          compiler_4.2.1       backports_1.4.1      assertthat_0.2.1     fastmap_1.1.0       
     [26] cli_3.4.1            later_1.3.0          htmltools_0.5.3      prettyunits_1.1.1    tools_4.2.1         
     [31] igraph_1.3.5         coda_0.19-4          gtable_0.3.1         glue_1.6.2           reshape2_1.4.4      
     [36] dplyr_1.0.10         posterior_1.3.1      V8_4.2.1             vctrs_0.4.1          svglite_2.1.0       
     [41] iterators_1.0.14     crosstalk_1.2.0      tensorA_0.36.2       xfun_0.33            stringr_1.4.1       
     [46] ps_1.7.1             mime_0.12            miniUI_0.1.1.1       lifecycle_1.0.2      gtools_3.9.3        
     [51] MASS_7.3-58.1        zoo_1.8-11           scales_1.2.1         colourpicker_1.1.1   promises_1.2.0.1    
     [56] Brobdingnag_1.2-7    faraway_1.0.9        inline_0.3.19        shinystan_2.6.0      yaml_2.3.5          
     [61] curl_4.3.2           gridExtra_2.3        loo_2.5.1            StanHeaders_2.26.13  stringi_1.7.8       
     [66] highr_0.9            dygraphs_1.1.1.6     checkmate_2.1.0      boot_1.3-28          pkgbuild_1.3.1      
     [71] systemfonts_1.0.4    rlang_1.0.6          pkgconfig_2.0.3      matrixStats_0.62.0   distributional_0.3.1
     [76] evaluate_0.16        lattice_0.20-45      purrr_0.3.4          labeling_0.4.2       rstantools_2.2.0    
     [81] htmlwidgets_1.5.4    tidyselect_1.1.2     processx_3.7.0       plyr_1.8.7           magrittr_2.0.3      
     [86] R6_2.5.1             generics_0.1.3       DBI_1.1.3            pillar_1.8.1         withr_2.5.0         
     [91] xts_0.12.1           abind_1.4-5          tibble_3.1.8         crayon_1.5.1         utf8_1.2.2          
     [96] rmarkdown_2.16       grid_4.2.1           callr_3.7.2          threejs_0.3.3        digest_0.6.29       
    [101] xtable_1.8-4         httpuv_1.6.6         RcppParallel_5.1.5   stats4_4.2.1         munsell_0.5.0       
    [106] shinyjs_2.1.0       
