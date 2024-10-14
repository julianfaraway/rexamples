# Poisson GLMM
[Julian Faraway](https://julianfaraway.github.io/)
2024-10-14

- [Data and Model](#data-and-model)
- [LME4](#lme4)
- [INLA](#inla)
- [BRMS](#brms)
- [MGCV](#mgcv)
- [GINLA](#ginla)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](../index.md) for an overview.

This example is discussed in more detail in the book [Bayesian
Regression Modeling with
INLA](https://julianfaraway.github.io/brinlabook/chaglmm.html#sec:poissonglmm)

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

In [Davison and Hinkley,
1997](https://doi.org/10.1017/CBO9780511802843), the results of a study
on Nitrofen, a herbicide, are reported. Due to concern regarding the
effect on animal life, 50 female water fleas were divided into five
groups of ten each and treated with different concentrations of the
herbicide. The number of offspring in three subsequent broods for each
flea was recorded. We start by loading the data from the `boot` package:
(the `boot` package comes with base R distribution so there is no need
to download this)

``` r
data(nitrofen, package="boot")
head(nitrofen)
```

      conc brood1 brood2 brood3 total
    1    0      3     14     10    27
    2    0      5     12     15    32
    3    0      6     11     17    34
    4    0      6     12     15    33
    5    0      6     15     15    36
    6    0      5     14     15    34

We need to rearrange the data to have one response value per line:

``` r
lnitrofen = data.frame(conc = rep(nitrofen$conc,each=3),
  live = as.numeric(t(as.matrix(nitrofen[,2:4]))),
  id = rep(1:50,each=3),
  brood = rep(1:3,50))
head(lnitrofen)
```

      conc live id brood
    1    0    3  1     1
    2    0   14  1     2
    3    0   10  1     3
    4    0    5  2     1
    5    0   12  2     2
    6    0   15  2     3

Make a plot of the data:

``` r
lnitrofen$jconc <- lnitrofen$conc + rep(c(-10,0,10),50)
lnitrofen$brood = factor(lnitrofen$brood)
ggplot(lnitrofen, aes(x=jconc,y=live, shape=brood, color=brood)) + 
       geom_point(position = position_jitter(w = 0, h = 0.5)) + 
       xlab("Concentration") + labs(shape = "Brood")
```

<img src="figs/fig-nitrodat-1..svg" id="fig-nitrodat"
alt="Figure 1: The number of live offspring varies with the concentration of Nitrofen and the brood number." />

Since the response is a small count, a Poisson model is a natural
choice. We expect the rate of the response to vary with the brood and
concentration level. The plot of the data suggests these two predictors
may have an interaction. The three observations for a single flea are
likely to be correlated. We might expect a given flea to tend to produce
more, or less, offspring over a lifetime. We can model this with an
additive random effect. The linear predictor is:

``` math
\eta_i = x_i^T \beta + u_{j(i)}, \quad i=1, \dots, 150. \quad j=1, \dots 50,
```

where $x_i$ is a vector from the design matrix encoding the information
about the $i^{th}$ observation and $u_j$ is the random affect associated
with the $j^{th}$ flea. The response has distribution
$Y_i \sim Poisson(\exp(\eta_i))$.

# LME4

We fit a model using penalized quasi-likelihood (PQL) using the `lme4`
package:

``` r
glmod <- glmer(live ~ I(conc/300)*brood + (1|id), nAGQ=25, 
             family=poisson, data=lnitrofen)
summary(glmod, correlation = FALSE)
```

    Generalized linear mixed model fit by maximum likelihood (Adaptive Gauss-Hermite Quadrature, nAGQ = 25) ['glmerMod']
     Family: poisson  ( log )
    Formula: live ~ I(conc/300) * brood + (1 | id)
       Data: lnitrofen

         AIC      BIC   logLik deviance df.resid 
       313.9    335.0   -150.0    299.9      143 

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -2.208 -0.606 -0.008  0.618  3.565 

    Random effects:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 0.0911   0.302   
    Number of obs: 150, groups:  id, 50

    Fixed effects:
                       Estimate Std. Error z value Pr(>|z|)
    (Intercept)          1.6386     0.1367   11.99  < 2e-16
    I(conc/300)         -0.0437     0.2193   -0.20     0.84
    brood2               1.1687     0.1377    8.48  < 2e-16
    brood3               1.3512     0.1351   10.00  < 2e-16
    I(conc/300):brood2  -1.6730     0.2487   -6.73  1.7e-11
    I(conc/300):brood3  -1.8312     0.2451   -7.47  7.9e-14

We scaled the concentration by dividing by 300 (the maximum value is
310) to avoid scaling problems encountered with `glmer()`. This is
helpful in any case since it puts all the parameter estimates on a
similar scale. The first brood is the reference level so the slope for
this group is estimated as $-0.0437$ and is not statistically
significant, confirming the impression from the plot. We can see that
numbers of offspring in the second and third broods start out
significantly higher for zero concentration of the herbicide, with
estimates of $1.1688$ and $1.3512$. But as concentration increases, we
see that the numbers decrease significantly, with slopes of $-1.6730$
and $-1.8312$ relative to the first brood. The individual SD is
estimated at $0.302$ which is noticeably smaller than the estimates
above, indicating that the brood and concentration effects outweigh the
individual variation.

We can make a plot of the mean predicted response as concentration and
brood vary. I have chosen not specify a particular individual in the
random effects with the option `re.form=~0` . We have $u_i = 0$ and so
this represents the the response for a `typical` individual.

``` r
predf = data.frame(conc=rep(c(0,80,160,235,310),each=3),brood=factor(rep(1:3,5)))
predf$live = predict(glmod, newdata=predf, re.form=~0, type="response")
ggplot(predf, aes(x=conc,y=live,group=brood,color=brood)) + 
  geom_line() + xlab("Concentration")
```

<img src="figs/fig-prednitro-1..svg" id="fig-prednitro"
alt="Figure 2: Predicted number of live offspring" />

We see that if only the first brood were considered, the herbicide does
not have a large effect. In the second and third broods, the (negative)
effect of the herbicide becomes more apparent with fewer live offspring
being produced as the concentration rises.

# INLA

Integrated nested Laplace approximation is a method of Bayesian
computation which uses approximation rather than simulation. More can be
found on this topic in [Bayesian Regression Modeling with
INLA](http://julianfaraway.github.io/brinla/) and the [chapter on
GLMMs](https://julianfaraway.github.io/brinlabook/chaglmm.html)

The same model, with default priors, can be fitted with INLA as:

``` r
formula <- live ~ I(conc/300)*brood + f(id, model="iid")
imod <- inla(formula, family="poisson", data=lnitrofen)
```

The fixed effects summary is:

``` r
imod$summary.fixed |> kable()
```

|                    |     mean |      sd | 0.025quant | 0.5quant | 0.975quant |     mode | kld |
|:-------------------|---------:|--------:|-----------:|---------:|-----------:|---------:|----:|
| (Intercept)        |  1.63596 | 0.13642 |    1.36744 |  1.63617 |    1.90334 |  1.63617 |   0 |
| I(conc/300)        | -0.04216 | 0.21861 |   -0.47449 | -0.04116 |    0.38439 | -0.04118 |   0 |
| brood2             |  1.16798 | 0.13766 |    0.89852 |  1.16780 |    1.43841 |  1.16780 |   0 |
| brood3             |  1.35051 | 0.13506 |    1.08620 |  1.35033 |    1.61588 |  1.35033 |   0 |
| I(conc/300):brood2 | -1.66847 | 0.24851 |   -2.15743 | -1.66789 |   -1.18281 | -1.66788 |   0 |
| I(conc/300):brood3 | -1.82604 | 0.24499 |   -2.30823 | -1.82541 |   -1.34741 | -1.82541 |   0 |

The posterior means are very similar to the PQL estimates. We can get
plots of the posteriors of the fixed effects:

``` r
fnames = names(imod$marginals.fixed)
par(mfrow=c(2,2))
for(i in 1:4){
  plot(imod$marginals.fixed[[i]],
       type="l",
       ylab="density",
       xlab=fnames[i])
  abline(v=0)
}
par(mfrow=c(1,1))
```

<img src="figs/fig-nitrofpd-1..svg" id="fig-nitrofpd"
alt="Figure 3: Posterior densities of the fixed effects model for the Nitrofen data." />

We can also see the summary for the random effect SD:

``` r
hpd = inla.tmarginal(function(x) 1/sqrt(x), imod$marginals.hyperpar[[1]])
inla.zmarginal(hpd)
```

    Mean            0.293403 
    Stdev           0.0574257 
    Quantile  0.025 0.188438 
    Quantile  0.25  0.253485 
    Quantile  0.5   0.290358 
    Quantile  0.75  0.329905 
    Quantile  0.975 0.415035 

Again the result is very similar to the PQL output although notice that
INLA provides some assessment of uncertainty in this value in contrast
to the PQL result. We can also see the posterior density:

``` r
plot(hpd,type="l",xlab="linear predictor",ylab="density")
```

<img src="figs/fig-nitrohyppd-1..svg" id="fig-nitrohyppd"
alt="Figure 4: Posterior density of the SD of id" />

# BRMS

For this example, I did not write my own STAN program. I am not that
experienced in writing STAN programmes so it is better rely on the
superior experience of others.

[BRMS](https://paul-buerkner.github.io/brms/) stands for Bayesian
Regression Models with STAN. It provides a convenient wrapper to STAN
functionality.

Fitting the model is very similar to `lmer` as seen above:

``` r
bmod <- brm(live ~ I(conc/300)*brood + (1|id), 
            family=poisson, 
            data=lnitrofen, 
            refresh=0, silent=2, cores=4)
```

We can check the MCMC diagnostics and the posterior densities with:

``` r
plot(bmod)
```

<img src="figs/fig-nitrobrmsdiag-1..svg" id="fig-nitrobrmsdiag-1" />

<img src="figs/fig-nitrobrmsdiag-2..svg" id="fig-nitrobrmsdiag-2" />

Looks quite similar to the INLA results.

We can look at the STAN code that `brms` used with:

``` r
stancode(bmod)
```

    // generated with brms 2.21.0
    functions {
    }
    data {
      int<lower=1> N;  // total number of observations
      array[N] int Y;  // response variable
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
      vector<lower=0>[M_1] sd_1;  // group-level standard deviations
      array[M_1] vector[N_1] z_1;  // standardized group-level effects
    }
    transformed parameters {
      vector[N_1] r_1_1;  // actual group-level effects
      real lprior = 0;  // prior contributions to the log posterior
      r_1_1 = (sd_1[1] * (z_1[1]));
      lprior += student_t_lpdf(Intercept | 3, 1.9, 2.5);
      lprior += student_t_lpdf(sd_1 | 3, 0, 2.5)
        - 1 * student_t_lccdf(0 | 3, 0, 2.5);
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
        target += poisson_log_glm_lpmf(Y | Xc, mu, b);
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

     Family: poisson 
      Links: mu = log 
    Formula: live ~ I(conc/300) * brood + (1 | id) 
       Data: lnitrofen (Number of observations: 150) 
      Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
             total post-warmup draws = 4000

    Multilevel Hyperparameters:
    ~id (Number of levels: 50) 
                  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    sd(Intercept)     0.33      0.06     0.22     0.45 1.00     1588     2223

    Regression Coefficients:
                     Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    Intercept            1.64      0.14     1.37     1.91 1.00     1223     2486
    IconcD300           -0.06      0.23    -0.50     0.39 1.00     1424     2263
    brood2               1.17      0.14     0.91     1.45 1.00     1395     2598
    brood3               1.36      0.13     1.11     1.62 1.00     1457     2730
    IconcD300:brood2    -1.69      0.25    -2.17    -1.18 1.00     1626     2612
    IconcD300:brood3    -1.85      0.25    -2.32    -1.37 1.00     1466     2681

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
lnitrofen$fid = factor(lnitrofen$id)
gmod = gam(live ~ I(conc/300)*brood + s(fid,bs="re"), 
           data=lnitrofen, family="poisson", method="REML")
```

and look at the summary output:

``` r
summary(gmod)
```


    Family: poisson 
    Link function: log 

    Formula:
    live ~ I(conc/300) * brood + s(fid, bs = "re")

    Parametric coefficients:
                       Estimate Std. Error z value Pr(>|z|)
    (Intercept)          1.6470     0.1388   11.86  < 2e-16
    I(conc/300)         -0.0334     0.2222   -0.15     0.88
    brood2               1.1737     0.1370    8.57  < 2e-16
    brood3               1.3565     0.1342   10.11  < 2e-16
    I(conc/300):brood2  -1.6843     0.2464   -6.84  8.1e-12
    I(conc/300):brood3  -1.8435     0.2421   -7.61  2.7e-14

    Approximate significance of smooth terms:
            edf Ref.df Chi.sq p-value
    s(fid) 31.7     48   81.1  <2e-16

    R-sq.(adj) =  0.711   Deviance explained = 66.5%
    -REML = 405.68  Scale est. = 1         n = 150

We get the fixed effect estimates. We also get a test on the random
effect (as described in this
[article](https://doi.org/10.1093/biomet/ast038)). The hypothesis of no
variation between the ids is rejected.

We can get an estimate of the id SD:

``` r
gam.vcomp(gmod)
```


    Standard deviations and 0.95 confidence intervals:

           std.dev   lower   upper
    s(fid)  0.3163 0.22148 0.45171

    Rank: 1/1

which is the same as the REML estimate from `lmer` earlier.

The random effect estimates for the fields can be found with:

``` r
coef(gmod)
```

           (Intercept)        I(conc/300)             brood2             brood3 I(conc/300):brood2 I(conc/300):brood3 
              1.647020          -0.033400           1.173657           1.356469          -1.684338          -1.843512 
              s(fid).1           s(fid).2           s(fid).3           s(fid).4           s(fid).5           s(fid).6 
             -0.329841          -0.211252          -0.166687          -0.188777          -0.123622          -0.166687 
              s(fid).7           s(fid).8           s(fid).9          s(fid).10          s(fid).11          s(fid).12 
             -0.188777          -0.257407          -0.406447          -0.234124           0.125999           0.125999 
             s(fid).13          s(fid).14          s(fid).15          s(fid).16          s(fid).17          s(fid).18 
              0.173063           0.125999           0.195973          -0.053239          -0.026132           0.077187 
             s(fid).19          s(fid).20          s(fid).21          s(fid).22          s(fid).23          s(fid).24 
              0.101818           0.026517           0.301452           0.301452           0.123974           0.244729 
             s(fid).25          s(fid).26          s(fid).27          s(fid).28          s(fid).29          s(fid).30 
              0.328967           0.355942           0.328967           0.215484           0.301452           0.301452 
             s(fid).31          s(fid).32          s(fid).33          s(fid).34          s(fid).35          s(fid).36 
              0.334777           0.265780          -0.320622          -0.087909           0.463915           0.078692 
             s(fid).37          s(fid).38          s(fid).39          s(fid).40          s(fid).41          s(fid).42 
             -0.044725           0.038544           0.265780           0.117890          -0.243476          -0.243476 
             s(fid).43          s(fid).44          s(fid).45          s(fid).46          s(fid).47          s(fid).48 
             -0.189884          -0.594160           0.193946          -0.298415          -0.243476          -0.354727 
             s(fid).49          s(fid).50 
             -0.243476          -0.298415 

We make a Q-Q plot of the ID random effects:

``` r
qqnorm(coef(gmod)[-(1:6)])
```

<img src="figs/fig-gamqq-1..svg" id="fig-gamqq" />

Nothing unusual here - none of the IDs standout in particular.

# GINLA

In [Wood (2019)](https://doi.org/10.1093/biomet/asz044), a simplified
version of INLA is proposed. The first construct the GAM model without
fitting and then use the `ginla()` function to perform the computation.

``` r
gmod = gam(live ~ I(conc/300)*brood + s(fid,bs="re"), 
           data=lnitrofen, family="poisson", fit = FALSE)
gimod = ginla(gmod)
```

We get the posterior densities for the fixed effects as:

``` r
par(mfrow=c(3,2))
for(i in 1:6){
plot(gimod$beta[i,],gimod$density[i,],type="l",
     xlab=gmod$term.names[i],ylab="density")
}
par(mfrow=c(1,1))
```

<img src="figs/fig-nitroginlareff-1..svg" id="fig-nitroginlareff"
alt="Figure 5: Posteriors of the fixed effects" />

It is not straightforward to obtain the posterior densities of the
hyperparameters.

# Discussion

- No strong differences in the results between the different methods.

- LME4, MGCV and GINLA were very fast. INLA was fast. BRMS was slowest.
  But this is a small dataset and a simple model so we cannot draw too
  general a conclusion from this.

# Package version info

``` r
sessionInfo()
```

    R version 4.4.1 (2024-06-14)
    Platform: x86_64-apple-darwin20
    Running under: macOS Sonoma 14.7

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
     [1] mgcv_1.9-1    nlme_3.1-166  brms_2.21.0   Rcpp_1.0.13   knitr_1.48    INLA_24.06.27 sp_2.1-4      lme4_1.1-35.5
     [9] Matrix_1.7-0  ggplot2_3.5.1

    loaded via a namespace (and not attached):
     [1] tidyselect_1.2.1     dplyr_1.1.4          farver_2.1.2         loo_2.8.0            fastmap_1.2.0       
     [6] tensorA_0.36.2.1     digest_0.6.37        estimability_1.5.1   lifecycle_1.0.4      Deriv_4.1.3         
    [11] sf_1.0-16            StanHeaders_2.32.10  magrittr_2.0.3       posterior_1.6.0      compiler_4.4.1      
    [16] rlang_1.1.4          tools_4.4.1          utf8_1.2.4           yaml_2.3.10          labeling_0.4.3      
    [21] bridgesampling_1.1-2 pkgbuild_1.4.4       classInt_0.4-10      plyr_1.8.9           abind_1.4-5         
    [26] KernSmooth_2.23-24   withr_3.0.1          grid_4.4.1           stats4_4.4.1         fansi_1.0.6         
    [31] xtable_1.8-4         e1071_1.7-14         colorspace_2.1-1     inline_0.3.19        emmeans_1.10.4      
    [36] scales_1.3.0         MASS_7.3-61          cli_3.6.3            mvtnorm_1.2-6        rmarkdown_2.28      
    [41] generics_0.1.3       RcppParallel_5.1.9   rstudioapi_0.16.0    reshape2_1.4.4       minqa_1.2.8         
    [46] DBI_1.2.3            proxy_0.4-27         rstan_2.32.6         stringr_1.5.1        splines_4.4.1       
    [51] bayesplot_1.11.1     parallel_4.4.1       matrixStats_1.3.0    vctrs_0.6.5          boot_1.3-31         
    [56] jsonlite_1.8.8       systemfonts_1.1.0    units_0.8-5          glue_1.7.0           nloptr_2.1.1        
    [61] codetools_0.2-20     distributional_0.4.0 stringi_1.8.4        gtable_0.3.5         QuickJSR_1.3.1      
    [66] munsell_0.5.1        tibble_3.2.1         pillar_1.9.0         htmltools_0.5.8.1    Brobdingnag_1.2-9   
    [71] R6_2.5.1             fmesher_0.1.7        evaluate_0.24.0      lattice_0.22-6       backports_1.5.0     
    [76] rstantools_2.4.0     class_7.3-22         svglite_2.1.3        coda_0.19-4.1        gridExtra_2.3       
    [81] checkmate_2.3.2      xfun_0.47            pkgconfig_2.0.3     
