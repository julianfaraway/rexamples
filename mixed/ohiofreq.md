# Binary response GLMM using Frequentist Methods
[Julian Faraway](https://julianfaraway.github.io/)
2024-10-14

- [Data and Model](#data-and-model)
- [LME4](#lme4)
- [NLME](#nlme)
- [MMRM](#mmrm)
- [GLMMTMB](#glmmtmb)
- [Discussion](#discussion)
- [Package version info](#package-version-info)

See the [introduction](../index.md) for an overview.

See a [mostly Bayesian analysis](ohio.md) analysis of the same data.

This example is discussed in more detail in the book [Bayesian
Regression Modeling with
INLA](https://julianfaraway.github.io/brinlabook/chaglmm.html)

Packages used:

``` r
library(knitr)
library(here)
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

Read in and examine the first two subjects worth of data:

``` r
ohio = read.csv(here("data","ohio.csv"),header = TRUE)
kable(ohio[1:8,])
```

| resp |  id | age | smoke |
|-----:|----:|----:|------:|
|    0 |   0 |  -2 |     0 |
|    0 |   0 |  -1 |     0 |
|    0 |   0 |   0 |     0 |
|    0 |   0 |   1 |     0 |
|    0 |   1 |  -2 |     0 |
|    0 |   1 |  -1 |     0 |
|    0 |   1 |   0 |     0 |
|    0 |   1 |   1 |     0 |

We sum the number of smoking and non-smoking mothers:

``` r
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

``` math
\eta_{ij} = \beta_0 + \beta_1 age_j + \beta_2 smoke_i + u_i, \quad i=1, \dots ,537, \quad j=1,2,3,4,
```

with

``` math
P(Y_{ij} = 1) = {\exp(\eta_{ij}) \over 1+\exp(\eta_{ij})}.
```

The random effect $u_i$ models the propensity of child $i$ to wheeze.
Children are likely to vary in their health condition and this effect
enables us to include this unknown variation in the model. Because $u_i$
is added to all four observations for a child, we induce a positive
correlation among the four responses as we might naturally expect. The
response is Bernoulli or, in other words, binomial with trial size one.

# LME4

Here is the model fit penalized quasi-likelihood using the `lme4`
package:

``` r
library(lme4)
modagh <- glmer(resp ~ age + smoke + (1|id), 
              family=binomial, data=ohio)
summary(modagh, correlation = FALSE)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) ['glmerMod']
     Family: binomial  ( logit )
    Formula: resp ~ age + smoke + (1 | id)
       Data: ohio

         AIC      BIC   logLik deviance df.resid 
      1597.9   1620.6   -794.9   1589.9     2144 

    Scaled residuals: 
       Min     1Q Median     3Q    Max 
    -1.403 -0.180 -0.158 -0.132  2.518 

    Random effects:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 5.49     2.34    
    Number of obs: 2148, groups:  id, 537

    Fixed effects:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)   -3.374      0.275  -12.27   <2e-16
    age           -0.177      0.068   -2.60   0.0093
    smoke          0.415      0.287    1.45   0.1485

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

# NLME

Cannot fit GLMMs (other than normal response).

# MMRM

Cannot fit GLMMs (other than normal response).

# GLMMTMB

See the discussion for the [single random effect
example](pulpfreq.md#GLMMTMB) for some introduction.

``` r
library(glmmTMB)
```

We can fit the model with:

``` r
gtmod <- glmmTMB(resp ~ age + smoke + (1|id), 
             family=binomial, data=ohio)
summary(gtmod)
```

     Family: binomial  ( logit )
    Formula:          resp ~ age + smoke + (1 | id)
    Data: ohio

         AIC      BIC   logLik deviance df.resid 
      1597.9   1620.6   -794.9   1589.9     2144 

    Random effects:

    Conditional model:
     Groups Name        Variance Std.Dev.
     id     (Intercept) 5.49     2.34    
    Number of obs: 2148, groups:  id, 537

    Conditional model:
                Estimate Std. Error z value Pr(>|z|)
    (Intercept)   -3.374      0.275  -12.27   <2e-16
    age           -0.177      0.068   -2.60   0.0093
    smoke          0.415      0.287    1.45   0.1484

Same as `lme4`.

# Discussion

- In both cases, we do not find evidence of an effect for maternal
  smoking.

# Package version info

``` r
xfun::session_info()
```

    R version 4.4.1 (2024-06-14)
    Platform: x86_64-apple-darwin20
    Running under: macOS Sonoma 14.7

    Locale: en_US.UTF-8 / en_US.UTF-8 / en_US.UTF-8 / C / en_US.UTF-8 / en_US.UTF-8

    Package version:
      base64enc_0.1.3     boot_1.3-31         bslib_0.8.0         cachem_1.1.0        cli_3.6.3          
      coda_0.19-4.1       compiler_4.4.1      cpp11_0.5.0         digest_0.6.37       emmeans_1.10.4     
      estimability_1.5.1  evaluate_0.24.0     faraway_1.0.8       fastmap_1.2.0       fontawesome_0.5.2  
      fs_1.6.4            glmmTMB_1.1.10.9000 glue_1.8.0          graphics_4.4.1      grDevices_4.4.1    
      grid_4.4.1          here_1.0.1          highr_0.11          htmltools_0.5.8.1   jquerylib_0.1.4    
      jsonlite_1.8.8      knitr_1.48          lattice_0.22-6      lifecycle_1.0.4     lme4_1.1-35.5      
      MASS_7.3-61         Matrix_1.7-0        memoise_2.0.1       methods_4.4.1       mgcv_1.9-1         
      mime_0.12           minqa_1.2.8         mvtnorm_1.2-6       nlme_3.1-166        nloptr_2.1.1       
      numDeriv_2016.8-1.1 parallel_4.4.1      R6_2.5.1            rappdirs_0.3.3      rbibutils_2.3      
      Rcpp_1.0.13         RcppEigen_0.3.4.0.2 Rdpack_2.6.1        reformulas_0.3.0    rlang_1.1.4        
      rmarkdown_2.28      rprojroot_2.0.4     rstudioapi_0.16.0   sass_0.4.9          splines_4.4.1      
      stats_4.4.1         svglite_2.1.3       systemfonts_1.1.0   tinytex_0.52        TMB_1.9.15         
      tools_4.4.1         utils_4.4.1         xfun_0.47           xtable_1.8-4        yaml_2.3.10        
