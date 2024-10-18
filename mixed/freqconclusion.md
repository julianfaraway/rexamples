# Commentary on Frequentist Methods for fitting GLMMs

## LME4

`lme4` was able to fit all the models. This is partly because I featured this
package to fit this class of models in my *Extending the Linear Model with R*
where I avoided covering models that `lme4` could not fit. Nevertheless, `lme4`
is capable of fitting a wide range of models. There is the problem of inference
which `lme4` chooses to avoid but `lmerTest` does a convenient job of testing
the fixed effects for LMMs. You need to work harder to test the random effects 
but one could argue that the random effects reflect the assumed correlated 
structure of the data so there is no benefit in testing them (at least not
for the current data). If you have a GLMM (i.e. a non-Gaussian response) then
the inference gets harder.

## NLME

`nlme` has the advantage of being part of the standard distribution of R which
is great when you are sharing or collaborating because you can be sure that the
other party has `nlme` installed. This is a big advantage when dealing with
inexperienced R users. Furthermore, it's a very mature package that's been
extensively tested and is very stable. (`lme4` is also mature and widely
used but `nlme` even more so). `nlme` can fit some models that are unavailable
in `lme4` that have correlated error structures of various kinds. It also
can fit GLS and some non-linear models. But it cannot manage some models that 
`lme4` can handle
such as crossed effect models (at least not without some unnatural manipulations).
It's also slower for larger datasets although this speed difference was not
important for the smaller datasets considered here. Furthermore, it cannot
do GLMMs. It's also less cautious about inference than `lme4`. Sometimes the
results will be OK but `lme4` + `lmerTest` is a safer bet unless you are very 
well-informed
about LMM inference.

## MMRM

I was curious about this package and thought this would be a good way of discovering
its capabilities. But now I realise that it is special purpose software built by
those interested in a particular type of longitudinal data commonly arising
in clinical trials (although it can be useful for data from other sources). It
was able to fit some of the models here but it just was not designed to handle
many of them. Like `nlme` it has the capacity to model correlated error structures.
It is more capable than `nlme` in this respect and has better inferential tools.
If that's what you need, then
`mmrm` is for you but it's not a general purpose package.

## GLMMTMB

Again, I tried this package out of curiousity but now realise that I did not
understand its purpose. For the models considered here, it gives essentially
the same results as `lme4`. It's less well connected to other packages such
as `emmeans` and `lmerTest` and it's harder to install. I did not discover
any additional capabilities for the models fitted here. The strength of
`glmmTMB` is in fitting less common response types, such as the zero-inflated
Poisson. If you need to fit such a model, then `glmmTMB` is for you. It also promises to fit faster than `lme4` for
some combinations of data size and model. This was not an issue for these 
examples but perhaps may be useful in bigger problems.

## Other packages

There is a wide selection of packages that can fit some smaller subset
of GLMMs but nothing with particularly general capabilities. If you want
to take a Bayesian approach, there are some widely capable options such
as `brms`. It's also easier to fit your own custom model in Stan (or other
general purpose Bayesian software).

