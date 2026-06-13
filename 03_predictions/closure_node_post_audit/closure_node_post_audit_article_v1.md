---
title: "Closure-Node Registration after Audit"
subtitle: "Current conclusions, experimental ledger, and a non-predictor framing for the RGM/ROSI chemistry line"
author: "L. D. Mata Sánchez"
date: "Version 1.0 — June 2026"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

This note closes the current stage of the closure-node / redox-richness programme. The initial record, *Closure-Node Registration*, proposed that redox richness can be read as where unresolved relational difference is registered relative to an electronic closure node. In the transition-metal seed dataset, the descriptor

\[
d_{\mathrm{balance}}=1-|N_d-5|/5
\]

organizes oxidation-state count and span better than linear or monotonic baselines. The post-audit conclusion is narrower: this is a compact quantitative encoding of the known half-filled-shell pattern, not a demonstrated predictor superior to a fair quadratic baseline in the same filling coordinate.

The subsequent actinide and radius--complement tests sharpen the epistemic status further. Static \(f\)-balance, static \(s:d:f\) fractions, and a quadratic radius--complement representation all fail to establish predictive novelty once degree-of-freedom-matched baselines and label audits are applied. A first non-circular redox-couple potential test also fails to show predictive advantage. The programme should therefore be framed as a **generative mechanism framework**: it creates relational coordinates, hypotheses, and falsifiable experimental designs, but it is not yet a validated predictor of chemical redox behaviour.

# 1. Position

The correct current claim is:

\[
\boxed{\text{The framework generates interpretable relational descriptions and falsifiable tests; it is not yet a validated predictive model.}}
\]

The descriptor and geometry developed so far are useful as *mechanism screens*. They are not yet useful as independently validated predictors. This distinction should be explicit in the abstract, introduction, and conclusion of any public article.

# 2. Experiment 1: Closure-node registration record

The first associated experiment is the Zenodo v1 record:

- Title: *Closure-Node Registration: Redox richness as the signature of where unresolved relational difference is registered relative to a closure node — and two regimes of one event in the d- and f-blocks*.
- DOI: `10.5281/zenodo.20573845`.
- Published: 8 June 2026.
- Resource type: preprint.
- Files: `CLOSURE_NODE_REGISTRATION.pdf` and `ROSI_article_support_2026-06-06.zip`.

The original d-block result remains reproducible from the archived support data: `d_balance` obtains LOSO-RMSE \(1.2259\) for oxidation-state count and \(1.4755\) for oxidation span, versus \(1.6483\) and \(1.9708\) for a linear \(N_d\) baseline.

However, the fair comparison is not against linear \(N_d\). Since \(d_{\mathrm{balance}}\) is an affine transform of \(|N_d-5|\), it already imposes a tent-shaped peak at half filling. The fair baseline is a quadratic in \(N_d\), or a quadratic in \(|N_d-5|\).

# 3. Resolution of the lyo_bench critique

The Reddit critique correctly identifies four required fixes:

1. compare against quadratic filling baselines, not only linear \(N_d\);
2. bootstrap model-comparison error, not only the \(d^5\) local mean contrast;
3. respect the leave-one-series-out design by block/cluster bootstrapping at the series level;
4. lock the oxidation-state source and model set before fitting.

The series-block bootstrap below uses the same three blocks as the LOSO design: \(3d\), \(4d\), and \(5d\). With only three clusters the intervals are still coarse, but they are more honest than element-wise resampling.

## 3.1 Count models

| model | LOSO_RMSE | AICc_in_sample |
| --- | --- | --- |
| d_balance | 1.226 | 10.743 |
| Nd linear | 1.648 | 34.641 |
| Nd quadratic | 1.291 | 11.700 |
| |Nd-5| quadratic | 1.222 | 12.534 |
| group | 1.657 | 34.616 |
| period+group | 1.751 | 36.303 |

## 3.2 Span models

| model | LOSO_RMSE | AICc_in_sample |
| --- | --- | --- |
| d_balance | 1.475 | 24.950 |
| Nd linear | 1.971 | 45.061 |
| Nd quadratic | 1.509 | 22.701 |
| |Nd-5| quadratic | 1.477 | 26.624 |
| group | 1.993 | 45.056 |
| period+group | 1.994 | 46.574 |

## 3.3 Series-block bootstrap of RMSE differences

Here

\[
\Delta RMSE = RMSE(d_{\mathrm{balance}}) - RMSE(\text{baseline}).
\]

Negative values favour `d_balance`.

| target | baseline | delta_full | ci_low | ci_high | prop_candidate_better |
| --- | --- | --- | --- | --- | --- |
| count | Nd linear | -0.422 | -0.686 | -0.024 | 1.000 |
| count | Nd quadratic | -0.065 | -0.292 | 0.109 | 0.704 |
| count | |Nd-5| quadratic | 0.004 | -0.018 | 0.021 | 0.260 |
| span | Nd linear | -0.495 | -0.633 | -0.416 | 1.000 |
| span | Nd quadratic | -0.034 | -0.522 | 0.306 | 0.704 |
| span | |Nd-5| quadratic | -0.001 | -0.062 | 0.039 | 0.591 |

The interpretation is now stable:

\[
\boxed{d_{\mathrm{balance}}\ \text{beats baselines that cannot represent a peak, but not the fair curved filling baselines.}}
\]

For count, the difference against \(N_d+N_d^2\) crosses zero. Against quadratic \(|N_d-5|\), it is essentially a tie. For span, the same pattern holds.

# 4. What survives from the d-block result

The d-block experiment should be described as a compact quantitative restatement of known chemistry:

\[
\boxed{d_{\mathrm{balance}}\ \text{encodes the half-filled-shell symmetry, not a new independent chemical law.}}
\]

The useful parts are:

- the effect is organized by symmetry about \(d^5\);
- it is not well described by strictly monotonic properties alone;
- it provides a clean sanity check for the framework;
- it forces a falsifiable distinction between compact representation and predictive novelty.

The not-yet-supported parts are:

- “out-predicts electron count” as a headline;
- “new predictor” relative to quadratic filling;
- any claim that RGM/ROSI is validated by the d-block dataset.

# 5. Associated experiments and current status

| experiment_id | name | main_candidate | strongest_baseline | status |
| --- | --- | --- | --- | --- |
| E1 | Closure-node d-block seed (Zenodo 20573845) | d_balance = 1-|Nd-5|/5 | quadratic in Nd / quadratic in |Nd-5| | compact encoding of known half-filled-shell symmetry; not a validated superior predictor |
| E2 | Chemical-property confound audit | d_balance added after monotonic property quadratics | EN+EN^2, IE+IE^2, radius+radius^2 | not a proxy for monotonic properties; still textbook filling symmetry |
| E3 | Actinide f-balance stress test | f_balance around f7 | Nf + Nf^2 | negative/partial stress test |
| E4 | Actinide s-d-f static fraction test | P3, entropy, electron-hole terms | Nf + Nf^2 / Nf(14-Nf) | static orbital fractions insufficient |
| E5 | Quadratic radius-complement state/edge screen | R=sqrt(x), Rbar=sqrt(1-x) | DOF-matched polynomial in same occupancy fractions; state-output edge baseline | interpretive representation only; predictor claim falsified in this screen |
| E6 | External redox-couple potential test | radius-complement deltas | Z + charge/delta-charge | first non-circular predictive test negative |

# 6. Post-audit conclusion for radius--complement geometry

The radius--complement construction

\[
R_i=\sqrt{x_i},\qquad \bar R_i=\sqrt{1-x_i},\qquad R_i^2+\bar R_i^2=1
\]

is interpretively attractive because it separates occupation from complement. The product \(R_i\bar R_i\) highlights half filling without manually adding a half-filled rule.

But the post-audit result is negative:

\[
\boxed{\text{The transformation does not add predictive information beyond flexible polynomials in the same occupancies.}}
\]

The constructed transition-edge task is also not independent when the label is defined as the conjunction of two endpoint state labels. Therefore it cannot validate a transition geometry without external redox-couple data.

# 7. First non-circular redox-couple test

The first non-circular test used redox-couple potentials for U, Np, Pu, and Am in acidic aqueous solution. The target was no longer “state observed” or “edge observed”, but an external potential \(E^0\).

The result is negative: simple \(Z\)-and-charge baselines outperform radius--complement deltas in leave-one-element-out testing.

This closes the current predictive claim:

\[
\boxed{\text{No current experiment establishes predictive novelty for the closure-node or radius--complement descriptors.}}
\]

# 8. What the article should now say

A publication-quality article should state:

> The closure-node and radius--complement constructions are generative relational coordinates. They can propose hypotheses, organize known patterns, and define falsifiable experiments. In the present experiments they do not yet outperform fair baselines, so they should not be presented as validated predictors.

Recommended framing:

\[
\boxed{\text{generation of falsifiable mechanisms, not prediction.}}
\]

# 9. Response to lyo_bench

A concise public response can be:

> You were right on the fair baseline and the bootstrap granularity. I reran the model comparison with quadratic filling baselines and a series-block bootstrap. The descriptor still robustly beats linear \(N_d\), but it does not beat \(N_d+N_d^2\) or quadratic \(|N_d-5|\). The current status is therefore: compact encoding of half-filled-shell symmetry, not a predictor established beyond the fair curved filling baseline. I also agree that the oxidation-state source must be locked before any stronger claim. The follow-up actinide and radius--complement tests were treated as falsifier-driven screens; they did not establish predictive novelty. The framework is useful as a generator of hypotheses and tests, not yet as a validated chemical predictor.

# 10. Final conclusion

The most honest closure is:

\[
\boxed{\text{The framework has generated useful coordinates and falsifiers, but the predictor claim is currently negative.}}
\]

The scientific value at this stage is not a successful predictor. It is the disciplined conversion of a relational intuition into reproducible tests, fair baselines, and falsifiable next steps.

# References

1. L. D. Mata Sánchez, *Closure-Node Registration*, Zenodo record `10.5281/zenodo.20573845`, 2026.
2. `ROSI_article_support_2026-06-06.zip`, supplementary support archive attached to Zenodo record `10.5281/zenodo.20573845`.
3. Public lyo_bench Reddit critique, treated here as an audit prompt rather than a bibliographic source.
