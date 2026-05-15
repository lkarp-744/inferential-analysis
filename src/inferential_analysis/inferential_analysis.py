# ruff: noqa: D101, D102, N806, C901, N802, D107
"""This module provides tools for conducting inferential reproducibility analysis.

The original implementation is sourced from the StatNLP repository on GitHub:
https://github.com/StatNLP/empirical_methods/blob/master/inferential_reproducibility/inferential_analysis.py.

Modifications have been made to ensure compliance with our code quality standards.

This module is licensed under the Apache 2.0 License, given by the original authors.
"""

# The authors of pymer4 recommend to add the following lines when pymer is run inside a jupyter notebook.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
from dataclasses import dataclass
from typing import Literal, Self, TypedDict

import marginaleffects
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from plotnine import (
    aes,
    element_blank,
    facet_wrap,
    geom_boxplot,
    geom_line,
    geom_point,
    geom_pointrange,
    geom_smooth,
    ggplot,
    theme,
    theme_bw,
    xlab,
    ylab,
)
from pymer4.models import Lmer
from scipy.stats import chi2
from statsmodels.formula.api import mixedlm
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

Distribution = Literal["gaussian", "binomial"]


class GeneralizedLikelihoodTestResult(TypedDict):
    p: float
    chi_square: float
    df: int


@dataclass(frozen=True, kw_only=True)
class LinearMixedEffectsModelResults:
    log_likelihood: float
    fixed_effects_count: int
    residual_standard_deviation: float

    # NOTE: The emmeans package actually calculates two average things:
    # “marginal effects at the means” (MEM), or average slopes using emtrends(),
    # and “estimated marginal means” (EMM), or average predictions using emmeans().
    # It’s named after the second of these, hence the name emmeans

    model: Lmer | MixedLMResultsWrapper

    @classmethod
    def from_lmer_model(cls, model: Lmer) -> Self:
        return cls(
            log_likelihood=model.logLike,
            fixed_effects_count=len(model.coefs),
            residual_standard_deviation=model.ranef_var.loc["Residual", "Std"],
            model=model,
        )

    @classmethod
    def from_mixed_model(cls, model: MixedLMResultsWrapper) -> Self:
        return cls(
            log_likelihood=model.llf,
            fixed_effects_count=model.k_fe,
            residual_standard_deviation=math.sqrt(model.scale),
            model=model,
        )

    def compute_marginal_effects(
        self,
        marginal_vars: str,
        grouping_vars: str | None = None,
        grouping_type: Literal["means", "slopes"] = "means",
    ) -> list[pd.DataFrame]:
        if isinstance(self.model, MixedLMResultsWrapper):
            if grouping_vars is None:
                return [
                    marginaleffects.avg_predictions(self.model, by=marginal_vars)
                    .to_pandas()
                    .drop(columns=["statistic", "s_value", "p_value"])
                    .round(3)
                    .rename(
                        columns={
                            "estimate": "Estimate",
                            "contrast": "Contrast",
                            "std_error": "SE",
                            "conf_low": "2.5_ci",
                            "conf_high": "97.5_ci",
                        }
                    ),
                    marginaleffects.avg_comparisons(
                        self.model, variables={marginal_vars: "revpairwise"}
                    )
                    .to_pandas()
                    .drop(columns=["term", "statistic", "s_value"])
                    .round(3)
                    .rename(
                        columns={
                            "estimate": "Estimate",
                            "contrast": "Contrast",
                            "std_error": "SE",
                            "p_value": "P-val",
                            "conf_low": "2.5_ci",
                            "conf_high": "97.5_ci",
                        }
                    ),
                ]
            else:
                return [
                    marginaleffects.avg_slopes(
                        self.model,
                        by=grouping_vars,
                        variables=marginal_vars,
                        newdata="mean",
                    )
                    .to_pandas()
                    .drop(
                        columns=["statistic", "s_value", "p_value", "contrast", "term"]
                    )
                    .round(3)
                    .rename(
                        columns={
                            "estimate": "Estimate",
                            "contrast": "Contrast",
                            "std_error": "SE",
                            "conf_low": "2.5_ci",
                            "conf_high": "97.5_ci",
                        }
                    )
                    if grouping_type == "slopes"
                    else None,  # TODO: Handle this case
                    marginaleffects.avg_comparisons(
                        self.model,
                        by=grouping_vars,
                        variables=marginal_vars,
                        hypothesis="difference ~ revpairwise",
                    )
                    .to_pandas()
                    .drop(columns=["s_value"])
                    .round(3)
                    .rename(
                        columns={
                            "estimate": "Estimate",
                            "statistic": "T-stat",
                            "term": "Contrast",
                            "std_error": "SE",
                            "p_value": "P-val",
                            "conf_low": "2.5_ci",
                            "conf_high": "97.5_ci",
                        }
                    ),
                ]

        return [
            r for r in self.model.post_hoc(marginal_vars, grouping_vars=grouping_vars)
        ]


def generalized_likelihood_ratio_test(
    model_a: Lmer | LinearMixedEffectsModelResults,
    model_b: Lmer | LinearMixedEffectsModelResults,
) -> GeneralizedLikelihoodTestResult:
    """Compute the generalized likelihood ratio test between two models."""
    if isinstance(model_a, LinearMixedEffectsModelResults):
        a1 = model_a.log_likelihood
        a2 = model_a.fixed_effects_count

    else:
        a1 = model_a.logLike
        a2 = len(model_a.coefs)

    if isinstance(model_b, LinearMixedEffectsModelResults):
        b1 = model_b.log_likelihood
        b2 = model_b.fixed_effects_count

    else:
        b1 = model_b.logLike
        b2 = len(model_b.coefs)

    chi_square = 2 * abs(a1 - b1)
    delta_params = abs(a2 - b2)

    return {
        "chi_square": chi_square,
        "df": delta_params,
        "p": 1 - chi2.cdf(chi_square, df=delta_params),
    }


@dataclass(kw_only=True)
class SystemComparison:
    glrt: GeneralizedLikelihoodTestResult
    means: pd.DataFrame
    contrasts: pd.DataFrame

    @classmethod
    def from_results(
        cls,
        model_h0: LinearMixedEffectsModelResults,
        model_h1: LinearMixedEffectsModelResults,
        system: str,
    ) -> Self:
        glrt = generalized_likelihood_ratio_test(model_h0, model_h1)

        # create means and contrasts
        means, contrasts = model_h1.compute_marginal_effects(system)

        # add effect size (a Hodge's g derivate) to contrasts
        contrasts = contrasts.assign(
            Effect_size_g=lambda df: df.Estimate / model_h1.residual_standard_deviation
        )

        return cls(
            glrt=glrt,
            means=means.drop(columns="DF", errors="ignore").rename(
                columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"}
            ),
            contrasts=contrasts.drop(
                columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore"
            ).rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"}),
        )


@dataclass(kw_only=True)
class ConditionalSystemComparison(SystemComparison):
    slopes: pd.DataFrame
    interaction_plot: ggplot
    data_property: str


@dataclass(kw_only=True)
class Reliability:
    algorithm: str
    icc: pd.DataFrame


@dataclass(kw_only=True)
class HyperParameterAssessment:
    algorithm: str
    glrt: GeneralizedLikelihoodTestResult
    means: pd.DataFrame
    contrasts: pd.DataFrame


@dataclass(kw_only=True)
class ConditionalHyperParameterAssessment(HyperParameterAssessment):
    data_property: str
    interaction_plot: ggplot
    slopes: pd.DataFrame


class InferentialAnalysis:
    def __init__(
        self,
        evaluation_data: pd.DataFrame,
        eval_metric_col: str,
        system_col: str,
        input_identifier_col: str,
        distribution: Distribution = "gaussian",
    ) -> None:
        self.data = evaluation_data
        self.metric = eval_metric_col
        self.system = system_col
        self.input_id = input_identifier_col
        self.distribution = distribution

        # check input consistency
        if distribution == "gaussian":
            if not is_numeric_dtype(self.data[self.metric]):
                raise ValueError(
                    f"Data type of provided evaluation metric {self.metric} is not numerical!"
                )
        elif distribution == "binomial":
            if not len(self.data[self.metric].unique()) == 2:
                raise ValueError(
                    f"Provided evaluation metric data column {self.metric} has more or less than 2 values. You can't run a binomial model."
                )

        # make sure that self.system and input_identifer_var are proper categoricals
        if not isinstance(self.data[self.input_id].dtype, pd.CategoricalDtype):
            self.data = self.data.astype({self.input_id: "string"})
            self.data = self.data.astype({self.input_id: "categorical"})

        if not isinstance(self.data[self.system].dtype, pd.CategoricalDtype):
            print(
                f"WARNING: {self.system} is not categorical! Datatype will be converted."
            )
            self.data = self.data.astype({self.system: "string"})
            self.data = self.data.astype({self.system: "categorical"})

        # make sure that self.system categories are all strings. This is important for Lmer.
        self.data[self.system] = self.data[self.system].cat.rename_categories(
            lambda x: str(x)
        )

    def system_comparison(
        self, alpha: float = 0.05, verbose: bool = True, row_filter: str = ""
    ) -> SystemComparison:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # minimize data to speed processing and remove rows with missing values
        if row_filter:
            model_data = self.data.query(row_filter).copy()
            model_data = model_data[[self.system, self.metric, self.input_id]].dropna()
        else:
            model_data = self.data[[self.system, self.metric, self.input_id]].dropna()

        model_data[self.system] = model_data[self.system].cat.remove_unused_categories()

        # instantiate and fit models
        print("Fitting H0-model.")
        model_H0 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(f"{self.metric} ~ 1", model_data, groups=self.input_id).fit(
                reml=False
            )
        )

        print("Fitting H1-model.")
        # NOTE: https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
        # shows that this formula is equivalent to the one used in lme4.
        model_H1 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ C({self.system})",
                model_data,
                groups=self.input_id,
            ).fit(reml=False)
        )

        result = SystemComparison.from_results(model_H0, model_H1, self.system)

        if result.glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        return result

    def conditional_system_comparison(
        self,
        data_prop_col: str,
        alpha: float = 0.05,
        scale_data_prop: bool = False,
        verbose: bool = True,
        row_filter: str = "",
    ) -> ConditionalSystemComparison:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # check data type of data_prop_col
        if isinstance(self.data[data_prop_col].dtype, pd.CategoricalDtype):
            self.data[data_prop_col] = self.data[data_prop_col].cat.rename_categories(
                lambda x: str(x)
            )
            print(
                "Data property is a categorical variable. Applying cell means model and reporting means."
            )
            reported_estimates = "means"
        elif is_string_dtype(self.data[data_prop_col]):
            self.data = self.data.astype({data_prop_col: "categorical"})
            print(
                "Data property is a categorical variable. Applying cell means model and reporting means."
            )
            reported_estimates = "means"
        elif is_numeric_dtype(self.data[data_prop_col]):
            print(
                "Data property is a numeric variable. Applying individual trends model and reporting slopes."
            )
            reported_estimates = "slopes"
        else:
            raise ValueError(
                f"Data property column {data_prop_col} data type is neither numeric nor categorical/string."
            )

        # minimize data to speed processing and removing rows with missing values
        if row_filter:
            model_data = self.data.query(row_filter).copy()
            model_data = model_data[
                [self.system, self.metric, data_prop_col, self.input_id]
            ].dropna()
        else:
            model_data = self.data[
                [self.system, self.metric, data_prop_col, self.input_id]
            ].dropna()

        model_data[self.system] = model_data[self.system].cat.remove_unused_categories()

        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            model_data[data_prop_col] = model_data[
                data_prop_col
            ].cat.remove_unused_categories()

        # instantiate and fit models
        if scale_data_prop:
            data_prop_col_m = f"scale({data_prop_col})"
        else:
            data_prop_col_m = data_prop_col

        model_factors = {
            self.system: [s for s in self.data[self.system].cat.categories]
        }

        if isinstance(self.data[data_prop_col].dtype, pd.CategoricalDtype):
            model_factors[data_prop_col] = [
                p for p in model_data[data_prop_col].cat.categories
            ]

        print("Fitting H0-model.")
        model_H0 = Lmer(
            formula=f"{self.metric} ~ {self.system} + {data_prop_col_m} + ( 1 | {self.input_id} )",
            data=model_data,
            family=self.distribution,
        )
        model_H0.fit(factors=model_factors, REML=False, summarize=False)
        h0_results = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ {self.system} + {data_prop_col_m}",
                model_data,
                groups=self.input_id,
            ).fit(reml=False)
        )
        # h0_results = LinearMixedEffectsModelResults.from_lmer_model(model_H0)

        print("Fitting H1-model.")
        model_H1 = Lmer(
            formula=f"{self.metric} ~ {self.system} + {data_prop_col} + {self.system}:{data_prop_col_m} + ( 1 | {self.input_id} )",
            data=model_data,
            family=self.distribution,
        )
        model_H1.fit(factors=model_factors, REML=False, summarize=False)
        h1_results = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ {self.system} + {data_prop_col} + {self.system}:{data_prop_col_m}",
                model_data,
                groups=self.input_id,
            ).fit(reml=False)
        )
        # h1_results = LinearMixedEffectsModelResults.from_lmer_model(model_H1)

        # compare models and calculate postHoc
        glrt = generalized_likelihood_ratio_test(h0_results, h1_results)

        # FOR CATEGORICAL data property!!!!
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            postHoc_result = h1_results.compute_marginal_effects(
                marginal_vars=self.system,
                grouping_vars=data_prop_col,
                grouping_type=reported_estimates,  # type: ignore[arg-type]
            )

        elif is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = h1_results.compute_marginal_effects(
                marginal_vars=data_prop_col,
                grouping_vars=self.system,
                grouping_type=reported_estimates,  # type: ignore[arg-type]
            )

        # simplify postHoc result
        means = pd.DataFrame()
        slopes = pd.DataFrame()
        if reported_estimates == "means":
            means = (
                postHoc_result[0]
                .drop(columns="DF", errors="ignore")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )
        else:
            slopes = (
                postHoc_result[0]
                .drop(columns="DF", errors="ignore")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )

        contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            contrasts = contrasts.assign(
                Effect_size_g=lambda df: (
                    df.Estimate / h1_results.residual_standard_deviation
                )
            )

        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            interaction_plot = (
                ggplot(postHoc_result[0])
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="top",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Estimated Expectation of Evaluation Metric")
                + geom_pointrange(
                    aes(
                        x=data_prop_col,
                        y="Estimate",
                        ymin="Estimate - SE",
                        ymax="Estimate + SE",
                        colour=self.system,
                    ),
                    alpha=0.7,
                )
                + geom_line(
                    aes(
                        x=data_prop_col,
                        y="Estimate",
                        group=self.system,
                        colour=self.system,
                    ),
                    alpha=0.3,
                )
            )

        if is_numeric_dtype(model_data[data_prop_col]):
            interaction_plot = (
                ggplot(data=model_data)
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="top",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Estimated Expected Evaluation Metric")
                + geom_smooth(
                    aes(
                        x=data_prop_col,
                        y=self.metric,
                        group=self.system,
                        linetype=self.system,
                    ),
                    method="lm",
                    colour="black",
                    se=False,
                )
            )

        if glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        return ConditionalSystemComparison(
            means=means,
            glrt=glrt,
            data_property=data_prop_col,
            slopes=slopes,
            contrasts=contrasts,
            interaction_plot=interaction_plot,
        )

    def icc(
        self, algorithm_id: str, facet_cols: str | list[str], row_filter: str = ""
    ) -> Reliability:
        # check if variables for random interceps have the correct data type
        if isinstance(facet_cols, str):
            facet_cols = [facet_cols]

        var_components = [self.input_id] + facet_cols

        for c in var_components:
            if not isinstance(self.data[c].dtype, pd.CategoricalDtype):
                print(f"WARNING: {c} is not categorical! Datatype will be converted.")
                self.data = self.data.astype({c: "string"})
                self.data = self.data.astype({c: "category"})

        # minimize data to speed processing and removing rows with missing values
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()

        if row_filter:
            model_data = model_data.query(row_filter).copy()

        model_data = model_data[[self.metric] + var_components].dropna()

        # instantiate and fit models
        formula_var_decomposition_model = (
            f"{self.metric} ~ {' + '.join([f'( 1 | {c})' for c in var_components])}"
        )
        var_decomposition_model = Lmer(
            formula_var_decomposition_model, data=model_data, family=self.distribution
        )

        print("Calculating variance decomposition.")
        var_decomposition_model.fit(summarize=False, control="calc.derivs = FALSE")

        # calculate icc based on the variance decomposition
        icc_data_frame = var_decomposition_model.ranef_var.drop(columns=["Name", "Std"])
        icc_data_frame["ICC"] = icc_data_frame["Var"] * 100 / sum(icc_data_frame["Var"])

        return Reliability(algorithm=algorithm_id, icc=icc_data_frame)

    def hyperparameter_assessment(
        self,
        algorithm_id: str,
        hyperparameter_col: str,
        alpha: float = 0.05,
        verbose: bool = True,
        row_filter: str = "",
    ) -> HyperParameterAssessment:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # make sure that hyperparameter_col and input_identifer_var are proper categoricals
        if not isinstance(self.data[hyperparameter_col].dtype, pd.CategoricalDtype):
            print(
                f"WARNING: {hyperparameter_col} is not categorical! Datatype will be converted."
            )
            self.data = self.data.astype({hyperparameter_col: "string"})
            self.data = self.data.astype({hyperparameter_col: "category"})

        # make sure that hyperparameter_col categories are all strings. This is important for Lmer.
        self.data[hyperparameter_col] = self.data[
            hyperparameter_col
        ].cat.rename_categories(lambda x: str(x))

        # minimize data to speed processing and remove rows with missing values
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()

        if row_filter:
            model_data = model_data.query(row_filter).copy()

        model_data = model_data[
            [hyperparameter_col, self.metric, self.input_id]
        ].dropna()
        model_data[hyperparameter_col] = model_data[
            hyperparameter_col
        ].cat.remove_unused_categories()

        # instantiate and fit models
        print("Fitting H0-model.")
        result_h0 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(f"{self.metric} ~ 1", data=model_data, groups=self.input_id).fit(
                reml=False
            )
        )

        print("Fitting H1-model.")
        result_h1 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ {hyperparameter_col}",
                data=model_data,
                groups=self.input_id,
            ).fit(reml=False)
        )

        # compare models and calculate postHoc stats
        glrt = generalized_likelihood_ratio_test(result_h0, result_h1)
        means, contrasts = result_h1.compute_marginal_effects(hyperparameter_col)

        # simplify postHoc result
        means = means.drop(columns="DF", errors="ignore").rename(
            columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"}
        )
        contrasts = contrasts.drop(
            columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore"
        ).rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})

        # add effect size (a Hedge's g derivate) to mean model contrasts
        contrasts = contrasts.assign(
            Effect_size_g=lambda df: df.Estimate / result_h1.residual_standard_deviation
        )

        if glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        return HyperParameterAssessment(
            means=means, algorithm=algorithm_id, contrasts=contrasts, glrt=glrt
        )

    def conditional_hyperparameter_assessment(
        self,
        algorithm_id: str,
        hyperparameter_col: str,
        data_prop_col: str,
        alpha: float = 0.05,
        scale_data_prop: bool = False,
        verbose: bool = True,
        row_filter: str = "",
    ) -> ConditionalHyperParameterAssessment:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # make sure that hyperparameter_col is a proper categorical
        if not isinstance(self.data[hyperparameter_col].dtype, pd.CategoricalDtype):
            print(
                f"WARNING: {hyperparameter_col} is not categorical! Datatype will be converted."
            )
            self.data = self.data.astype({hyperparameter_col: "string"})
            self.data = self.data.astype({hyperparameter_col: "category"})

        # make sure that hyperparameter_col categories are all strings. This is important for Lmer.
        self.data[hyperparameter_col] = self.data[
            hyperparameter_col
        ].cat.rename_categories(lambda x: str(x))

        # check data type of data_prop_col
        if isinstance(self.data[data_prop_col].dtype, pd.CategoricalDtype):
            self.data[data_prop_col] = self.data[data_prop_col].cat.rename_categories(
                lambda x: str(x)
            )
            print(
                "Data property is a categorical variable. Applying cell means model and reporting means."
            )
            reported_estimates = "means"
        elif is_string_dtype(self.data[data_prop_col]):
            self.data = self.data.astype({data_prop_col: "categorical"})
            print(
                "Data property is a categorical variable. Applying cell means model and reporting means."
            )
            reported_estimates = "means"
        elif is_numeric_dtype(self.data[data_prop_col]):
            print(
                "Data property is a numeric variable. Applying individual trends model and reporting slopes."
            )
            reported_estimates = "slopes"
        else:
            raise ValueError(
                f"Data property column {data_prop_col} data type is neither numeric nor categorical/string."
            )

        # minimize data to speed processing and remove rows with missing values
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()

        if row_filter:
            model_data = model_data.query(row_filter).copy()

        model_data = model_data[
            [hyperparameter_col, data_prop_col, self.metric, self.input_id]
        ].dropna()
        model_data[hyperparameter_col] = model_data[
            hyperparameter_col
        ].cat.remove_unused_categories()
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            model_data[data_prop_col] = model_data[
                data_prop_col
            ].cat.remove_unused_categories()

        # instantiate and fit models
        data_prop_col_m = (
            f"scale({data_prop_col})" if scale_data_prop else data_prop_col
        )
        model_factors = {
            hyperparameter_col: [
                s for s in model_data[hyperparameter_col].cat.categories
            ]
        }
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            model_factors[data_prop_col] = [
                p for p in model_data[data_prop_col].cat.categories
            ]

        print("Fitting H0-model.")
        results_h0 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ {hyperparameter_col} + {data_prop_col_m}",
                groups=self.input_id,
                data=model_data,
            ).fit(reml=False)
        )
        print("Fitting H1-model.")
        # model_H1 = Lmer(
        #     formula=f"{self.metric} ~ {hyperparameter_col} + {data_prop_col_m} + {hyperparameter_col}:{data_prop_col_m} + ( 1 | {self.input_id} )",
        #     data=model_data,
        #     family=self.distribution,
        # )
        # model_H1.fit(factors=model_factors, REML=False, summarize=False)
        results_h1 = LinearMixedEffectsModelResults.from_mixed_model(
            mixedlm(
                f"{self.metric} ~ {hyperparameter_col} + {data_prop_col_m} + {hyperparameter_col}:{data_prop_col_m}",
                data=model_data,
                groups=self.input_id,
            ).fit(reml=False)
        )
        # results_h1 = LinearMixedEffectsModelResults.from_lmer_model(model_H1)

        # compare models and calculate postHoc
        glrt = generalized_likelihood_ratio_test(results_h0, results_h1)

        # FOR CATEGORICAL data property!!!!
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            postHoc_result = results_h1.compute_marginal_effects(
                marginal_vars=hyperparameter_col,
                grouping_vars=data_prop_col,
                grouping_type=reported_estimates,  # type: ignore[arg-type]
            )

        if is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = results_h1.compute_marginal_effects(
                marginal_vars=data_prop_col,
                grouping_vars=hyperparameter_col,
                grouping_type=reported_estimates,  # type: ignore[arg-type]
            )

        # simplify postHoc result
        means = pd.DataFrame()
        slopes = pd.DataFrame()
        if reported_estimates == "means":
            means = (
                postHoc_result[0]
                .drop(columns="DF", errors="ignore")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )
        else:
            slopes = (
                postHoc_result[0]
                .drop(columns="DF", errors="ignore")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )

        contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            contrasts = contrasts.assign(
                Effect_size_g=lambda df: (
                    df.Estimate / results_h1.residual_standard_deviation
                )
            )

        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            interaction_plot = (
                ggplot(postHoc_result[0])
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="top",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Estimated Expectation of Evaluation Metric")
                + geom_pointrange(
                    aes(
                        x=data_prop_col,
                        y="Estimate",
                        ymin="Estimate - SE",
                        ymax="Estimate + SE",
                        colour=hyperparameter_col,
                    ),
                    alpha=0.7,
                )
                + geom_line(
                    aes(
                        x=data_prop_col,
                        y="Estimate",
                        group=hyperparameter_col,
                        colour=hyperparameter_col,
                    ),
                    alpha=0.3,
                )
            )

        if is_numeric_dtype(model_data[data_prop_col]):
            interaction_plot = (
                ggplot(data=model_data)
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="top",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Estimated Expected Evaluation Metric")
                + geom_smooth(
                    aes(
                        x=data_prop_col,
                        y=self.metric,
                        group=hyperparameter_col,
                        linetype=hyperparameter_col,
                    ),
                    method="lm",
                    colour="black",
                    se=False,
                )
            )

        if glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        return ConditionalHyperParameterAssessment(
            algorithm=algorithm_id,
            glrt=glrt,
            contrasts=contrasts,
            data_property=data_prop_col,
            means=means,
            slopes=slopes,
            interaction_plot=interaction_plot,
        )

    def conditional_system_comparison_plot(
        self, data_prop_col: str, row_filter: str = ""
    ) -> ggplot:
        # minimize data to speed processing and removing rows with missing values
        if row_filter:
            model_data = self.data.query(row_filter).copy()
            model_data = model_data[
                [self.system, self.metric, data_prop_col, self.input_id]
            ].dropna()
        else:
            model_data = self.data[
                [self.system, self.metric, data_prop_col, self.input_id]
            ].dropna()

        model_data[self.system] = model_data[self.system].cat.remove_unused_categories()
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            model_data[data_prop_col] = model_data[
                data_prop_col
            ].cat.remove_unused_categories()

        if is_numeric_dtype(model_data[data_prop_col]):
            descriptive_plot = (
                ggplot(
                    data=model_data,
                    mapping=aes(x=data_prop_col, y=self.metric, color=self.system),
                )
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="none",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Evaluation Metric")
                + facet_wrap("system")
                + geom_point(alpha=0.01)
                + geom_smooth(method="loess", se=False)
            )
        elif isinstance(
            model_data[data_prop_col].dtype, pd.CategoricalDtype
        ) or is_string_dtype(model_data[data_prop_col]):
            descriptive_plot = (
                ggplot(
                    data=model_data,
                    mapping=aes(x=data_prop_col, y=self.metric, fill=self.system),
                )
                + theme_bw()
                + theme(
                    panel_grid=element_blank(),
                    legend_position="none",
                    legend_title=element_blank(),
                )
                + xlab(data_prop_col)
                + ylab("Evaluation Metric")
                + geom_boxplot(alpha=0.3)
            )
        else:
            raise ValueError("No plot defined for the data type of data_prop_var.")

        return descriptive_plot
