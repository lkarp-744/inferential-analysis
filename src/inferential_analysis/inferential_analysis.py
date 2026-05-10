# ruff: noqa
"""
This module provides tools for conducting inferential reproducibility analysis.

The original implementation is sourced from the StatNLP repository on GitHub:
https://github.com/StatNLP/empirical_methods/blob/master/inferential_reproducibility/inferential_analysis.py.

Modifications have been made to ensure compliance with our code quality standards.

This module is licensed under the Apache 2.0 License, given by the original authors.
"""

# The authors of pymer4 recommend to add the following lines when pymer is run inside a jupyter notebook.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Literal, TypedDict

import pandas as pd
from dataclasses import dataclass, field
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

GLRT = TypedDict("GLRT", {"p": float, "chi_square": float, "df": int})
Distribution = Literal["gaussian", "binomial"]


@dataclass(kw_only=True)
class SystemComparison:
    glrt: GLRT
    means: pd.DataFrame
    contrasts: pd.DataFrame


@dataclass(kw_only=True)
class ConditionalSystemComparison:
    glrt: GLRT
    means: pd.DataFrame
    slopes: pd.DataFrame
    contrasts: pd.DataFrame
    interaction_plot: ggplot
    data_property: str


@dataclass(kw_only=True)
class Reliability:
    algorithm: str
    icc: pd.DataFrame


class HyperParameterAssessment:
    def __init__(self) -> None:
        self.algorithm = ""
        self.glrt: GLRT = dict(p=1.0, chi_square=0.0, df=0)
        self.means = pd.DataFrame()
        self.contrasts = pd.DataFrame()


class ConditionalHyperParameterAssessment:
    def __init__(self) -> None:
        self.algorithm = ""
        self.glrt: GLRT = dict(p=1.0, chi_square=0.0, df=0)
        self.means = pd.DataFrame()
        self.slopes = pd.DataFrame()
        self.contrasts = pd.DataFrame()
        self.interaction_plot = ggplot()
        self.data_property = ""


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
        self.HyperParameterAssessment = HyperParameterAssessment()
        self.ConditionalHyperParameterAssessment = ConditionalHyperParameterAssessment()

        # check input consistency
        if distribution == "gaussian":
            if not is_numeric_dtype(self.data[self.metric]):
                raise ValueError(
                    "Data type of provided evaluation metric"
                    + self.metric
                    + " is not numerical!"
                )
        elif distribution == "binomial":
            if not len(self.data[self.metric].unique()) == 2:
                raise ValueError(
                    "Provided evaluation metric data column "
                    + self.metric
                    + " has more or less than 2 values. You can't run a binomial model."
                )
        else:
            raise ValueError("You have choosen an currently unsupported distribution.")

        # make sure that self.system and input_identifer_var are proper categoricals
        if not isinstance(self.data[self.input_id].dtype, pd.CategoricalDtype):
            self.data = self.data.astype({self.input_id: "string"})
            self.data = self.data.astype({self.input_id: "categorical"})

        if not isinstance(self.data[self.system].dtype, pd.CategoricalDtype):
            print(
                "WARNING: "
                + self.system
                + " is not categorical! Datatype will be converted."
            )
            self.data = self.data.astype({self.system: "string"})
            self.data = self.data.astype({self.system: "categorical"})

        # make sure that self.system categories are all strings. This is important for Lmer.
        self.data[self.system] = self.data[self.system].cat.rename_categories(
            lambda x: str(x)
        )

    def GLRT(self, mod1: Lmer, mod2: Lmer) -> GLRT:
        chi_square = 2 * abs(mod1.logLike - mod2.logLike)
        delta_params = abs(len(mod1.coefs) - len(mod2.coefs))

        return {
            "chi_square": chi_square,
            "df": delta_params,
            "p": 1 - chi2.cdf(chi_square, df=delta_params),
        }

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
        formula_H1 = f"{self.metric} ~ {self.system} + ( 1 | {self.input_id} )"
        formula_H0 = f"{self.metric} ~ ( 1 | {self.input_id} )"

        model_H1 = Lmer(formula=formula_H1, data=model_data, family=self.distribution)
        model_H0 = Lmer(formula=formula_H0, data=model_data, family=self.distribution)

        model_factors = {}
        model_factors[self.system] = [s for s in model_data[self.system].cat.categories]

        print("Fitting H0-model.")
        model_H0.fit(REML=False, summarize=False)
        print("Fitting H1-model.")
        model_H1.fit(factors=model_factors, REML=False, summarize=False)

        # compare models via GLRT
        glrt = self.GLRT(model_H0, model_H1)

        # create means and contasts
        postHoc_result = [r for r in model_H1.post_hoc(marginal_vars=self.system)]

        means = (
            postHoc_result[0]
            .drop(columns="DF")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )
        contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hodge's g derivate) to contrasts
        sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
        contrasts = contrasts.assign(
            Effect_size_g=lambda df: df.Estimate / sigma_residuals
        )

        if glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        return SystemComparison(glrt=glrt, means=means, contrasts=contrasts)

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
                "Data property is a numeric variable. Applying indivdual trends model and reporting slopes."
            )
            reported_estimates = "slopes"
        else:
            raise ValueError(
                "Data property column "
                + data_prop_col
                + " data type is neither numeric nor categorical/string."
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
            data_prop_col_m = "scale(" + data_prop_col + ")"
        else:
            data_prop_col_m = data_prop_col

        formula_H1 = (
            self.metric
            + " ~ "
            + self.system
            + " + "
            + data_prop_col_m
            + " + "
            + self.system
            + ":"
            + data_prop_col_m
            + " + ( 1 | "
            + self.input_id
            + " )"
        )
        formula_H0 = (
            self.metric
            + " ~ "
            + self.system
            + " + "
            + data_prop_col_m
            + " + ( 1 | "
            + self.input_id
            + " )"
        )

        model_H1 = Lmer(formula=formula_H1, data=model_data, family=self.distribution)
        model_H0 = Lmer(formula=formula_H0, data=model_data, family=self.distribution)

        model_factors = {}
        model_factors[self.system] = [s for s in self.data[self.system].cat.categories]

        if isinstance(self.data[data_prop_col].dtype, pd.CategoricalDtype):
            model_data[data_prop_col] = model_data[
                data_prop_col
            ].cat.remove_unused_categories()
            model_factors[data_prop_col] = [
                p for p in model_data[data_prop_col].cat.categories
            ]

        print("Fitting H0-model.")
        model_H0.fit(factors=model_factors, REML=False, summarize=False)
        print("Fitting H1-model.")
        model_H1.fit(factors=model_factors, REML=False, summarize=False)

        # compare models and calculate postHoc
        glrt = self.GLRT(model_H0, model_H1)

        # FOR CATEGORICAL data property!!!!
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            postHoc_result = [
                r
                for r in model_H1.post_hoc(
                    marginal_vars=self.system, grouping_vars=data_prop_col
                )
            ]

        if is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = [
                r
                for r in model_H1.post_hoc(
                    marginal_vars=data_prop_col, grouping_vars=self.system
                )
            ]

        # simplify postHoc result
        means = pd.DataFrame()
        slopes = pd.DataFrame()
        if reported_estimates == "means":
            means = (
                postHoc_result[0]
                .drop(columns="DF")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )
        else:
            slopes = (
                postHoc_result[0]
                .drop(columns="DF")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )

        contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
            contrasts = contrasts.assign(
                Effect_size_g=lambda df: df.Estimate / sigma_residuals
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
                print(
                    "WARNING: " + c + " is not categorical! Datatype will be converted."
                )
                self.data = self.data.astype({c: "string"})
                self.data = self.data.astype({c: "category"})

        # minimize data to speed processing and removing rows with missing values
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()

        if row_filter:
            model_data = model_data.query(row_filter).copy()

        model_data = model_data[[self.metric] + var_components].dropna()

        # instantiate and fit models
        formula_var_decomposition_model = (
            self.metric + " ~ " + " + ".join([f"( 1 | {c})" for c in var_components])
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
    ) -> None:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # make sure that hyperparameter_col and input_identifer_var are proper categoricals
        if not isinstance(self.data[hyperparameter_col].dtype, pd.CategoricalDtype):
            print(
                "WARNING: "
                + hyperparameter_col
                + " is not categorical! Datatype will be converted."
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
        formula_H1 = f"{self.metric} ~ {hyperparameter_col} + ( 1 | {self.input_id} )"
        formula_H0 = f"{self.metric} ~ (1 | {self.input_id})"

        model_H1 = Lmer(formula=formula_H1, data=model_data, family=self.distribution)
        model_H0 = Lmer(formula=formula_H0, data=model_data, family=self.distribution)

        model_factors = {}
        model_factors[hyperparameter_col] = [
            s for s in model_data[hyperparameter_col].cat.categories
        ]

        print("Fitting H0-model.")
        model_H0.fit(REML=False, summarize=False)
        print("Fitting H1-model.")
        model_H1.fit(factors=model_factors, REML=False, summarize=False)

        # compare models and calculate postHoc stats
        self.HyperParameterAssessment.glrt = self.GLRT(model_H0, model_H1)
        postHoc_result = [
            r for r in model_H1.post_hoc(marginal_vars=hyperparameter_col)
        ]

        # simplify postHoc result
        self.HyperParameterAssessment.means = (
            postHoc_result[0]
            .drop(columns="DF")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )
        self.HyperParameterAssessment.contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hedge's g derivate) to mean model contrasts
        sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
        self.HyperParameterAssessment.contrasts = (
            self.HyperParameterAssessment.contrasts.assign(
                Effect_size_g=lambda df: df.Estimate / sigma_residuals
            )
        )

        if self.HyperParameterAssessment.glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        self.HyperParameterAssessment.algorithm = algorithm_id

    def conditional_hyperparameter_assessment(
        self,
        algorithm_id: str,
        hyperparameter_col: str,
        data_prop_col: str,
        alpha: float = 0.05,
        scale_data_prop: bool = False,
        verbose: bool = True,
        row_filter: str = "",
    ) -> None:
        # check input consistency
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        # make sure that hyperparameter_col is a proper categorical
        if not isinstance(self.data[hyperparameter_col].dtype, pd.CategoricalDtype):
            print(
                "WARNING: "
                + hyperparameter_col
                + " is not categorical! Datatype will be converted."
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
                "Data property is a numeric variable. Applying indivdual trends model and reporting slopes."
            )
            reported_estimates = "slopes"
        else:
            raise ValueError(
                "Data property column "
                + data_prop_col
                + " data type is neither numeric nor categorical/string."
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
        if scale_data_prop:
            data_prop_col_m = "scale(" + data_prop_col + ")"
        else:
            data_prop_col_m = data_prop_col

        formula_H1 = f"{self.metric} ~ {hyperparameter_col} + {data_prop_col_m} + {hyperparameter_col}:{data_prop_col_m} + ( 1 | {self.input_id} )"
        formula_H0 = f"{self.metric} ~ {hyperparameter_col} + {data_prop_col_m} + ( 1 | {self.input_id} )"

        model_H1 = Lmer(formula=formula_H1, data=model_data, family=self.distribution)
        model_H0 = Lmer(formula=formula_H0, data=model_data, family=self.distribution)

        model_factors = {}
        model_factors[hyperparameter_col] = [
            s for s in model_data[hyperparameter_col].cat.categories
        ]

        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            model_data[data_prop_col] = model_data[
                data_prop_col
            ].cat.remove_unused_categories()
            model_factors[data_prop_col] = [
                p for p in model_data[data_prop_col].cat.categories
            ]

        print("Fitting H0-model.")
        model_H0.fit(factors=model_factors, REML=False, summarize=False)
        print("Fitting H1-model.")
        model_H1.fit(factors=model_factors, REML=False, summarize=False)

        # compare models and calculate postHoc
        self.ConditionalHyperParameterAssessment.glrt = self.GLRT(model_H0, model_H1)

        # FOR CATEGORICAL data property!!!!
        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            postHoc_result = [
                r
                for r in model_H1.post_hoc(
                    marginal_vars=hyperparameter_col, grouping_vars=data_prop_col
                )
            ]

        if is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = [
                r
                for r in model_H1.post_hoc(
                    marginal_vars=data_prop_col, grouping_vars=hyperparameter_col
                )
            ]

        # simplify postHoc result
        if reported_estimates == "means":
            self.ConditionalHyperParameterAssessment.means = (
                postHoc_result[0]
                .drop(columns="DF")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )
        else:
            self.ConditionalHyperParameterAssessment.slopes = (
                postHoc_result[0]
                .drop(columns="DF")
                .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
            )

        self.ConditionalHyperParameterAssessment.contrasts = (
            postHoc_result[1]
            .drop(columns=["DF", "T-stat", "Z-stat", "Sig"], errors="ignore")
            .rename(columns={"2.5_ci": "95CI_lo", "97.5_ci": "95CI_up"})
        )

        # add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
            self.ConditionalHyperParameterAssessment.contrasts = (
                self.ConditionalHyperParameterAssessment.contrasts.assign(
                    Effect_size_g=lambda df: df.Estimate / sigma_residuals
                )
            )

        if isinstance(model_data[data_prop_col].dtype, pd.CategoricalDtype):
            self.ConditionalHyperParameterAssessment.interaction_plot = (
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
            self.ConditionalHyperParameterAssessment.interaction_plot = (
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

        if self.ConditionalHyperParameterAssessment.glrt["p"] <= alpha and verbose:
            print(
                "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons."
            )
        elif verbose:
            print(
                "GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems."
            )

        self.ConditionalHyperParameterAssessment.algorithm = algorithm_id
        self.ConditionalHyperParameterAssessment.data_property = data_prop_col

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
                # + geom_density_2d(alpha = .3)
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
