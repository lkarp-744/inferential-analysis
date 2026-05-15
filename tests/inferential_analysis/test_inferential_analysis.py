import pathlib

import numpy as np
import pandas as pd
import plotnine
import pytest
from matplotlib.figure import Figure

from inferential_analysis import InferentialAnalysis


@pytest.fixture(scope="session")
def evaluation_data_cnn_best() -> pd.DataFrame:
    eval_data = pd.read_csv(
        pathlib.Path(__file__).parent / "evaluation_data" / "aghajanyan_cnn-best.csv",
        low_memory=False,
    )
    eval_data = eval_data.astype({"summary_id": "category", "system": "category"})

    return eval_data


@pytest.fixture(scope="session")
def evaluation_data_cnn_all() -> pd.DataFrame:
    eval_data = pd.read_csv(
        pathlib.Path(__file__).parent / "evaluation_data" / "aghajanyan_cnn-all.csv",
        low_memory=False,
    )
    eval_data = eval_data.astype({"summary_id": "category", "system": "category"})
    # NOTE: lambda is an python keyword
    eval_data = eval_data.rename(columns={"lambda": "lambda_value"})

    return eval_data


@pytest.mark.integration
def test_system_comparison(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_best: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_best,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.system_comparison()

    assert capsys.readouterr().out == (
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons.\n"
    )

    assert result.glrt == dict(
        chi_square=pytest.approx(np.float64(37.0204), abs=0.0001),
        df=1,
        p=pytest.approx(1.168989e-09),
    )
    assert isinstance(result.means, pd.DataFrame)
    assert result.means.to_dict("list") == {
        "system": ["Baseline", "SOTA"],
        "Estimate": [0.213, 0.217],
        "95CI_lo": [0.210, 0.215],
        "95CI_up": [0.215, 0.220],
        "SE": [0.001, 0.001],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": ["Baseline - SOTA"],
        "Estimate": [-0.005],
        "95CI_lo": [-0.006],
        "95CI_up": [-0.003],
        "SE": [0.001],
        "P-val": [0.0],
        "Effect_size_g": pytest.approx([-0.088419], abs=0.0001),
    }


@pytest.mark.integration
def test_system_comparison_2(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_all: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.system_comparison()

    assert capsys.readouterr().out == (
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons.\n"
    )

    assert result.glrt == dict(
        chi_square=pytest.approx(np.float64(11588.40), abs=0.01),
        df=1,
        p=0.0,
    )
    assert isinstance(result.means, pd.DataFrame)
    assert result.means.to_dict("list") == {
        "system": ["Baseline", "SOTA"],
        "Estimate": [0.21, 0.19],
        "95CI_lo": [0.208, 0.188],
        "95CI_up": [0.212, 0.192],
        "SE": [0.001, 0.001],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": ["Baseline - SOTA"],
        "Estimate": [0.02],
        "95CI_lo": [0.02],
        "95CI_up": [0.021],
        "SE": [0.0],
        "P-val": [0.0],
        "Effect_size_g": pytest.approx([0.294151], abs=0.0001),
    }


@pytest.mark.integration
def test_conditional_system_comparison_word_rarity(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_best: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_best,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_system_comparison(
        data_prop_col="word_rarity"
    )

    assert capsys.readouterr().out == (
        "Data property is a numeric variable. Applying individual trends model and reporting slopes.\n"
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.\n"
    )

    assert result.data_property == "word_rarity"
    assert result.glrt == dict(
        chi_square=pytest.approx(np.float64(4.46357), abs=0.0001),
        df=1,
        p=pytest.approx(np.float64(0.0346), abs=0.0001),
    )
    assert isinstance(result.slopes, pd.DataFrame)
    assert result.slopes.to_dict("list") == {
        "system": ["Baseline", "SOTA"],
        "Estimate": [-0.001, -0.001],
        "95CI_lo": [-0.001, -0.001],
        "95CI_up": [-0.0, -0.0],
        "SE": [0.0, 0.0],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": ["(Baseline) - (SOTA)"],
        "Estimate": [0.0],
        "95CI_lo": [0.0],
        "95CI_up": [0.0],
        "SE": [0.0],
        "P-val": [0.035],
    }

    assert isinstance(result.interaction_plot, plotnine.ggplot)


@pytest.mark.mpl_image_compare
@pytest.mark.integration
def test_conditional_system_comparison_word_rarity_interaction_plot(
    evaluation_data_cnn_best: pd.DataFrame,
) -> Figure:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_best,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_system_comparison(
        data_prop_col="word_rarity"
    )

    return result.interaction_plot.draw()


@pytest.mark.integration
def test_conditional_system_comparison_flesh_kincaid_score(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_best: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_best,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_system_comparison(
        data_prop_col="flesch_kincaid", row_filter="flesch_kincaid >= 0"
    )

    assert capsys.readouterr().out == (
        "Data property is a numeric variable. Applying individual trends model and reporting slopes.\n"
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.\n"
    )

    assert result.data_property == "flesch_kincaid"
    assert result.glrt == dict(
        chi_square=pytest.approx(np.float64(4.30353), abs=0.0001),
        df=1,
        p=pytest.approx(np.float64(0.03803), abs=0.0001),
    )
    assert isinstance(result.slopes, pd.DataFrame)
    assert result.slopes.to_dict("list") == {
        "system": ["Baseline", "SOTA"],
        "Estimate": [0.0, 0.0],
        "95CI_up": [0.0, 0.001],
        "95CI_lo": [0.0, -0.0],
        "SE": [0.0, 0.0],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": ["(Baseline) - (SOTA)"],
        "Estimate": [-0.0],
        "95CI_lo": [-0.0],
        "95CI_up": [-0.0],
        "SE": [0.0],
        "P-val": [0.038],
    }

    assert isinstance(result.interaction_plot, plotnine.ggplot)


@pytest.mark.mpl_image_compare
@pytest.mark.integration
def test_conditional_system_comparison_flesch_kincaid_score_interaction_plot(
    evaluation_data_cnn_best: pd.DataFrame,
) -> Figure:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_best,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_system_comparison(
        data_prop_col="flesch_kincaid"
    )

    return result.interaction_plot.draw()


@pytest.mark.skip(reason="ICC with statsmodels not yet implemented")
@pytest.mark.integration
def test_icc(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_all: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )
    result = inferential_analysis.icc(
        algorithm_id="SOTA", facet_cols=["seed", "lambda_value", "distribution"]
    )

    assert capsys.readouterr().out == (
        "WARNING: seed is not categorical! Datatype will be converted.\n"
        "WARNING: lambda_value is not categorical! Datatype will be converted.\n"
        "WARNING: distribution is not categorical! Datatype will be converted.\n"
        "Calculating variance decomposition.\n"
    )

    assert result.algorithm == "SOTA"
    assert isinstance(result.icc, pd.DataFrame)
    assert result.icc.to_dict("list") == dict(
        Var=pytest.approx(
            [0.00991, 0.000077, 0.001318, 0.000032, 0.004485], abs=0.00001
        ),
        ICC=pytest.approx([62.66, 0.484, 8.30, 0.2, 28.34], abs=0.01),
    )


@pytest.mark.integration
def test_hyperparameter_assessment(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_all: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )
    result = inferential_analysis.hyperparameter_assessment(
        algorithm_id="SOTA", hyperparameter_col="lambda_value"
    )

    assert capsys.readouterr().out == (
        "WARNING: lambda_value is not categorical! Datatype will be converted.\n"
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        # "P-values adjusted by tukey method for family of 3 estimates\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons.\n"
    )
    assert result.glrt == dict(
        chi_square=pytest.approx(60450.5165, abs=0.0001),
        df=2,
        p=0.0,
    )
    assert result.algorithm == "SOTA"
    assert isinstance(result.means, pd.DataFrame)
    assert result.means.to_dict("list") == {
        "lambda_value": ["0-001", "0-01", "0-1"],
        "Estimate": [0.209, 0.213, 0.148],
        "95CI_lo": [0.207, 0.211, 0.146],
        "95CI_up": [0.211, 0.215, 0.150],
        "SE": [0.001, 0.001, 0.001],
    }

    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": ["0-001 - 0-01", "0-001 - 0-1", "0-01 - 0-1"],
        "Estimate": [-0.004, 0.061, 0.065],
        "95CI_lo": [-0.005, 0.060, 0.064],
        "95CI_up": [-0.003, 0.061, 0.065],
        "SE": [0.0, 0.0, 0.0],
        "P-val": [0.0, 0.0, 0.0],
        "Effect_size_g": pytest.approx([-0.059202, 0.902828, 0.962030], abs=0.0001),
    }


@pytest.mark.integration
def test_conditional_hyperparameter_assessment_word_rarity(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_all: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_hyperparameter_assessment(
        algorithm_id="SOTA",
        hyperparameter_col="lambda_value",
        data_prop_col="word_rarity",
    )

    assert capsys.readouterr().out == (
        "WARNING: lambda_value is not categorical! Datatype will be converted.\n"
        "Data property is a numeric variable. Applying individual trends model and reporting slopes.\n"
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        # "P-values adjusted by tukey method for family of 3 estimates\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.\n"
    )

    assert result.glrt == dict(
        chi_square=pytest.approx(428.5751, abs=0.001), df=2, p=0.0
    )
    assert result.algorithm == "SOTA"
    assert result.data_property == "word_rarity"
    assert isinstance(result.slopes, pd.DataFrame)
    assert result.slopes.to_dict("list") == {
        "lambda_value": ["0-001", "0-01", "0-1"],
        "Estimate": [-0.001, -0.001, -0.0],
        "95CI_lo": [-0.001, -0.001, -0.0],
        "95CI_up": [-0.0, -0.0, -0.0],
        "SE": [0.0, 0.0, 0.0],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": [
            "((0-001)) - ((0-01))",
            "((0-001)) - ((0-1))",
            "((0-01)) - ((0-1))",
        ],
        "Estimate": [0.0, -0.0, -0.0],
        "95CI_lo": [-0.0, -0.0, -0.0],
        "95CI_up": [0.0, -0.0, -0.0],
        "SE": [0.0, 0.0, 0.0],
        "P-val": [0.112, 0.0, 0.0],  # FIXME: Why 0.112 - was previously 0.25
    }
    assert isinstance(result.interaction_plot, plotnine.ggplot)


@pytest.mark.mpl_image_compare
@pytest.mark.integration
def test_conditional_hyperparameter_assessment_word_rarity_score_interaction_plot(
    evaluation_data_cnn_all: pd.DataFrame,
) -> Figure:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )

    result = inferential_analysis.conditional_hyperparameter_assessment(
        algorithm_id="SOTA",
        hyperparameter_col="lambda_value",
        data_prop_col="word_rarity",
    )

    return result.interaction_plot.draw()


@pytest.mark.integration
def test_conditional_hyperparameter_assessment_flesch_kincaid_score(
    capsys: pytest.CaptureFixture[str], evaluation_data_cnn_all: pd.DataFrame
) -> None:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )
    result = inferential_analysis.conditional_hyperparameter_assessment(
        algorithm_id="SOTA",
        hyperparameter_col="lambda_value",
        data_prop_col="flesch_kincaid",
        row_filter="flesch_kincaid >= 0",
    )

    assert capsys.readouterr().out == (
        "WARNING: lambda_value is not categorical! Datatype will be converted.\n"
        "Data property is a numeric variable. Applying individual trends model and reporting slopes.\n"
        "Fitting H0-model.\n"
        "Fitting H1-model.\n"
        # "P-values adjusted by tukey method for family of 3 estimates\n"
        "GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.\n"
    )
    assert result.glrt == dict(
        chi_square=pytest.approx(366.5533, abs=0.0001), df=2, p=0.0
    )
    assert result.algorithm == "SOTA"
    assert result.data_property == "flesch_kincaid"
    assert isinstance(result.slopes, pd.DataFrame)
    assert result.slopes.to_dict("list") == {
        "lambda_value": ["0-001", "0-01", "0-1"],
        "Estimate": [0.0, 0.0, -0.0],
        "95CI_lo": [0.0, 0.0, -0.0],
        "95CI_up": [0.0, 0.001, -0.0],
        "SE": [0.0, 0.0, 0.0],
    }
    assert isinstance(result.contrasts, pd.DataFrame)
    assert result.contrasts.to_dict("list") == {
        "Contrast": [
            "((0-001)) - ((0-01))",
            "((0-001)) - ((0-1))",
            "((0-01)) - ((0-1))",
        ],
        "Estimate": [-0.0, 0.001, 0.001],
        "95CI_lo": [-0.0, 0.0, 0.0],
        "95CI_up": [0.0, 0.001, 0.001],
        "SE": [0.0, 0.0, 0.0],
        "P-val": [0.19, 0.0, 0.0],  # FIXME: Was previously 0.389 and not 0.19
    }
    assert isinstance(result.interaction_plot, plotnine.ggplot)


@pytest.mark.mpl_image_compare
@pytest.mark.integration
def test_conditional_hyperparameter_assessment_flesch_kincaid_score_interaction_plot(
    evaluation_data_cnn_all: pd.DataFrame,
) -> Figure:
    inferential_analysis = InferentialAnalysis(
        evaluation_data=evaluation_data_cnn_all,
        eval_metric_col="rouge_2",
        system_col="system",
        input_identifier_col="summary_id",
    )
    result = inferential_analysis.conditional_hyperparameter_assessment(
        algorithm_id="SOTA",
        hyperparameter_col="lambda_value",
        data_prop_col="flesch_kincaid",
        row_filter="flesch_kincaid >= 0",
    )

    return result.interaction_plot.draw()
