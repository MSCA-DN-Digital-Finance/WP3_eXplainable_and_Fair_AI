import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import sys
import os


# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from prediction.adapters import chronos_input_adapter, chronos_output_adapter, timesfm_input_adapter, timesfm_output_adapter


############ CHRONOS INPUT ADAPTER TESTS ############

chronos_input_testdata = [
    # Tensor is (Samples, Time Steps, Dimensions)


    # Case 0: 1 sample, 1 dimension
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]]]), # input sample
        pd.DataFrame({ 
            "item_id": [0, 0, 0, 0, 0],
            "timestamp": pd.date_range("1750-01-01", periods=5, freq="h"),
            "target": [0.0, 1.0, 2.0, 3.0, 4.0]
        })               
    ),
    # Case 1: 2 sample, 1 dimension
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]], [[0.], [1.], [2.], [3.], [4.]]]), # input sample
        pd.DataFrame({ 
            "item_id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "timestamp": np.tile(pd.date_range("1750-01-01", periods=5, freq="h"), 2),
            "target": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        })               
    ),
        # Case 2: 1 sample, 2 dimensions
    (
        np.array([[[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]]]), # input sample
        pd.DataFrame({ 
            "item_id": [0, 0, 0, 0, 0],
            "timestamp": pd.date_range("1750-01-01", periods=5, freq="h"),
            "target": [0.0, 1.0, 2.0, 3.0, 4.0],
            "var1": [0.0, 1.0, 2.0, 3.0, 4.0]
        })               
    ),
]


@pytest.mark.parametrize("input_data,expected_df", chronos_input_testdata)
def test_chronos_input_adapter(input_data, expected_df):

    actual_df = chronos_input_adapter(input_data=input_data)

    assert assert_frame_equal(actual_df, expected_df) == None


############ CHRONOS OUTPUT ADAPTER TESTS ############

chronos_output_testdata = [
    # Case 0: 1 sample, 1 dimension
    (
        pd.DataFrame({ 
            "item_id": [0, 0, 0, 0, 0],
            "timestamp": pd.date_range("1750-01-01", periods=5, freq="h"),
            "predictions": [0.0, 1.0, 2.0, 3.0, 4.0]
        }),
        np.array([[[0.], [1.], [2.], [3.], [4.]]]) # expected output
    ),

    # Case 1: 2 sample, 1 dimension
    (
        pd.DataFrame({ 
            "item_id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "timestamp": np.tile(pd.date_range("1750-01-01", periods=5, freq="h"), 2),
            "predictions": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        }),
        np.array([[[0.], [1.], [2.], [3.], [4.]], [[0.], [1.], [2.], [3.], [4.]]]) # expected output
    )
]

@pytest.mark.parametrize("input_df,expected_array", chronos_output_testdata)
def test_chronos_output_adapter(input_df, expected_array):
    
    actual_array = chronos_output_adapter(pred_df=input_df)

    assert np.array_equal(actual_array, expected_array), "Output array does not match expected array"






############ TIMESFM INPUT ADAPTER TESTS ############


timesfm_input_testdata = [
    # Tensor is (Samples, Time Steps, Dimensions)


    # Case 0: 1 sample, 1 dimension
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]]]), # input sample
        [np.array([0., 1., 2., 3., 4.])]
    ),
    # Case 1: 2 sample, 1 dimension
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]], [[0.], [1.], [2.], [3.], [4.]]]), # input sample
        [np.array([0., 1., 2., 3., 4.]), np.array([0., 1., 2., 3., 4.])]
    )
]


@pytest.mark.parametrize("input_data,expected_output", timesfm_input_testdata)
def test_timesfm_input_adapter(input_data, expected_output):

    actual_output = timesfm_input_adapter(input_data=input_data)

    for i, (actual, expected) in enumerate(zip(actual_output, expected_output)):
        assert np.array_equal(actual, expected), f"Mismatch in sample {i}"



############## TIMESFM OUTPUT ADAPTER TESTS ############

timesfm_output_testdata = [
    # Case 0: 1 sample, 1 dimension
    (
        [np.array([0., 1., 2., 3., 4.])],
        np.array([[[0.], [1.], [2.], [3.], [4.]]])
    ),
    # Case 1: 2 sample, 1 dimension
    (
        [np.array([0., 1., 2., 3., 4.]), np.array([0., 1., 2., 3., 4.])],
        np.array([[[0.], [1.], [2.], [3.], [4.]], [[0.], [1.], [2.], [3.], [4.]]])
    )
]

@pytest.mark.parametrize("input_list,expected_array", timesfm_output_testdata)
def test_timesfm_output_adapter(input_list, expected_array):

    actual_array = timesfm_output_adapter(input_list=input_list)

    assert np.array_equal(actual_array, expected_array), "Output array does not match expected array"
