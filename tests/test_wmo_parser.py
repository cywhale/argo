import pytest

from cli.odbargo_cli import parse_wmo_input


def test_parse_wmo_input_single():
    assert parse_wmo_input("5903377") == [5903377]


def test_parse_wmo_input_commas_with_spaces():
    assert parse_wmo_input("5903377, 5903594,5904000") == [5903377, 5903594, 5904000]


def test_parse_wmo_input_invalid_text():
    with pytest.raises(ValueError):
        parse_wmo_input("view open tmp/argo.nc")


def test_parse_wmo_input_bad_separator():
    with pytest.raises(ValueError):
        parse_wmo_input("5903377 5903594")
