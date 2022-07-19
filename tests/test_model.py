from mrtools import model


def test_SampleFile():
    """Test instance of SampleFile."""
    sf = model.SampleFile(
        None,
        "/store/test",
        23456,
        entries=12345,
        checksum=0x12345789,
    )
    assert str(sf) == "/store/test"
    assert sf.size == 23456
    assert sf.entries == 12345
    assert sf.checksum == 0x12345789
