from nero_pi05_bridge import msgpack_numpy
import numpy as np


def test_numpy_round_trip():
    value = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.asarray([1.0, 2.0], dtype=np.float64),
    }
    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(value))
    np.testing.assert_array_equal(unpacked["image"], value["image"])
    np.testing.assert_array_equal(unpacked["state"], value["state"])
