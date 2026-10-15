"""Python value adapter for the native GF(2) row-space basis."""
from . import _native


def image_basis(rows):
    return list(_native.image_basis(rows))
