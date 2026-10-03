"""Aggregate ordered slice predictions into side-level and case-level grades."""

from __future__ import annotations

from collections.abc import Sequence


def aggregate_side(
    slice_grades: Sequence[int],
    minimum_consecutive: int = 2,
    slice_orders: Sequence[int] | None = None,
) -> int:
    if minimum_consecutive < 1:
        raise ValueError("minimum_consecutive must be positive")
    if slice_orders is not None and len(slice_orders) != len(slice_grades):
        raise ValueError("slice_orders and slice_grades must have equal length")
    valid = {0}
    previous = None
    previous_order = None
    run = 0
    orders = slice_orders if slice_orders is not None else range(len(slice_grades))
    for order, grade in zip(orders, slice_grades):
        order = int(order)
        grade = int(grade)
        if grade < 0 or grade > 4:
            raise ValueError(f"Grade must be in [0, 4], got {grade}")
        if grade == previous and (previous_order is None or order == previous_order + 1):
            run += 1
        else:
            previous, run = grade, 1
        previous_order = order
        if grade > 0 and run >= minimum_consecutive:
            valid.add(grade)
    return max(valid)


def aggregate_case(
    left_slice_grades: Sequence[int],
    right_slice_grades: Sequence[int],
    minimum_consecutive: int = 2,
    left_slice_orders: Sequence[int] | None = None,
    right_slice_orders: Sequence[int] | None = None,
) -> int:
    left = aggregate_side(left_slice_grades, minimum_consecutive, left_slice_orders)
    right = aggregate_side(right_slice_grades, minimum_consecutive, right_slice_orders)
    return max(left, right)

